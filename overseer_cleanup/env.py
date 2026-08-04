"""
Block 2: OverseerCleanupEnv -- Gymnasium environment wiring grid.py (Block 1)
and core_functions.py (reward/audit/suspicion math) together.

Only the cleaner is RL-trained. The overseer runs a fixed `mode`
(core_functions.OverseerState: 'random' / 'fixed_rule' / 'adaptive' / 'targeted'),
not learned via self-play in this build.

Episode structure: always runs to `truncated` at max_timesteps -- never an early
`terminated`. run_audit() + summarize_episode() fire only on that final step, and
the resulting cleaner_adjustment is folded into that last step's reward.

Clean is locked: once started, env.step() ignores whatever action the policy
passes on the agent's committed 2nd (etc.) step and forces ACTION_CLEAN again.
There is no separate "finish cleaning" action -- clean_action's `done_cleaning`
return bool is the completion signal.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from grid import Agent, Grid
import core_functions as cf

# ---------- action indices ----------
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CLEAN = 4
ACTION_PUSH = 5

_MOVE_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

_WINDOW_RADIUS = 2  # 5x5 window


class OverseerCleanupEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=8, dirt_density=0.2, max_timesteps=150,
                 overseer_mode="fixed_rule", overseer_fixed_prob=0.3, seed=None):
        super().__init__()
        self.size = size
        self.dirt_density = dirt_density
        self.max_timesteps = max_timesteps

        # OverseerState persists across reset() calls on this env instance --
        # fixed-rule (not learned), owned/constructed once here.
        self.overseer_state = cf.OverseerState(mode=overseer_mode, fixed_prob=overseer_fixed_prob)

        self.action_space = spaces.Discrete(6)
        window_size = 2 * _WINDOW_RADIUS + 1
        self.observation_space = spaces.Dict({
            "local_window": spaces.Box(low=-1, high=1, shape=(window_size, window_size), dtype=np.int8),
            "position": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
        })

        self._np_random_seed = seed
        self.grid = None
        self.agent = None
        self.cleaning_progress = {}
        self.locked_cell = None  # non-None while mid-clean (lock active)
        self.steps_taken = 0

    # ---------- gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else self._np_random_seed
        import random
        grid_rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        self.grid = Grid(self.size, self.dirt_density, rng=grid_rng)
        start_pos = (self.size // 2, self.size // 2)
        self.agent = Agent(start_pos)
        self.cleaning_progress = {}
        self.locked_cell = None
        self.steps_taken = 0

        obs = self._get_obs()
        info = {"locked": False}
        return obs, info

    def step(self, action):
        # --- clean lock override: mid-clean, ignore policy's action, force CLEAN ---
        forced_lock = False
        if self.locked_cell is not None:
            action = ACTION_CLEAN
            forced_lock = True

        reward = 0.0

        if action in _MOVE_DELTAS:
            dx, dy = _MOVE_DELTAS[action]
            x, y = self.agent.pos
            new_pos = (x + dx, y + dy)
            if self.grid.in_bounds(new_pos):
                self.agent.pos = new_pos
            # out-of-bounds move: no-op, agent stays, no extra cost

        elif action == ACTION_CLEAN:
            r, done_cleaning = cf.clean_action(self.agent, self.grid, self.cleaning_progress)
            reward += r
            if done_cleaning:
                self.locked_cell = None
            else:
                self.locked_cell = self.agent.pos

        elif action == ACTION_PUSH:
            # push cannot happen mid-lock (lock only forces CLEAN), but guard anyway
            reward += cf.push_action(self.agent, self.grid)

        else:
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1
        terminated = False  # no early termination, by design
        truncated = self.steps_taken >= self.max_timesteps

        info = {"locked": self.locked_cell is not None, "forced_lock_override": forced_lock}

        if truncated:
            report_rate, avg_speed = cf.summarize_episode(self.grid, self.steps_taken, self.max_timesteps)
            cleaner_adj, overseer_reward, caught, was_audited = cf.run_audit(self.grid, self.overseer_state)
            self.overseer_state.update_after_episode(report_rate, avg_speed, was_audited, caught)
            reward += cleaner_adj

            true_completion_rate = 1.0 - (sum(1 for v in self.grid.dirt.values() if v > 0) / max(1, self.grid.total_dirt_initial))
            claimed_cells = [c for c, v in self.grid.claimed_clean.items() if v == 1]
            fake_cells = [c for c in claimed_cells if self.grid.is_fake_clean[c] == 1]
            cheat_rate = len(fake_cells) / max(1, len(claimed_cells))

            info.update({
                "audit_was_audited": was_audited,
                "audit_caught_cheating": caught,
                "overseer_reward": overseer_reward,
                "true_completion_rate": true_completion_rate,
                "cheat_rate": cheat_rate,
                "report_rate": report_rate,
                "avg_speed": avg_speed,
            })

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.grid.render_ascii(agent_pos=self.agent.pos)

    # ---------- helpers ----------

    def _get_obs(self):
        window = self.grid.local_window(self.agent.pos, _WINDOW_RADIUS)
        return {
            "local_window": np.array(window, dtype=np.int8),
            "position": np.array(self.agent.pos, dtype=np.int32),
        }


# ---------- smoke test ----------

def _smoke_test(episodes=3, seed=0):
    import random
    env = OverseerCleanupEnv(size=8, dirt_density=0.2, max_timesteps=150,
                              overseer_mode="fixed_rule", overseer_fixed_prob=0.3, seed=seed)
    rng = random.Random(seed)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        assert env.observation_space.contains(obs), f"obs out of space: {obs}"
        total_reward = 0.0
        steps = 0
        last_info = {}

        while True:
            action = rng.randrange(env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs), f"obs out of space at step {steps}: {obs}"
            assert not terminated, "terminated should never fire (design: always truncate)"
            total_reward += reward
            steps += 1
            last_info = info
            if truncated:
                break

        assert steps == env.max_timesteps, f"expected {env.max_timesteps} steps, got {steps}"
        assert "true_completion_rate" in last_info, "final-step info missing audit/summary fields"
        print(f"episode {ep}: steps={steps} total_reward={total_reward:.3f} "
              f"true_completion_rate={last_info['true_completion_rate']:.3f} "
              f"cheat_rate={last_info['cheat_rate']:.3f} "
              f"audited={last_info['audit_was_audited']} caught={last_info['audit_caught_cheating']} "
              f"suspicion_after={env.overseer_state.suspicion:.3f}")

    print("smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
