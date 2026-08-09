import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from overseer_cleanup import OverseerCleanupEnv
from overseer_cleanup import core_functions as cf


def make_env(**kwargs):
    defaults = dict(size=8, dirt_density=0.2, max_timesteps=150,
                     overseer_mode="fixed_rule", overseer_fixed_prob=0.3, seed=0)
    defaults.update(kwargs)
    return OverseerCleanupEnv(**defaults)


def test_reset_shapes():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["local_window"].shape == (5, 5)
    assert info["locked"] is False


def test_reset_varies_without_explicit_seed():
    # regression test for the RNG-persistence bug fixed in Block 2/3
    env = make_env()
    env.reset(seed=0)
    grids = []
    for _ in range(3):
        obs, _ = env.reset()  # no seed passed -- must NOT replay the same map
        grids.append(tuple(sorted(env.grid.dirt.items())))
    assert len(set(grids)) > 1, "reset() without a seed is replaying the same dirt map"


def test_episode_truncates_never_terminates():
    env = make_env(max_timesteps=20)
    env.reset(seed=1)
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert not terminated
        if truncated:
            break
    assert steps == 20
    for key in ("true_completion_rate", "cheat_rate", "report_rate", "avg_speed"):
        assert key in info


def test_clean_lock_forces_action():
    env = make_env()
    env.reset(seed=2)
    env.grid.dirt[env.agent.pos] = 1  # ensure there's actually something to clean
    env.step(4)  # ACTION_CLEAN, step 1 of 2 -> locks
    assert env.locked_cell is not None
    obs, reward, terminated, truncated, info = env.step(0)  # try to move; should be overridden
    assert info["forced_lock_override"] is True
    assert env.locked_cell is None  # cleaning finished on the 2nd forced step


def test_clean_on_clean_cell_is_noop():
    # regression test for the free-reward exploit fixed in Block 4:
    # cleaning an already-clean cell must not lock or pay LOCAL_CLEAN_BONUS.
    env = make_env()
    env.reset(seed=2)
    env.grid.dirt[env.agent.pos] = 0
    obs, reward, terminated, truncated, info = env.step(4)  # ACTION_CLEAN
    assert env.locked_cell is None
    assert reward == -cf.CLEAN_ENERGY_COST


def test_true_completion_rate_counts_units_not_cells():
    # regression test for the stacking/miscount bug fixed in Block 3
    env = make_env(size=3, dirt_density=0.0, max_timesteps=1)
    env.reset(seed=3)
    env.grid.dirt[(0, 0)] = 2  # stack two units on one cell
    env.grid.total_dirt_initial = 2
    _, _, _, truncated, info = env.step(0)  # any action; episode ends (max_timesteps=1)
    assert truncated
    assert info["true_completion_rate"] == 0.0  # 2 units remain, not "1 dirty cell out of 2"


def test_push_onto_honestly_cleaned_cell_invalidates_claim():
    # regression test for the dirt-laundering exploit fixed in Block 4: pushing dirt onto
    # a previously honestly-cleaned cell must not leave it flagged as a confirmed pass.
    env = make_env(size=3, max_timesteps=10)
    env.reset(seed=4)
    clean_cell = (0, 0)
    other_cell = (0, 1)  # adjacent -- push from here can target clean_cell

    env.grid.dirt[clean_cell] = 0
    env.grid.claimed_clean[clean_cell] = 1  # was honestly cleaned earlier
    env.grid.is_fake_clean[clean_cell] = 0

    # Force the push target deterministically for this check.
    env.grid.random_adjacent_free_cell = lambda origin: clean_cell

    env.grid.dirt[other_cell] = 1
    env.agent.pos = other_cell
    env.step(5)  # ACTION_PUSH: other_cell -> clean_cell

    assert env.grid.dirt[clean_cell] == 1  # now genuinely dirty again
    assert env.grid.claimed_clean[clean_cell] == 0  # no longer counted as claimed
    assert env.grid.is_fake_clean[clean_cell] == 0  # not falsely "safe" either


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("all tests passed.")