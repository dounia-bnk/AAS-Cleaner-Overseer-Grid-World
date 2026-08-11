"""
Block 4: single-condition, single-seed PPO training for the cleaner agent.

Only the cleaner is RL-trained (project_spec.md: overseer is fixed-rule, not
learned, in this build). One run end-to-end before scaling to the full
condition x seed sweep in run_experiment.py.

Usage:
    python scripts/train.py --condition adaptive --seed 0
    python scripts/train.py --condition fixed_rule --seed 3 --timesteps 200000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# make the overseer_cleanup package importable when run as `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from overseer_cleanup import OverseerCleanupEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "conditions.yaml"


def load_config(condition: str) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if condition not in cfg["conditions"]:
        raise ValueError(
            f"Unknown condition '{condition}'. Choose from {list(cfg['conditions'])}."
        )
    return cfg


def make_env(cfg: dict, condition: str, seed: int, log_dir: Path):
    cond_cfg = cfg["conditions"][condition]
    env = OverseerCleanupEnv(
        size=cfg["env"]["size"],
        dirt_density=cfg["env"]["dirt_density"],
        max_timesteps=cfg["env"]["max_timesteps"],
        overseer_mode=cond_cfg["overseer_mode"],
        overseer_fixed_prob=cond_cfg["overseer_fixed_prob"] or 0.3,  # unused unless fixed_rule
        seed=seed,
    )
    # Monitor logs per-episode reward/length and writes the audit/completion
    # info fields to results/logs/<run_name>/monitor.csv for later analysis.
    env = Monitor(env, filename=str(log_dir / "monitor.csv"),
                  info_keywords=("true_completion_rate", "cheat_rate", "report_rate", "audit_prob"))
    return env


def train(condition: str, seed: int, timesteps_override: Optional[int] = None, overwrite: bool = False):
    cfg = load_config(condition)
    train_cfg = cfg["training"]
    total_timesteps = timesteps_override or train_cfg["total_timesteps"]

    run_name = f"{condition}_seed{seed}"
    log_dir = REPO_ROOT / "results" / "logs" / run_name
    model_dir = REPO_ROOT / "results" / "models" / run_name

    final_path = model_dir / "final_model.zip"
    if final_path.exists() and not overwrite:
        raise FileExistsError(
            f"{final_path} already exists. Pass --overwrite to replace it, "
            f"or use a different --seed."
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, condition, seed, log_dir)

    model = PPO(
        train_cfg["policy"],
        env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(log_dir),
        **train_cfg["ppo_kwargs"],
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1),
        save_path=str(model_dir),
        name_prefix="ppo_cleaner",
    )

    print(f"[train] condition={condition} seed={seed} total_timesteps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb, progress_bar=True)

    model.save(str(final_path))
    print(f"[train] saved final model to {final_path}")

    env.close()
    return final_path


def parse_args():
    p = argparse.ArgumentParser(description="Train the cleaner agent for one overseer condition/seed.")
    p.add_argument("--condition", required=True,
                    choices=["random", "fixed_rule", "adaptive", "targeted"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=None,
                    help="Override total_timesteps from configs/conditions.yaml")
    p.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing final_model.zip for this condition/seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.condition, args.seed, args.timesteps, args.overwrite)