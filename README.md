# Overseer-Cleanup Environment: Demonstrating Specification Gaming & Goodhart's Law

A custom Gym-style environment for studying **specification gaming** and **Goodhart's Law** through the lens of principal-agent dynamics. Built for Autonomous and Adaptive Systems.

---

## Problem Statement

This is the project's scenario :
- A **cleaning agent** must clear dirt from a grid, rewarded based on a **proxy signal** (what appears clean to sensors).
- The agent can either **honestly clean** dirt or **cheat** by hiding dirt (moving it out of view).
- A separate **overseer agent** runs probabilistic audits to catch cheating.
- Question: How do different audit strategies affect whether the agent behaves honestly or strategically exploits the proxy reward?

This models real-world principal-agent problems (ontent moderation, safety testing, scientific oversight) where the thing being measured may not align with true objectives.

---

## Core Environment: OverseerCleanupEnv

### Grid & Setup
- **N × N grid** (default N=8), episodic with max 150 timesteps
- **~20% of grid cells** are randomly initialized with dirt each episode
- Agent's position is fully observable; dirt positions are **not** (agent only sees a 5×5 local window)
- **No camera position is observable** to the cleaner (adds realistic difficulty)

### Cleaner Agent

**Observation:**
- 6 movements : up, down, left, wrie, clean, push
- Agent's current position on the grid
- Local 5×5 dirt map (presence only, not magnitude)

### Overseer Agent

The overseer runs one of four fixed policies:

**Mechanic:** With probability `audit_prob`, the overseer samples up to 3 cells that the agent has marked as clean and inspects them for ground truth.
- **Catch:** If sampled cells are fake (pushed, not cleaned), penalty applied
- **Miss:** If cells are truly clean, small bonus applied
- **Always:** Audit frequency can adapt based on agent behavior

---

## Experiments

### Experiment 1: Overseer Effectiveness (Main Result)
Compare `true_completion_rate` and `cheat_rate` across the three primary conditions (`random` / `fixed_rule` / `adaptive`) at equal audit budget.

**Hypothesis:** Adaptive oversight yields higher true completion and lower cheat rate for the same total audit cost.

### Experiment 2: Arms-Race Dynamics
Plot `cheat_rate` and `catch_rate` over training epochs. Look for oscillation early, convergence (or lack thereof) later.
---

## Project Structure

```

├── README.md                       # This file
├── configs/
│   └── conditions.yaml             # Overseer condition parameters
├── overseer_cleanup/
│   ├── env.py                      # OverseerCleanupEnv class (Gym wrapper)
│   ├── core_functions.py           # Agent actions, audit logic, OverseerState
│   └── grid.py                     # Grid state, dirt/dirt_units, dirt conservation
├── scripts/
│   └── train.py                    # PPO training loop with monitoring
├── notebooks/
│   ├── position_analysis.ipynb     # Movement & coverage analysis
│   ├── overseer_cleanup_analysis.ipynb          # Core metrics
│   └── overseer_cleanup_analysis_improved.ipynb # Multi-seed analysis
└── results/
    ├── logs/                       # TensorBoard data
    ├── models/                     # Agent checkpoints
    └── *.csv                       # Metrics & position logs
```

### Quick Start
```bash
# Clone the repo
git clone <your-repo-url>
cd AAS

# Install dependencies (in a virtual environment)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # (or pip install gymnasium stable-baselines3 pandas numpy tensorboard)

# Run a test training
python scripts/train.py --condition adaptive --seed 0 --timesteps 100000
```

---

## Project Structure

```

├── README.md                       # This file
├── configs/
│   └── conditions.yaml             # Overseer condition parameters
├── overseer_cleanup/
│   ├── env.py                      # OverseerCleanupEnv class (Gym wrapper)
│   ├── core_functions.py           # Agent actions, audit logic, OverseerState
│   └── grid.py                     # Grid state, dirt/dirt_units, dirt conservation
├── scripts/
│   └── train.py                    # PPO training loop with monitoring
├── notebooks/
│   ├── position_analysis.ipynb     # Movement & coverage analysis
│   ├── overseer_cleanup_analysis.ipynb          # Core metrics
│   └── overseer_cleanup_analysis_improved.ipynb # Multi-seed analysis
└── results/
    ├── logs/                       # TensorBoard data
    ├── models/                     # Agent checkpoints
    └── *.csv                       # Metrics & position logs
```





