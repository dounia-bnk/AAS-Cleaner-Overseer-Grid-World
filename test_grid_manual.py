import random
import sys
sys.path.insert(0, "/home/claude/overseer_cleanup")
sys.path.insert(0, "/mnt/user-data/uploads")  # core_functions.py lives here

from grid import Grid, Agent
from core_functions import push_action, clean_action

rng = random.Random(42)
grid = Grid(size=8, dirt_density=0.20, rng=rng)

print(f"Grid size: {grid.size}, total_dirt_initial: {grid.total_dirt_initial} "
      f"(~{grid.total_dirt_initial / 64:.0%} of 64 cells)")

# find a dirty cell to start the agent on
start = next(c for c, v in grid.dirt.items() if v == 1)
agent = Agent(start)
print(f"\nAgent starts at {agent.pos} (dirty: {grid.dirt[agent.pos] == 1})")
print(grid.render_ascii(agent.pos))

# --- test push_action ---
reward = push_action(agent, grid)
print(f"\nAfter push at {agent.pos}: reward={reward}")
assert grid.dirt[start] == 0, "origin should no longer have dirt"
assert grid.claimed_clean[start] == 1, "origin should be claimed clean"
assert grid.is_fake_clean[start] == 1, "origin should be flagged fake"
print("push_action: OK (dirt relocated, origin flagged fake-clean)")

live_dirt_now = sum(grid.dirt.values())
print(f"Live dirt cells now: {live_dirt_now} (was {grid.total_dirt_initial} at episode start)")
if live_dirt_now != grid.total_dirt_initial:
    print("  NOTE: not conserved -- push landed on an already-dirty neighbor, "
          "see writeup below the test output.")

# --- test clean_action over 2 steps ---
start2 = next(c for c, v in grid.dirt.items() if v == 1)
agent2 = Agent(start2)
progress = {}
r1, done1 = clean_action(agent2, grid, progress)
print(f"\nClean step 1 at {agent2.pos}: reward={r1}, done={done1}")
assert not done1
r2, done2 = clean_action(agent2, grid, progress)
print(f"Clean step 2 at {agent2.pos}: reward={r2}, done={done2}")
assert done2
assert grid.dirt[start2] == 0
assert grid.claimed_clean[start2] == 1
assert grid.is_fake_clean[start2] == 0
print("clean_action: OK (dirt truly removed after 2 steps, not flagged fake)")

# --- test bounds / neighbors ---
corner = (0, 0)
neighbors = grid.neighbors(corner)
print(f"\nNeighbors of corner {corner}: {neighbors}")
assert all(grid.in_bounds(n) for n in neighbors)
assert len(neighbors) == 2, "corner cell should have exactly 2 in-bounds neighbors"
print("neighbors/in_bounds: OK")

# --- test local_window padding ---
win = grid.local_window((0, 0), radius=2)
print(f"\nlocal_window around (0,0), radius=2 (should show -1 padding for off-grid cells):")
for row in win:
    print(row)
assert win[0][0] == -1, "top-left of window should be off-grid (-1)"
print("local_window: OK (off-grid padded with -1)")

print("\nAll Block 1 checks passed.")
