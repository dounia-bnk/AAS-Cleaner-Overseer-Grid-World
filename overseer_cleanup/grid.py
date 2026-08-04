"""
Block 1: Grid and Agent.

Defines the data structures that core_functions.py expects:
  - grid.dirt[cell]           -> 0/1 ground-truth dirt
  - grid.claimed_clean[cell]  -> 0/1 what the agent reports/appears as
  - grid.is_fake_clean[cell]  -> 0/1 ground-truth flag for audits
  - grid.total_dirt_initial   -> int, dirt count at episode start
  - grid.random_adjacent_free_cell(origin) -> a neighbor cell for push_action
  - agent.pos                 -> (x, y) tuple

No gym/env logic here on purpose -- keep this block small and testable in isolation.
"""

import random


class Agent:
    __slots__ = ("pos",)

    def __init__(self, pos):
        self.pos = pos


class Grid:
    """
    N x N grid of cells, each a (x, y) tuple with 0 <= x, y < size.

    dirt density is randomized at construction time (one Grid per episode --
    call Grid(...) fresh in env.reset(), don't reuse across episodes).
    """

    def __init__(self, size, dirt_density, rng=None):
        self.size = size
        self.rng = rng if rng is not None else random.Random()

        self.dirt = {}
        self.claimed_clean = {}
        self.is_fake_clean = {}

        for x in range(size):
            for y in range(size):
                cell = (x, y)
                self.dirt[cell] = 1 if self.rng.random() < dirt_density else 0
                self.claimed_clean[cell] = 0
                self.is_fake_clean[cell] = 0

        self.total_dirt_initial = sum(self.dirt.values())

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbors(self, pos):
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.in_bounds(c)]

    def random_adjacent_free_cell(self, origin):
        """
        Pick a random in-bounds neighbor to relocate dirt to, for push_action.
        dirt is stackable (push_action accumulates via +=), so any in-bounds
        neighbor is a valid target -- no preference needed, conservation holds
        unconditionally. Falls back to origin itself if no neighbor exists at
        all (shouldn't happen for grid_size >= 2).
        """
        candidates = self.neighbors(origin)
        return self.rng.choice(candidates) if candidates else origin

    def local_window(self, center, radius):
        """
        Ground-truth ternary observation of dirt PRESENCE (0/1) in a
        (2r+1)x(2r+1) window around center, in row-major [y][x] order matching
        typical image/array conventions. Cells outside the grid boundary are
        padded with -1 (so the agent's policy can distinguish 'no dirt' from
        'off the grid'). Note: dirt is stored as a stackable count internally
        (see push_action), but the agent only ever observes presence, never
        the magnitude of a stack.
        """
        cx, cy = center
        size = 2 * radius + 1
        window = [[-1 for _ in range(size)] for _ in range(size)]
        for j, y in enumerate(range(cy - radius, cy + radius + 1)):
            for i, x in enumerate(range(cx - radius, cx + radius + 1)):
                if self.in_bounds((x, y)):
                    window[j][i] = 1 if self.dirt[(x, y)] > 0 else 0
        return window

    def render_ascii(self, agent_pos=None):
        """Quick text render for debugging: A=agent, x=fake-clean, c=claimed-clean, .=dirt, ' '=empty."""
        rows = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                cell = (x, y)
                if agent_pos == cell:
                    row.append("A")
                elif self.is_fake_clean.get(cell):
                    row.append("x")
                elif self.claimed_clean.get(cell):
                    row.append("c")
                elif self.dirt.get(cell):
                    row.append(".")
                else:
                    row.append(" ")
            rows.append("".join(row))
        return "\n".join(rows)
