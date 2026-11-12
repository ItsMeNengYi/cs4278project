"""Simulator-side helpers: building the env and ground truth.

Nothing in here is visible to the agent. The agent only sees what UncertainNavWorld hands it
(lidar scans, noisy odometry, the goal and the arena bounds); ground truth is used for placing
the robot and for scoring.
"""
import contextlib
import io
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pick_nav_reach_env as E  # noqa: E402

BASE_X0 = -2.0                  # world x of the base when its x joint is 0 (_load_agent)
CUBE = 1.0                      # maze block size (grid_xy in _load_maze)
TUCK = np.zeros(15)
TUCK[6:13] = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]   # arm folded over the base


def make_env(seed=0, gui=False, realtime=False):
    """Build PickNavReachEnv, headless by default and without the per-step sleep."""
    os.chdir(ROOT)              # the env loads its assets with relative paths
    with contextlib.redirect_stdout(io.StringIO()):
        env = E.PickNavReachEnv(seed=seed, gui=gui, realtime=realtime)
        env.reset()
    return env


def close_env(env):
    p.disconnect(env.pb_physics_client)


class GroundTruth:
    """Exact maze geometry, read from the env. Used only by the simulator side and for scoring."""

    def __init__(self, env):
        self.cubes = np.asarray(env.cube_positions, dtype=float)[:, :2]
        self.goal = np.asarray(env.goal_pos[:2], dtype=float)
        h = CUBE / 2
        self.boxes = np.column_stack([self.cubes - h, self.cubes + h])     # (N, 4): xmin, ymin, xmax, ymax
        self.bounds = (self.boxes[:, 0].min(), self.boxes[:, 1].min(),
                       self.boxes[:, 2].max(), self.boxes[:, 3].max())

    def entrance_xy(self):
        """Centre of the gap in the maze's left border wall (first column after the room with > 2 walls)."""
        xs = np.unique(np.round(self.cubes[:, 0], 3))
        ys = np.unique(np.round(self.cubes[:, 1], 3))
        for x in xs[1:]:
            col_ys = np.round(self.cubes[np.isclose(self.cubes[:, 0], x), 1], 3)
            if len(col_ys) > 2:
                free = np.setdiff1d(ys, col_ys)
                return np.array([x, free[0]])
        raise RuntimeError("maze entrance not found")

    def occupancy(self, spec):
        """Boolean grid on `spec`: True where the cell centre lies inside a wall block."""
        cx, cy = spec.centres()
        occ = np.zeros((spec.nx, spec.ny), dtype=bool)
        for xmin, ymin, xmax, ymax in self.boxes:
            ix = (cx > xmin) & (cx < xmax)
            iy = (cy > ymin) & (cy < ymax)
            occ[np.ix_(ix, iy)] = True
        return occ

    def raycast(self, origin, dirs, max_range):
        """Exact 2D distance from `origin` along each unit vector in `dirs` to the first wall (inf if none)."""
        o = np.asarray(origin, float)
        d = np.asarray(dirs, float)[:, None, :]                       # (B, 1, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (self.boxes[None, :, :2] - o) * inv                  # (B, N, 2)
            t2 = (self.boxes[None, :, 2:] - o) * inv
        t_near = np.nanmax(np.minimum(t1, t2), axis=2)
        t_far = np.nanmin(np.maximum(t1, t2), axis=2)
        hit = (t_far >= t_near) & (t_near >= 0)
        t = np.where(hit, t_near, np.inf).min(axis=1)
        return np.where(t <= max_range, t, np.inf)
