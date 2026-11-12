"""The robot's interface to the world once the oracles are removed.

The agent gets:
- a 2D lidar scan (PyBullet ray casts from the true pose, plus range noise and dropouts),
- odometry: the true displacement with a per-episode scale error and per-step noise,
- velocity-style motion commands: a desired displacement for this step.

The agent never sees qpos[0:2] or cube_positions. Yaw is held at 0 and is exact (the base is
holonomic, so the state being estimated is (x, y)).
"""
import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pybullet as p

from .sim import BASE_X0, TUCK


@dataclass
class NoiseConfig:
    beams: int = 180
    max_range: float = 4.0
    range_sigma: float = 0.02
    dropout: float = 0.01
    ray_start: float = 0.42         # beams start outside the base footprint so they never hit the robot
    ray_z: float = 0.3
    odom_scale_sigma: float = 0.03  # per-episode scale error on each axis (systematic drift)
    odom_sigma_rel: float = 0.05    # per-step noise, proportional to the distance moved
    odom_sigma_abs: float = 0.0005  # per-step noise floor (m)
    init_sigma: float = 0.05        # error of the start pose the agent is told


class UncertainNavWorld:
    def __init__(self, env, noise, start_xy, seed=0):
        self.env = env
        self.cfg = noise
        self.rng = np.random.default_rng(seed)
        self.odom_scale = 1.0 + self.rng.normal(0.0, noise.odom_scale_sigma, 2)

        a = np.linspace(-np.pi, np.pi, noise.beams, endpoint=False)
        self.angles = a
        self.dirs = np.column_stack([np.cos(a), np.sin(a)])

        self.hold_q = TUCK.copy()
        self.hold_q[0], self.hold_q[1] = start_xy[0] - BASE_X0, start_xy[1]
        env._set_qpos(self.hold_q)
        self._last_xy = self.true_xy()
        self.step_count = 0

    # ---- ground truth (simulator side; the agent must not call these) ----
    def true_xy(self):
        q = self.env.agent_qpos
        return np.array([q[0] + BASE_X0, q[1]])

    def in_collision(self):
        return any(c[2] not in (0, self.env.object_id)
                   for c in p.getContactPoints(bodyA=self.env.robot_id))

    # ---- what the agent may use ----
    def noisy_start(self):
        return self.true_xy() + self.rng.normal(0.0, self.cfg.init_sigma, 2)

    def step(self, cmd_xy):
        """Ask the base to move by cmd_xy (m, world frame) this step. Returns the odometry reading."""
        q = self.env.agent_qpos
        action = self.hold_q.copy()
        action[0], action[1] = q[0] + cmd_xy[0], q[1] + cmd_xy[1]
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.step(action)
        self.step_count += 1

        xy = self.true_xy()
        delta, self._last_xy = xy - self._last_xy, xy
        sigma = self.cfg.odom_sigma_rel * np.linalg.norm(delta) + self.cfg.odom_sigma_abs
        return delta * self.odom_scale + self.rng.normal(0.0, sigma, 2)

    def scan(self):
        """Returns (angles, ranges). Range is inf for no return within max_range, nan for a dropout."""
        c = self.cfg
        xy = self.true_xy()
        origin = np.array([xy[0], xy[1], c.ray_z])
        d3 = np.column_stack([self.dirs, np.zeros(len(self.dirs))])
        froms = origin + c.ray_start * d3
        tos = origin + c.max_range * d3
        hits = p.rayTestBatch(froms.tolist(), tos.tolist())

        frac = np.array([h[2] for h in hits])
        hit = np.array([h[0] != -1 for h in hits])
        ranges = np.where(hit, c.ray_start + frac * (c.max_range - c.ray_start), np.inf)
        ranges = ranges + np.where(hit, self.rng.normal(0.0, c.range_sigma, len(ranges)), 0.0)
        ranges[self.rng.random(len(ranges)) < c.dropout] = np.nan
        return self.angles, ranges
