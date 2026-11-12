"""End-to-end checks in PyBullet (headless). Slower: about a minute in total."""
import numpy as np

from nav_uncertainty.agent import AgentConfig, run_episode
from nav_uncertainty.sim import GroundTruth, close_env, make_env
from nav_uncertainty.world import NoiseConfig, UncertainNavWorld


def _episode(seed, **cfg):
    noise = NoiseConfig()
    env = make_env(seed)
    try:
        gt = GroundTruth(env)
        world = UncertainNavWorld(env, noise, gt.entrance_xy(), seed=seed)
        return run_episode(world, gt, AgentConfig(**cfg), noise, seed=seed)[0]
    finally:
        close_env(env)


def test_lidar_matches_geometry():
    """PyBullet ray casts must agree with exact ray/box intersection against the true walls."""
    noise = NoiseConfig(range_sigma=0.0, dropout=0.0)
    env = make_env(0)
    try:
        gt = GroundTruth(env)
        world = UncertainNavWorld(env, noise, gt.entrance_xy(), seed=0)
        angles, ranges = world.scan()
        expected = gt.raycast(world.true_xy(), world.dirs, noise.max_range)
        both = np.isfinite(ranges) & np.isfinite(expected)
        assert both.sum() > 100
        assert np.abs(ranges[both] - expected[both]).max() < 0.02
        assert np.array_equal(np.isfinite(ranges), np.isfinite(expected))
    finally:
        close_env(env)


def test_mapping_dstar_matches_astar():
    """Regression: on this run D* Lite once reported a cost 0.12 below A* after ~100 repairs,
    because float rounding broke key ties (see DStarLite._key). Checks every replan against A*."""
    m = _episode(0, map_mode="unknown", pose_mode="true", check_astar=True)
    assert m["astar_gap"] < 1e-6, m["astar_gap"]
    assert m["success"], m


def test_oracle_reaches_goal():
    m = _episode(1, map_mode="known", pose_mode="true")
    assert m["success"] and m["collision_steps"] == 0, m
