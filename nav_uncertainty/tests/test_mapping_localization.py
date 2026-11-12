"""Mapping, cost grid and particle filter on hand-made walls (no PyBullet needed)."""
import numpy as np

from nav_uncertainty.localization import ParticleFilter
from nav_uncertainty.occupancy import INSCRIBED, GridSpec, LikelihoodField, OccupancyMap, cost_grid
from nav_uncertainty.sim import GroundTruth


def fake_world(boxes):
    gt = GroundTruth.__new__(GroundTruth)          # skip the env; only boxes are needed
    gt.boxes = np.asarray(boxes, float)
    return gt


def beams(n=180):
    a = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return a, np.column_stack([np.cos(a), np.sin(a)])


def test_scan_marks_wall_and_free_space():
    gt = fake_world([[3.0, -2.0, 4.0, 2.0]])        # wall at x = 3..4
    spec = GridSpec(-1, -3, 7, 3, 0.1)
    m = OccupancyMap(spec)
    a, d = beams()
    pose = np.array([1.0, 0.0])
    for _ in range(3):
        m.update(pose, a, gt.raycast(pose, d, 4.0), 4.0)

    def state(x, y):
        ix, iy = spec.to_idx([x, y])
        return m.occupied()[ix, iy], m.known()[ix, iy]

    assert state(3.05, 0.0) == (True, True)         # surface cell, on the wall side
    assert state(2.95, 0.0) == (False, True)        # just in front of the wall: free
    assert state(1.5, 0.5) == (False, True)
    assert state(5.0, 0.0) == (False, False)        # behind the wall: never observed


def test_cost_grid_corridor_profile():
    """A 1 m corridor leaves 4 cells (0.4 m) where the footprint fits, cheapest in the middle."""
    gt = fake_world([[0, 0.5, 10, 1.5], [0, -1.5, 10, -0.5]])
    spec = GridSpec(0, -2, 10, 2, 0.1)
    cost = cost_grid(gt.occupancy(spec), 0.1, 0.32, 0.30)
    ix = spec.to_idx([5.0, 0.0])[0]
    column = cost[ix]
    passable = np.flatnonzero(column < INSCRIBED)
    ys = spec.to_xy(np.full(len(passable), ix), passable)[:, 1]
    ys = np.sort(ys[np.abs(ys) < 0.5])              # inside the corridor (the strips outside the walls also pass)
    assert np.allclose(ys, [-0.15, -0.05, 0.05, 0.15]), ys
    centre = column[spec.to_idx([5.0, 0.05])[1]]
    edge = column[spec.to_idx([5.0, 0.15])[1]]
    assert centre < edge


def test_particle_filter_corrects_biased_odometry():
    walls = [[0, 0.5, 10, 1.0], [0, -1.0, 10, -0.5], [-1, -1, 0, 1], [10, -1, 11, 1],
             [2.0, 0.2, 2.5, 0.5], [5.0, 0.2, 5.5, 0.5], [8.0, 0.2, 8.5, 0.5],      # bumps make x observable
             [3.5, -0.5, 4.0, -0.3], [6.5, -0.5, 7.0, -0.3]]
    gt = fake_world(walls)
    fine = GridSpec(-1, -1, 11, 1, 0.05)
    field = LikelihoodField(fine, gt.occupancy(fine))
    rng = np.random.default_rng(0)
    a, d = beams()

    true = np.array([1.0, 0.0])
    dead = true + np.array([0.04, -0.03])
    pf = ParticleFilter(300, dead, 0.05, rng)
    for t in range(400):
        delta = np.array([0.02, 0.0])
        true = true + delta
        odom = delta * 1.04 + rng.normal(0, 0.0005, 2)        # 4% scale error
        dead = dead + odom
        pf.predict(odom)
        if t % 5 == 0:
            ranges = gt.raycast(true, d, 4.0) + rng.normal(0, 0.02, len(a))
            pf.update(a, ranges, field)
    est, _ = pf.estimate()
    assert np.linalg.norm(dead - true) > 0.2         # dead reckoning drifts ~0.3 m over 8 m
    assert np.linalg.norm(est - true) < 0.08, (est, true)
    assert abs(pf.scale_estimate()[0] - 1.04) < 0.02, pf.scale_estimate()   # learned the x bias
