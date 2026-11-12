"""Small dependency-free checks for the analytic grasp policy."""

import numpy as np

from util_find_best_pair import find_best_pair, find_grasp_candidates


def test_antipodal_pair_and_ranking():
    points = np.array([
        [-0.04, 0.00, 0.10],
        [0.04, 0.00, 0.10],
        [-0.03, 0.03, 0.10],
        [0.03, 0.03, 0.10],
    ])
    normals = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    candidates = find_grasp_candidates(
        points, normals, object_center=np.array([0.0, 0.0, 0.10]),
        max_width=0.09, max_candidates=4,
    )
    assert candidates
    assert candidates[0]["indices"] == (0, 1)
    np.testing.assert_allclose(candidates[0]["width"], 0.08, atol=1e-12)
    assert candidates[0]["friction_margin_deg"] > 19.9
    np.testing.assert_allclose(candidates[0]["center_fraction"], 0.5, atol=1e-12)


def test_rejects_non_antipodal_and_too_wide_pairs():
    points = np.array([
        [-0.04, 0.0, 0.1],
        [0.04, 0.0, 0.1],
        [-0.06, 0.0, 0.1],
        [0.06, 0.0, 0.1],
    ])
    normals = np.array([
        [1.0, 0.0, 0.0],    # same-facing: invalid with the first point
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    candidates = find_grasp_candidates(
        points, normals, object_center=np.zeros(3), max_width=0.09,
    )
    assert candidates == []


def test_legacy_wrapper_uses_same_policy():
    points = np.array([
        [-0.02, 0.0, 0.1],
        [0.02, 0.0, 0.1],
    ])
    normals = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    pair = find_best_pair(points, normals, np.zeros(3), 0.05)
    assert pair is not None
    assert pair["indices"] == (0, 1)


if __name__ == "__main__":
    test_antipodal_pair_and_ranking()
    test_rejects_non_antipodal_and_too_wide_pairs()
    test_legacy_wrapper_uses_same_policy()
    print("grasp policy tests passed")
