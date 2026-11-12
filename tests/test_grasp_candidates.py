"""Tests for grasp mechanics and proposal ranking; no simulator required."""
import unittest

import numpy as np

from grasp_candidates import generate_grasp_candidates


class GraspCandidatesTests(unittest.TestCase):
    def pair(self, normals=None, **kwargs):
        points = np.array([[-0.03, 0, 0.1], [0.03, 0, 0.1]])
        normals = [[-1, 0, 0], [1, 0, 0]] if normals is None else normals
        return generate_grasp_candidates(points, normals, [0, 0, 0.1], 0.09,
                                         min_patch_points=1, **kwargs)

    def test_antipodal_axis_and_friction_margin(self):
        candidate = self.pair()[0]
        np.testing.assert_allclose(candidate["axis"], [1, 0, 0])
        self.assertAlmostEqual(candidate["friction_margin_deg"], np.degrees(np.arctan(0.5)) - 5)
        self.assertAlmostEqual(candidate["required_friction_mu"], 0.0)

    def test_friction_cone_rejects_near_boundary_under_uncertainty(self):
        a = np.deg2rad(24)
        normals = [[-np.cos(a), np.sin(a), 0], [np.cos(a), np.sin(a), 0]]
        self.assertEqual(self.pair(normals), [])
        self.assertEqual(len(self.pair(normals, normal_uncertainty_deg=0)), 1)

    def test_inward_normals_are_not_accepted(self):
        self.assertEqual(self.pair([[1, 0, 0], [-1, 0, 0]]), [])

    def test_support_filter_applies_to_every_candidate(self):
        self.assertEqual(self.pair(support_top=0.09), [])
        self.assertEqual(len(self.pair(support_top=0.08)), 1)

    def test_local_opposing_wall_is_not_patch_support(self):
        self.assertEqual(generate_grasp_candidates([[-0.003, 0, 0.1], [0.003, 0, 0.1]],
                         [[-1, 0, 0], [1, 0, 0]], [0, 0, 0.1], 0.08, min_width=0.001), [])

    def test_deterministic_diverse_ranked_candidates(self):
        points, normals = [], []
        for x, sign in [(-0.03, -1), (0.03, 1)]:
            for y in np.linspace(-0.035, 0.035, 8):
                for z in [0.09, 0.10, 0.11]:
                    points.append([x, y, z])
                    normals.append([sign, 0, 0])
        a = generate_grasp_candidates(points, normals, [0, 0, 0.1], 0.09, max_candidates=5)
        b = generate_grasp_candidates(points, normals, [0, 0, 0.1], 0.09, max_candidates=5)
        self.assertEqual([g["indices"] for g in a], [g["indices"] for g in b])
        self.assertGreater(len(a), 1)
        self.assertEqual(sorted([g["score"] for g in a], reverse=True), [g["score"] for g in a])
        for i, g in enumerate(a):
            for old in a[:i]:
                self.assertTrue(np.linalg.norm(g["midpoint"] - old["midpoint"]) >= 0.012
                                or abs(g["axis"] @ old["axis"]) < np.cos(np.deg2rad(15)))

    def test_finite_segment_distance(self):
        points = [[0.02, 0, 0.1], [0.08, 0, 0.1]]
        result = generate_grasp_candidates(points, [[-1, 0, 0], [1, 0, 0]],
                                            [0, 0, 0.1], 0.09, min_patch_points=1)
        self.assertAlmostEqual(result[0]["center_distance"], 0.02)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.pair(friction_mu=0.01)
        with self.assertRaises(ValueError):
            self.pair([[0, 0, 0], [1, 0, 0]])
        with self.assertRaises(ValueError):
            self.pair(max_candidates=True)
        with self.assertRaises(ValueError):
            self.pair([[np.nan, 0, 0], [1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
