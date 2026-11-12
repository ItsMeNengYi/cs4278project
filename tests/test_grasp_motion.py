"""Meaningful kinematic/isolation checks; run from the project root."""
import unittest
from pathlib import Path
import sys

import numpy as np
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None
    pybullet_data = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_motion import MotionPlanner, ROOT, rotation_error


class _LiveScene:
    def __init__(self):
        self.pb_physics_client = p.connect(p.DIRECT)
        c = self.pb_physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
        p.loadURDF("plane.urdf", physicsClientId=c)
        self.robot_id = p.loadURDF(str(ROOT / "assets/fetch/fetch.urdf"),
                                   [-2, 0, 0], useFixedBase=True, physicsClientId=c)
        p.loadURDF("table/table.urdf", useFixedBase=True, physicsClientId=c)
        self.object_idx = 5
        obj = sorted((ROOT / "assets/ycb_objects").glob("*"))[self.object_idx]
        self.object_id = p.loadURDF(str(obj / "coacd_decomposed_object_one_link.urdf"),
                                    [0, 0, .8], physicsClientId=c)
        self.joint_indices = []
        lo, hi = [], []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=c)):
            i = p.getJointInfo(self.robot_id, j, physicsClientId=c)
            if i[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_indices.append(j)
                lo.append(i[8]); hi.append(i[9])
        self.joint_lower, self.joint_upper = np.asarray(lo), np.asarray(hi)

    @property
    def agent_qpos(self):
        return np.array([s[0] for s in p.getJointStates(self.robot_id, self.joint_indices,
                                                      physicsClientId=self.pb_physics_client)])

    def close(self):
        p.disconnect(physicsClientId=self.pb_physics_client)


@unittest.skipIf(p is None, "PyBullet is not installed on this test host")
class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _LiveScene()
        cls.planner = MotionPlanner(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.planner.close()
        cls.env.close()

    def test_half_turn_is_not_zero(self):
        R = np.diag([1., -1., -1.])
        self.assertAlmostEqual(np.linalg.norm(rotation_error(R, np.eye(3))), np.pi)

    def test_jacobian_matches_fk(self):
        m = self.planner
        q = self.env.agent_qpos.copy()
        q[m.arm] = [.2, .2, -.6, .3, 1., .2, .5, -.3]
        pos, R = m.tcp_pose(q)
        jp, jr = m._jacobian()
        for j in m.arm:
            perturbed = q.copy(); perturbed[j] += 1e-4
            pos1, R1 = m.tcp_pose(perturbed)
            np.testing.assert_allclose((pos1 - pos) / 1e-4, jp[:, j], atol=.004, rtol=.02)
            np.testing.assert_allclose(rotation_error(R1, R) / 1e-4, jr[:, j], atol=.004, rtol=.02)

    def test_ik_roundtrip_and_live_isolation(self):
        m, e = self.planner, self.env
        c = e.pb_physics_client
        before = e.agent_qpos.copy()
        obj_before = p.getBasePositionAndOrientation(e.object_id, physicsClientId=c)
        n_before = p.getNumBodies(physicsClientId=c)
        q = before.copy()
        q[m.arm] = [.2, .2, -.6, .3, 1., .2, .5, -.3]
        target, rotation = m.tcp_pose(q)
        seed = q.copy(); seed[m.arm] += .05
        solved = m.solve_pose(seed, target, rotation)
        self.assertIsNotNone(solved)
        pos, R = m.tcp_pose(solved)
        self.assertLess(np.linalg.norm(target - pos), .0031)
        self.assertLess(np.linalg.norm(rotation_error(rotation, R)), .061)
        np.testing.assert_array_equal(e.agent_qpos, before)
        self.assertEqual(p.getBasePositionAndOrientation(e.object_id, physicsClientId=c), obj_before)
        self.assertEqual(p.getNumBodies(physicsClientId=c), n_before)

    def test_rejects_table_intersection(self):
        m = self.planner
        q = self.env.agent_qpos.copy()
        # Translation of the robot base into the table is deliberately invalid.
        q[m.name_to_action['root_x_axis_joint']] = 2.
        self.assertFalse(m.collision_free(q))


class PureMotionTests(unittest.TestCase):
    def test_half_turn_is_not_zero(self):
        self.assertAlmostEqual(np.linalg.norm(rotation_error(np.diag([1., -1., -1.]), np.eye(3))), np.pi)

    def test_rotation_residual_restores_orientation(self):
        from scipy.spatial.transform import Rotation
        current = Rotation.from_euler('xyz', [.3, -.2, .6]).as_matrix()
        target = Rotation.from_euler('xyz', [-.4, .8, -.1]).as_matrix()
        residual = rotation_error(target, current)
        np.testing.assert_allclose(Rotation.from_rotvec(residual).as_matrix() @ current, target, atol=1e-12)

    def test_numerical_jacobian_uses_requested_columns_and_restores_state(self):
        from scipy.spatial.transform import Rotation
        class FakePlanner(MotionPlanner):
            def set_q(self, q):
                self.q = np.array(q, float)
            def tcp_pose(self, q=None):
                if q is not None:
                    self.set_q(q)
                return np.array([self.q[0] ** 2, 2 * self.q[1], 0.]), Rotation.from_rotvec([0, 0, self.q[2]]).as_matrix()
        planner = object.__new__(FakePlanner)
        planner.arm = [0, 1, 2]
        planner.set_q([.4, .3, .2, .7])
        original = planner.q.copy()
        jp, jr = planner._jacobian([0, 2])
        np.testing.assert_allclose(jp[:, 0], [.8, 0, 0], atol=1e-8)
        np.testing.assert_allclose(jr[:, 2], [0, 0, 1], atol=1e-8)
        np.testing.assert_array_equal(jp[:, 1], [0, 0, 0])
        np.testing.assert_array_equal(planner.q, original)


    def test_shortcuts_recheck_obstacles(self):
        class FakePlanner(MotionPlanner):
            def _interpolate(self, a, b, resolution=.045):
                return [a + t * (b - a) for t in np.linspace(0, 1, 101)]
            def collision_free(self, q, **kwargs):
                return not (.35 < q[0] < .65 and q[1] < .7)
        planner = object.__new__(FakePlanner)
        path = [np.array(v, float) for v in ((0, 0), (0, 1), (1, 1), (1, 0))]
        result = planner.simplify_path(path)
        self.assertGreater(len(result), 2)
        for a, b in zip(result, result[1:]):
            self.assertTrue(all(planner.collision_free(q) for q in planner._interpolate(a, b)))

    def test_cartesian_shortcut_rejects_tcp_curvature(self):
        class FakePlanner(MotionPlanner):
            def _interpolate(self, a, b, resolution=.045):
                return [a + t * (b - a) for t in np.linspace(0, 1, 30)]
            def collision_free(self, q, **kwargs):
                self.q = q
                return True
            def tcp_pose(self, q=None):
                if q is not None:
                    self.q = q
                angle = self.q[0]
                return np.array([np.cos(angle), np.sin(angle), 0.]), np.eye(3)
        planner = object.__new__(FakePlanner)
        path = [np.array([v]) for v in np.linspace(0, np.pi / 2, 100)]
        self.assertEqual(len(planner.simplify_path(path)), 2)
        cartesian = planner.simplify_path(path, cartesian=True)
        self.assertIsNotNone(cartesian)
        self.assertGreater(len(cartesian), 2)
        np.testing.assert_array_equal(cartesian[0], path[0])
        np.testing.assert_array_equal(cartesian[-1], path[-1])


if __name__ == '__main__':
    unittest.main()
