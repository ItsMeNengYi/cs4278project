"""Controller contract tests with mocked physics; no Bullet installation required.

These check timing, contact classification and failure gates, not physical grasp
success. Run the separate Bullet integration tests for kinematics and collisions.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from scipy.spatial.transform import Rotation
except ImportError:
    np = Rotation = None

G = None
if np is not None:
    # Isolate imports so this suite runs without Bullet and never contaminates
    # the real simulator tests' pybullet or grasp_motion modules.
    shadow_motion = types.ModuleType("grasp_motion")
    shadow_motion.MotionPlanner = mock.Mock()
    spec = importlib.util.spec_from_file_location("_execution_under_test", ROOT / "grasping.py")
    G = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"pybullet": mock.MagicMock(), "grasp_motion": shadow_motion}):
        spec.loader.exec_module(G)


def contact(link=-1, point=(0., 0., 0.), force=1., other=3):
    out = [0] * 14
    out[2], out[3], out[5], out[9] = other, link, point, force
    return tuple(out)


class FakeEnv:
    pb_physics_client = 7
    robot_id = 1
    object_id = 2
    substeps = 1

    def __init__(self):
        self.q = np.zeros(15)
        self.pos = np.array([0., 0., .8])
        self.quat = np.array([0., 0., 0., 1.])
        self.commands = []
        self.step_count = 0

    @property
    def agent_qpos(self):
        return self.q.copy()

    def _get_state(self):
        return self.q.copy(), np.zeros(15), self.pos.copy(), self.quat.copy()

    def step(self, action):
        self.q = np.asarray(action).copy()
        self.commands.append(self.q.copy())
        self.step_count += 1
        return {"object_pos": self.pos.copy(), "object_xyzw": self.quat.copy()}, {}


@unittest.skipIf(G is None, "NumPy and SciPy are required for controller tests")
class GraspExecutionTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.action = self.env.agent_qpos.copy()

    def test_zero_direction_rejected_and_half_turn_preserved(self):
        with self.assertRaises(ValueError):
            G.unit([0., 0., 0.])
        axis, angle = G.axis_angle(np.diag([1., -1., -1.]))
        self.assertAlmostEqual(angle, np.pi)
        np.testing.assert_allclose(G.rotation_about(axis, angle), np.diag([1., -1., -1.]), atol=1e-10)

    def test_gap_conversion_respects_joint_travel(self):
        self.assertEqual(G.opening_q(0.), 0.)
        self.assertAlmostEqual(G.opening_q(G.MAX_OPENING), .05)
        self.assertEqual(G.opening_q(1.), .05)
        gap = .06
        self.assertAlmostEqual(2 * (G.opening_q(gap) + .001425), gap)

    def test_hand_frame_aligns_flat_pads_with_normals_despite_staggered_samples(self):
        pair = dict(p1=np.array([0., .03, .10]), p2=np.array([0., -.03, .105]),
                    n1=np.array([0., 1., 0.]), n2=np.array([0., -1., 0.]), width=.061)
        options = G._approaches(pair, [pair['p1'], pair['p2']], 0., [-1., 0.])
        self.assertTrue(options)
        for option in options:
            np.testing.assert_allclose(option['rotation'][:,1], [0., 1., 0.])
            np.testing.assert_allclose(option['rotation'].T @ option['rotation'], np.eye(3), atol=1e-12)

    def test_settling_requires_support_and_small_angular_velocity(self):
        velocities = [([0, 0, 0], [0, 0, .2]), ([0, 0, 0], [0, 0, 0]),
                      ([0, 0, 0], [0, 0, 0]), ([0, 0, 0], [0, 0, 0])]
        supports = [[contact()], [], [contact()], [contact()]]
        with mock.patch.object(G.p, "getBaseVelocity", side_effect=velocities), \
             mock.patch.object(G.p, "getContactPoints", side_effect=supports):
            G.wait_until_rest(self.env, self.action, settle=2, max_steps=4)
        self.assertEqual(self.env.step_count, 4)

    def test_settling_rejects_free_object(self):
        with mock.patch.object(G.p, "getBaseVelocity", return_value=([0, 0, 0], [0, 0, 0])), \
             mock.patch.object(G.p, "getContactPoints", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "did not settle"):
                G.wait_until_rest(self.env, self.action, settle=2, max_steps=3)

    def test_contacts_require_loaded_inner_faces(self):
        right = G.R_FINGER
        left = G.L_FINGER
        samples = [contact(right, (0., -.014, 0.)),
                   contact(left, (0., .014, 0.)),
                   contact(right, (0., 0., 0.)),       # outer/side face
                   contact(left, (.1, .014, 0.)),     # beyond pad length
                   contact(left, (0., .014, .1)),     # beyond pad height
                   contact(left, (0., .014, 0.), 0.), # unloaded manifold
                   contact(G.GRIPPER_LINK, (0., 0., 0.))]
        with mock.patch.object(G, "link_frame", return_value=(np.zeros(3), np.eye(3), None)), \
             mock.patch.object(G.p, "getContactPoints", return_value=samples):
            self.assertEqual(G.fingers_touching(self.env), {right, left})
        with mock.patch.object(G, "link_frame", return_value=(np.zeros(3), np.eye(3), None)), \
             mock.patch.object(G.p, "getContactPoints", return_value=samples[2:]):
            self.assertEqual(G.fingers_touching(self.env), set())

    def test_holding_rejects_separation_but_allows_loaded_rolling_contact(self):
        self.env._grasp_reference = (np.zeros(3), np.eye(3))
        with mock.patch.object(G, "fingers_touching", return_value={G.R_FINGER, G.L_FINGER}):
            with mock.patch.object(G, "object_relative_pose", return_value=(np.array([.08, 0, 0]), np.eye(3))):
                self.assertFalse(G.holding(self.env))
            with mock.patch.object(G, "object_relative_pose", return_value=(np.array([.02, 0, 0]), np.eye(3))):
                self.assertTrue(G.holding(self.env))
            with mock.patch.object(G, "object_relative_pose", return_value=(np.zeros(3), Rotation.from_rotvec([0, 0, .3]).as_matrix())):
                self.assertTrue(G.holding(self.env))
            with mock.patch.object(G, "object_relative_pose", return_value=(np.zeros(3), np.eye(3))):
                self.assertTrue(G.holding(self.env))

    def test_stale_position_and_rotation_fail_before_any_motion(self):
        for new_position, new_quaternion in [
            (np.array([.02, 0, .8]), np.array([0., 0., 0., 1.])),
            (np.array([0., 0., .8]), Rotation.from_rotvec([0, 0, .2]).as_quat()),
        ]:
            self.env.pos, self.env.quat = new_position, new_quaternion
            plan = {"object_pose": (np.array([0., 0., .8]), np.array([0., 0., 0., 1.]))}
            with mock.patch.object(G, "open_gripper") as opened, mock.patch.object(G, "MotionPlanner") as planner:
                with self.assertRaisesRegex(RuntimeError, "moved after planning"):
                    G.execute_grasp(self.env, self.action, plan)
                opened.assert_not_called()
                planner.assert_not_called()
            self.assertEqual(self.env.step_count, 0)

    def test_capture_range_accounts_for_an_object_held_near_its_end(self):
        self.env._grasp_reference = (np.array([.02,0,0]),np.eye(3))
        self.env._grasp_capture_radius = .19
        with mock.patch.object(G,'fingers_touching',return_value={G.R_FINGER,G.L_FINGER}):
            with mock.patch.object(G,'object_relative_pose',return_value=(np.array([.15,0,0]),np.eye(3))):
                self.assertTrue(G.holding(self.env))
            with mock.patch.object(G,'object_relative_pose',return_value=(np.array([.3,0,0]),np.eye(3))):
                self.assertFalse(G.holding(self.env))

    def test_execution_targets_obey_velocity_and_acceleration_caps(self):
        target = self.action.copy()
        target[G.TORSO], target[G.ARM[0]] = .04, .4
        limits = np.ones(15)
        limits[G.TORSO] = .1
        dt, speed = .01, .45
        with mock.patch.object(G.p, "getPhysicsEngineParameters", return_value={"fixedTimeStep": dt}), \
             mock.patch.object(G, "_joint_limits", return_value=limits):
            G.execute_path(self.env, self.action, [np.zeros(15), target], speed=speed)
        commands = np.vstack([np.zeros(15), self.env.commands, target])
        velocity = np.diff(commands, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        vmax = limits * speed
        amax = np.maximum(2 * vmax, .05)
        self.assertLessEqual(float(np.max(np.abs(velocity) / vmax)), 1.001)
        self.assertLessEqual(float(np.max(np.abs(acceleration) / amax)), 1.01)
        np.testing.assert_allclose(self.action, target)

    def test_carried_motion_preserves_closure_command(self):
        self.env.q[G.FINGERS] = .02
        self.action[:] = self.env.q
        self.action[G.FINGERS] = .018
        target = self.env.q.copy()
        target[G.ARM[0]] = .1
        target[G.FINGERS] = .05  # planned opening must not reopen a held object
        with mock.patch.object(G.p, "getPhysicsEngineParameters", return_value={"fixedTimeStep": .02}), \
             mock.patch.object(G, "_joint_limits", return_value=np.ones(15)), \
             mock.patch.object(G, "holding", return_value=True):
            G.execute_path(self.env, self.action, [self.env.q.copy(), target], carrying=True)
        for command in self.env.commands:
            np.testing.assert_allclose(command[G.FINGERS], .018)

    def test_execution_uses_the_same_short_continuous_arc_as_planning(self):
        joint = G.ARM[2]
        self.env.joint_upper = np.ones(15)
        self.env.joint_upper[joint] = 314.
        self.env.q[joint] = np.pi-.05
        self.action[:] = self.env.q
        target = self.action.copy()
        target[joint] = -np.pi+.05
        with mock.patch.object(G.p, 'getPhysicsEngineParameters', return_value={'fixedTimeStep':.02}), \
             mock.patch.object(G, '_joint_limits', return_value=np.ones(15)):
            G.execute_path(self.env, self.action, [self.env.q.copy(), target])
        self.assertAlmostEqual(self.action[joint], np.pi+.05)
        self.assertLess(len(self.env.commands), 60)
        self.assertLess(np.ptp(np.asarray(self.env.commands)[:,joint]), .101)

    def test_collinear_validation_samples_do_not_add_execution_stops(self):
        target = self.action.copy()
        target[G.TORSO], target[G.ARM[0]] = .04, .4
        coarse = [self.action.copy(), target]
        dense = list(np.linspace(self.action, target, 25))
        commands = []
        for path in (coarse, dense):
            env = FakeEnv()
            action = env.agent_qpos.copy()
            with mock.patch.object(G.p, "getPhysicsEngineParameters", return_value={"fixedTimeStep": .02}), \
                 mock.patch.object(G, "_joint_limits", return_value=np.ones(15)):
                G.execute_path(env, action, path)
            commands.append(np.asarray(env.commands))
        np.testing.assert_allclose(commands[0], commands[1])

    def test_carried_contact_loss_during_final_settle_is_failure(self):
        with mock.patch.object(G.p, "getPhysicsEngineParameters", return_value={"fixedTimeStep": .02}), \
             mock.patch.object(G, "_joint_limits", return_value=np.ones(15)), \
             mock.patch.object(G, "holding", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "slipped while settling"):
                G.execute_path(self.env, self.action, [self.action.copy()], carrying=True)

    def test_object_movement_during_transit_blocks_insertion(self):
        q = self.action.copy()
        plan = {"object_pose": (self.env.pos.copy(), self.env.quat.copy()),
                "fingers": .04, "q_pre": q, "rotation": np.eye(3),
                "tcp": np.array([0., 0., .8])}
        planner = mock.MagicMock()
        planner.__enter__.return_value = planner
        planner.tcp_pose.return_value = (np.array([0., 0., .9]), np.eye(3))
        planner.solve_pose.return_value = q
        planner.plan_joint_path.return_value = [q, q]

        def move_object(*args, **kwargs):
            self.env.pos[0] += .02

        with mock.patch.object(G, "MotionPlanner", return_value=planner), \
             mock.patch.object(G, "open_gripper"), \
             mock.patch.object(G, "execute_path", side_effect=move_object), \
             mock.patch.object(G, "close_gripper") as close:
            with self.assertRaisesRegex(RuntimeError, "moved during arm transit"):
                G.execute_grasp(self.env, self.action, plan, log=lambda _: None)
            planner.plan_cartesian.assert_not_called()
            close.assert_not_called()

    def test_empty_path_and_failed_opening_are_explicit_errors(self):
        with self.assertRaisesRegex(RuntimeError, "No valid motion path"):
            G.execute_path(self.env, self.action, None)
        self.action[G.FINGERS] = .05
        # Jammed fingers never move even while commands are sent.
        with mock.patch.object(self.env, "step", return_value=({}, {})) as step:
            with self.assertRaisesRegex(RuntimeError, "did not reach"):
                G.settle_fingers(self.env, self.action, max_steps=3)
            self.assertEqual(step.call_count, 3)

    def test_closure_requires_consecutive_two_pad_contacts(self):
        two = {G.R_FINGER, G.L_FINGER}
        observations = [two, {G.R_FINGER}, two, two, two, two]
        self.env.q[G.FINGERS] = .03
        self.action[:] = self.env.q
        with mock.patch.object(G, "fingers_touching", side_effect=observations), \
             mock.patch.object(G, "object_relative_pose", return_value=(np.zeros(3), np.eye(3))):
            G.close_gripper(self.env, self.action, stable_steps=3, max_steps=10)
        self.assertEqual(self.env.step_count, 5 + 12)
        self.assertIsNotNone(self.env._grasp_reference)

    def test_pressure_maintenance_only_tightens_the_unloaded_pad(self):
        self.env.q[G.FINGERS] = .032
        self.env._grip_command = np.array([.03,.03])
        self.env._grip_min_force = 5.
        self.action[G.FINGERS] = .03
        with mock.patch.object(G.p, 'getContactPoints', return_value=[
            contact(G.R_FINGER,force=10.), contact(G.L_FINGER,force=1.)]):
            target = G.maintain_grip_pressure(self.env, self.action)
        self.assertEqual(target[G.FINGERS[0]], .03)
        self.assertLess(target[G.FINGERS[1]], .03)
        np.testing.assert_array_equal(target[:13], self.action[:13])
        with mock.patch.object(G.p, 'getContactPoints', return_value=[
            contact(G.R_FINGER,force=10.), contact(G.L_FINGER,force=10.)]):
            self.env.q[G.FINGERS] = .04  # widened under load: never yield by reopening
            np.testing.assert_array_equal(G.maintain_grip_pressure(self.env,self.action),target)

    def test_failed_closure_never_establishes_reference(self):
        with mock.patch.object(G, "fingers_touching", return_value=set()):
            with self.assertRaisesRegex(RuntimeError, "sustained contact"):
                G.close_gripper(self.env, self.action, max_steps=4)
        self.assertIsNone(getattr(self.env, "_grasp_reference", None))

    def _lift_patches(self, bottom):
        def lift(*args, **kwargs):
            self.env.pos[2] += .15
        return [mock.patch.object(G, "move_object_by", side_effect=lift),
                mock.patch.object(G, "support_body", return_value=(3, .65)),
                mock.patch.object(G, "object_relative_pose", return_value=(np.zeros(3), np.eye(3))),
                mock.patch.object(G, "holding", return_value=True),
                mock.patch.object(G.p, "getAABB", return_value=([-.03, -.03, bottom], [.03, .03, bottom+.1]))]

    def test_lift_checks_at_least_fifty_steps_and_bottom_clearance(self):
        patches = self._lift_patches(.8)
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        result = G.lift_and_hold(self.env, self.action, hold_steps=3, log=lambda _: None)
        self.assertTrue(result["held"])
        self.assertEqual(result["hold_steps"], 50)
        self.assertAlmostEqual(result["bottom_clearance"], .15)
        self.assertAlmostEqual(result["lift"], .15)

    def test_center_rise_alone_does_not_pass_lift(self):
        patches = self._lift_patches(.70)
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with self.assertRaisesRegex(RuntimeError, "Lift/hold validation failed"):
            G.lift_and_hold(self.env, self.action, log=lambda _: None)


class AddedGeometryTests(unittest.TestCase):
    def test_old_pad_metric_accepts_point_outside_pad(self):
        env = FakeEnv()
        with mock.patch.object(G, 'link_frame', return_value=(np.zeros(3),np.eye(3),None)):
            target = np.array([0.,-.014,.025])
            distance = G.grasp_error(env,{G.R_FINGER:target})[G.R_FINGER]
        self.assertAlmostEqual(distance,.013)
        self.assertLess(np.hypot(.025,.013),.0315)
        self.assertGreater(distance,G.TOLERANCE)

    def test_floor_pickup_still_has_to_clear_table(self):
        env = FakeEnv()
        physics = types.SimpleNamespace(
            getNumBodies=lambda **kw: 1, getBodyUniqueId=lambda i,**kw: 3,
            getBodyInfo=lambda *a,**kw: (b'table',b'table'),
            getNumJoints=lambda *a,**kw: 0,
            getCollisionShapeData=lambda *a,**kw: [True],
            getAABB=lambda body,*a,**kw: ([0.,0.,.01],[.1,.1,.05]) if body==env.object_id else ([-1.,-1.,0.],[1.,1.,.625]))
        with mock.patch.object(G,'p',physics), mock.patch.object(G,'support_body',return_value=(0,0.)):
            height,top = G.required_lift(env)
        self.assertAlmostEqual(top,.625)
        self.assertAlmostEqual(height,.725)


if __name__ == "__main__":
    unittest.main()
