"""Kinematic IK and collision-checked arm paths in a separate Bullet client.

The live robot is only read here. Planning never calls env.step(), resets a live
joint, or saves/restores a live simulation. The returned joint arrays are plans;
the execution controller decides when to send them to env.step().
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
try:
    import pybullet as p
    import pybullet_data
except ImportError:  # Rotation helpers remain usable in a dependency-limited test host.
    p = None
    pybullet_data = None
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
ARM_NAMES = ("torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint",
             "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint",
             "wrist_flex_joint", "wrist_roll_joint")
FINGER_NAMES = ("r_gripper_finger_joint", "l_gripper_finger_joint")


def rotation_error(target, current):
    """World rotation vector; unlike a cross product it detects 180 degree errors."""
    return Rotation.from_matrix(np.asarray(target) @ np.asarray(current).T).as_rotvec()


class MotionPlanner:
    """Shadow copy of this project's Fetch, target object, table, floor and maze.

    All q vectors use the same controllable-DOF order as env.step(). The TCP is
    gripper_link's origin, the exact pad midpoint with symmetric finger opening
    in this URDF. x points along the fingers and y along jaw separation.
    """
    def __init__(self, env, seed=0):
        if p is None:
            raise ImportError("MotionPlanner requires pybullet; install requirements-grasping.txt")
        self.env = env
        self.live_client = env.pb_physics_client
        self.client = p.connect(p.DIRECT)
        self.rng = np.random.default_rng(seed)
        self.attached = None
        self.last_failure = ""
        self.world = []
        self.live_to_shadow = {}
        try:
            self._load_scene()
            self.sync_scene()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None

    def _live_pose(self, body):
        return p.getBasePositionAndOrientation(body, physicsClientId=self.live_client)

    def _load(self, path, live, fixed=True, flags=0):
        pos, orn = self._live_pose(live)
        b = p.loadURDF(str(path), pos, orn, useFixedBase=fixed, flags=flags,
                       physicsClientId=self.client)
        self.live_to_shadow[live] = b
        return b

    def _load_scene(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.robot = self._load(ROOT / "assets/fetch/fetch.urdf", self.env.robot_id)
        self.joints = []
        self.name_to_action = {}
        self.link_by_name = {"root": -1}
        self.info = {}
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            i = p.getJointInfo(self.robot, j, physicsClientId=self.client)
            self.info[j] = i
            self.link_by_name[i[12].decode()] = j
            if i[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.name_to_action[i[1].decode()] = len(self.joints)
                self.joints.append(j)
        self.arm = [self.name_to_action[n] for n in ARM_NAMES]
        self.finger_actions = [self.name_to_action[n] for n in FINGER_NAMES]
        self.finger_links = {self.link_by_name["r_gripper_finger_link"],
                             self.link_by_name["l_gripper_finger_link"]}
        self.tcp = self.link_by_name["gripper_link"]
        self.lower = np.array(self.env.joint_lower, float)
        self.upper = np.array(self.env.joint_upper, float)
        self.continuous = self.upper > 300
        self.floor_links = {self.link_by_name[n] for n in
                            ("base_link", "r_wheel_link", "l_wheel_link")}
        self._build_self_exclusions()
        objects = sorted((ROOT / "assets/ycb_objects").glob("*"))
        self.object = self._load(objects[self.env.object_idx] /
                                 "coacd_decomposed_object_one_link.urdf",
                                 self.env.object_id, fixed=False)
        self.floor = None
        for k in range(p.getNumBodies(physicsClientId=self.live_client)):
            live = p.getBodyUniqueId(k, physicsClientId=self.live_client)
            if live in (self.env.robot_id, self.env.object_id):
                continue
            names = p.getBodyInfo(live, physicsClientId=self.live_client)
            label = " ".join(n.decode().lower() for n in names)
            if "plane" in label:
                b = self._load(Path(pybullet_data.getDataPath()) / "plane.urdf", live)
                self.floor = b
            elif "table" in label:
                b = self._load(Path(pybullet_data.getDataPath()) / "table/table.urdf", live)
            elif "maze" in label:
                b = self._load(ROOT / "assets/maze.urdf", live,
                               flags=p.URDF_MERGE_FIXED_LINKS)
            elif not any(p.getCollisionShapeData(live, link, physicsClientId=self.live_client)
                         for link in range(-1, p.getNumJoints(live, physicsClientId=self.live_client))):
                continue  # collision-free goal marker
            else:
                raise ValueError(f"Unrecognised scene collision body {live}: {label}")
            self.world.append(b)
        self.root_rotation = np.asarray(p.getMatrixFromQuaternion(
            self._live_pose(self.env.robot_id)[1])).reshape(3, 3)

    def _build_self_exclusions(self):
        # Adjacent links, fixed rigid clusters and the supplied semantic model.
        # Never learn exclusions from a potentially colliding arm start pose.
        self.self_exclusions = set()
        links = [-1] + list(self.info)
        parent = {j: i[16] for j, i in self.info.items()}
        fixed = {j for j, i in self.info.items() if i[2] == p.JOINT_FIXED}
        for j in self.info:
            self.self_exclusions.add(tuple(sorted((j, parent[j]))))
        def rigid_ancestors(j):
            out = {j}
            while j in fixed:
                j = parent[j]
                out.add(j)
            return out
        # Treat each fixed-connected cluster as one rigid body. Adjacency
        # carries through its fixed links (e.g. base -> fixed torso bracket ->
        # prismatic torso); these remain adjacent physical rigid bodies.
        clusters = {j: min(rigid_ancestors(j)) for j in links}
        adjacent_clusters = {tuple(sorted((clusters[j], clusters[parent[j]])))
                             for j in self.info}
        for a in links:
            for b in links:
                pair = tuple(sorted((clusters[a], clusters[b])))
                if a < b and (clusters[a] == clusters[b] or pair in adjacent_clusters):
                    self.self_exclusions.add((a, b))
        for e in ET.parse(ROOT / "assets/fetch/fetch.srdf").getroot().findall("disable_collisions"):
            a, b = self.link_by_name[e.attrib["link1"]], self.link_by_name[e.attrib["link2"]]
            self.self_exclusions.add(tuple(sorted((a, b))))
        self.collision_links = [j for j in links if p.getCollisionShapeData(
            self.robot, j, physicsClientId=self.client)]

    def sync_scene(self):
        """Read live poses and joints into the shadow; never advances live physics."""
        for live, shadow in self.live_to_shadow.items():
            if shadow == self.robot:
                continue
            pos, orn = self._live_pose(live)
            p.resetBasePositionAndOrientation(shadow, pos, orn, physicsClientId=self.client)
        self.set_q(np.asarray(self.env.agent_qpos, float))
        self.object_pose = self._live_pose(self.env.object_id)

    def set_q(self, q):
        q = np.asarray(q, float)
        if q.shape != (len(self.joints),) or not np.isfinite(q).all():
            raise ValueError("q must be a finite full action vector")
        for j, value in zip(self.joints, q):
            p.resetJointState(self.robot, j, float(value), physicsClientId=self.client)
        self.q = q.copy()

    def tcp_pose(self, q=None):
        if q is not None:
            self.set_q(q)
        state = p.getLinkState(self.robot, self.tcp, computeForwardKinematics=True,
                               physicsClientId=self.client)
        return np.asarray(state[4]), np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)

    def _jacobian(self, joints=None):
        """Central FK differences avoid Bullet's localPosition frame ambiguity.

        A 0.2 mm/rad perturbation is large enough for getLinkState's float32
        output. Only requested action columns are evaluated; the live client is
        never involved. Position and rotation derivatives use world axes.
        """
        active = self.arm if joints is None else list(joints)
        q = self.q.copy()
        linear = np.zeros((3, len(q)))
        angular = np.zeros_like(linear)
        h = .0002
        for j in active:
            plus, minus = q.copy(), q.copy()
            plus[j] += h
            minus[j] -= h
            pp, rp = self.tcp_pose(plus)
            pm, rm = self.tcp_pose(minus)
            linear[:, j] = (pp - pm) / (2 * h)
            angular[:, j] = rotation_error(rp, rm) / (2 * h)
        self.set_q(q)
        return linear, angular

    def solve_pose(self, q0, position, rotation, joints=None, position_tolerance=0.003,
                   orientation_tolerance=0.06, max_iters=150, restarts=2):
        """Damped least-squares SE(3) IK, fixed base; None on non-convergence.

        Limits apply at every iterate. Multiple deterministic seeds address
        local minima; every accepted answer independently meets both tolerances.
        Collision checking belongs to the path planner, not this IK routine.
        """
        q0 = np.asarray(q0, float)
        target = np.asarray(position, float)
        Rtarget = np.asarray(rotation, float)
        active = self.arm if joints is None else list(joints)
        if target.shape != (3,) or Rtarget.shape != (3, 3) or not np.isfinite(target).all() or not np.isfinite(Rtarget).all():
            raise ValueError("position and rotation must be finite shapes (3,) and (3,3)")
        if not np.allclose(Rtarget.T @ Rtarget, np.eye(3), atol=1e-5) or np.linalg.det(Rtarget) < .999:
            raise ValueError("rotation must be a proper orthonormal matrix")
        for attempt in range(restarts + 1):
            best_cost = np.inf
            q = q0.copy()
            if attempt:
                q[active] += self.rng.normal(0, .3, len(active))
                if self.name_to_action["torso_lift_joint"] in active:
                    q[self.name_to_action["torso_lift_joint"]] = .18 + .08 * (attempt % 2)
                q = np.clip(q, self.lower, self.upper)
            stalled = 0
            for _ in range(max_iters):
                pos, R = self.tcp_pose(q)
                ep, er = target - pos, rotation_error(Rtarget, R)
                pe, re = np.linalg.norm(ep), np.linalg.norm(er)
                cost = pe * pe + .16 * re * re
                if pe <= position_tolerance and re <= orientation_tolerance:
                    return q.copy()
                if cost < best_cost - 1e-9:
                    best_cost, stalled = cost, 0
                else:
                    stalled += 1
                if stalled > 25:
                    break
                Jp, Jr = self._jacobian(active)
                J = np.vstack((Jp, .4 * Jr))[:, active]
                e = np.r_[ep, .4 * er]
                damping = .015 + .03 * min(1., pe + re)
                dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), e)
                bound = np.array([.02 if self.info[self.joints[j]][2] == p.JOINT_PRISMATIC else .16 for j in active])
                dq *= min(1., float(np.min(bound / np.maximum(np.abs(dq), 1e-12))))
                # Backtracking prevents a poor Jacobian step from increasing error.
                accepted = False
                for fraction in (1., .5, .25, .125):
                    qt = q.copy()
                    qt[active] = np.clip(q[active] + fraction * dq, self.lower[active], self.upper[active])
                    pp, rr = self.tcp_pose(qt)
                    c = np.sum((target - pp) ** 2) + .16 * np.sum(rotation_error(Rtarget, rr) ** 2)
                    if c < cost - 1e-13:
                        q, accepted = qt, True
                        break
                if not accepted:
                    break
        self.last_failure = f"IK did not reach {position_tolerance:g} m and {orientation_tolerance:g} rad tolerances"
        return None

    def attach_object(self, q=None):
        """Capture actual live object pose relative to the current/planned TCP."""
        if q is None:
            q = np.asarray(self.env.agent_qpos, float)
        pos, R = self.tcp_pose(q)
        op, oq = self._live_pose(self.env.object_id)
        oR = np.asarray(p.getMatrixFromQuaternion(oq)).reshape(3, 3)
        self.attached = (R.T @ (np.asarray(op) - pos), R.T @ oR)

    def collision_free(self, q, allow_finger_object=False, carrying=False, margin=0.002):
        """All scene bodies and non-exempt self pairs; only intended pad contact allowed.

        Attached-object/world tests allow tangency (1 mm contact slop) while
        lifting away from its support; this is distinct from robot clearance.
        """
        q = np.asarray(q, float)
        if (q.shape != self.lower.shape or not np.isfinite(q).all() or
                np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6)):
            self.last_failure = "joint configuration violates finite joint limits"
            return False
        self.set_q(q)
        if carrying:
            if self.attached is None:
                raise RuntimeError("attach_object() must precede carrying collision checks")
            pos, R = self.tcp_pose()
            local_p, local_R = self.attached
            op = pos + R @ local_p
            oq = Rotation.from_matrix(R @ local_R).as_quat()
            p.resetBasePositionAndOrientation(self.object, op, oq, physicsClientId=self.client)
        else:
            p.resetBasePositionAndOrientation(self.object, *self.object_pose, physicsClientId=self.client)
        for body in self.world:
            for c in p.getClosestPoints(self.robot, body, margin, physicsClientId=self.client):
                if body == self.floor and c[3] in self.floor_links:
                    continue
                self.last_failure = f"robot link {c[3]} near scene body {body} ({c[8]:.4f} m)"
                return False
        for c in p.getClosestPoints(self.robot, self.object, margin, physicsClientId=self.client):
            if (allow_finger_object or carrying) and c[3] in self.finger_links:
                continue
            self.last_failure = f"robot link {c[3]} near object ({c[8]:.4f} m)"
            return False
        for c in p.getClosestPoints(self.robot, self.robot, margin, physicsClientId=self.client):
            a, b = sorted((c[3], c[4]))
            if a == b or (a, b) in self.self_exclusions:
                continue
            self.last_failure = f"self collision {a},{b} ({c[8]:.4f} m)"
            return False
        if carrying:
            for body in self.world:
                if any(c[8] < -.001 for c in p.getClosestPoints(
                        self.object, body, 0., physicsClientId=self.client)):
                    self.last_failure = f"carried object intersects scene body {body}"
                    return False
        return True

    def _interpolate(self, q0, q1, resolution=.045):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        delta = q1 - q0
        # Choose the nearest equivalent representation for continuous axes.
        delta[self.continuous] = (delta[self.continuous] + np.pi) % (2 * np.pi) - np.pi
        scales = np.ones(len(delta))
        for a, j in enumerate(self.joints):
            if self.info[j][2] == p.JOINT_PRISMATIC:
                scales[a] = 5.  # 9 mm translation per default validation sample
        n = max(1, int(np.ceil(np.max(np.abs(delta) * scales) / resolution)))
        return [q0 + delta * t / n for t in range(n + 1)]

    def edge_free(self, q0, q1, allow_finger_object=False, carrying=False, margin=.002,
                  resolution=.045):
        return all(self.collision_free(q, allow_finger_object, carrying, margin)
                   for q in self._interpolate(q0, q1, resolution))

    def plan_joint_path(self, q_start, q_goal, joints=None, allow_finger_object=False,
                        carrying=False, margin=.002, max_iterations=350):
        """Straight edge first, then seeded bidirectional RRT-Connect.

        Collision tests sample the entire edge at <=45 mrad / 9 mm joint
        resolution; this is discrete validation, not continuous collision proof.
        """
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        active = self.arm if joints is None else list(joints)
        opts = dict(allow_finger_object=allow_finger_object, carrying=carrying, margin=margin)
        if not self.collision_free(q_start, **opts) or not self.collision_free(q_goal, **opts):
            return None
        if self.edge_free(q_start, q_goal, **opts):
            return self.simplify_path(self._interpolate(q_start, q_goal), **opts)
        scale = np.array([5. if self.info[self.joints[j]][2] == p.JOINT_PRISMATIC else 1. for j in active])
        lower = self.lower[active].copy()
        upper = self.upper[active].copy()
        for k, a in enumerate(active):
            if self.continuous[a]:
                lower[k], upper[k] = q_start[a] - np.pi, q_start[a] + np.pi
        trees = [([q_start.copy()], [-1]), ([q_goal.copy()], [-1])]
        def extend(tree, target):
            nodes, parents = tree
            distances = np.linalg.norm((np.asarray(nodes)[:, active] - target[active]) * scale, axis=1)
            index = int(np.argmin(distances))
            source = nodes[index]
            delta = target[active] - source[active]
            length = np.linalg.norm(delta * scale)
            q = source.copy()
            q[active] += delta * min(1., .35 / max(length, 1e-12))
            if not self.edge_free(source, q, **opts):
                return None, False
            nodes.append(q)
            parents.append(index)
            return len(nodes) - 1, length <= .35
        def trace(tree, index):
            nodes, parents = tree
            path = []
            while index >= 0:
                path.append(nodes[index])
                index = parents[index]
            return path[::-1]
        for iteration in range(max_iterations):
            a, b = iteration % 2, 1 - iteration % 2
            target = q_start.copy()
            if self.rng.random() < .2:
                target = trees[b][0][0].copy()
            else:
                target[active] = self.rng.uniform(lower, upper)
            ia, _ = extend(trees[a], target)
            if ia is None:
                continue
            target = trees[a][0][ia]
            for _ in range(50):
                ib, reached = extend(trees[b], target)
                if ib is None:
                    break
                if reached:
                    pa, pb = trace(trees[a], ia), trace(trees[b], ib)
                    # Nonactive dimensions (e.g. finger settling error) may
                    # differ between trees. Validate that bridge explicitly.
                    if not self.edge_free(pa[-1], pb[-1], **opts):
                        break
                    path = pa + pb[::-1] if a == 0 else pb + pa[::-1]
                    dense = [path[0]]
                    for x, y in zip(path, path[1:]):
                        dense.extend(self._interpolate(x, y)[1:])
                    return self.simplify_path(dense, **opts)
        self.last_failure = "RRT-Connect exhausted its iteration budget"
        return None

    def plan_cartesian(self, q_start, positions, rotation, joints=None,
                       allow_finger_object=False, carrying=False, margin=.001,
                       step_size=.008):
        """Solve sequential TCP waypoints and validate every connecting joint edge.

        Each requested translation is subdivided at 8 mm. Unreachable, colliding
        or partial paths return None; never execute a partial insertion.
        """
        q = np.asarray(q_start, float).copy()
        path = [q.copy()]
        current, _ = self.tcp_pose(q)
        rotations = rotation if np.asarray(rotation).ndim == 3 else None
        for index, target in enumerate(positions):
            target = np.asarray(target, float)
            n = max(1, int(np.ceil(np.linalg.norm(target - current) / step_size)))
            start = current.copy()
            R = np.asarray(rotations[index] if rotations is not None else rotation)
            for k in range(1, n + 1):
                point = start + (target - start) * k / n
                nxt = self.solve_pose(q, point, R, joints=joints, max_iters=90, restarts=0)
                if nxt is None or not self.edge_free(q, nxt, allow_finger_object,
                                                     carrying, margin, resolution=.025):
                    return None
                path.extend(self._interpolate(q, nxt, resolution=.025)[1:])
                q = nxt
            current = target.copy()
        return self.simplify_path(path, allow_finger_object=allow_finger_object,
                                  carrying=carrying, margin=margin, cartesian=True)

    def simplify_path(self, path, carrying=False, allow_finger_object=False,
                      cartesian=False, margin=.001):
        """Greedy checked shortcuts, with bounded candidate tests per waypoint.

        Cartesian shortcuts additionally keep sampled TCPs within 4 mm of the
        endpoint segment and 0.06 rad of interpolated endpoint orientation.
        Every accepted edge is collision checked again. The capped lookahead
        avoids a quadratic search on densely sampled long trajectories.
        """
        if path is None or len(path) < 3:
            return path
        path = [np.asarray(q, float).copy() for q in path]
        opts = dict(allow_finger_object=allow_finger_object, carrying=carrying, margin=margin)
        def valid(a, b):
            samples = self._interpolate(a, b, resolution=.025 if cartesian else .045)
            if cartesian:
                p0, r0 = self.tcp_pose(a)
                p1, r1 = self.tcp_pose(b)
                delta = p1 - p0
                length2 = float(delta @ delta)
                angle = rotation_error(r1, r0)
            for k, q in enumerate(samples):
                if not self.collision_free(q, **opts):
                    return False
                if cartesian:
                    pos, R = self.tcp_pose()
                    t = k / max(1, len(samples) - 1)
                    along = np.clip((pos - p0) @ delta / max(length2, 1e-16), 0., 1.)
                    if np.linalg.norm(pos - (p0 + along * delta)) > .004:
                        return False
                    reference = Rotation.from_rotvec(t * angle).as_matrix() @ r0
                    if np.linalg.norm(rotation_error(reference, R)) > .06:
                        return False
            return True
        out = [path[0]]
        index = 0
        while index < len(path) - 1:
            limit = min(len(path) - 1, index + 250)
            # Longest tested valid shortcut wins; at most 12 candidates.
            candidates = sorted(set(int(v) for v in np.linspace(index + 1, limit, 12)), reverse=True)
            chosen = None
            for end in candidates:
                if valid(path[index], path[end]):
                    chosen = end
                    break
            if chosen is None:
                self.last_failure = "path simplification found an invalid original edge"
                return None
            out.append(path[chosen])
            index = chosen
        return out
