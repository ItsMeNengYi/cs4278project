"""Grasp planning and execution. Planning uses an isolated Bullet client.

World coordinates are metres. A grasp frame has x pointing into the object,
y along jaw opening and z=x cross y. Its origin is the symmetric pad midpoint.
Navigation helpers are retained for compatibility with run_task.py.
"""
import numpy as np
try:
    import pybullet as p
except ImportError:
    p = None
from scipy.spatial.transform import Rotation

from grasp_motion import MotionPlanner
from grasp_candidates import generate_grasp_candidates

BASE_X0 = -2.0
TORSO, ARM, FINGERS = 3, list(range(6, 13)), [13, 14]
GRIPPER_LINK, R_FINGER, L_FINGER = 20, 21, 22
FINGER_HALF_LEN, FINGER_HALF_H = 0.0315, 0.012
PAD = {R_FINGER: (np.array([0., -.014, 0.]), np.array([0., -1., 0.])),
       L_FINGER: (np.array([0., .014, 0.]), np.array([0., 1., 0.]))}
TOLERANCE = .006
MAX_OPENING = .10285
CARRY_HALF_SPAN = .48  # 2 cm nominal clearance per side in a 1 m corridor
PALM_X, PALM_HALF_Y, PALM_HALF_Z = (.032, .154), .063, .04
RAISED_ARM = {6:0., 7:-1., 8:0., 9:0., 10:0., 11:0., 12:0.}


def client(env):
    return env.pb_physics_client


def step(env, action):
    return env.step(action)


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-10 or not np.all(np.isfinite(v)):
        raise ValueError('Expected a finite nonzero vector')
    return v / n


def wait_until_rest(env, action, tol=.01, settle=20, max_steps=1200):
    """Require support and a stable measured pose window, including orientation.

    Contact-solver jitter can alternate angular velocity while the actual pose
    is stable. Verify both displacement and rotation over the settling window
    instead of demanding one very small instantaneous angular velocity.
    """
    window = []
    for _ in range(max_steps):
        obs, _ = step(env, action)
        vel, omega = p.getBaseVelocity(env.object_id, physicsClientId=client(env))
        supported = any(c[2] != env.robot_id and c[9] > .001
                        for c in p.getContactPoints(bodyA=env.object_id, physicsClientId=client(env)))
        ok = supported and np.linalg.norm(vel) < tol and np.linalg.norm(omega) < .5
        window = (window + [obs])[-settle:] if ok else []
        if len(window) == settle:
            origin = window[0]
            displacement = max(np.linalg.norm(o['object_pos']-origin['object_pos']) for o in window)
            initial_R = Rotation.from_quat(origin['object_xyzw'])
            rotation = max((Rotation.from_quat(o['object_xyzw'])*initial_R.inv()).magnitude() for o in window)
            if displacement < .002 and rotation < .02:
                return obs
    raise RuntimeError('Object did not settle on a support surface')


def support_body(env, obj_pos):
    best, top = None, -np.inf
    for i in range(p.getNumBodies(physicsClientId=client(env))):
        b = p.getBodyUniqueId(i, physicsClientId=client(env))
        if b in (env.robot_id, env.object_id, getattr(env, 'goal_marker_id', None)):
            continue
        for link in range(-1, p.getNumJoints(b, physicsClientId=client(env))):
            if not p.getCollisionShapeData(b, link, physicsClientId=client(env)):
                continue
            lo, hi = map(np.array, p.getAABB(b, link, physicsClientId=client(env)))
            if np.all(lo[:2] <= obj_pos[:2]) and np.all(obj_pos[:2] <= hi[:2]) and hi[2] <= obj_pos[2] and hi[2] > top:
                best, top = b, float(hi[2])
    if best is None:
        raise RuntimeError('No support surface found under the object')
    return best, top


def required_lift(env, height=.15):
    """Also clear the actual table if a dropped object landed on the floor."""
    _, top = support_body(env, env._get_state()[2])
    for i in range(p.getNumBodies(physicsClientId=client(env))):
        body = p.getBodyUniqueId(i,physicsClientId=client(env))
        label = ' '.join(n.decode().lower() for n in p.getBodyInfo(body,physicsClientId=client(env)))
        if 'table' in label:
            for link in range(-1,p.getNumJoints(body,physicsClientId=client(env))):
                if p.getCollisionShapeData(body,link,physicsClientId=client(env)):
                    top = max(top,p.getAABB(body,link,physicsClientId=client(env))[1][2])
    bottom = p.getAABB(env.object_id,physicsClientId=client(env))[0][2]
    return max(height,float(top+.11-bottom)),float(top)


def link_frame(env, link):
    st = p.getLinkState(env.robot_id, link, computeForwardKinematics=1, physicsClientId=client(env))
    return np.array(st[4]), Rotation.from_quat(st[5]).as_matrix(), st


def pad_points(env, link, target):
    o, R, st = link_frame(env, link)
    mid, _ = PAD[link]
    local = R.T @ (np.asarray(target) - o)
    closest = np.array([np.clip(local[0], -FINGER_HALF_LEN, FINGER_HALF_LEN), mid[1],
                        np.clip(local[2], -FINGER_HALF_H, FINGER_HALF_H)])
    return o + R @ mid, o + R @ closest, st


def pad_centres(env):
    return {k: pad_points(env, k, np.zeros(3))[0] for k in (R_FINGER, L_FINGER)}


def grasp_error(env, targets):
    """Distance to the actual pad rectangle; pad-center error is reported separately."""
    return {k: float(np.linalg.norm(pad_points(env, k, t)[1] - t)) for k, t in targets.items()}


def gripper_axis(env):
    return link_frame(env, GRIPPER_LINK)[1][:, 0]


def opening_q(gap):
    return float(np.clip(gap / 2 - .001425, 0., .05))


def planning_friction(env):
    """Conservative coefficient from the unchanged Bullet contact materials."""
    obj_mu = p.getDynamicsInfo(env.object_id,-1,physicsClientId=client(env))[1]
    finger_mu = min(p.getDynamicsInfo(env.robot_id,k,physicsClientId=client(env))[1]
                    for k in (R_FINGER,L_FINGER))
    mu = obj_mu * finger_mu
    if mu <= np.tan(np.deg2rad(3.)):
        raise RuntimeError('Insufficient friction margin for this antipodal policy')
    return float(mu)


def object_relative_pose(env):
    pos, R, _ = link_frame(env, GRIPPER_LINK)
    op, oq = p.getBasePositionAndOrientation(env.object_id, physicsClientId=client(env))
    return R.T @ (np.array(op) - pos), R.T @ Rotation.from_quat(oq).as_matrix()


def fingers_touching(env, min_force=.01):
    """Loaded contacts on the inner pad faces, not just any finger collision."""
    touched = set()
    for c in p.getContactPoints(bodyA=env.robot_id, bodyB=env.object_id, physicsClientId=client(env)):
        k = c[3]
        if k not in PAD or c[9] < min_force:
            continue
        pos, R, _ = link_frame(env, k)
        loc = R.T @ (np.array(c[5]) - pos)
        if abs(loc[1] - PAD[k][0][1]) < .005 and abs(loc[0]) <= FINGER_HALF_LEN + .003 and abs(loc[2]) <= FINGER_HALF_H + .003:
            touched.add(k)
    return touched


def holding(env):
    touched = fingers_touching(env)
    if len(touched) != 2:
        env._grasp_failure = f'loaded pads {sorted(touched)}'
        return False
    reference = getattr(env, '_grasp_reference', None)
    if reference is None:
        return True  # only a contact observation until close_gripper establishes the reference
    pos, rot = object_relative_pose(env)
    # An antipodal pinch is not a rigid weld. Curved objects can roll while
    # retaining two loaded inner-pad contacts. Rotation is reported separately
    # during the stationary hold test; it alone does not establish a drop.
    delta_R = rot @ reference[1].T
    swing = np.arccos(np.clip(delta_R[1, 1], -1., 1.))
    contact_local = getattr(env, '_grasp_contact_local', np.zeros(3))
    contact_drift = (pos + rot @ contact_local) - (reference[0] + reference[1] @ contact_local)
    drift = np.linalg.norm(contact_drift)
    env._grasp_failure = f'contact drift {drift:.4f} m, swing {swing:.3f} rad'
    # Reject separation beyond the whole pad travel, rather than treating a
    # few millimetres of loaded rolling/sliding as a dropped object. The actual
    # contacts above must remain on both inner pad rectangles at every action.
    maximum = getattr(env, '_grasp_capture_radius',
                      np.linalg.norm(reference[0])+2*FINGER_HALF_LEN)
    return bool(np.linalg.norm(pos) < maximum)


def _joint_limits(env):
    velocities = []
    for j in env.joint_indices:
        info = p.getJointInfo(env.robot_id, j, physicsClientId=client(env))
        velocities.append(info[11] if info[11] > 0 else .8)
    return np.array(velocities)


def execute_path(env, action, path, *, carrying=False, speed=.45):
    """Execute a prevalidated path with smooth, URDF-velocity-limited interpolation."""
    if path is None or len(path) == 0:
        raise RuntimeError('No valid motion path')
    path = [np.asarray(q, float).copy() for q in path]
    reduced = [path[0]]
    for i in range(1,len(path)-1):
        a,b = path[i]-reduced[-1],path[i+1]-path[i]
        if np.linalg.norm(a) > 1e-10 and np.linalg.norm(b) > 1e-10 and np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b)) < 1e-7:
            continue
        reduced.append(path[i])
    if len(path)>1:
        reduced.append(path[-1])
    path = reduced
    dt = p.getPhysicsEngineParameters(physicsClientId=client(env))['fixedTimeStep'] * env.substeps
    vmax = _joint_limits(env) * speed
    # quintic time scaling max s'=1.875, max s''~5.78; conservative acceleration caps
    amax = np.maximum(vmax * 2., .05)
    for q_target in path[1:]:
        start = env.agent_qpos.astype(float)
        delta = np.asarray(q_target) - start
        continuous = np.asarray(getattr(env, 'joint_upper', np.zeros_like(delta))) > 300
        delta[continuous] = (delta[continuous] + np.pi) % (2*np.pi) - np.pi
        # A carried grasp keeps the actual closure setpoint, not the planner's open jaws.
        if carrying:
            delta[FINGERS] = 0.
        seconds = max(float(np.max(1.875 * np.abs(delta) / vmax)),
                      float(np.max(np.sqrt(5.78 * np.abs(delta) / amax))), dt)
        n = int(np.ceil(seconds / dt))
        for k in range(1, n + 1):
            t = k / n
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            target = start + s * delta
            if carrying:
                target[FINGERS] = action[FINGERS]
            action[:] = target
            step(env, action)
            if carrying and not holding(env):
                raise RuntimeError(f'Lost loaded pad contact or object slipped during motion: {getattr(env, "_grasp_failure", "")}')
    for _ in range(5):
        step(env, action)
        if carrying and not holding(env):
            raise RuntimeError('Object slipped while settling at a waypoint')
    checked = [TORSO] + ARM
    if np.max(np.abs(env.agent_qpos[checked] - action[checked])) > .04:
        raise RuntimeError('Arm failed to track the planned motion')


def settle_fingers(env, action, tol=.0005, max_steps=150):
    for _ in range(max_steps):
        if np.all(np.abs(env.agent_qpos[FINGERS] - action[FINGERS]) < tol):
            return
        step(env, action)
    raise RuntimeError('Gripper did not reach its requested opening')


def open_gripper(env, action, gap=MAX_OPENING):
    env.grip_controller = None
    env._grasp_reference = None
    env._grasp_contact_local = np.zeros(3)
    action[FINGERS] = opening_q(gap)
    settle_fingers(env, action)


def close_gripper(env, action, max_steps=180, stable_steps=8):
    stable = 0
    for _ in range(max_steps):
        action[FINGERS] = np.maximum(action[FINGERS] - .0004, 0.)
        step(env, action)
        stable = stable + 1 if len(fingers_touching(env)) == 2 else 0
        if stable >= stable_steps:
            # Maintain a small compression target; no 5 cm overclosure or physics weld.
            action[FINGERS] = np.maximum(env.agent_qpos[FINGERS] - .002, 0.)
            for _ in range(12):
                step(env, action)
            if len(fingers_touching(env)) != 2:
                raise RuntimeError('Pad contact was not sustained during closure')
            env._grasp_reference = object_relative_pose(env)
            return
    raise RuntimeError('Could not establish sustained contact on both inner pads')


def maintain_grip_pressure(env, command):
    """Take up lost jaw pressure without reopening an established grasp.

    A curved object can settle onto a narrower cross-section during transport.
    Keep the original compression, with small extra position corrections only
    when a finger's measured load falls below the friction/weight requirement.
    No force, friction coefficient, constraint or object pose is changed here.
    """
    loads = {R_FINGER:0., L_FINGER:0.}
    for c in p.getContactPoints(bodyA=env.robot_id, bodyB=env.object_id,
                               physicsClientId=client(env)):
        if c[3] in loads:
            loads[c[3]] += max(0., c[9])
    force = np.array([loads[R_FINGER], loads[L_FINGER]])
    correction = .0002*np.clip(1.-force/env._grip_min_force, 0., 1.)
    minimum = getattr(env, '_grip_min_command', np.zeros(2))
    env._grip_command = np.clip(env._grip_command-correction, minimum, .05)
    command = np.asarray(command).copy()
    command[FINGERS] = env._grip_command
    return command


def _approaches(pair, points, support_top, robot_xy, n_dirs=16):
    p1, p2 = np.asarray(pair['p1']), np.asarray(pair['p2'])
    mid = (p1 + p2) / 2
    # Parallel jaws should align with the opposing surface normals. Sampled
    # contacts can be staggered tangentially: aligning the jaws with their
    # connecting segment tilts flat pads against flat faces, reducing a broad
    # patch to unstable edge contact. The finite pad rectangles accommodate
    # the tangential offset; execution still checks both selected contacts.
    y = unit(np.asarray(pair['n1']) - np.asarray(pair['n2']))
    if y @ (p1-p2) < 0:
        y = -y
    ref = np.array([0., 0., 1.]) if abs(y[2]) < .95 else np.array([1., 0., 0.])
    up = unit(ref - y * (ref @ y))
    side = np.cross(y, up)
    to_robot = unit(np.r_[np.asarray(robot_xy) - mid[:2], 0.])
    options = []
    for th in np.linspace(0., 2 * np.pi, n_dirs, endpoint=False):
        d = np.cos(th) * up + np.sin(th) * side
        if d[2] < -.05:
            continue
        x = -d
        R = np.column_stack((x, y, np.cross(x, y)))
        face = float(np.max((np.asarray(points) - mid) @ d))
        insert = max(0., face - PALM_X[0] + .004)
        if insert > FINGER_HALF_LEN - .001:
            continue
        tcp = mid + insert * d
        # Conservative palm and finger support clearance; exact mesh checks follow.
        finger_low = tcp[2] - abs(y[2]) * (pair['width'] / 2 + .012) - abs(x[2]) * FINGER_HALF_LEN - abs(R[2, 2]) * FINGER_HALF_H
        palm_low = tcp[2] + min(-.154 * x[2], -.032 * x[2]) - .063 * abs(y[2]) - .04 * abs(R[2, 2])
        deficit = support_top + .001 - min(finger_low, palm_low)
        if deficit > 0 and d[2] > .1:
            insert += deficit / d[2]
            tcp = mid + insert * d
            finger_low += deficit
            palm_low += deficit
        if insert > FINGER_HALF_LEN - .001 or min(finger_low, palm_low) < support_top + .0005:
            continue
        for extra in (0., .003, .006, .009):
            if insert + extra <= FINGER_HALF_LEN - .001:
                options.append(dict(dir=d, rotation=R, tcp=tcp+extra*d,
                                    insert=insert+extra, face=face,
                                    score=.3*d[2] - .8*(d @ to_robot) + 4*(insert+extra)))
    return sorted(options, key=lambda a: a['score'])


def plan_grasp(env, pair, points, robot_xy, clearance=.008, standoff=.05,
               n_dirs_tried=24, standoffs=(.65, .75, .55, .85), log=print, planner=None,
               vertical_range=(-.1,1.1)):
    """Return checked transit, straight approach and lift trajectories; never move live state."""
    own = planner is None
    mp = MotionPlanner(env) if own else planner
    if not hasattr(mp, 'grasp_rejections'):
        mp.grasp_rejections = {}
    def rejected(stage):
        mp.grasp_rejections[stage] = mp.grasp_rejections.get(stage, 0) + 1
    try:
        mp.sync_scene()
        _, top = support_body(env, env._get_state()[2])
        gap = MAX_OPENING
        if gap - pair['width'] < .0004:
            rejected('jaw_width')
            mp.last_failure = (f'contact width {1000*pair["width"]:.2f} mm exceeds the '
                               f'{1000*gap:.2f} mm jaw gap with 0.40 mm opening reserve')
            return None
        q_now = env.agent_qpos.astype(float)
        approaches = _approaches(pair, points, top, robot_xy)[:n_dirs_tried]
        if not approaches:
            rejected('hand_approach')
            mp.last_failure = ('no hand approach fits the palm, 30.5 mm available finger reach '
                               'and support clearance at the sampled contacts; an antipodal '
                               'pair alone does not establish hand accessibility')
            return None
        for ap in approaches:
            d, R, target = ap['dir'], ap['rotation'], ap['tcp']
            if not vertical_range[0] <= d[2] <= vertical_range[1]:
                continue
            if not mp.fingers_clear(target, R, gap):
                rejected('finger_geometry')
                continue
            mid = (pair['p1'] + pair['p2']) / 2
            dh = unit(d[:2] if np.linalg.norm(d[:2]) > .3 else np.asarray(robot_xy) - mid[:2])
            out = max(.06, ap['face'] - ap['insert'] + FINGER_HALF_LEN + standoff)
            pre = target + out * d
            stances = [(r, np.array([[np.cos(angle),-np.sin(angle)],
                                     [np.sin(angle),np.cos(angle)]]) @ dh)
                       for r in standoffs for angle in (0., np.pi/2, -np.pi/2, np.pi)]
            for r, stance_dir in stances:
                q_start = q_now.copy()
                base = mid[:2] + r * stance_dir
                q_start[:3] = [base[0] - BASE_X0, base[1], np.arctan2(-stance_dir[1], -stance_dir[0])]
                q_start[FINGERS] = opening_q(gap)
                if not mp.collision_free(q_start):
                    rejected('base_stance_collision')
                    continue
                q_pre = mp.solve_pose(q_start, pre, R, collision_options={})
                if q_pre is None:
                    rejected('pregrasp_ik')
                    continue
                if not mp.collision_free(q_pre):
                    rejected('pregrasp_collision')
                    continue
                positions = np.linspace(pre, target, max(3, int(out / .008) + 1))[1:]
                insertion = mp.plan_cartesian(q_pre, positions, R, margin=.0002)
                if insertion is None:
                    rejected('insertion')
                    continue
                transit = mp.plan_joint_path(q_start, q_pre)
                if transit is None:
                    rejected('arm_transit')
                    continue
                # Preflight a closed-grasp lift including object geometry. Actual measured
                # contact transform is used to replan the final lift after closure.
                q_contact = insertion[-1].copy()
                q_contact[FINGERS] = opening_q(pair['width'])
                mp.attach_object(q_contact)
                lift_height, _ = required_lift(env, height=.30)
                lift_preview = mp.plan_cartesian(q_contact,
                    target + np.linspace(0., lift_height, max(20,int(lift_height/.008)))[1:,None]*np.array([0.,0.,1.]), R, carrying=True)
                if lift_preview is None:
                    rejected('lift_preview')
                    continue
                log(f"  feasible grasp width {pair['width']:.3f} m; base {np.round(base, 2)}, approach {np.round(d, 2)}")
                return dict(pair=pair, dir=d, rotation=R, tcp=target, base=base, yaw=float(q_start[2]),
                            fingers=opening_q(gap), transit=transit, insertion=insertion, lift_preview=lift_preview,
                            q_pre=q_pre, q_grasp=insertion[-1], support_top=top,
                            object_pose=env._get_state()[2:], approach=ap,
                            contact={R_FINGER: np.array(pair['p1']), L_FINGER: np.array(pair['p2'])},
                            assign={R_FINGER:(pair['p1'],pair['n1']), L_FINGER:(pair['p2'],pair['n2'])})
        return None
    finally:
        if own:
            mp.close()


def execute_grasp(env, action, plan, log=print):
    pos, quat = env._get_state()[2:]
    oldpos, oldquat = plan['object_pose']
    if np.linalg.norm(pos - oldpos) > .006 or (Rotation.from_quat(quat).inv() * Rotation.from_quat(oldquat)).magnitude() > .08:
        raise RuntimeError('Object moved after planning; refresh the grasp')
    open_gripper(env, action, 2 * (plan['fingers'] + .001425))
    # Replan from measured joints at the actual arrived base pose.
    with MotionPlanner(env) as mp:
        q0 = env.agent_qpos.astype(float)
        pre, _ = mp.tcp_pose(plan['q_pre'])
        seed = plan['q_pre'].copy()
        seed[:3] = q0[:3]
        qpre = mp.solve_pose(seed, pre, plan['rotation'])
        if qpre is None:
            raise RuntimeError('Pregrasp IK failed after base arrival')
        path = mp.plan_joint_path(q0, qpre)
        if path is None:
            raise RuntimeError(f'Arm transit planning failed: {mp.last_failure}')
        log('  executing checked arm transit')
        execute_path(env, action, path)
        pos_now, quat_now = env._get_state()[2:]
        if np.linalg.norm(pos_now-oldpos) > .006 or (Rotation.from_quat(quat_now).inv()*Rotation.from_quat(oldquat)).magnitude() > .08:
            raise RuntimeError('Object moved during arm transit; refresh the grasp')
        mp.sync_scene()
        cur, _ = mp.tcp_pose(env.agent_qpos)
        positions = np.linspace(cur, plan['tcp'], max(3, int(np.linalg.norm(cur-plan['tcp'])/.006)+1))[1:]
        path = mp.plan_cartesian(env.agent_qpos.astype(float), positions, plan['rotation'], margin=.0002)
        if path is None:
            raise RuntimeError(f'Insertion planning failed: {mp.last_failure}')
        log('  executing checked Cartesian insertion')
        execute_path(env, action, path)
    current_pos, current_quat = env._get_state()[2:]
    if np.linalg.norm(current_pos-oldpos) > .006 or (Rotation.from_quat(current_quat).inv()*Rotation.from_quat(oldquat)).magnitude() > .08:
        raise RuntimeError('Object moved during insertion; refresh the grasp')
    close_gripper(env, action)
    current_pos,current_quat = env._get_state()[2:]
    object_motion = Rotation.from_quat(current_quat).as_matrix() @ Rotation.from_quat(oldquat).as_matrix().T
    contacts = {k: current_pos + object_motion @ (point-oldpos) for k,point in plan['contact'].items()}
    errors = grasp_error(env,contacts)
    if max(errors.values()) > TOLERANCE:
        raise RuntimeError(f'Actual grasp missed the selected pad contacts: {errors}')
    # Monitor the pinched material points, rather than the COM: a cylinder
    # rotating about an offset contact line moves its COM without dropping.
    pressure_centres = []
    manifolds = p.getContactPoints(bodyA=env.robot_id, bodyB=env.object_id,
                                   physicsClientId=client(env))
    for finger in (R_FINGER, L_FINGER):
        loaded = [c for c in manifolds if c[3] == finger and c[9] > .01]
        pressure_centres.append(np.average([c[6] for c in loaded], axis=0,
                                           weights=[c[9] for c in loaded]))
    env._grasp_contact_local = Rotation.from_quat(current_quat).inv().apply(
        np.mean(pressure_centres, axis=0) - current_pos)
    lo,hi = p.getAABB(env.object_id,physicsClientId=client(env))
    corners = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])
                        for z in (lo[2],hi[2])])
    object_radius = float(np.linalg.norm(corners-current_pos,axis=1).max())
    pad_radius = np.linalg.norm([FINGER_HALF_LEN,MAX_OPENING/2,FINGER_HALF_H])
    env._grasp_capture_radius = object_radius+pad_radius+.006
    mass = p.getDynamicsInfo(env.object_id,-1,physicsClientId=client(env))[0]
    env._grip_min_force = max(1., 1.5*mass*9.8/(2*planning_friction(env)))
    env._grip_command = action[FINGERS].copy()
    env._grip_min_command = np.maximum(env._grip_command-.002, 0.)
    env.grip_controller = lambda command: maintain_grip_pressure(env, command)
    return dict(assign=plan['assign'], contact=contacts, err_points=errors,
                approach=plan['approach'], touching=fingers_touching(env))


def move_joints(env, action, q_target, joints, max_dq=None):
    target = env.agent_qpos.astype(float)
    target[joints] = np.asarray(q_target)[joints]
    carrying = getattr(env, '_grasp_reference', None) is not None
    with MotionPlanner(env) as mp:
        if carrying:
            mp.attach_object(env.agent_qpos)
        path = mp.plan_joint_path(env.agent_qpos.astype(float), target, carrying=carrying)
        if path is None:
            raise RuntimeError(f'Joint motion planning failed: {mp.last_failure}')
        execute_path(env, action, path, carrying=carrying)


def raise_arm(env, action, steps=None):
    target = env.agent_qpos.astype(float)
    target[TORSO] = .15
    for k, v in RAISED_ARM.items():
        target[k] = v
    move_joints(env, action, target, [TORSO] + ARM)


def ik_move(env, action, pad_targets, mid_targets=None, joints=None, axis=(0.,0.,-1.),
            opening=None, tol=.004, **kwargs):
    """Compatibility wrapper: plan the rigid TCP pose, validate a path, then execute."""
    targets = mid_targets if mid_targets is not None else pad_targets
    position = np.mean(list(targets.values()), axis=0)
    currentR = link_frame(env, GRIPPER_LINK)[1]
    x = currentR[:,0] if axis is None else unit(axis)
    y = currentR[:,1] if opening is None else unit(opening)
    y = unit(y - x * (x @ y))
    R = np.column_stack((x,y,np.cross(x,y)))
    carrying = getattr(env, '_grasp_reference', None) is not None
    with MotionPlanner(env) as mp:
        if carrying:
            mp.attach_object(env.agent_qpos)
        cur, _ = mp.tcp_pose(env.agent_qpos)
        positions = np.linspace(cur, position, max(2,int(np.linalg.norm(position-cur)/.006)+1))[1:]
        path = mp.plan_cartesian(env.agent_qpos.astype(float), positions, R, carrying=carrying, joints=joints)
        execute_path(env, action, path, carrying=carrying)
    return grasp_error(env, pad_targets)


def move_object_by(env, action, delta, step_len=.006, tol=.004, joints=None):
    if not holding(env):
        raise RuntimeError('Object is not securely held')
    with MotionPlanner(env) as mp:
        q0 = env.agent_qpos.astype(float)
        mp.attach_object(q0)
        pos, R = mp.tcp_pose(q0)
        delta = np.asarray(delta)
        positions = pos + np.linspace(0., 1., max(2,int(np.ceil(np.linalg.norm(delta)/step_len))+1))[1:,None] * delta
        path = mp.plan_cartesian(q0, positions, R, carrying=True, joints=joints, margin=.0002)
        if path is None:
            raise RuntimeError(f'Held-object motion planning failed: {mp.last_failure}')
        execute_path(env, action, path, carrying=True)
    return True


def lift_and_hold(env, action, height=.30, hold_steps=60, log=print):
    z0 = float(env._get_state()[2][2])
    height, top = required_lift(env,height)
    move_object_by(env, action, [0., 0., height])
    lifted_at = env.step_count
    rel0, R0 = object_relative_pose(env)
    positions, drifts, rotations, bottoms = [], [], [], []
    for _ in range(max(50,hold_steps)):
        obs, _ = step(env, action)
        if not holding(env):
            raise RuntimeError('Object slipped during hold verification')
        rel, R = object_relative_pose(env)
        positions.append(obs['object_pos'])
        drifts.append(np.linalg.norm(rel-rel0))
        rotations.append(Rotation.from_matrix(R@R0.T).magnitude())
        bottoms.append(p.getAABB(env.object_id, physicsClientId=client(env))[0][2])
    positions = np.asarray(positions)
    rise = float(positions[:,2].min()-z0)
    clearance = float(min(bottoms)-top)
    drift, angle = float(max(drifts)), float(max(rotations))
    held = rise >= .10 and clearance >= .10 and drift <= .01 and angle <= .15
    result = dict(lift=rise, lift_step=lifted_at, hold_drift=drift, rotation_drift=angle,
                  bottom_clearance=clearance, held=bool(held), hold_steps=len(positions), within_step_budget=lifted_at<=3000)
    log(f'  lift {rise*100:.1f} cm, bottom clearance {clearance*100:.1f} cm; relative drift {drift*1000:.1f} mm')
    if not held:
        raise RuntimeError(f'Lift/hold validation failed: {result}')
    return result


def release(env, action, back_off=.10):
    open_gripper(env, action)
    with MotionPlanner(env) as mp:
        q0 = env.agent_qpos.astype(float)
        pos, R = mp.tcp_pose(q0)
        target = pos - back_off*R[:,0] + [0.,0.,.04]
        path = mp.plan_cartesian(q0, np.linspace(pos,target,15)[1:], R, allow_finger_object=True)
        if path is None:
            raise RuntimeError('No checked retreat after release')
        execute_path(env, action, path)


def rotation_about(axis, angle):
    return Rotation.from_rotvec(unit(axis)*angle).as_matrix()


def reach_goal(env, action, goal):
    """Use the torso and holonomic base before changing the arm's grasp pose.

    The assignment constrains the object's position, not a fixed base standoff.
    Whole-body positioning avoids forcing a tucked hand through the torso just
    to attain an arbitrary arm-only reach. All moves remain collision checked.
    """
    goal = np.asarray(goal, float)
    for _ in range(3):
        if np.linalg.norm(goal-env._get_state()[2]) < .06:
            return
        with MotionPlanner(env) as mp:
            q0 = env.agent_qpos.astype(float)
            mp.attach_object(q0)
            q = q0.copy()
            q[TORSO] = np.clip(q0[TORSO]+goal[2]-env._get_state()[2][2],
                               mp.lower[TORSO], mp.upper[TORSO])
            path = mp.plan_joint_path(q0, q, joints=[TORSO], carrying=True, margin=.0002)
            if path is not None:
                execute_path(env, action, path, carrying=True)
        delta = goal-env._get_state()[2]
        if np.linalg.norm(delta[:2]) < .02:
            break
        with MotionPlanner(env) as mp:
            q0 = env.agent_qpos.astype(float)
            mp.attach_object(q0)
            for fraction in (1., .7, .4):
                q = q0.copy()
                q[:2] += fraction*delta[:2]
                path = mp.plan_joint_path(q0, q, joints=[0,1], carrying=True, margin=.0002)
                if path is not None:
                    execute_path(env, action, path, carrying=True)
                    break
            else:
                break
    if np.linalg.norm(goal-env._get_state()[2]) >= .06:
        move_object_by(env, action, goal-env._get_state()[2], joints=[TORSO]+ARM)


def axis_angle(R):
    v = Rotation.from_matrix(R).as_rotvec()
    a = np.linalg.norm(v)
    return (v/a if a>1e-10 else np.array([0.,0.,1.])), float(a)


def carry_pose(env, action, heading, reach, height, **kwargs):
    """Plan a compact pose with the object included in collision checks."""
    if not holding(env):
        raise RuntimeError('Cannot carry an unverified grasp')
    with MotionPlanner(env) as mp:
        q0 = env.agent_qpos.astype(float)
        mp.attach_object(q0)
        _, R0 = mp.tcp_pose(q0)
        if not mp.collision_free(q0, carrying=True, margin=.0002):
            print(f'  carry start clearance failed: {mp.last_failure}')
        heading = np.asarray(heading, float)
        sideways = np.array([-heading[1], heading[0]])
        cartesian_paths, smallest_radius = 0, np.inf
        # Prefer a Cartesian transfer that preserves the grasp orientation.
        # Turning a top-down pinch over while folding can spill a real object,
        # even when the rigid attached-object model predicts a valid path.
        for z in (height+.1, height+.2, height+.3):
            for forward in (reach+.2, reach+.175, reach+.15, reach+.1, reach+.05, reach, reach-.1):
                for lateral in (0., .1, -.1, .2, -.2):
                    object_target = np.r_[base_xy(env)+forward*heading+lateral*sideways, z]
                    target = object_target - R0 @ mp.attached[0]
                    path = mp.plan_cartesian(q0, [target], R0, carrying=True,
                                              margin=.0002, step_size=.012)
                    if path is None:
                        continue
                    cartesian_paths += 1
                    lo, hi = mp.footprint_bounds(path[-1])
                    radius = float(np.max((hi-lo)/2))
                    smallest_radius = min(smallest_radius, radius)
                    if radius > CARRY_HALF_SPAN:
                        continue
                    getattr(env,'task_result',{})['carry_method'] = 'orientation_preserving_cartesian'
                    execute_path(env, action, path, carrying=True)
                    return env._get_state()[2]
        print(f'  carry Cartesian paths: {cartesian_paths}; smallest radius {smallest_radius:.3f} m')
        starts = [(q0, [q0])]
        for pan in (q0[ARM[0]], 0., .9, -.9):
            raised = q0.copy()
            raised[TORSO] = .30
            raised[ARM] = [pan, -1., 0., 0., 0., 0., 0.]
            if not mp.collision_free(raised, carrying=True, margin=.0002):
                continue
            path = mp.plan_joint_path(q0, raised, carrying=True, margin=.0002,
                                      max_iterations=600)
            if path is not None:
                starts.insert(0, (raised, path))
                break
        # Carrying only needs a compact, collision-free configuration, not an
        # arbitrarily fixed six-DOF TCP target. Search folded postures first;
        # validate the held object's geometry and the full turning envelope.
        rng = np.random.default_rng(91)
        nominal = np.array([1.2, 1.2, 0., 1.8, 0., 1.5, 0.])
        path_attempts = 0
        for index in range(400):
            q = q0.copy()
            q[TORSO] = .20 + .15*rng.random()
            q[ARM] = nominal + rng.normal(0., [.25,.3,.4,.25,.6,.35,.8])
            q = np.clip(q, mp.lower, mp.upper)
            if mp.radial_extent(q) > .445 or not mp.collision_free(q, carrying=True, margin=.0002):
                continue
            path = None
            for start, prefix in starts:
                suffix = mp.plan_joint_path(start, q, carrying=True, margin=.0002,
                                            max_iterations=600)
                if suffix is not None:
                    path = prefix + suffix[1:]
                    break
            path_attempts += 1
            if path is not None:
                getattr(env,'task_result',{})['carry_method'] = 'checked_folded_posture'
                execute_path(env, action, path, carrying=True)
                return env._get_state()[2]
            if path_attempts >= 8:
                break
        x = unit([heading[0],heading[1],0.])
        y = np.array([-heading[1], heading[0], 0.])
        options = [np.column_stack((x,[0.,0.,s],np.cross(x,[0.,0.,s]))) for s in (1.,-1.)]
        options += [np.column_stack(([0.,0.,s], y, np.cross([0.,0.,s],y))) for s in (-1.,1.)]
        for z in (height, height+.2, height+.35):
            object_target = np.r_[base_xy(env)+reach*np.asarray(heading), z]
            for R in options:
                target = object_target - R @ mp.attached[0]
                q = mp.solve_pose(q0,target,R, restarts=8,
                                  collision_options={'carrying':True, 'margin':.0002})
                if q is not None:
                    path = mp.plan_joint_path(q0,q,carrying=True, margin=.0002)
                    if path is not None:
                        execute_path(env,action,path,carrying=True)
                        return target
    raise RuntimeError(f'No collision-free compact carry pose: {mp.last_failure}')


# Existing base-motion helpers.

def find_support(env, obj_pos):
    """AABB of the highest body under the object (the table, or the floor), found by geometry."""
    best = None
    for b in range(p.getNumBodies()):
        if b in (env.robot_id, env.object_id):
            continue
        lo, hi = map(np.array, p.getAABB(b))
        if lo[0] <= obj_pos[0] <= hi[0] and lo[1] <= obj_pos[1] <= hi[1] and hi[2] <= obj_pos[2]:
            if best is None or hi[2] > best[1][2]:
                best = (lo, hi)
    if best is None:
        raise RuntimeError("no support surface under the object")
    return best

def base_xy(env):
    q = env.agent_qpos
    return np.array([q[0] + BASE_X0, q[1]])

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def touching_obstacle(env):
    return any(c[2] not in (0, env.object_id) for c in p.getContactPoints(bodyA=env.robot_id))

def approach(env, action, obj_pos, dist=0.5, v_max=0.01, w_max=0.02, max_steps=3000):
    """Raise the arm, turn to face the object and drive straight at it until the base centre is
    within `dist` m of the object (horizontal distance). Stops early, backing off 2 cm, if the robot
    touches anything other than the floor or the object (e.g. the table edge)."""
    for j, v in RAISED_ARM.items():
        action[j] = v
    action[TORSO] = 0.15
    for _ in range(150):
        step(env, action)

    obj_xy = np.asarray(obj_pos[:2], dtype=float)
    heading = np.arctan2(*(obj_xy - base_xy(env))[::-1])
    for _ in range(max_steps):                                  # turn in place
        err = wrap(heading - env.agent_qpos[2])
        if abs(err) < 0.01:
            break
        action[2] = env.agent_qpos[2] + np.clip(err, -w_max, w_max)
        step(env, action)

    for _ in range(max_steps):                                  # drive
        cur = base_xy(env)
        gap = np.linalg.norm(obj_xy - cur) - dist
        if gap <= 0.003:
            break
        d = unit(obj_xy - cur) * min(gap, v_max)
        action[0], action[1] = cur[0] + d[0] - BASE_X0, cur[1] + d[1]
        step(env, action)
        if touching_obstacle(env):
            back = base_xy(env) - unit(obj_xy - cur) * 0.02
            action[0], action[1] = back[0] - BASE_X0, back[1]
            for _ in range(30):
                step(env, action)
            break
    return float(np.linalg.norm(obj_xy - base_xy(env)))

def back_out(env, action, boxes, clearance=0.4, v_max=0.004, max_steps=2000):
    """Reverse straight back (opposite to the way the base faces) until the base centre is at least
    `clearance` m outside every box in `boxes` (xy AABBs, e.g. the table)."""
    def gap():
        b = base_xy(env)
        return min(np.linalg.norm(np.maximum(0, np.maximum(lo[:2] - b, b - hi[:2]))) for lo, hi in boxes)
    if not boxes:
        return
    yaw = env.agent_qpos[2]
    back = -np.array([np.cos(yaw), np.sin(yaw)])
    velocity = 0.
    acceleration = .0001
    for _ in range(max_steps):
        remaining = clearance - gap()
        if remaining <= .001:
            return
        velocity = min(velocity + acceleration, v_max,
                       np.sqrt(2 * acceleration * remaining), remaining)
        nxt = base_xy(env) + back * velocity
        action[0], action[1] = nxt[0] - BASE_X0, nxt[1]
        step(env, action)
        if not holding(env):
            raise RuntimeError(f'Object slipped while backing away from the table: {getattr(env, "_grasp_failure", "")}')
    raise RuntimeError('Could not back away from the table')

def turn_to(env, action, yaw, w_max=0.01, w_accel=0.0003):
    """Turn the base in place to `yaw` with a ramped turn rate."""
    yaw0 = float(env.agent_qpos[2])
    turn = wrap(yaw - yaw0)
    done, w = 0.0, 0.0
    while abs(turn) - done > 1e-4:
        left = abs(turn) - done
        w = min(w + w_accel, w_max, np.sqrt(2 * w_accel * left), left)
        done += w
        action[2] = yaw0 + np.copysign(done, turn)
        step(env, action)
    for _ in range(30):
        step(env, action)
