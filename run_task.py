"""Full task: wait for the object, drive to it, grasp, lift, hold, carry it through the maze, reach the goal.

    python run_task.py                     # GUI, seed 42, object 5
    python run_task.py --seed 3 --object 2 --headless
"""
import argparse

import numpy as np
import pybullet as p

import grasping as G
import navigation as N
from grasp_candidates import generate_grasp_candidates
from grasp_motion import MotionPlanner

CARRY_REACH = 0.40                      # pad midpoint this far in front of the base centre while driving
PLACE_REACH = 0.60                      # at the goal the base stops this far short and reaches forward and down
MAX_EXTENT = 0.45                       # robot + object must stay this close to the base centre (corridor half-width 0.5)
CLEARANCE = 0.02                        # opening reserved beyond the grasp width


def make_env(seed, obj, headless):
    if headless:
        import pick_nav_reach_env as E
        from nav_uncertainty.sim import make_env as make_headless
        init = E.PickNavReachEnv.__init__
        E.PickNavReachEnv.__init__ = lambda self, seed=0, object_idx=obj: init(self, seed, object_idx)
        try:
            return make_headless(seed)
        finally:
            E.PickNavReachEnv.__init__ = init
    env = N.PickNavReachEnv(seed=seed, object_idx=obj)
    env.reset()
    return env


def place_on_table(env, gap=0.003):
    """Put the object upright on the middle of the table, at rest, instead of the env's drop from z = 5
    (which can bounce it off the table or wedge it into the table top). The env file is not changed."""
    lo, hi = max(furniture(env), key=lambda b: b[1][2])               # the table top
    centre = (lo[:2] + hi[:2]) / 2
    p.resetBasePositionAndOrientation(env.object_id, [*centre, 2.0], [0, 0, 0, 1])
    olo, _ = map(np.array, p.getAABB(env.object_id))
    z = 2.0 - olo[2] + hi[2] + gap                                    # bottom of the object just above the top
    p.resetBasePositionAndOrientation(env.object_id, [*centre, z], [0, 0, 0, 1])
    p.resetBaseVelocity(env.object_id, [0, 0, 0], [0, 0, 0])


def furniture(env):
    """AABBs of free-standing obstacles outside the maze (the table): bodies 0.3-2 m across with a top
    between 0.3 and 1.5 m, other than the robot and the object. The floor and the maze are bigger, the
    goal marker is smaller. (getAABB covers the base link only: for the table that is just its top.)"""
    out = []
    for b in range(p.getNumBodies()):
        if b in (env.robot_id, env.object_id):
            continue
        lo, hi = map(np.array, p.getAABB(b))
        size = hi[:2] - lo[:2]
        if 0.3 < hi[2] < 1.5 and np.all(size < 2.0) and np.all(size > 0.3):
            out.append((lo, hi))
    return out


def block_box(m, lo, hi, margin=0.0):
    """Mark every 1 m cell overlapping the box [lo, hi] (+margin) as a wall."""
    r0, c0 = m.world_to_cell(lo[0] - margin, hi[1] + margin)
    r1, c1 = m.world_to_cell(hi[0] + margin, lo[1] - margin)
    m.grid[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = N.WALL


def nearest_free(m, cell):
    free = np.argwhere(m.grid == N.FREE)
    return tuple(int(v) for v in free[np.argmin(np.sum((free - np.array(cell)) ** 2, axis=1))])


def footprint(env):
    """How far the robot and the held object reach from the base centre, in the base frame:
    (forward, backward, sideways). Uses the corners of every link's AABB."""
    b = G.base_xy(env)
    yaw = env.agent_qpos[2]
    fwd, side = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    boxes = [p.getAABB(env.robot_id, l) for l in range(p.getNumJoints(env.robot_id))
             if p.getCollisionShapeData(env.robot_id, l)]
    boxes.append(p.getAABB(env.object_id))
    corners = np.array([[x, y] for lo, hi in boxes for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - b
    f, sd = corners @ fwd, corners @ side
    return float(f.max()), float(-f.min()), float(np.abs(sd).max())


def footprint_xy(env):
    """Largest |x| and |y| reach of the robot and the held object from the base centre."""
    b = G.base_xy(env)
    # skip the root link (-1) and the geometry-free links of the virtual x/y joints: they don't follow
    # the base in both axes
    boxes = [p.getAABB(env.robot_id, l) for l in range(p.getNumJoints(env.robot_id))
             if p.getCollisionShapeData(env.robot_id, l)]
    boxes.append(p.getAABB(env.object_id))
    lo = np.min([bx[0] for bx in boxes], axis=0)[:2] - b
    hi = np.max([bx[1] for bx in boxes], axis=0)[:2] - b
    return np.maximum(np.abs(lo), np.abs(hi))


def choose_pair(obs, keep=1.0, rng=None, support_top=None):
    """Compatibility helper; pick() evaluates the complete ranked candidate list."""
    candidates = generate_grasp_candidates(obs["object_pc"], obs["object_normals"],
        obs["object_pos"], G.MAX_OPENING - .012, support_top=support_top,
        pad_half_height=0.0)
    return candidates[0] if candidates else None


def route(start, goal, boxes, margin=0.3, clear=0.6):
    """Waypoints from `start` to `goal` for the base. If the straight line would take the base through a
    table (AABB grown by `margin`; the last `clear` m are exempt, since the planned base pose may sit
    partly under the table top), go round it on the side nearer the goal."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    for lo, hi in boxes:
        lo2, hi2 = lo[:2] - margin, hi[:2] + margin
        length = np.linalg.norm(goal - start)
        ts = np.linspace(0, max(0.0, 1 - clear / max(length, 1e-9)), 50)
        pts = start + ts[:, None] * (goal - start)
        if not np.any(np.all((pts > lo2) & (pts < hi2), axis=1)):
            continue
        wy = hi[1] + clear if goal[1] >= (lo[1] + hi[1]) / 2 else lo[1] - clear
        wx0 = lo[0] - clear if start[0] < (lo[0] + hi[0]) / 2 else hi[0] + clear
        in_band = lo[1] - margin < goal[1] < hi[1] + margin
        wx1 = (hi[0] + clear if wx0 < lo[0] else lo[0] - clear) if in_band else goal[0]
        via = [np.array([wx0, wy]), np.array([wx1, wy])]
        if in_band:
            via.append(np.array([wx1, goal[1]]))
        return [start] + via + [goal]
    return [start, goal]


def drive_to(env, action, xy, yaw):
    """Drive the base to (xy, yaw): facing the direction of travel, round the table if needed, then turn
    in place to `yaw`."""
    wps = route(G.base_xy(env), xy, furniture(env))
    N.follow_path_facing(env, wps, hold_q=action.copy())
    action[:3] = env.agent_qpos[:3]
    G.turn_to(env, action, yaw)


def pick(env, action, attempts=5, draw=False):
    """Try distinct ranked grasps; plan all arm motion before execution."""
    G.raise_arm(env, action)
    failed_local = []
    last_error = "no feasible trajectory"
    for attempt in range(attempts):
        obs = G.wait_until_rest(env, action)
        _, top = G.support_body(env, obs["object_pos"])
        candidates = generate_grasp_candidates(obs["object_pc"], obs["object_normals"],
            obs["object_pos"], G.MAX_OPENING - .012, support_top=top,
            pad_half_height=0., max_candidates=40, friction_mu=G.planning_friction(env),
            normal_uncertainty_deg=3.)
        print(f"  {len(candidates)} diverse antipodal candidates")
        plan = None
        with MotionPlanner(env, seed=attempt) as mp:
            object_R = np.array(p.getMatrixFromQuaternion(obs["object_xyzw"])).reshape(3,3)
            for pair in candidates:
                local_mid = object_R.T @ (pair["midpoint"] - obs["object_pos"])
                local_axis = object_R.T @ pair["axis"]
                if any(np.linalg.norm(local_mid-m) < .015 and abs(local_axis @ a) > .95
                       for m,a in failed_local):
                    continue
                plan = G.plan_grasp(env,pair,obs["object_pc"],G.base_xy(env),planner=mp)
                if plan is not None:
                    break
        if plan is None:
            raise RuntimeError("No candidate has a complete checked arm trajectory")
        if draw:
            env.draw_or_update_arrow(pair["p1"],pair["n1"])
            env.draw_or_update_arrow(pair["p2"],pair["n2"])
        try:
            drive_to(env,action,plan["base"],plan["yaw"])
            # execute_grasp reobserves object pose and replans against actual arrived joints.
            grasp = G.execute_grasp(env,action,plan)
            lift = G.lift_and_hold(env,action)
            return grasp,lift
        except RuntimeError as error:
            last_error = str(error)
            print(f"  attempt {attempt+1} failed: {error}")
            failed_local.append((local_mid.copy(),local_axis.copy()))
            G.release(env,action)
            G.raise_arm(env,action)
    raise RuntimeError(f"No verified pickup in {attempts} attempts: {last_error}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--object", type=int, default=5)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--drop", action="store_true", help="compatibility flag: official drop is already the default")
    ap.add_argument("--place-on-table", action="store_true", help="debug only: replace the official drop with upright placement")
    ap.add_argument("--grasp-only", action="store_true", help="stop after pickup and 60-step hold verification")
    args = ap.parse_args()

    env = make_env(args.seed, args.object, args.headless)
    if args.place_on_table:
        place_on_table(env)
    action = env.agent_qpos.astype(float).copy()

    print("1. waiting for the object to come to rest")
    obs = G.wait_until_rest(env, action)
    print(f"   object at {np.round(obs['object_pos'], 3)}")

    print("2-4. planning the grasp, driving there, grasping, lifting and holding")
    g, lift = pick(env, action, draw=not args.headless)
    print(f"   pickup verified; within 3000 steps: {lift['lift_step'] <= 3000}")
    if args.grasp_only:
        if not args.headless:
            input("Press Enter to close...")
        p.disconnect(physicsClientId=env.pb_physics_client)
        return

    print("5. backing away from the table and moving the object into the carry pose")
    G.back_out(env, action, furniture(env), clearance=0.6)   # room to turn without sweeping the table
    yaw = env.agent_qpos[2]
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    reach = CARRY_REACH
    G.carry_pose(env, action, heading, reach, obs["goal_pos"][2])
    fwd, back, side = footprint(env)
    if fwd > MAX_EXTENT:                                      # long object: pull it in towards the body
        reach -= fwd - MAX_EXTENT
        G.carry_pose(env, action, heading, reach, obs["goal_pos"][2])
        fwd, back, side = footprint(env)
    reach = float(np.linalg.norm(env._get_state()[2][:2] - G.base_xy(env)))
    print(f"   robot + object reach {fwd:.2f} m forward, {back:.2f} m back, {side:.2f} m sideways "
          f"(limit {MAX_EXTENT} m); object {reach:.2f} m in front of the base")
    if max(fwd, back, side) > MAX_EXTENT:
        print("   WARNING: the carry pose is wider than the corridor clearance")

    print("6. navigating the maze, facing the direction of travel")
    m = N.GridMap(obs["cube_positions"], obs["goal_pos"])
    for lo, hi in furniture(env):
        block_box(m, lo, hi)
    start = nearest_free(m, m.world_to_cell(*G.base_xy(env)))
    path = m.astar(start)
    if path is None:
        raise RuntimeError("no path to the goal")
    cells = [np.asarray(m.cell_to_world(r, c)) for r, c in path]
    waypoints = [G.base_xy(env)] + cells
    # the robot arrives facing along the last stretch and stops PLACE_REACH m short of the goal; it then
    # reaches forward and down to it (the tucked carry pose can't get the object as low as the goal)
    last_dir = G.unit(cells[-1] - cells[-2]) if len(cells) > 1 else heading
    waypoints[-1] = np.asarray(obs["goal_pos"][:2]) - PLACE_REACH * last_dir
    ok, steps, wall_steps, obj_hits = N.follow_path_facing(env, waypoints, hold_q=action.copy(),
                                                           still_ok=G.holding)
    action[:3] = env.agent_qpos[:3]
    print(f"   reached: {ok}, {steps} steps, robot touching walls {wall_steps} steps, "
          f"object touching anything but the gripper {obj_hits} steps, still holding: {G.holding(env)}")

    print("7. placing the object at the goal")
    try:
        G.move_object_by(env, action, np.asarray(obs["goal_pos"]) - env._get_state()[2],
                         joints=[G.TORSO] + G.ARM)
    except RuntimeError as e:
        print(f"   {e}")
    obs, info = env.step(action)

    print("\n---- summary ----")
    print(f"contact-to-pad error: right {g['err_points'][G.R_FINGER]*1000:.1f} mm, "
          f"left {g['err_points'][G.L_FINGER]*1000:.1f} mm (tolerance {G.TOLERANCE*1000:.1f} mm)")
    print(f"lift {lift['lift']*100:.1f} cm by step {lift['lift_step']}, hold drift {lift['hold_drift']*1000:.1f} mm")
    print(f"total steps {env.step_count}, dist_to_goal {info['dist_to_goal']:.3f} m, success {info['success']}")
    if not args.headless:
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
