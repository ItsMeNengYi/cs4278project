"""Full task: wait for the object, drive to it, grasp, lift, hold, carry it through the maze, reach the goal.

    python run_task.py                     # GUI, fastest execution, seed 42, object 5
    python run_task.py --seed 3 --object 2 --headless
    python run_task.py --realtime          # pace physics for watching the GUI
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pybullet as p

import grasping as G
import navigation as N
from grasp_candidates import generate_grasp_candidates
from grasp_motion import MotionPlanner

CARRY_REACH = 0.40                      # seed reach for the compact object-centre search
PLACE_REACH = 0.60                      # footprint-centre standoff before final whole-body positioning
MAX_EXTENT = G.CARRY_HALF_SPAN           # maximum half-size of the centred world-axis footprint
CLEARANCE = 0.02                        # opening reserved beyond the grasp width


def make_env(seed, obj, headless, realtime=False):
    env = N.PickNavReachEnv(seed=seed, object_idx=obj, gui=not headless,
                            realtime=realtime)
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


def connected_start(env, m, offset):
    """Connect the measured footprint to the grid without cutting the table."""
    centre = G.base_xy(env)+offset
    free = [tuple(int(v) for v in rc) for rc in np.argwhere(m.grid == N.FREE)]
    free.sort(key=lambda rc: np.linalg.norm(np.asarray(m.cell_to_world(*rc))-centre))
    with MotionPlanner(env) as mp:
        q0 = env.agent_qpos.astype(float)
        mp.attach_object(q0)
        for cell in free[:40]:
            if m.astar(cell) is None:
                continue
            xy = np.asarray(m.cell_to_world(*cell))-offset
            target = q0.copy()
            target[:2] = [xy[0]-G.BASE_X0,xy[1]]
            if mp.edge_free(q0,target,carrying=True,margin=.0002):
                return cell
    raise RuntimeError('No collision-checked connection from the table to the maze grid')


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


def footprint_bounds(env):
    boxes = [p.getAABB(env.robot_id,l) for l in range(p.getNumJoints(env.robot_id))
             if p.getCollisionShapeData(env.robot_id,l)]
    boxes.append(p.getAABB(env.object_id))
    base = G.base_xy(env)
    return (np.min([box[0] for box in boxes],axis=0)[:2]-base,
            np.max([box[1] for box in boxes],axis=0)[:2]-base)


def choose_pair(obs, keep=1.0, rng=None, support_top=None):
    """Compatibility helper; pick() evaluates the complete ranked candidate list."""
    candidates = generate_grasp_candidates(obs["object_pc"], obs["object_normals"],
        obs["object_pos"], G.MAX_OPENING - .012, support_top=support_top,
        pad_half_height=0.0)
    return candidates[0] if candidates else None


def route(start, goal, boxes, margin=0.4, clear=0.6):
    """Visibility route to a stance, ending perpendicular to its table edge.

    Entering along an edge can hit a table leg even when the final stance is
    free. Stay outside the expanded table until the final normal approach.
    """
    import heapq
    start, goal = np.asarray(start,float), np.asarray(goal,float)
    if not boxes:
        return [start,goal]
    lo,hi = boxes[0]
    lo,hi = lo[:2],hi[:2]
    def outside(point):
        distances = [lo[0]-point[0],point[0]-hi[0],lo[1]-point[1],point[1]-hi[1]]
        edge = int(np.argmax(distances))
        result = point.copy()
        axis = edge//2
        result[axis] = (lo[axis]-clear) if edge%2==0 else (hi[axis]+clear)
        return result
    entry = outside(goal)
    prefix = [start]
    if np.all(start>lo-margin) and np.all(start<hi+margin):
        prefix.append(outside(start))
    nodes = [prefix[-1],entry]+[np.array([x,y]) for x in (lo[0]-clear,hi[0]+clear)
                               for y in (lo[1]-clear,hi[1]+clear)]
    def visible(a,b):
        # Slab intersection with the table's expanded rectangle.
        lower,upper = 0.,1.
        for k in range(2):
            delta = b[k]-a[k]
            if abs(delta)<1e-12:
                if not lo[k]-margin<a[k]<hi[k]+margin:
                    return True
            else:
                t0,t1 = sorted(((lo[k]-margin-a[k])/delta,(hi[k]+margin-a[k])/delta))
                lower,upper = max(lower,t0),min(upper,t1)
        return upper<lower
    queue = [(0.,0)]
    costs, parents = {0:0.},{}
    while queue:
        cost,i = heapq.heappop(queue)
        if cost>costs[i]:
            continue
        if i==1:
            path = [1]
            while path[-1]!=0:
                path.append(parents[path[-1]])
            return prefix+[nodes[j] for j in path[::-1][1:]]+[goal]
        for j,target in enumerate(nodes):
            if i==j or not visible(nodes[i],target):
                continue
            total = cost+float(np.linalg.norm(target-nodes[i]))
            if total<costs.get(j,np.inf):
                costs[j],parents[j] = total,i
                heapq.heappush(queue,(total,j))
    raise RuntimeError('No route around the table to the grasp stance')


def drive_to(env, action, xy, yaw):
    """Drive the base to (xy, yaw): facing the direction of travel, round the table if needed, then turn
    in place to `yaw`."""
    wps = route(G.base_xy(env), xy, furniture(env))
    # The supplied base is holonomic. Keep the raised arm facing its current
    # direction during unladen repositioning instead of sweeping it across the
    # table at every intermediate corner.
    ok, _, _ = N.follow_path(env, wps[:-1], hold_q=action.copy(), max_iters=1200)
    action[:3] = env.agent_qpos[:3]
    if not ok:
        raise RuntimeError('Base could not reach the planned grasp stance')
    G.turn_to(env, action, yaw)
    ok,_,_ = N.follow_path(env,[G.base_xy(env),np.asarray(xy)],hold_q=action.copy(),max_iters=400)
    action[:3] = env.agent_qpos[:3]
    if not ok:
        raise RuntimeError('Base could not complete the final approach to the grasp stance')


def pick(env, action, attempts=5, draw=False):
    """Try distinct ranked grasps; plan all arm motion before execution."""
    G.raise_arm(env, action)
    failed_local = []
    last_error = "no feasible trajectory"
    for attempt in range(attempts):
        obs = G.wait_until_rest(env, action)
        _, top = G.support_body(env, obs["object_pos"])
        options = dict(support_top=top, pad_half_height=0., max_candidates=160,
                       friction_mu=G.planning_friction(env), normal_uncertainty_deg=1.,
                       diversity_distance=.004)
        candidates = generate_grasp_candidates(obs["object_pc"], obs["object_normals"],
            obs["object_pos"], G.MAX_OPENING-.0004, min_width=.003, **options)
        if not candidates:
            # Thin rims can have only one sample on each face. Keep them as
            # low-confidence proposals; exact hand geometry and a physical
            # lift/hold must still pass, including the real minimum jaw gap.
            candidates = generate_grasp_candidates(obs['object_pc'],obs['object_normals'],
                obs['object_pos'],G.MAX_OPENING-.0004,min_width=.001,min_patch_points=1,**options)
        print(f"  {len(candidates)} diverse antipodal candidates")
        plan = None
        with MotionPlanner(env, seed=attempt) as mp:
            object_R = np.array(p.getMatrixFromQuaternion(obs["object_xyzw"])).reshape(3,3)
            # Rank complete hand poses, not just contact pairs. Try side
            # grasps across the entire diverse set before accepting a tip-heavy
            # top-down pinch of the very first pair.
            for vertical_range in ((-.1,.2),(.200001,1.1)):
                for pair in candidates:
                    local_mid = object_R.T @ (pair["midpoint"] - obs["object_pos"])
                    local_axis = object_R.T @ pair["axis"]
                    if any(np.linalg.norm(local_mid-m) < .015 and abs(local_axis @ a) > .95
                           for m,a in failed_local):
                        continue
                    plan = G.plan_grasp(env,pair,obs["object_pc"],G.base_xy(env),planner=mp,
                                       vertical_range=vertical_range)
                    if plan is not None:
                        break
                if plan is not None:
                    break
            rejection_detail = f'{getattr(mp, "grasp_rejections", {})}; last: {mp.last_failure}'
        if plan is None:
            raise RuntimeError(f'No candidate has a complete checked arm trajectory: {rejection_detail}')
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
            try:
                G.release(env,action)
                G.raise_arm(env,action)
            except RuntimeError as recovery_error:
                raise RuntimeError(f'Pickup attempt failed ({error}); recovery failed ({recovery_error})') from recovery_error
    raise RuntimeError(f"No verified pickup in {attempts} attempts: {last_error}")


def run_episode(env, grasp_only=False, draw=False):
    """Execute the assignment and retain stage metrics even if a later stage fails."""
    result = env.task_result = dict(stage="settling", verified_pickup=False,
                                  navigation_reached=False, success=False)
    def checkpoint(stage, **metrics):
        result.update(stage=stage, **metrics)
        path = getattr(env, 'result_path', None)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(result, total_env_steps=env.step_count), indent=2)+'\n')
    checkpoint('settling')
    action = env.agent_qpos.astype(float).copy()

    print("1. waiting for the object to come to rest")
    obs = G.wait_until_rest(env, action)
    print(f"   object at {np.round(obs['object_pos'], 3)}")

    print("2-4. planning the grasp, driving there, grasping, lifting and holding")
    checkpoint('pickup')
    g, lift = pick(env, action, draw=draw)
    result.update(lift, verified_pickup=True,
                  contact_error_m=max(g['err_points'].values()))
    checkpoint('pickup_verified')
    print(f"   pickup verified; within 3000 steps: {lift['lift_step'] <= 3000}")
    if not lift['within_step_budget']:
        raise RuntimeError('Pickup completed after the 3000-action budget')
    if grasp_only:
        result.update(stage='complete', success=bool(lift['within_step_budget']))
        return result

    print("5. moving into a compact carry pose, then backing away from the table")
    checkpoint('carry')
    yaw = env.agent_qpos[2]
    # The exit corridor runs along x. A cross-body carry can preserve the
    # grasp's world orientation while avoiding a large lateral goal offset.
    heading = np.array([1. if abs(np.cos(yaw))<.7 else np.sign(np.cos(yaw)), 0.])
    reach = CARRY_REACH
    G.carry_pose(env, action, heading, reach, obs["goal_pos"][2])
    G.back_out(env, action, furniture(env), clearance=0.6)
    fwd, back, side = footprint(env)
    lo, hi = footprint_bounds(env)
    offset, half_size = (lo+hi)/2, (hi-lo)/2
    reach = float(np.linalg.norm(env._get_state()[2][:2] - G.base_xy(env)))
    print(f"   robot + object reach {fwd:.2f} m forward, {back:.2f} m back, {side:.2f} m sideways "
          f"(limit {MAX_EXTENT} m); object {reach:.2f} m in front of the base")
    print(f'   footprint half-size {np.round(half_size,3)}, centre offset {np.round(offset,3)} m')
    if max(half_size) > MAX_EXTENT:
        raise RuntimeError('The carry pose exceeds corridor clearance')
    result['footprint_m'] = [fwd, back, side]
    result.update(footprint_half_size_m=half_size.tolist(), footprint_offset_m=offset.tolist())

    print("6. navigating the maze with the holonomic base, keeping grasp orientation fixed")
    checkpoint('navigation')
    m = N.GridMap(obs["cube_positions"], obs["goal_pos"])
    for lo, hi in furniture(env):
        block_box(m, lo, hi)
    start = connected_start(env,m,offset)
    path = m.astar(start)
    if path is None:
        raise RuntimeError("no path to the goal")
    cells = [np.asarray(m.cell_to_world(r, c)) for r, c in path]
    # Plan the centre of the whole footprint, rather than its asymmetric base.
    # Fixed yaw makes this offset constant during holonomic translation.
    waypoints = [G.base_xy(env)] + [cell-offset for cell in cells]
    # the robot arrives facing along the last stretch and stops PLACE_REACH m short of the goal; it then
    # reaches forward and down to it (the tucked carry pose can't get the object as low as the goal)
    last_dir = G.unit(cells[-1] - cells[-2]) if len(cells) > 1 else heading
    waypoints[-1] = np.asarray(obs["goal_pos"][:2]) - PLACE_REACH * last_dir - offset
    ok, steps, wall_steps, obj_hits = N.follow_path_facing(env, waypoints, hold_q=action.copy(),
        still_ok=G.holding, face_path=False, v_max=.015, accel=.0002)
    action[:3] = env.agent_qpos[:3]
    result.update(navigation_reached=bool(ok), navigation_steps=steps,
                  robot_wall_steps=wall_steps, object_hit_steps=obj_hits)
    print(f"   reached: {ok}, {steps} steps, robot touching walls {wall_steps} steps, "
          f"object touching anything but the gripper {obj_hits} steps, still holding: {G.holding(env)}")
    if not ok or not G.holding(env):
        raise RuntimeError(f'Navigation failed or the object was dropped: {getattr(env,"_grasp_failure","")}')

    print("7. placing the object at the goal")
    checkpoint('reach')
    G.reach_goal(env, action, np.asarray(obs["goal_pos"]))
    obs, info = env.step(action)

    print("\n---- summary ----")
    print(f"contact-to-pad error: right {g['err_points'][G.R_FINGER]*1000:.1f} mm, "
          f"left {g['err_points'][G.L_FINGER]*1000:.1f} mm (tolerance {G.TOLERANCE*1000:.1f} mm)")
    print(f"lift {lift['lift']*100:.1f} cm by step {lift['lift_step']}, hold drift {lift['hold_drift']*1000:.1f} mm")
    print(f"total steps {env.step_count}, dist_to_goal {info['dist_to_goal']:.3f} m, success {info['success']}")
    result.update(dist_to_goal=float(info['dist_to_goal']),
                  success=bool(info['success'] and lift['within_step_budget']
                               and env.step_count <= 15000), stage='complete')
    if not result['success']:
        raise RuntimeError('The object did not reach the goal within the assignment limits')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--object', type=int, default=5)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--realtime', action='store_true',
                    help='pace physics at real time; by default run as fast as possible')
    ap.add_argument('--drop', action='store_true', help='official drop is already the default')
    ap.add_argument('--place-on-table', action='store_true', help='debug upright placement instead of official drop')
    ap.add_argument('--grasp-only', action='store_true')
    ap.add_argument('--result', type=Path, help='write stage metrics as JSON, including on failure')
    args = ap.parse_args()
    env = None
    result = dict(seed=args.seed, object_index=args.object,
                  solver_iterations=200,
                  internal_substeps=4, realtime=args.realtime,
                  setup='debug_upright' if args.place_on_table else 'official_drop')
    started = time.perf_counter()
    try:
        env = make_env(args.seed, args.object, args.headless, realtime=args.realtime)
        env.result_path = args.result
        if args.place_on_table:
            place_on_table(env)
        run_episode(env, grasp_only=args.grasp_only, draw=not args.headless)
        if not args.headless:
            input('Press Enter to close...')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if env is not None:
            result.update(getattr(env, 'task_result', {}), total_env_steps=env.step_count)
            p.disconnect(physicsClientId=env.pb_physics_client)
            env._maze_dir.cleanup()
        result['wall_seconds'] = time.perf_counter() - started
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == "__main__":
    main()
