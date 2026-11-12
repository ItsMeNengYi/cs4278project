"""The navigation agent: sense, localise, map, replan, then follow the path.

Each switch removes or keeps one oracle:
    map_mode  "known"  : costs from the true walls      | "unknown": occupancy grid built from lidar
    pose_mode "true"   : exact pose                     | "odom": dead reckoning | "pf": particle filter
    planner   "dstar"  : D* Lite repairs its search     | "astar": A* from scratch on every map change
"""
import time
from dataclasses import dataclass

import numpy as np

from .localization import ParticleFilter
from .occupancy import GridSpec, LikelihoodField, OccupancyMap, cost_grid
from .planner import Planner, GridGraph, astar


@dataclass
class AgentConfig:
    map_mode: str = "unknown"
    pose_mode: str = "pf"
    planner: str = "dstar"
    res: float = 0.1
    half_x: float = 0.32            # base footprint half-extents (0.287 x 0.264 m) plus a 3 cm margin
    half_y: float = 0.30
    scan_every: int = 5             # one scan every 5 env steps (25 sim substeps, about 10 Hz)
    v_max: float = 0.01             # metres per env step
    lookahead: float = 0.3
    goal_tol: float = 0.03
    max_steps: int = 15000
    n_particles: int = 300
    estimate_scale: bool = True     # particle filter also estimates the odometry scale bias
    field_res: float = 0.05         # likelihood-field resolution when the map is known
    check_astar: bool = False       # on every replan, also run A* from scratch and record the cost gap


def nearest_passable(cost, ix, iy, radius=6):
    """The estimate can land inside a wall cell; plan from the closest cell that isn't one."""
    nx, ny = cost.shape
    ix, iy = int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))
    if np.isfinite(cost[ix, iy]):
        return ix, iy
    x0, x1 = max(ix - radius, 0), min(ix + radius + 1, nx)
    y0, y1 = max(iy - radius, 0), min(iy + radius + 1, ny)
    win = cost[x0:x1, y0:y1]
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
    d = np.where(np.isfinite(win), (gx - ix) ** 2 + (gy - iy) ** 2, np.inf)
    k = np.unravel_index(np.argmin(d), d.shape)
    return int(gx[k]), int(gy[k])


def pursuit_cmd(path_xy, est, goal, cfg):
    """Aim at the first path point at least `lookahead` ahead of the closest one; move at most v_max."""
    if np.linalg.norm(goal - est) < cfg.lookahead:
        carrot = goal
    else:
        k = int(np.argmin(np.linalg.norm(path_xy - est, axis=1)))
        far = np.flatnonzero(np.linalg.norm(path_xy[k:] - est, axis=1) >= cfg.lookahead)
        carrot = path_xy[k + far[0]] if len(far) else goal
    cmd = carrot - est
    n = np.linalg.norm(cmd)
    return cmd if n <= cfg.v_max else cmd * (cfg.v_max / n)


def run_episode(world, gt, cfg, noise, seed=0):
    rng = np.random.default_rng(seed + 1000)
    t_start = time.perf_counter()
    spec = GridSpec(*gt.bounds, cfg.res)      # the arena's outer size is known, its interior is not
    goal = gt.goal                            # obs["goal_pos"] gives this in the real task
    goal_cell = spec.to_idx(goal)

    if cfg.map_mode == "known":
        omap = OccupancyMap.from_truth(spec, gt.occupancy(spec))
    else:
        omap = OccupancyMap(spec)

    est = world.true_xy() if cfg.pose_mode == "true" else world.noisy_start()
    pf, field = None, None
    if cfg.pose_mode == "pf":
        scale_args = {} if cfg.estimate_scale else {"scale_sigma": 0.0, "scale_walk": 0.0}
        pf = ParticleFilter(cfg.n_particles, est, noise.init_sigma, rng, **scale_args)
        if cfg.map_mode == "known":
            fine = GridSpec(*gt.bounds, cfg.field_res)
            field = LikelihoodField(fine, gt.occupancy(fine))
        else:
            field = LikelihoodField(spec, omap.occupied())

    cost = cost_grid(omap.occupied(), cfg.res, cfg.half_x, cfg.half_y)
    planner = Planner(cfg.planner, cost, nearest_passable(cost, *spec.to_idx(est)), goal_cell)
    path_xy = None

    log = {"true": [], "est": [], "spread": [], "collision": []}
    astar_gap = 0.0
    status = "timeout"
    spread = 0.0
    for t in range(cfg.max_steps):
        if t % cfg.scan_every == 0:
            angles, ranges = world.scan()
            if pf is not None:
                pf.update(angles, ranges, field)
                est = pf.estimate()[0]
            new_cost = None
            if cfg.map_mode == "unknown":
                omap.update(est, angles, ranges, noise.max_range)
                occ = omap.occupied()
                new_cost = cost_grid(occ, cfg.res, cfg.half_x, cfg.half_y)
                cost = new_cost
                if pf is not None:
                    field = LikelihoodField(spec, occ)
            start_cell = nearest_passable(cost, *spec.to_idx(est))
            replanned = planner.update(start_cell, new_cost)
            if replanned and cfg.check_astar:
                _, c_ref, _ = astar(GridGraph(cost), planner.start, planner.goal)
                if np.isfinite(c_ref) or np.isfinite(planner.cost):
                    astar_gap = max(astar_gap, abs(c_ref - planner.cost))
            cells = planner.path_cells()
            if cells is None:
                status = "no_path"
                break
            path_xy = spec.to_xy(*np.array(cells).T)

        cmd = pursuit_cmd(path_xy, est, goal, cfg)
        odom = world.step(cmd)
        if cfg.pose_mode == "true":
            est = world.true_xy()
        elif cfg.pose_mode == "odom":
            est = est + odom
        else:
            pf.predict(odom)
            est, spread = pf.estimate()

        true = world.true_xy()
        log["true"].append(true)
        log["est"].append(est.copy())
        log["spread"].append(spread)
        log["collision"].append(world.in_collision())
        if np.linalg.norm(goal - est) < cfg.goal_tol:
            status = "arrived"
            break

    true = np.array(log["true"])
    est_arr = np.array(log["est"])
    err = np.linalg.norm(true - est_arr, axis=1) if len(true) else np.zeros(1)
    final_err = float(np.linalg.norm(world.true_xy() - goal))
    metrics = {
        "status": status,
        "success": status == "arrived" and final_err < 0.1,
        "steps": len(true),
        "final_goal_err": final_err,
        "path_length": float(np.sum(np.linalg.norm(np.diff(true, axis=0), axis=1))) if len(true) > 1 else 0.0,
        "collision_steps": int(np.sum(log["collision"])),
        "loc_rmse": float(np.sqrt(np.mean(err ** 2))),
        "loc_max": float(err.max()),
        "replans": planner.replans,
        "plan_time_s": planner.plan_time,
        "expansions": planner.expansions,
        "astar_gap": astar_gap,
        "wall_time_s": time.perf_counter() - t_start,
    }
    if pf is not None:
        # world.odom_scale is simulator-side truth, read here only for scoring
        metrics["scale_err"] = float(np.abs(pf.scale_estimate() - world.odom_scale).max())
    if cfg.map_mode == "unknown":
        truth = gt.occupancy(spec)
        occ, known = omap.occupied(), omap.known()
        metrics["map_precision"] = float((occ & truth).sum() / max(occ.sum(), 1))
        metrics["map_accuracy"] = float((occ == truth)[known].mean())
        metrics["map_known_frac"] = float(known.mean())
    log.update({"spec": spec, "occ": omap.occupied(), "known": omap.known(), "final_path": path_xy})
    return metrics, log
