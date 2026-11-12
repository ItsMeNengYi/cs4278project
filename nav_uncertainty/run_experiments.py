"""Run every condition over several seeds, write a CSV and per-run plots.

    python -m nav_uncertainty.run_experiments                      # all conditions, seeds 0-4
    python -m nav_uncertainty.run_experiments --conditions slam_lite --seeds 0 --gui
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from .agent import AgentConfig, run_episode
from .sim import GroundTruth, close_env, make_env
from .world import NoiseConfig, UncertainNavWorld

CONDITIONS = {
    # name:           (map,      pose,   planner, PF estimates odometry scale)
    "oracle":         ("known",   "true", "dstar", True),    # same information as navigation.py
    "mapping":        ("unknown", "true", "dstar", True),    # remove the map oracle
    "mapping_astar":  ("unknown", "true", "astar", True),    # same, replanning from scratch (planner comparison)
    "dead_reckoning": ("known",   "odom", "dstar", True),    # remove the pose oracle, no correction
    "mcl_no_bias":    ("known",   "pf",   "dstar", False),   # particle filter over (x, y) only
    "mcl":            ("known",   "pf",   "dstar", True),    # particle filter that also learns the odometry bias
    "slam_lite":      ("unknown", "pf",   "dstar", True),    # remove both oracles
}

OUT = Path(__file__).resolve().parent / "results"


def run_one(name, seed, noise, gui=False, check_astar=False, plot=True):
    map_mode, pose_mode, planner, estimate_scale = CONDITIONS[name]
    cfg = AgentConfig(map_mode=map_mode, pose_mode=pose_mode, planner=planner,
                      estimate_scale=estimate_scale, check_astar=check_astar)
    env = make_env(seed, gui=gui, realtime=gui)
    try:
        gt = GroundTruth(env)
        world = UncertainNavWorld(env, noise, gt.entrance_xy(), seed=seed)
        metrics, log = run_episode(world, gt, cfg, noise, seed=seed)
    finally:
        close_env(env)
    if plot:
        save_plot(name, seed, gt, log, metrics)
    return metrics


def save_plot(name, seed, gt, log, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = log["spec"]
    true, est = np.array(log["true"]), np.array(log["est"])
    extent = [spec.x_min, spec.x_min + spec.nx * spec.res, spec.y_min, spec.y_min + spec.ny * spec.res]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    truth = gt.occupancy(spec).astype(float)
    ax1.imshow(truth.T, origin="lower", extent=extent, cmap="Greys", alpha=0.35)
    mapped = np.where(log["known"], log["occ"].astype(float), np.nan)
    ax1.imshow(mapped.T, origin="lower", extent=extent, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
    if len(true):
        ax1.plot(true[:, 0], true[:, 1], "g-", lw=1.5, label="true")
        ax1.plot(est[:, 0], est[:, 1], "r--", lw=1, label="estimate")
    ax1.plot(*gt.goal, "r*", ms=12)
    ax1.set_xlim(1.5, extent[1])
    ax1.set_title(f"{name}, seed {seed}: {metrics['status']}, {metrics['steps']} steps")
    ax1.legend(loc="lower right")

    if len(true):
        err = np.linalg.norm(true - est, axis=1)
        ax2.plot(err, label="|true - est|")
        ax2.plot(log["spread"], label="PF spread")
    ax2.set_xlabel("step")
    ax2.set_ylabel("m")
    ax2.set_title("localisation error")
    ax2.legend()
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}_seed{seed}.png", dpi=110)
    plt.close(fig)


def summarise(rows):
    keys = ["success", "steps", "collision_steps", "loc_rmse", "loc_max", "final_goal_err",
            "replans", "plan_time_s", "expansions"]
    print(f"\n{'condition':<15}" + "".join(f"{k:>16}" for k in keys))
    for name in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == name]
        vals = []
        for k in keys:
            v = np.array([float(r[k]) for r in rs])
            vals.append(f"{v.mean():.3f}±{v.std():.3f}" if k != "success" else f"{int(v.sum())}/{len(v)}")
        print(f"{name:<15}" + "".join(f"{v:>16}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--check-astar", action="store_true", help="verify D* Lite against A* on every replan")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--odom-scale-sigma", type=float, default=None)
    args = ap.parse_args()

    noise = NoiseConfig()
    if args.odom_scale_sigma is not None:
        noise = replace(noise, odom_scale_sigma=args.odom_scale_sigma)

    rows = []
    OUT.mkdir(exist_ok=True)
    for name in args.conditions:
        for seed in args.seeds:
            m = run_one(name, seed, noise, gui=args.gui, check_astar=args.check_astar, plot=not args.no_plot)
            row = {"condition": name, "seed": seed, **m}
            rows.append(row)
            print(f"{name:<15} seed {seed}: {m['status']:<8} steps {m['steps']:>5}  "
                  f"collisions {m['collision_steps']:>4}  loc_rmse {m['loc_rmse']:.3f}  "
                  f"goal_err {m['final_goal_err']:.3f}  plan {m['plan_time_s']:.2f}s  "
                  f"wall {m['wall_time_s']:.0f}s", flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("condition", "seed"), k))
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summarise(rows)
    print(f"\nwrote {OUT / 'runs.csv'} and per-run plots to {OUT}")


if __name__ == "__main__":
    main()
