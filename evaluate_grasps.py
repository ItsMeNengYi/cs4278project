"""Repeatable grasp-only evaluation; reports failures without hiding debug setup.

    python evaluate_grasps.py --seeds 0 1 --objects 2 5 14 --output grasp_results.json
"""
import argparse
import json
from pathlib import Path
import time

import pybullet as p
import grasping as G
from run_task import make_env, pick, place_on_table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--objects', type=int, nargs='+', default=[0, 1, 2, 5, 14])
    ap.add_argument('--place-on-table', action='store_true', help='debug setup; excluded from official-drop results')
    ap.add_argument('--output', default='grasp_results.json')
    args = ap.parse_args()
    results = []
    for seed in args.seeds:
        for obj in args.objects:
            env = None
            start = time.perf_counter()
            row = dict(seed=seed, object_index=obj, setup='debug_upright' if args.place_on_table else 'official_drop')
            try:
                env = make_env(seed, obj, True)
                if args.place_on_table:
                    place_on_table(env)
                action = env.agent_qpos.astype(float)
                G.wait_until_rest(env, action)
                _, lift = pick(env, action)
                row.update(lift)
                row['verified_pickup'] = bool(lift['held'])
                row['within_3000_steps'] = bool(lift['lift_step'] <= 3000)
            except Exception as error:
                row.update(verified_pickup=False, within_3000_steps=False,
                           error=f'{type(error).__name__}: {error}')
            finally:
                row['wall_seconds'] = time.perf_counter() - start
                if env is not None:
                    row['total_env_steps'] = env.step_count
                    p.disconnect(physicsClientId=env.pb_physics_client)
            results.append(row)
            Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row))
    successful = sum(r['verified_pickup'] and r['within_3000_steps'] for r in results)
    print(f'Verified within budget: {successful}/{len(results)}; details: {args.output}')


if __name__ == '__main__':
    main()
