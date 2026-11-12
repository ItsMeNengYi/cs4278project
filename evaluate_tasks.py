"""Isolated, bounded evaluation of every supplied object and multiple maze seeds.

    python evaluate_tasks.py --seeds 0 1 3 --workers 4

Uses the official drop by default. Every trial writes a log and stage metrics;
timeouts, crashes, pickup-only results and full task success remain distinct.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 3])
    ap.add_argument('--objects', type=int, nargs='+', default=list(range(15)))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--grasp-only', action='store_true')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'evaluation')
    args = ap.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        ap.error('workers and timeout must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Freeze executable sources once: a long batch must not mix implementations
    # if development continues while its subprocesses are being scheduled.
    sources = {str(p.relative_to(ROOT)):p.read_bytes() for p in
               sorted(list(ROOT.glob('*.py'))+list((ROOT/'nav_uncertainty').rglob('*.py')))}
    digest = hashlib.sha256()
    for name,data in sorted(sources.items()):
        digest.update(name.encode()+b'\0'+data+b'\0')
    source_hash = digest.hexdigest()
    frozen = out / ('source-'+source_hash[:12])
    for name,data in sources.items():
        path = frozen/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:
            raise RuntimeError(f'Frozen source was modified: {path}')
        if not path.exists():
            path.write_bytes(data)
    assets = frozen/'assets'
    if not assets.exists():
        assets.symlink_to(ROOT/'assets',target_is_directory=True)
    (out/'manifest.json').write_text(json.dumps(dict(source_hash=source_hash,
        source_directory=str(frozen),python=sys.version,
        dependencies={name:version(name) for name in ['numpy','scipy','pybullet','trimesh','matplotlib']},
        seeds=args.seeds,objects=args.objects,timeout_seconds=args.timeout),indent=2)+'\n')
    names = [p.name for p in sorted((ROOT / 'assets/ycb_objects').iterdir())]
    if any(i < 0 or i >= len(names) for i in args.objects):
        ap.error(f'object indices must be between 0 and {len(names)-1}')

    def trial(seed, obj):
        stem = f'object{obj:02d}_seed{seed}'
        metrics = out / f'{stem}.json'
        log = out / f'{stem}.log'
        # The aggregate is based on this invocation, never stale trial files.
        metrics.unlink(missing_ok=True)
        cmd = [sys.executable, '-u', str(frozen / 'run_task.py'), '--headless',
               '--seed', str(seed), '--object', str(obj), '--result', str(metrics)]
        if args.grasp_only:
            cmd.append('--grasp-only')
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg',
                   MPLCONFIGDIR='/workspace/.cache/matplotlib')
        started = time.perf_counter()
        timed_out = False
        with log.open('w') as stream:
            try:
                proc = subprocess.run(cmd, cwd=frozen, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=args.timeout)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        row = json.loads(metrics.read_text()) if metrics.exists() else {}
        row.update(seed=seed, object_index=obj, object_name=names[obj],
                   source_hash=source_hash,
                   mode='pickup' if args.grasp_only else 'full_task',
                   exit_code=code, timed_out=timed_out, log=str(log),
                   wall_seconds=time.perf_counter()-started)
        if timed_out or code != 0:
            row['success'] = False
            row.setdefault('error', 'Trial timed out' if timed_out else f'Process exited {code}')
        metrics.write_text(json.dumps(row, indent=2) + '\n')
        return row

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(trial, seed, obj) for seed in args.seeds for obj in args.objects]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            results.sort(key=lambda r: (r['seed'], r['object_index']))
            (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    passed = sum(bool(r.get('success')) for r in results)
    print(f'{passed}/{len(results)} passed; results: {out / "results.json"}')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
