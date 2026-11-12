"""Runs every test_* function in this folder (pytest isn't installed in the pnr env).

    python -m nav_uncertainty.tests.run_all
"""
import importlib
import sys
import time
import traceback
from pathlib import Path


def main():
    failed = 0
    for f in sorted(Path(__file__).parent.glob("test_*.py")):
        mod = importlib.import_module(f"nav_uncertainty.tests.{f.stem}")
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
                print(f"PASS {f.stem}.{name} ({time.perf_counter() - t0:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {f.stem}.{name}")
                traceback.print_exc()
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
