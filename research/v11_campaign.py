"""Reproduce retrospectively completed cohorts without changing the chosen model."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys


def evaluate(task):
    module, revision = task
    subprocess.run(
        [sys.executable, "-B", "-m", "research." + module, str(revision)], check=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                evaluate,
                [(m, r) for r in range(13) for m in ("v11_suite", "v11_dynamics")],
            )
        )
