"""Fixed-size independent delayed-law replication."""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from .v11_dynamics import run
from .reference import ROOT


def changed(task):
    seed, rev = task
    return dict(seed=seed, metrics=run(seed, rev, True))


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import sys

    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    start = 16301 + 100 * (rev - 13)
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(changed, [(s, rev) for s in range(start, start + 96)]))
    (
        ROOT
        / (
            "measurements/v11_replication.json"
            if rev == 13
            else f"measurements/v11_replication_{rev}.json"
        )
    ).write_text(json.dumps(rows, indent=1))
