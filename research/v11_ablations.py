"""Retrospective mechanism ablations on identical candidate seeds."""

import json
import sys
from .reference import ROOT
from .v11_world import measure
from .v11_identity import run


if __name__ == "__main__":
    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rows = []
    for seed in range(11101 + 100 * rev, 11125 + 100 * rev):
        row = {"seed": seed}
        for key, variant in (("ordinary_source", 14), ("robust_source", 15)):
            row[key] = run(seed, variant)["joint"][0]
        for key, variant in (("local", 15), ("propagated", 16)):
            trace = measure(seed, variant)
            row[key] = [trace["novel"]["success"], trace["recovery"]]
        rows.append(row)
    (ROOT / f"measurements/v11_ablations_{rev}.json").write_text(
        json.dumps(rows, indent=1)
    )
