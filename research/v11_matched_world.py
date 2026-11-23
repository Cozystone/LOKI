"""Matched acquisition and identical physical endpoints for all three planners."""

import json
from .reference import ROOT
from .v11_world import measure


if __name__ == "__main__":
    import sys

    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    shift = (rev - 13) * 100
    result = {
        key: [
            measure(s, rev, holdout=changed, controls=True)
            for s in range(start + shift, start + shift + 24)
        ]
        for key, start, changed in (("sealed", 12401, False), ("holdout", 13701, True))
    }
    (
        ROOT
        / (
            "measurements/v11_matched_world.json"
            if rev == 13
            else f"measurements/v11_matched_world_{rev}.json"
        )
    ).write_text(json.dumps(result, indent=1))
