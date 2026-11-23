import json
from .reference import ROOT
from .v11_world import measure
from .v11_alias import probe
from .v11_identity import run as identity
from .v11_dynamics import run as dynamics
from .v11_detour import run as detour


if __name__ == "__main__":
    import sys

    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    start = 13701 + (rev - 13) * 100
    rows = []
    for s in range(start, start + 24):
        rows.append(
            dict(
                world=measure(s, rev, holdout=True),
                alias=probe(s, rev, True),
                entity=identity(s, rev, True),
                dynamics=dynamics(s, rev, True),
                detour=detour(s, rev),
            )
        )
    (
        ROOT
        / (
            "measurements/v11_holdout.json"
            if rev == 13
            else f"measurements/v11_holdout_{rev}.json"
        )
    ).write_text(json.dumps(rows, default=float, indent=1))
