"""Machine-check the registered numerical gates from frozen measurements."""

import math
from .v11_report import read


def sign_probability(wins, losses):
    n = wins + losses
    return min(
        1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    )


def advantages(pairs):
    pairs = list(pairs)
    wins = sum(a > b for a, b in pairs)
    losses = sum(a < b for a, b in pairs)
    return wins > losses and sign_probability(wins, losses) < 0.05


def evaluate():
    summary = read("summary_16")
    capacity = read("capacity_16")
    compact = read("compact")
    checks = {
        "fixed snapshot": len({r["bytes"] for r in capacity}) == 1
        and capacity[-1]["events"] == 10_000_000
        and capacity[-1]["bytes"] <= 8 * 1024**2,
        "core arrays and compact numeric workspace": compact["array_bytes"] + 28 * 1024
        <= 256 * 1024,
        "one-shot": all(
            r["one_shot"] > 0.99
            and r["intact_action"] == 1
            and r["edge_deleted_action"] != 1
            for r in capacity
        ),
    }
    for key in ("sealed", "holdout"):
        x = summary[key]
        suite = read("suite_16" if key == "sealed" else "holdout_16")
        worlds = read("world_16") if key == "sealed" else [r["world"] for r in suite]
        checks.update(
            {
                key + " " + k: v
                for k, v in {
                    "identity": min(x["entity"][:2]) >= 0.85
                    and x["entity"][0] > x["visual"],
                    "delayed dynamics": all(
                        x["dynamics"][b]["difference_ci95"][1] < 0
                        for b in ("v9", "v10", "matched_RLS", "recurrent")
                    ),
                    "held-out sensory prediction": x["transfer"][0] < x["transfer"][1],
                    "rotated/displaced": min(x["return_by_condition"][1:5]) >= 18,
                    "novel": x["novel"] >= 18,
                    "detour": sum(r["detour"][0][0] for r in suite) >= 18,
                    "active disambiguation": advantages(
                        (r["alias"]["True"], r["alias"]["False"]) for r in suite
                    ),
                    "closed-loop recovery": advantages(
                        (r["novel"]["success"], r["open_loop"]) for r in worlds
                    ),
                }.items()
            }
        )
    return checks


if __name__ == "__main__":
    results = evaluate()
    for name, passed in results.items():
        print("PASS" if passed else "FAIL", name)
    raise SystemExit(0 if all(results.values()) else 1)
