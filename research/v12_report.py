"""Seed-level summary with reproducible bootstrap intervals for randomized assignments."""

from pathlib import Path
import json
import numpy as np


def summarize(rows):
    out = {}
    for key in ("body", "other"):
        lives = [r[key] for r in rows]
        error = np.array([r["mae"] for r in lives])
        corr = np.array([np.mean(np.array(r["r3"])[:, 1:], axis=0) for r in lives])
        idx = np.random.default_rng(301).integers(len(lives), size=(10000, len(lives)))
        out[key] = dict(
            mae=error.mean(axis=(0, 2)).tolist(),
            channel_seed_count=error[:, 0].size,
            both=int(
                ((error[:, 0] < error[:, 2]) & (error[:, 0] <= error[:, 1])).sum()
            ),
            r3_mean=corr.mean(axis=0).tolist(),
            r3_bootstrap95=np.quantile(
                corr[idx].mean(1), [0.025, 0.975], axis=0
            ).tolist(),
            familiar=np.nanmean(
                np.array([r["familiar"] for r in lives], float), axis=0
            ).tolist(),
            shift=np.nanmean(
                np.array([r["shift"] for r in lives], float), axis=0
            ).tolist(),
            timings=np.mean([r["ms"] for r in lives], axis=0).tolist(),
        )
    for key in ("cycle", "shuffle", "ablated"):
        out[key] = dict(
            early=sum(r[key]["early"] for r in rows),
            opportunities=sum(r[key]["offered"] for r in rows),
            accuracy=np.mean([r[key]["accuracy"] for r in rows])
            if key != "shuffle"
            else None,
        )
    return out


if __name__ == "__main__":
    p = Path("measurements")
    r = summarize(json.loads((p / "v12_sealed.json").read_text()))
    (p / "v12_summary.json").write_text(json.dumps(r, indent=2))
