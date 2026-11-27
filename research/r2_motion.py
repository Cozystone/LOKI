"""Fresh paired test of support-aware motion error, including its novelty tradeoff."""

import argparse
import copy
import json
from pathlib import Path
import torch
from research.r2_geometry import generator, trial
from research.r2_places import inputs, replay, timing
from loki_memory.place_bundles import PlaceBundles


def controls(seed, flag):
    s = PlaceBundles(supported_motion=flag)
    for i in range(3):
        replay(s, inputs(seed + i, i, True, bool(seed % 2)), i)
        s.sleep()
    choices = []
    for i in (0, 1):
        read, _ = replay(
            copy.deepcopy(s), inputs(seed + i + 1, i, False, not bool(seed % 2)), i
        )
        choices.append([read[8], read[-1]])
    path, rows = inputs(seed + 77, 0)
    s.cut()
    for args in rows:
        s.observe(*args)
    p, d, _ = rows[-1]
    unknown = s.observe(p, d, [1.0])[0]
    return dict(probes=choices, unseen_collision=unknown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    gen = generator(args.source)
    rows = [
        dict(
            seed=i,
            baseline=trial(i, gen, False),
            revision=trial(i, gen, True),
            fixed=controls(i, True),
            old_fixed=controls(i, False),
        )
        for i in range(29301, 29325)
    ]
    data = dict(rows=rows, cpu1=timing(1), cpu64=timing(64))
    Path("measurements/r2_motion.json").write_text(
        json.dumps(data, indent=2, allow_nan=False)
    )
    print(
        {k: sum(r[k]["returned"] for r in rows) for k in ("baseline", "revision")},
        data["cpu1"],
        data["cpu64"],
    )
