"""Post-registration sensitivity checks on paired observations with alternate judge identities."""

import json
from pathlib import Path
import numpy as np
import torch
from research.v13_acceptance import batch, setup

if __name__ == "__main__":
    torch.set_num_threads(1)
    rows = []
    for seed in range(26741, 26749):
        rng = np.random.default_rng(seed)
        key = rng.random((64, 32), dtype="f4")
        ctx = np.tile(rng.uniform(0.2, 0.5, (1, 16)).astype("f4"), (64, 1))
        pos = np.stack((np.arange(64), np.zeros(64)), 1).astype("f4")
        s = setup(key, pos, ctx)
        a = batch(s, key, pos, ctx, True)
        b = batch(s, key, pos + [0, 3], ctx, True)

        def accuracy(left, right):
            return float(np.mean(np.r_[left == a, right == b]))

        seen = batch(s, key, pos, ctx)
        with_pos = accuracy(seen, batch(s, key, pos + [0, 3], ctx))
        ids = s.cue(key, pos, ctx, landmark=True)[0][:, 0].numpy()
        # Omit pose for BOTH conditions, so any deterministic ranking tops out at .5.
        ablated = accuracy(ids, ids)
        rows.append(dict(seed=seed, with_pose=with_pos, omit_pose=ablated))
        for delta in (0, 0.02, 0.05, 0.10, 0.20):
            # The judge says a new place. Evidence increasingly separates the two.
            _, _, conf, nov = s.cue(key, pos, ctx + delta)
            rows[-1][str(delta)] = dict(
                high_wrong=float((conf >= 0.7).float().mean()),
                mean_novelty=float(nov.mean()),
            )
    Path("measurements/v13_indistinguishable.json").write_text(
        json.dumps(rows, indent=2)
    )
