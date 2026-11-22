"""Independent raw-body evaluation for the general LOKI organ."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from high_dim_experiment import BUDGET, PIN, mse
from loki_memory import LokiMemory
from loki_memory.atanor import AtanorImageMemory


class PooledFIFO:
    def __init__(self, shape):
        h, w, c = shape
        self.grid = (h // 8, 8, w // 8, 8, c)
        self.frames = np.zeros(((BUDGET - 64) // (h*w*c//64), h//8, w//8, c), np.uint8)
        self.slot = self.used = 0

    def observe(self, frame):
        pool = frame.reshape(self.grid).mean(axis=(1, 3))
        self.frames[self.slot] = np.rint(pool * 255)
        self.slot = (self.slot + 1) % len(self.frames)
        self.used = min(self.used + 1, len(self.frames))

    def recall(self, frame, mask):
        # Sum squared error of piecewise-constant predictions on visible samples.
        query = np.where(mask, frame, 0).reshape(self.grid).sum(axis=(1, 3))
        count = mask.reshape(self.grid).sum(axis=(1, 3), dtype=np.float32)
        small = self.frames[:self.used].astype(np.float32) / 255
        score = (small * (small * count - 2 * query)).sum(axis=(1, 2, 3))
        best = small[int(score.argmin())]
        return best.repeat(8, axis=0).repeat(8, axis=1).ravel()


def run(seed):
    from atanor_os.emerge.room_body import RoomBody
    from atanor_os.emerge.room_vary import condition

    torch.manual_seed(seed)
    motor, judge = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    room = RoomBody(1, *condition(seed), seed=seed, device="cpu")
    observer = AtanorImageMemory(room.c)
    mem = observer.memory
    mean = LokiMemory(mem.width, BUDGET, atoms=0)
    fifo = PooledFIFO(observer.shape)
    keys = ("loki_hidden", "mean_hidden", "pooled_fifo_hidden", "last_hidden", "loki_next", "last_next")
    stats = {k: [] for k in keys}
    raw = room.raw()
    last = raw[observer.index][0].numpy().ravel() / 255
    observer.step(raw, cut=True)
    mean.observe(last)
    fifo.observe(last)
    for t in range(160):
        action = int(motor.integers(observer.actions))
        forecast = mem.imagine([action])
        room.step(torch.tensor([action]))
        raw = room.raw()
        frame = raw[observer.index][0].numpy().ravel() / 255
        if t >= 40:
            vis = np.zeros(mem.tiles, bool)
            vis[judge.choice(mem.tiles, size=mem.tiles//4, replace=False)] = True
            mask = np.repeat(vis, mem.tile)[:mem.width]
            recon = (mem.recall(frame, vis), mean.recall(frame, vis), fifo.recall(frame, mask), last)
            for k, x in zip(keys[:4], recon):
                stats[k].append(mse(x[~mask], frame[~mask]))
            stats["loki_next"].append(mse(forecast, frame))
            stats["last_next"].append(mse(last, frame))
        observer.step(raw, action)
        mean.observe(frame, action)
        fifo.observe(frame)
        last = frame
    return {"seed": seed, "scored": len(stats["loki_hidden"]), "mse": {k: float(np.mean(v)) for k, v in stats.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atanor", type=Path, default=Path(".git/atanor-source"))
    args = parser.parse_args()
    if subprocess.check_output(["git", "-C", str(args.atanor), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("unexpected ATANOR revision")
    sys.path.insert(0, str(args.atanor.resolve()))
    torch.set_num_threads(1)
    rows = [run(s) for s in range(401, 409)]
    means = {k: float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]}
    wins = sum(r["mse"]["loki_hidden"] < r["mse"]["pooled_fifo_hidden"] for r in rows)
    result = dict(rooms=rows, mean_mse=means, wins_over_pooled=wins,
                  gate=wins >= 7 and means["loki_hidden"] <= 0.9 * means["pooled_fifo_hidden"])
    print(means, result["gate"])
    Path("body_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
