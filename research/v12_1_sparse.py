"""I0 sparse visits. World variables appear only in the simulator and scorer."""

import argparse
import json
from pathlib import Path
import numpy as np
from loki_memory.journey import LokiV12
from loki_memory.journey_121 import LokiV121


class NoMiss(LokiV121):
    def observe(self, *args):
        super().observe(*args)
        self.misses.fill(0)


class NoPrior(LokiV121):
    next_resource = LokiV12.next_resource


def life(seed, shuffled):
    rng = np.random.default_rng(seed)
    codes = rng.random((9, 4), dtype="f4")
    order = rng.permutation(3)
    press = int(rng.integers(3))
    models = [
        cls(str(seed), [1, 2], [3], place_dim=4)
        for cls in (LokiV12, LokiV121, NoMiss, NoPrior)
    ]
    eye = np.eye(3, dtype="f4")
    pos = 0
    stock = 0.5
    smooth = 0.5
    noise = 0.5
    room = int(order[0])
    nxt = int(order[1])
    mae = np.zeros((3, 3))
    scores = np.zeros((4, 3))
    checks = 0
    miss_sum = np.zeros(4)
    num_miss = 0
    for tick in range(6000):
        if tick and tick % 300 == 0:
            old_room = room
            room = nxt
            nxt = (
                int(rng.choice([i for i in order if i != room]))
                if shuffled
                else int(order[(tick // 300 + 1) % 3])
            )
        # Random dwell, no policy consults the resource or its true location.
        previous = pos
        if rng.random() < 0.032:
            pos = int(rng.integers(9))
        a = int(rng.integers(3))
        u = eye[a]
        prev = np.array([stock, smooth, noise], dtype="f4")
        preds = [m.forecast(u) for m in models[:2]]
        stock = np.clip(
            stock
            + (0.15 if room == pos and previous == pos and a == press else -0.0006),
            0,
            1,
        )
        smooth = 0.98 * smooth + 0.02 * (0.46 + 0.04 * a)
        noise = np.clip(noise + rng.choice([-0.001, 0.001]), 0.01, 0.99)
        raw = np.array([stock, smooth, noise], dtype="f4")
        for m in models:
            m.observe(raw, u, None, codes[pos])
        if tick < 1200:
            continue
        mae += np.abs(np.stack([*preds, prev]) - raw)
        if tick % 10 == 0:
            checks += 1
            for i, m in enumerate(models):
                now = m.resource_at(0)["mass"]
                future = m.next_resource(0)
                for col, (p, truth) in enumerate([(now, room), (future, nxt)]):
                    if len(p) and p.max() > 0:
                        tied = np.flatnonzero(np.isclose(p, p.max()))
                        scores[i, col] += np.mean(
                            [np.allclose(m.centers[j], codes[truth]) for j in tied]
                        )
                scores[i, 2] += now.sum()
        if (
            tick // 300 > 0
            and tick % 300 < 90
            and pos == previous
            and pos == old_room
            and a == press
            and prev[0] < 0.85
        ):
            num_miss += 1
            for i, m in enumerate(models):
                p = m.resource_at(0)["mass"]
                miss_sum[i] += sum(
                    p[j] for j in range(len(p)) if np.allclose(m.centers[j], codes[pos])
                )
    return dict(
        mae=(mae / 4800).tolist(),
        scores=(scores / checks).tolist(),
        miss_sum=miss_sum.tolist(),
        miss_count=num_miss,
    )


def run(task):
    seed, shuffled = task
    return dict(seed=seed, shuffled=shuffled, **life(seed, shuffled))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    arg = ap.parse_args()
    start, num = (25001, 4) if arg.dev else (25101, 24)
    with ProcessPoolExecutor(6) as pool:
        rows = list(
            pool.map(
                run,
                [(s, sh) for s in range(start, start + num) for sh in (False, True)],
            )
        )
    Path("measurements/v12_1_" + ("dev" if arg.dev else "sealed") + ".json").write_text(
        json.dumps(rows, indent=1)
    )
