"""Actual writes, shuffled admission, and exact-search reference for the CPU atlas."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from loki_memory.trace_atlas import TraceAtlas


def exhaustive(store, key, pos, ctx, cold):
    # Independent row-by-row metric over the full populated table, no atlas bounds.
    f = store.features.numpy()
    cost = np.sum((f[:, :32] - key.numpy()[0]) ** 2, 1) / 0.32
    cost += np.sum((f[:, 32:48] - ctx.numpy()[0]) ** 2, 1) / 0.1296
    if not cold:
        cost += np.sum((f[:, 48:50] - pos.numpy()[0]) ** 2, 1) / 0.4225
    return np.argsort(cost, kind="stable")[:8]


def trial(n, shuffled):
    rng = np.random.default_rng(26441)
    s = TraceAtlas(3, n)
    w = int(np.ceil(np.sqrt(n)))
    pos = np.column_stack((np.arange(n) % w, np.arange(n) // w)).astype("f4")
    key = rng.random((n, 32), dtype="f4")
    ctx = np.full((1, 16), 0.5, "f4")
    live = [True]
    rise = torch.ones(1, 3) * 0.01
    drop = [0.0]
    order = rng.permutation(n) if shuffled else np.arange(n)
    begin = perf_counter()
    for t, j in enumerate(order):
        token = s.write(key[j : j + 1], pos[j : j + 1], ctx, rise, drop, t, live)
        if token.item() < 0:
            raise RuntimeError("missed admission")
    seconds = perf_counter() - begin
    rows = {}
    prev = token
    for cold in (False, True):
        times = []
        tick = []
        tested = []
        correct = []
        exact = []
        for t, j in enumerate(rng.choice(n, 128, replace=False)):
            args = s._inputs(key[j : j + 1], pos[j : j + 1], ctx, None)
            ref = exhaustive(s, *args[:3], cold)
            begin = perf_counter()
            ids, *_ = s.cue(*args[:3], landmark=cold)
            times.append((perf_counter() - begin) * 1000)
            tested.append(s.examined)
            if not cold:
                found = s.write(*args[:3], rise, drop, n + t, live)
                s.link(prev, found, live)
                tick.append((perf_counter() - begin) * 1000)
                prev = found
            correct.append(np.array_equal(ref, ids.numpy()[0]))
            exact.append(
                np.array_equal(
                    s.features.numpy()[int(ids[0, 0]) % n, :32], key[j].astype("f2")
                )
            )
        rows["cold" if cold else "registered"] = dict(
            top8=float(np.mean(correct)),
            top1=float(np.mean(exact)),
            p95_ms=float(np.percentile(times, 95)),
            tick_p95_ms=float(np.percentile(tick, 95)) if tick else None,
            examined_p50=float(np.median(tested)),
        )
    fused = []
    for t, j in enumerate(rng.choice(n, 256, replace=False)):
        args = (
            key[j : j + 1],
            pos[j : j + 1],
            ctx,
            rise,
            drop,
            n + 128 + t,
            live,
            prev,
        )
        start = perf_counter()
        _, prev = s.observe(*args)
        fused.append((perf_counter() - start) * 1000)
    return dict(
        fused_p95_ms=float(np.percentile(fused, 95)),
        n=n,
        shuffled=shuffled,
        active=int((s.visits > 0).sum()),
        state_bytes=len(s.state()),
        fixed_bytes=s.fixed_bytes,
        populate_s=seconds,
        sleep=s.consolidate(),
        queries=rows,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 65536])
    args = parser.parse_args()
    results = []
    for n in args.sizes:
        for sh in (False, True):
            row = trial(n, sh)
            print(row, flush=True)
            results.append(row)
            Path("measurements/v13_scale.json").write_text(
                json.dumps(results, indent=2)
            )
