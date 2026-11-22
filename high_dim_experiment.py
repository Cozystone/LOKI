"""High-dimensional streaming benchmark. The generator and the judge live here."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc

import numpy as np

from loki_memory import LokiMemory


D = 65536
TILE = 64
BUDGET = 262144
PIN = "dc05860ae6837bcdd4de0be34e811bf0848e444f"


def scenes(rng, n, waves):
    shape = (n, D // TILE, 1)
    centers = rng.uniform(0.32, 0.68, shape).astype(np.float32)
    gains = rng.uniform(-0.14, 0.14, shape).astype(np.float32)
    templates = waves[rng.integers(len(waves), size=shape[:2])]
    return (centers + gains * templates).clip(0, 1).reshape(n, D)


def noisy(rng, x):
    return (x + np.float32(0.005) * rng.standard_normal(x.shape, dtype=np.float32)).clip(0, 1)


class RawFIFO:
    """Raw uint8 FIFO plus the incoming action on each retained transition."""
    def __init__(self):
        self.frames = np.zeros(((BUDGET - 32) // (D + 4), D), np.uint8)
        self.actions = np.full(len(self.frames), -1, np.int32)
        self.slot = self.used = 0

    def observe(self, x, action=None):
        self.frames[self.slot] = np.rint(x * 255)
        self.actions[self.slot] = -1 if action is None else action
        self.slot = (self.slot + 1) % len(self.frames)
        self.used = min(len(self.frames), self.used + 1)

    def locate(self, x, visible):
        mask = np.repeat(visible, TILE)
        distance = ((self.frames[:self.used, mask].astype(np.float32) / 255 - x[mask]) ** 2).mean(axis=1)
        return int(distance.argmin())

    def imagine(self, actions, *, start):
        idx = start
        for action in actions:
            nxt = (idx + 1) % self.used
            if nxt != self.slot and self.actions[nxt] == action:
                idx = nxt
        return self.frames[idx].astype(np.float32) / 255

    def recall(self, x, visible):
        return self.imagine((), start=self.locate(x, visible))


def mse(a, b):
    return float(np.square(a - b).mean())


def world(seed):
    rng = np.random.default_rng(seed)
    waves = rng.uniform(-1, 1, (12, TILE)).astype(np.float32)
    waves -= waves.mean(axis=1, keepdims=True)
    waves /= np.sqrt((waves ** 2).mean(axis=1, keepdims=True))
    data = scenes(rng, 24, waves)
    wiring = np.stack([rng.permutation(len(data)) for _ in range(3)])
    memory = LokiMemory(D, BUDGET)
    memories = {"loki": memory, "tile_mean": LokiMemory(D, BUDGET, atoms=0), "fifo": RawFIFO()}
    loc = 0
    first = []
    timings = []
    for t in range(321):
        action = None if t == 0 else int(rng.integers(3))
        if action is not None:
            loc = wiring[action, loc]
        if loc not in first and len(first) < 4:
            first.append(loc)
        last = noisy(rng, data[loc])
        for name, model in memories.items():
            start = perf_counter()
            model.observe(last, action)
            if name == "loki":
                timings.append((perf_counter() - start) * 1000)
    values = {name: {k: [] for k in ("hidden", "rollout", "early")} for name in (*memories, "last")}
    for loc, target in enumerate(data):
        cue = noisy(rng, target)
        visible = np.zeros(D // TILE, bool)
        visible[rng.choice(len(visible), len(visible) // 4, replace=False)] = True
        hidden = ~np.repeat(visible, TILE)
        cue[hidden] = np.nan
        chain = rng.integers(3, size=8)
        end = loc
        for action in chain:
            end = wiring[action, end]
        anchor = noisy(rng, target)
        for name in values:
            restored, forecast = last, anchor
            if name != "last":
                model = memories[name]
                restored = model.recall(cue, visible)
                slot = model.locate(anchor, np.ones_like(visible))
                forecast = model.imagine(chain, start=slot)
            err = mse(restored[hidden], target[hidden])
            values[name]["hidden"].append(err)
            values[name]["rollout"].append(mse(forecast, data[end]))
            if loc in first:
                values[name]["early"].append(err)
    return {"seed": seed, "mse": {n: {k: float(np.mean(v)) for k, v in scores.items()} for n, scores in values.items()},
            "state_bytes": memory.state_bytes, "capacity": memory.capacity, "used": memory.used,
            "write_ms_p50_p95": np.percentile(timings, [50, 95]).tolist()}


def limits():
    rng = np.random.default_rng(522)
    model = LokiMemory(D, BUDGET)
    mask = np.arange(D // TILE) % 4 == 0
    hidden = ~np.repeat(mask, TILE)
    for _ in range(12):
        model.observe(rng.random(D, dtype=np.float32), 0)
    future = rng.random(D, dtype=np.float32)
    noise_mse = mse(model.recall(future, mask)[hidden], future[hidden])
    model = LokiMemory(D, BUDGET)
    waves = rng.normal(size=(8, TILE)).astype(np.float32)
    waves -= waves.mean(axis=1, keepdims=True)
    waves /= np.sqrt((waves ** 2).mean(axis=1, keepdims=True))
    initial = scenes(rng, 1, waves)[0]
    model.observe(initial)
    before = mse(model.recall(initial, mask)[hidden], initial[hidden])
    for _ in range(3 * model.capacity):
        model.observe(scenes(rng, 1, waves)[0], 1)
    after = mse(model.recall(initial, mask)[hidden], initial[hidden])
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "loki.npz"
        model.save(path)
        clone = LokiMemory.load(path)
        checkpoint_bytes = path.stat().st_size
        for _ in range(16):
            frame = scenes(rng, 1, waves)[0]
            model.observe(frame, 1)
            clone.observe(frame, 1)
            np.testing.assert_array_equal(model.imagine([1, 1]), clone.imagine([1, 1]))
        for name, original in model._arrays().items():
            np.testing.assert_array_equal(original, clone._arrays()[name])
    tracemalloc.start()
    model.observe(initial, 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(iid_mse=noise_mse, iid_constant_half_mse=mse(future[hidden], np.float32(0.5)),
                before_overflow_mse=before, after_overflow_mse=after, novel_intervening_scenes=3 * model.capacity,
                checkpoint_bytes=checkpoint_bytes, checkpoint_continuation_steps=16, traced_write_extra_bytes=peak)


def adapter_check(source):
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("unexpected ATANOR revision")
    sys.path.insert(0, str(source.resolve()))
    import torch
    from atanor_os.emerge.room_body import RoomBody
    from atanor_os.emerge.room_vary import condition
    from loki_memory.atanor import AtanorImageMemory

    torch.set_num_threads(1)
    torch.manual_seed(901)
    room = RoomBody(1, *condition(901), seed=901, device="cpu")
    observer = AtanorImageMemory(room.c)
    rng = np.random.default_rng(901)
    observer.step(room.raw(), cut=True)
    errs = []
    actions = set()
    for t in range(96):
        action = int(rng.integers(observer.actions))
        actions.add(action)
        room.step(torch.tensor([action]))
        raw = room.raw()
        target = raw[observer.index][0].numpy().ravel() / 255
        if t >= 32:
            vis = rng.random(observer.memory.tiles) < 0.25
            pred = observer.memory.recall(target, vis)
            hidden = ~np.repeat(vis, observer.memory.tile)[:len(target)]
            errs.append(mse(target[hidden], pred[hidden]))
        out = observer.step(raw, action)
        assert out.shape == observer.shape
    # A blank raw frame may move the address but cannot write a fabricated memory.
    raw[observer.index] = torch.zeros_like(raw[observer.index])
    before = observer.memory.used
    observer.step(raw, action)
    assert before == observer.memory.used
    return dict(shape=observer.shape, dimensions=observer.memory.width, steps=96, distinct_actions=len(actions),
                masked_raw_mse=float(np.mean(errs)), budget=BUDGET, state_bytes=observer.memory.state_bytes,
                blank_skipped_write=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atanor", type=Path, default=Path(".git/atanor-source"))
    args = parser.parse_args()
    rows = [world(seed) for seed in range(201, 209)]
    mean = {n: {k: float(np.mean([r["mse"][n][k] for r in rows])) for k in rows[0]["mse"][n]} for n in rows[0]["mse"]}
    wins = {n: {k: sum(r["mse"]["loki"][k] < r["mse"][n][k] for r in rows) for k in mean[n]} for n in ("fifo", "last", "tile_mean")}
    gate = all(wins[n]["hidden"] >= 7 and mean["loki"]["hidden"] <= 0.7 * mean[n]["hidden"] for n in ("fifo", "tile_mean"))
    result = dict(worlds=rows, mean_mse=mean, paired_wins=wins, hidden_gate=gate and all(r["state_bytes"] <= BUDGET for r in rows),
                  rollout_gate=all(wins[n]["rollout"] >= 7 for n in ("tile_mean", "last")),
                  limits=limits(), adapter=adapter_check(args.atanor), host=platform.processor(), numpy=np.__version__)
    Path("high_dim_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "worlds"}), flush=True)


if __name__ == "__main__":
    main()
