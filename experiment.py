"""Paired CPU test on a pinned ATANOR renderer and its sensory Inner organ."""
import argparse
from copy import deepcopy
import json
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from loki_memory.recurrence import RecurrenceMemory


PIN = "dc05860ae6837bcdd4de0be34e811bf0848e444f"
WIDTH = 96
PREFIX = 128


def record(seed, length, moving):
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room

    rng = np.random.default_rng(seed)
    world = World(1, make_room(seed), device="cpu", seed=seed, drain=0)
    wiring = rng.permutation([2, 3])
    actions = np.empty(length, dtype=np.int64)
    frames = np.empty((length + 1, WIDTH), dtype=np.float32)
    frames[0] = world.picture(8, 4).numpy().ravel()
    action = 0
    for i in range(length):
        if rng.random() < 0.42:
            action = int(rng.integers(2))
        actions[i] = action
        world.step(torch.tensor([wiring[action]]))
        if moving and i % 4 == 3:
            world.step(torch.tensor([1]))
        frames[i + 1] = world.picture(8, 4).numpy().ravel()
    return frames, actions


def train_inner(model, streams, iterations, seed):
    # Only visible self-experience enters these windows. Hide inputs artificially
    # to train multi-step recall without any real blackout targets.
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for _ in range(iterations):
        inputs, targets, actions = [], [], []
        for _ in range(16):
            frames, acts = streams[rng.integers(len(streams))]
            start = rng.integers(len(acts) - 24 + 1)
            inputs.append(frames[start:start + 24])
            targets.append(frames[start + 1:start + 25])
            actions.append(acts[start:start + 24])
        shut = np.arange(24)[None, :] >= rng.integers(4, 18, size=(16, 1))
        prediction = model.run(torch.tensor(np.stack(inputs)), torch.tensor(np.stack(actions)), torch.tensor(shut).float())
        loss = (prediction - torch.tensor(np.stack(targets))).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TransitionReplay:
    """Replay the successor of the closest raw view with matching action."""
    def __init__(self, width):
        self.source = np.zeros((128, width), np.uint8)
        self.target = np.zeros_like(self.source)
        self.actions = np.full(128, -1, np.int8)
        self.slot = 0

    def learn(self, previous, action, current):
        self.source[self.slot] = np.rint(255 * previous)
        self.target[self.slot] = np.rint(255 * current)
        self.actions[self.slot] = action
        self.slot = (self.slot + 1) % len(self.source)

    def predict(self, previous, action):
        indices = np.flatnonzero(self.actions == action)
        if not len(indices):
            return previous
        distance = ((self.source[indices] / 255.0 - previous) ** 2).mean(axis=1)
        return self.target[indices[distance.argmin()]] / 255.0


def evaluate(base, stream, seed):
    frames, acts = stream
    inner = deepcopy(base)
    train_inner(inner, [(frames[:PREFIX + 1], acts[:PREFIX])], 64, seed)
    memory = RecurrenceMemory(WIDTH)
    shuffled = RecurrenceMemory(WIDTH)
    replay = TransitionReplay(WIDTH)
    # The alternate intervention permutes only within each blackout block.
    wrong = acts.copy()
    rng = np.random.default_rng(9000 + seed)
    for start in range(PREFIX + 40, len(acts), 60):
        rng.shuffle(wrong[start:start + 20])

    last = simulated = frames[0]
    incoming = torch.tensor(frames[0:1])
    was_dark = False
    h = inner.zero(1)
    errors = {name: [] for name in ("recurrence", "inner", "last", "replay", "permuted")}
    with torch.no_grad():
        for i, action in enumerate(acts):
            dark = i >= PREFIX and (i - PREFIX) % 60 >= 40
            reading = None if dark else frames[i + 1]
            prediction = memory.step(action, reading)
            disrupted = shuffled.step(wrong[i], reading)
            h, predicted = inner.step(h, incoming, torch.tensor([action]), torch.tensor([float(was_dark)]))
            if dark:
                simulated = replay.predict(simulated, action)
                outputs = (prediction, predicted.numpy().ravel().clip(0, 1), last, simulated, disrupted)
                for key, output in zip(errors, outputs):
                    errors[key].append(float(np.square(output - frames[i + 1]).mean()))
                incoming = torch.zeros_like(incoming)
            else:
                if not was_dark:
                    replay.learn(last, action, reading)
                last = simulated = reading
                incoming = torch.tensor(reading[None])
            was_dark = dark
    return {
        "room": seed, "frames_scored": len(errors["inner"]),
        "mse": {k: float(np.mean(v)) for k, v in errors.items()},
        "late_mse": {k: float(np.array(v).reshape(4, 20)[:, -5:].mean()) for k, v in errors.items()},
        "period": int(memory.period[memory.winner]), "second_stride": int(memory.stride[memory.winner]),
        "state_bytes": memory.state_bytes,
    }


def summarize(rows):
    mean = {k: float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]}
    pairs = {}
    for k in mean.keys() - {"recurrence"}:
        delta = np.array([r["mse"][k] - r["mse"]["recurrence"] for r in rows])
        wins = int((delta > 0).sum())
        n = int(np.count_nonzero(delta))
        pairs[k] = {"wins": wins, "reduction": 1 - mean["recurrence"] / mean[k],
                    "one_sided_sign_p": sum(comb(n, i) for i in range(wins, n + 1)) / 2**n}
    passed = all(pairs[k]["wins"] >= 10 and pairs[k]["reduction"] >= 0.2 for k in ("last", "inner"))
    return {"mean_mse": mean, "paired": pairs, "gate": passed and max(r["state_bytes"] for r in rows) <= 65536}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atanor", type=Path, default=Path(".git/atanor-source"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.atanor), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        parser.error(f"Expected ATANOR revision {PIN}, found {revision}")
    sys.path.insert(0, str(args.atanor.resolve()))
    from atanor_os.emerge.inner import Inner

    torch.set_num_threads(1)
    torch.manual_seed(803)
    base = Inner(WIDTH, 2, h=32)
    training = [record(seed, 208, False) for seed in range(1, 13)]
    train_inner(base, training, 500, 803)
    result = {"upstream": revision, "numpy": np.__version__, "torch": str(torch.__version__),
              "inner_parameter_bytes": sum(p.numel() * p.element_size() for p in base.parameters())}
    for name, moving in (("rotation", False), ("translation_stress", True)):
        rows = []
        for seed in range(101, 113):
            rows.append(evaluate(base, record(seed, PREFIX + 4 * 60, moving), seed))
        result[name] = {"rooms": rows, "summary": summarize(rows)}
        print(name, json.dumps(result[name]["summary"]), flush=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
