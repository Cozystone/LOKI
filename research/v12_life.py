"""Prospective v12 measurements. The learner receives only past observations and executed commands."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from loki_memory.journey import LokiV12
from loki_memory.organ import LokiMemory


class Body:
    def __init__(self, seed, widths=(3, 41, 96), motors=(5, 6, 3)):
        rng = np.random.default_rng(seed)
        self.widths = widths
        self.motors = motors
        self.commands = np.array(
            [
                np.concatenate([np.eye(k, dtype="f4")[a] for a, k in zip(c, motors)])
                for c in np.ndindex(motors)
            ]
        )
        self.move = rng.permutation(np.resize(np.array([-1, 0, 1]), motors[0]))
        self.physical_option = self.commands[:, : motors[0]].argmax(axis=1)
        splits = np.split(np.arange(sum(motors)), np.cumsum(motors)[:-1])
        branch_order = rng.permutation(len(motors))
        self.motors = tuple(motors[i] for i in branch_order)
        self.commands = self.commands[
            :, np.concatenate([rng.permutation(splits[i]) for i in branch_order])
        ]
        self.channels = rng.uniform(0, 2 * np.pi, sum(widths) - 3)
        self.act_weights = rng.normal(0, 0.05, (sum(motors), len(self.channels)))
        self.pos = 0
        self.stores = np.array([0.5, 0.5, 1.0])
        self.t = 0
        self.perm = rng.permutation(3)
        self.raw = self.obs(self.commands[0])
        self.path = 0

    def place(self):
        # The synthetic body's odometer integrates its own displacement; origin is arbitrary.
        a = 2 * np.pi * self.path / 9
        return np.array([np.cos(a), np.sin(a)], "f4")

    def obs(self, u):
        signal = (
            0.5
            + 0.26 * np.sin(2 * np.pi * self.pos / 9 + self.channels)
            + u @ self.act_weights
        )
        return np.r_[self.stores[self.perm], np.clip(signal, 0, 1)].astype("f4")

    def step(self, act):
        u = self.commands[act]
        dx = self.move[self.physical_option[act]]
        self.path += dx
        self.pos = (self.pos + dx) % 9
        self.t += 1
        self.stores[:2] = np.maximum(0.03, self.stores[:2] - 0.009)
        for h, loc in enumerate((1, 5)):
            if self.pos == loc:
                self.stores[h] = min(1.0, self.stores[h] + 0.20)
        self.raw = self.obs(u)
        return self.raw


def check(seed, small=False):
    rng = np.random.default_rng(seed)
    w = Body(seed, (3, 8, 5), (3, 2)) if small else Body(seed)
    b = 3
    mem = LokiV12(str(seed), w.widths, w.motors, place_dim=2)
    v4 = LokiMemory(sum(w.widths), grid_shape=None if small else (4, 8, 3))
    mem.observe(w.raw, place=w.place())
    v4.observe(w.raw)
    err = np.zeros((3, len(w.raw)))
    timing = []
    cal = []
    foreign = []
    dy = []
    dx = []
    contrast = []
    orientation = []
    for episode in range(160):
        # Draw candidates before the 32-beat intervention. Execute only one.
        plans = rng.integers(len(w.commands), size=(2, 4)).repeat(8, axis=1)
        which = rng.integers(2)
        plan = plans[which]
        before = w.raw[:b].copy()
        if episode >= 96:
            predicted = [mem.foresee(w.commands[p]).bedrock_change for p in plans]
            dx.append(predicted[which])
            contrast.append(predicted[1] - predicted[0])
            orientation.append(2 * which - 1)
        for a in plan:
            u = w.commands[a]
            start = time.perf_counter_ns()
            pred = mem.forecast(u)
            mid = time.perf_counter_ns()
            old = v4.imagine([u])
            prev = w.raw.copy()
            raw = w.step(a)
            conf = mem.confidence()
            t = time.perf_counter_ns()
            mem.observe(raw, u, place=w.place())
            end = time.perf_counter_ns()
            if episode >= 96:
                err += np.abs(np.stack([pred, old, prev]) - raw)
                timing.append((mid - start + end - t) / 1e6)
                cal.extend(zip(conf.tolist(), (np.abs(pred - raw) <= 0.05).tolist()))
            v4.observe(raw, u)
        if episode >= 96:
            dy.append(raw[:b] - before)
    # Unannounced sensory inversion and altered actuation, same uninterrupted life.
    w.move = np.roll(w.move, 1)
    for tick in range(100):
        a = rng.integers(len(w.commands))
        u = w.commands[a]
        pred = mem.forecast(u)
        conf = mem.confidence()
        raw = 1 - w.step(a)
        foreign.extend(zip(conf.tolist(), (np.abs(pred - raw) <= 0.05).tolist()))
        mem.observe(raw, u, place=w.place())

    def calibration(pairs):
        p, hit = np.asarray(pairs).T
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(abs((p[bins == i] - hit[bins == i]).sum()) for i in range(10)) / len(
            p
        )
        high = p >= 0.8
        return [
            float(ece),
            float(p.mean()),
            float(hit.mean()),
            float(high.mean()),
            float(p[high].mean()) if high.any() else None,
            float(hit[high].mean()) if high.any() else None,
        ]

    x, y = np.asarray(dx), np.asarray(dy)
    cs = []
    for h in range(b):
        if np.std(y[:, h]) > 1e-8:
            cs.append(
                [
                    h,
                    float(np.corrcoef(x[:, h], y[:, h])[0, 1])
                    if np.std(x[:, h])
                    else 0.0,
                    float(np.std(x[:, h]) / np.std(y[:, h])),
                    float(np.sqrt(np.mean(np.asarray(contrast)[:, h] ** 2))),
                    float(np.mean(np.abs(np.asarray(contrast)[:, h]))),
                    float(
                        np.mean(
                            2
                            * np.asarray(orientation)
                            * np.sign(np.asarray(contrast)[:, h])
                            * y[:, h]
                        )
                    ),
                ]
            )
    return dict(
        seed=seed,
        width=w.widths,
        motors=w.motors,
        mae=(err / (64 * 32)).tolist(),
        r3=cs,
        familiar=calibration(cal),
        shift=calibration(foreign),
        ms=np.percentile(timing, [50, 95]).tolist() + [float(np.mean(timing))],
        array_bytes=mem.matrix.nbytes,
    )


def moving(seed, shuffled, shuffle_recall=False):
    # The clock is used by the world and scorer only. Common random motor probes.
    rng = np.random.default_rng(seed)
    world = Body(seed, (3, 8, 5), (3, 2))
    m = LokiV12(str(seed), world.widths, world.motors, place_dim=2)
    corners = rng.permutation([0, 3, 6])
    location = int(corners[0])
    pos = 0
    value = 0.4
    displacements = np.zeros(len(world.commands))
    visits = np.zeros(len(world.commands))
    motor_rng = np.random.default_rng(seed + 39000)
    clock = 0
    right = total = early = offered = 0
    for epoch in range(19):
        if epoch:
            location = (
                int(rng.choice([x for x in corners if x != location]))
                if shuffled
                else int(corners[epoch % 3])
            )
        for beat in range(300):
            phase = 2 * np.pi * pos / 9
            place = np.array([np.cos(phase), np.sin(phase)], "f4")
            raw = np.r_[value, 0.5, 1, np.resize((1 + place) / 2, 13)].astype("f4")
            m.observe(raw, None if not clock else world.commands[last], place=place)
            recalled = m.resource_belief(0)
            probs = recalled["mass"]
            if epoch >= 6 and beat == 299 and not shuffled:
                total += 1
                if len(probs) and probs.max() > 0:
                    nextpos = corners[(epoch + 1) % 3]
                    target = np.array(
                        [
                            np.cos(nextpos * 2 * np.pi / 9),
                            np.sin(nextpos * 2 * np.pi / 9),
                        ]
                    )
                    right += (
                        np.linalg.norm(recalled["cues"][probs.argmax()] - target) < 0.1
                    )
            occupancy = m.resource_at(0, ahead=30)["mass"]
            if shuffle_recall and probs.any():
                sites = np.flatnonzero(m.hits[: len(probs), 0] > 0)
                previous = int(m.last[0])
                if len(sites) > 1:
                    scale = 1 - occupancy[previous]
                    occupancy[sites] = scale * np.roll(probs[sites], 1)
                    occupancy[previous] += 1 - scale
            explore = motor_rng.random() < 0.35
            random_action = int(motor_rng.integers(len(world.commands)))
            if occupancy.any() and not explore:
                target = recalled["cues"][occupancy.argmax()]
                heading = np.arctan2(target[1], target[0])
                predicted = phase + displacements / np.maximum(1, visits)
                last = int(np.argmax(np.cos(predicted - heading)))
            else:
                last = random_action
            if epoch >= 6 and beat >= 270:
                # Store occupancy; score against the actual next corner after it is drawn.
                pending.append(pos)
            dx = world.move[world.physical_option[last]]
            pos = (pos + dx) % 9
            clock += 1
            displacements[last] += 2 * np.pi * dx / 9
            visits[last] += 1
            value = (
                min(0.99, value + 0.22) if pos == location else max(0.01, value - 0.025)
            )
            if beat == 0:
                if epoch >= 7:
                    early += sum(x == location for x in pending)
                    offered += len(pending)
                pending = []
    return dict(accuracy=right / total if total else None, early=early, offered=offered)


def run(s):
    return dict(
        body=check(s),
        other=check(s, True),
        cycle=moving(s, False),
        shuffle=moving(s, True),
        ablated=moving(s, False, True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    start, n = (301, 4) if args.dev else (18031, 24)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run, range(start, start + n)))
    path = Path("measurements") / ("v12_dev.json" if args.dev else "v12_sealed.json")
    path.write_text(
        json.dumps(rows, allow_nan=False, default=lambda value: value.item())
    )
