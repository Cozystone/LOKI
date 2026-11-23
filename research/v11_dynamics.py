"""Delayed nonlinear dynamics. Every model sees the same physical probe and lag."""

import json
import numpy as np
from .reference import ROOT
from .v10_stream import Samples
from loki_memory.belief import BeliefModel
from loki_memory.lifelong import LifelongMemory
from loki_memory.episodic import EpisodeField


class Recurrent:
    def __init__(self, recurrent):
        # Online random-feature state-space baseline, readout trained with RLS.
        d = 155 if recurrent else 220
        rng = np.random.default_rng(112)
        self.U = rng.normal(0, 0.6, (16, d)).astype("f4")
        self.H = (
            rng.normal(0, 0.7 / np.sqrt(d), (d, d)).astype("f4") if recurrent else None
        )
        self.P = np.eye(d, dtype="f4") * 8
        self.W = np.zeros((d, 8), "f4")
        self.hidden = np.zeros(d, "f4")

    def step(self, k, a, target=None):
        inputs = np.r_[k, np.eye(4)[a]]
        h = np.tanh(
            inputs @ self.U + (self.hidden @ self.H if self.H is not None else 0)
        ).astype("f4")
        self.hidden = h
        pred = h @ self.W
        if target is not None:
            v = self.P @ h
            gain = v / (1 + h @ v)
            self.W += np.outer(gain, target - pred)
            self.P -= np.outer(gain, v)
        return pred


def run(seed, rev=3, holdout=False):
    rng = np.random.default_rng(seed)
    m = LifelongMemory()
    plain = LifelongMemory()
    plain.flags[0] = 0
    nohistory = LifelongMemory()
    nohistory.flags[2] = 0
    v9 = EpisodeField()
    fifo = Samples(512, False, rng)
    reservoir = Samples(512, True, rng)
    model = BeliefModel(rev)
    esn = Recurrent(True)
    matched = Recurrent(False)
    regime = 1

    def initial():
        return rng.uniform(0.1, 0.9, 8)

    def advance(x, a, h):
        if holdout:
            return np.clip(
                0.90 * x
                + 0.05
                + 0.09 * np.sin(4 * x + a)
                + 0.035 * np.cos(8 * x - a)
                + regime * 0.035 * np.sin(x * 3)
                + 0.09 * np.roll(h, 2).repeat(2),
                0,
                1,
            )
        return np.clip(
            x
            + 0.12 * np.sin(5 * x + 1.57 * a)
            + regime * 0.04 * np.sin(3 * x)
            + 0.11 * np.roll(h, 1).repeat(2)
            - 0.15 * (x - 0.5),
            0,
            1,
        )

    def command(a):
        return np.eye(4)[a]

    def state9(x, h):
        return np.resize(np.r_[x, h], 20).astype("f4")

    def action9(a):
        return np.resize(command(a), 8).astype("f4")

    def feat(x, a, h):
        return np.r_[1, x, command(a), h]

    P = np.eye(17) * 10
    W = np.zeros((17, 8))
    xx = []
    aa = []
    yy = []
    x = initial()
    h = command(0)
    for t in range(1500):
        if t % 16 == 0:
            x = initial()
            regime = rng.choice([-1, 1])
            esn.hidden.fill(0)
        a = int(rng.integers(4))
        y = advance(x, a, h)
        model.fit(np.resize(x, 24), a, np.resize(y, 24), last=h)
        for baseline in (esn, matched):
            baseline.step(np.r_[x, h], a, y - x)
        xx.append(np.r_[x, h])
        aa.append(a)
        yy.append(y - x)
        z = feat(x, a, h)
        v = P @ z
        g = v / (1 + z @ v)
        W += np.outer(g, y - x - z @ W)
        P -= np.outer(g, v)
        v9.record(state9(x, h), action9(a), state9(y, command(a)), "rig")
        x, h = y, command(a)
    X, A, Y = map(np.asarray, (xx, aa, yy))
    for control in (m, plain, nohistory, fifo, reservoir):
        control.learn(X, A, Y)
    # VQ quantizes sensory context only, then averages effects by cluster/action.
    code = X[:48].copy()
    avg = np.zeros((48, 4, 8))
    count = np.zeros((48, 4))
    for key, a, delta in zip(X, A, Y):
        i = np.argmin(np.sum((code - key) ** 2, 1))
        count[i, a] += 1
        avg[i, a] += (delta - avg[i, a]) / count[i, a]

    def vq(k, a):
        return avg[np.argmin(np.sum((code - k) ** 2, 1)), a]

    methods = {
        "v11": lambda k, a: (
            model.forecast(
                np.resize(k[:8], 24), a, k[8:], model.infer(np.resize(k[:8], 24))
            )[:8]
            - k[:8]
        ),
        "recurrent": esn.step,
        "matched_RLS": matched.step,
        "v10": m.predict,
        "no_entity": lambda k, a: m.predict(k, a, entities=False),
        "prototype_only": lambda k, a: m.predict(k, a, shared=False),
        "no_merge": plain.predict,
        "no_sensorimotor": nohistory.predict,
        "VQ": vq,
        "FIFO": fifo.predict,
        "reservoir": reservoir.predict,
        "RLS": lambda k, a: feat(k[:8], a, k[8:]) @ W,
        "v9": lambda k, a: v9.rollout(state9(k[:8], k[8:]), [action9(a)])[:8] - k[:8],
        "hold": lambda k, a: np.zeros(8),
    }
    scores = {k: [] for k in methods}
    for _ in range(24):
        x = initial()
        regime = rng.choice([-1, 1])
        esn.hidden.fill(0)
        h = command(1)
        pred = {k: x.copy() for k in methods}
        commands = rng.integers(4, size=16)
        ensemble = model.imagine(np.resize(x, 24), commands, h) if rev >= 14 else None
        for j in range(16):
            a = int(commands[j])
            x = advance(x, a, h)
            for k, fn in methods.items():
                last = pred[k]
                pred[k] = np.clip(last + fn(np.r_[last, h], a), 0, 1)
                if k == "v11" and ensemble is not None and j in (0, 3, 7, 15):
                    scores.setdefault("single_path", []).append(
                        np.mean((pred[k] - x) ** 2)
                    )
                    scores[k].append(np.mean((ensemble[j, :8] - x) ** 2))
                    continue
                if j in (0, 3, 7, 15):
                    scores[k].append(np.mean((pred[k] - x) ** 2))
            h = command(a)
    return {k: np.mean(np.reshape(v, (-1, 4)), 0).tolist() for k, v in scores.items()}


if __name__ == "__main__":
    import sys

    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    start = 11101 + 100 * rev
    rows = [run(s, rev) for s in range(start, start + 24)]
    (ROOT / f"measurements/v11_dynamics_{rev}.json").write_text(
        json.dumps(rows, indent=2)
    )
