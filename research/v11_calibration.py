"""Score action forecasts by observable outcome, with train-room and new-room bins."""

import json
import numpy as np
from .reference import ROOT
from .v11_world import Maze, descriptor
from loki_memory import BeliefModel


def calibrate(seed):
    rng = np.random.default_rng(seed)
    m = BeliefModel(16)
    poses = [(x, y, h) for x in range(5) for y in range(5) for h in range(4)]

    def table(w):
        return np.stack([descriptor(w, p) for p in poses])

    train = Maze(seed)
    known = table(train)
    for _ in range(1000):
        i = rng.integers(100)
        a = rng.integers(4)
        m.fit(known[i], a, known[poses.index(train.step(poses[i], a))], adapt=False)
    transitions = m.probabilities()
    report = {}
    for name, w in (("experienced", train), ("unseen", Maze(seed + 9900))):
        obs = known if name == "experienced" else table(w)
        bins = np.zeros((5, 3))
        for i, x in enumerate(obs):
            b = m.infer(x)
            for a in range(4):
                prior = b @ transitions[a]
                winner = int(np.argmax(prior))
                same = (
                    np.mean((m.emission[: m.n] - m.emission[winner]) ** 2, axis=1)
                    < 1.0e-9
                )
                confidence = float(prior[same].sum())
                y = obs[poses.index(w.step(poses[i], a))]
                hit = np.mean((m.emission[winner] - y) ** 2) < 1.0e-9
                bins[min(4, int(5 * confidence))] += [1, confidence, int(hit)]
        report[name] = bins.tolist()
    return report


if __name__ == "__main__":
    data = [calibrate(i) for i in range(12701, 12725)]
    (ROOT / "measurements/v11_calibration.json").write_text(json.dumps(data, indent=1))
