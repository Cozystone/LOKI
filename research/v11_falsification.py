"""Fresh raw-frame and collision diagnostics. Pose access belongs to the evaluator."""

import json
import numpy as np
from .reference import ROOT
from .v11_world import Maze, descriptor
from loki_memory import BeliefModel


def diagnose(seed):
    rng = np.random.default_rng(seed)
    model = BeliefModel(13)
    anchors = rng.random((192, 24), dtype="f4")
    outcomes = rng.random((2, 24), dtype="f4")
    bits = rng.integers(2, size=len(anchors))
    for x, bit in zip(anchors, bits):
        model.fit(x, 0, outcomes[bit], adapt=False)
    law = model.probabilities()[0]
    decode = np.stack([model.infer(y) for y in outcomes])
    guesses = [np.argmax(decode @ (model.infer(x) @ law)) for x in anchors]
    accuracy = np.mean((guesses == bits).reshape(2, -1), axis=1)

    model = BeliefModel(13)
    warm, test = Maze(seed + 5000), Maze(seed)
    poses = [(*xy, h) for xy in sorted(warm.edges) for h in range(4)]
    for tick in range(600):
        p = poses[rng.integers(len(poses))]
        a = rng.integers(4)
        x, y = descriptor(warm, p), descriptor(warm, warm.step(p, a))
        if tick < 480:
            model.fit(x, a, y, topology=False)
        else:
            delta = model.features(x, np.zeros(4)) @ model.theta[a]
            model.cross[a] += [delta @ (y - x), delta @ delta]
    errors = []
    for p in poses:
        frame = test.view(p).reshape(2, 8, 4, 4, 3)
        x = descriptor(test, p)
        for a in range(4):
            nxt = test.view(test.step(p, a)).reshape(frame.shape)
            delta = (model.forecast(x, a, np.zeros(4)) - x).reshape(2, 1, 4, 1, 3)
            errors.append(
                [np.mean((frame + delta - nxt) ** 2), np.mean((frame - nxt) ** 2)]
            )
    return dict(
        seed=seed, recall=accuracy.tolist(), pixel_mse=np.mean(errors, axis=0).tolist()
    )


if __name__ == "__main__":
    rows = [diagnose(s) for s in range(14901, 14925)]
    (ROOT / "measurements/v11_falsification.json").write_text(
        json.dumps(rows, indent=1)
    )
