import json
import numpy as np
from .reference import ROOT
from loki_memory.belief import BeliefModel


def run(seed, rev=3, holdout=False):
    rng = np.random.default_rng(seed)
    m = BeliefModel(rev)
    colors = rng.uniform(0.15, 0.75, (8, 3))
    patterns = rng.normal(0, 0.06, (8, 12, 12, 3))
    laws = rng.uniform(0.1, 0.9, (4, 8))
    assignment = rng.permutation(np.tile(np.arange(4), 4))

    def view(i, changed):
        im = np.rot90(colors[i // 2] + patterns[i // 2], rng.integers(4))
        im = np.roll(im, rng.integers(-3, 4, 2), axis=(0, 1)) * rng.uniform(0.93, 1.07)
        if changed:
            im[:3] = 0.4
        if changed and holdout:
            im = im * rng.uniform(0.85, 1.15) + rng.normal(0, 0.012, im.shape)
        return im.reshape(2, 6, 4, 3, 3).mean((1, 3)).ravel().astype("f4")

    def sense(i):
        return np.tile(laws[assignment[i]] * (0.85 if holdout else 1), 3).astype("f4")

    def consequence(i):
        return (0.5 + 0.25 * np.sin(5 * sense(i))).astype("f4")

    ids = []
    for i in range(16):
        x = view(i, False)
        idx, _ = m.fit(x, 0, sense(i))
        ids.append(idx)
        m.fit(x, 1, consequence(i))
    result = {}
    for mode in ("visual", "joint"):
        pair = np.zeros(4, int)
        errors = []
        for i in range(16):
            x = view(i, True)
            b = m.infer(x) if mode == "visual" else m.source(x, 0, sense(i))
            idx = np.array(ids)[np.argmax(b[ids])]
            for k in range(16):
                pair[(0 if i == k else 2) + int(idx != ids[k])] += 1
            # Held-out random texture with same physical probe law.
            x = rng.random(24).astype("f4")
            b = (
                m.infer(x)
                if mode == "visual"
                else m.source(x, 0, sense(i), appearance=False)
            )
            pred = (b @ m.probabilities()[1]) @ m.emission[: m.n]
            errors.append(np.mean((pred - consequence(i)) ** 2))
        tp, fn, fp, tn = pair
        result[mode] = [
            tp / (tp + fp),
            tp / (tp + fn),
            fp / (fp + tn),
            fn / (tp + fn),
            np.mean(errors),
        ]
    indistinguishable = np.sum(assignment[::2] == assignment[1::2])
    result["ideal_expected_accuracy_ceiling"] = 1 - float(indistinguishable) / 16
    return result


if __name__ == "__main__":
    r = [run(s) for s in range(11401, 11425)]
    (ROOT / "measurements/v11_identity.json").write_text(
        json.dumps(r, indent=1, default=float)
    )
    print({k: np.mean([x[k] for x in r], 0) for k in r[0]})
