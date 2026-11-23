import json, tempfile
from pathlib import Path
import numpy as np
from .reference import ROOT
from loki_memory.belief import BeliefModel
from loki_memory.belief_stream import accumulate
from loki_memory.lifelong import LifelongMemory


def run(seed, revision=13):
    rng = np.random.default_rng(seed)
    symbols = rng.random((16, 24), dtype="f4")
    m = BeliefModel(revision)
    v10 = LifelongMemory()
    v10.flags[3] = 2
    rows = []
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state"
        for i in range(1000):
            context = [0, 1, 0, 2][i // 250]
            src = rng.integers(0, 3, 10000) + 3 * context
            act = rng.integers(0, 4, 10000)
            dst = (src % 3 + act + context) % 3 + 3 * context
            if i == 0:
                src[0], act[0], dst[0] = 14, 1, 15
            x, y = symbols[src], symbols[dst]
            m.n = accumulate(x, act, y, m.emission, m.n, m.count)
            if i < 100:
                v10.learn(x[:, :12], act, y[:, :8] - x[:, :8])
            if i + 1 in (1, 10, 100, 250, 500, 750, 1000):
                m.save(path)
                m = BeliefModel.load(path)
                b = m.infer(symbols[14])
                pred = b @ m.probabilities()[1]
                # Which action reaches the once-observed consequence? Never alter the live stream.
                q, t = m.policy(symbols[15])
                intact_action = m.choose(b, q, t, active=False)
                edge = (1, int(b.argmax()), int(m.infer(symbols[15]).argmax()))
                original = m.count[edge]
                m.count[edge] = 0
                q, t = m.policy(symbols[15])
                removed_action = m.choose(b, q, t, active=False)
                m.count[edge] = original
                error = []
                for c in range(3):
                    err = []
                    for j in range(3):
                        for a in range(4):
                            z = m.infer(symbols[3 * c + j]) @ m.probabilities()[a]
                            err.append(
                                np.mean(
                                    (
                                        z @ m.emission[: m.n]
                                        - symbols[3 * c + (j + a + c) % 3]
                                    )
                                    ** 2
                                )
                            )
                    error.append(float(np.mean(err)))
                rows.append(
                    dict(
                        events=(i + 1) * 10000,
                        bytes=path.stat().st_size,
                        one_shot=float(pred @ m.infer(symbols[15])),
                        intact_action=intact_action,
                        edge_deleted_action=removed_action,
                        regime_mse=error,
                        v10_at_1M=float(
                            np.mean(
                                (
                                    v10.predict(symbols[14, :12], 1, exceptions=True)
                                    + symbols[14, :8]
                                    - symbols[15, :8]
                                )
                                ** 2
                            )
                        )
                        if i == 99
                        else None,
                    )
                )
    return rows


if __name__ == "__main__":
    import sys

    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    r = run(11101 + 100 * rev, rev)
    print(r[-1])
    (
        ROOT
        / (
            "measurements/v11_capacity.json"
            if rev == 13
            else f"measurements/v11_capacity_{rev}.json"
        )
    ).write_text(json.dumps(r, indent=2))
