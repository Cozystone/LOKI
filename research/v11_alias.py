"""Visually identical junctions. Independent shuffled sensor codes and motor wiring."""

import json
import numpy as np
from .reference import ROOT
from loki_memory.belief import BeliefModel


def probe(seed, rev, holdout=False):
    rng = np.random.default_rng(seed)
    obs = rng.random((7, 24), dtype="f4")
    approach, hall, landmark, good, bad = obs[:2], obs[2], obs[3:5], obs[5], obs[6]
    motors = rng.permutation(4)
    m = BeliefModel(rev)
    # Both hidden junctions show the very same frame. A one-step probe reveals a landmark.
    for _ in range(16):
        for side in rng.permutation(2):
            for raw in (1, 2, 0) if holdout else (2, 0, 1):
                i, j = m.fit(approach[side], motors[3], hall)
                y = landmark[side] if raw == 2 else good if raw == side else bad
                m.fit(hall, motors[raw], y, incoming=(i, motors[3]))
                if raw == 2:
                    m.fit(y, motors[side], good)
    q, t = m.policy(good)
    b = m.infer(hall)
    results = {}
    for active in (True, False):
        m.attempts.fill(0)
        a = m.choose(b, q, t, active=active)
        hits = []
        for side in range(2):
            if a == motors[2]:
                z = m.infer(landmark[side], b @ t[a])
                hits.append(m.choose(z, q, t) == motors[side])
            else:
                hits.append(a == motors[side])
        results[str(active)] = sum(hits) / 2

    def entropy(b):
        return float(-np.sum(b * np.log(b + 1e-20)))

    mass = []
    calibration = []
    prem = []
    identity = []
    for side in range(2):
        prior = m.infer(approach[side]) @ t[motors[3]]
        z = m.infer(hall, prior)
        label = np.argmax((t[motors[2]] @ m.infer(landmark[side])) * m.infer(hall))
        mass.append(float(z[label]))
        identity.append(int(np.argmax(z)))
        prem.append(float(b.max() > 0.9))
        for raw in range(3):
            nxt = z @ t[motors[raw]]
            target = landmark[side] if raw == 2 else good if raw == side else bad
            calibration.append(float(1 - nxt @ m.infer(target)))
    same = int(identity[0] == identity[1])
    results.update(
        mass=mass,
        entropy=entropy(b),
        collapsed=np.mean(prem),
        splits=m.splits,
        false_merge=same,
        precision=1 / (1 + same),
        recall=1.0,
        nearest_mass=0.5,
        cf_missing_mass=np.mean(calibration),
    )
    return results


if __name__ == "__main__":
    rows = {r: [probe(11101 + 100 * r + i, r) for i in range(24)] for r in range(3)}
    (ROOT / "measurements/v11_alias.json").write_text(json.dumps(rows, indent=1))
    print(rows[2][0])
