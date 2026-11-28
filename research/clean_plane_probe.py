"""Evaluator alone knows orientations and the random physical-to-sensor wiring."""

import json
from pathlib import Path
import numpy as np
from research.clean_body import SensoryPlane, TurningCue
from research.clean_probe import unit


def run(seed):
    rng = np.random.default_rng(seed)
    wiring = np.linalg.qr(rng.normal(size=(12, 12)))[0]

    def sense(angle):
        return 0.5 + 0.38 * (wiring[:, :2] @ unit(angle)) + 0.015 * rng.normal(size=12)

    angles = rng.uniform(-np.pi, np.pi, 80)
    turns = rng.uniform(-0.4, 0.4, 80)
    pairs = [(sense(a), sense(a + t)) for a, t in zip(angles, turns)]
    results = []
    for shuffle in (False, True):
        eye = SensoryPlane(12)
        tracker = TurningCue()
        measured = rng.permutation(turns) if shuffle else turns
        for (a, b), t in zip(pairs, measured):
            eye.pair(a, b, t)
        eye.consolidate()
        for (a, b), t in zip(pairs, measured):
            tracker.cut()
            tracker.update(eye.encode(a), 0.0)
            tracker.update(eye.encode(b), t)
        # Separate evaluator calibration decodes arbitrary learned axes to angles.
        refs = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        codes = np.array([eye.encode(sense(a)) for a in refs])
        decoder = np.linalg.lstsq(codes, np.array([unit(a) for a in refs]), rcond=None)[
            0
        ]
        angle = 0.57
        tracker.cut()
        first = tracker.update(eye.encode(sense(angle)), 0.0)
        for _ in range(12):
            last = tracker.update([0.0, 0.0], np.pi / 18)
        truth = unit(angle + 2 * np.pi / 3)

        def error(q, target):
            cosine = (q @ target) / (np.linalg.norm(q) * np.linalg.norm(target))
            return float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))

        actual = eye.encode(sense(angle + 2 * np.pi / 3))
        results.append(
            [
                error(last @ decoder, truth),
                error(first @ decoder, truth),
                error(last, actual),
            ]
        )
    return dict(seed=seed, aligned=results[0], shuffled=results[1])


if __name__ == "__main__":
    rows = [run(s) for s in range(11701, 11725)]
    Path("measurements/clean_plane.json").write_text(json.dumps(rows, indent=2))
    print({k: np.median([r[k] for r in rows], axis=0) for k in ("aligned", "shuffled")})
