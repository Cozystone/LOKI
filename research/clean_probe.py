"""Synthetic sensory/motor assay. Hidden geometry and answer live here only."""

import json
import time
from pathlib import Path
import numpy as np
from research.clean_body import OutcomeField, TurningCue


def unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def assay(seed):
    rng = np.random.default_rng(seed)
    eye = TurningCue()
    angle = rng.uniform(-np.pi, np.pi)
    sign = rng.choice([-1, 1])
    for i in range(60):
        turn = rng.uniform(-0.18, 0.18)
        angle += sign * turn
        eye.update(unit(angle + rng.normal(0, 0.012)), turn)
    frozen = eye.vector.copy()
    for _ in range(12):
        turn = np.pi / 18
        angle += turn * sign
        hidden = eye.update([0, 0], turn + rng.normal(0, 0.002))

    def err(vec):
        return float(
            np.rad2deg(
                np.arccos(np.clip(unit(angle) @ vec / np.linalg.norm(vec), -1, 1))
            )
        )

    field, shuffle = [OutcomeField(2, 3, 2) for _ in range(2)]
    dirs = np.eye(3)
    axis = unit(rng.uniform(-np.pi, np.pi))
    mapping = int(rng.integers(2))
    costs = []
    initial = field.array_bytes
    body = np.full(2, 0.5)
    other = body.copy()
    for event in range(135):
        side = int(rng.integers(2))
        pos = (2 * side - 1) * axis + rng.normal(0, 0.07, 2)
        act = int(rng.integers(2))
        channel = side ^ act ^ mapping
        shuffled = int(rng.integers(2))
        delay = int(rng.choice([8, 14]))
        next_pos = rng.uniform(-1, 1, 2)
        for tick in range(22):
            cue = pos if tick < 2 else next_pos
            following = pos if tick < 1 else next_pos
            onset = tick in (0, 4)
            motor = dirs[act if tick < 4 else 2]
            delta = (0.5 - body) * 0.025
            noise = (0.5 - other) * 0.025
            if tick == delay:
                delta[channel] -= 0.08
                noise[shuffled] -= 0.08
            body += delta
            other += noise
            t = time.perf_counter_ns()
            field.update(cue, following, motor, delta, onset)
            for m in dirs[:2]:
                field.read(cue, m)
            costs.append((time.perf_counter_ns() - t) / 1e6)
            shuffle.update(cue, following, motor, noise, onset)
    scores = []
    for model in (field, shuffle):
        wins = 0
        for side in range(2):
            for need in range(2):
                for _ in range(8):
                    q = (2 * side - 1) * axis + rng.normal(0, 0.07, 2)
                    deficits = np.full(2, 0.15)
                    deficits[need] = 0.85
                    estimates = [
                        np.square(np.clip(deficits + model.read(q, m)[0], 0, 1)).mean()
                        for m in dirs[:2]
                    ]
                    wins += int(np.argmin(estimates) == (side ^ need ^ mapping))
        scores.append(wins / 32)
    weights = field.weights.copy()
    field.cut()
    assert not field.history.any() and np.array_equal(weights, field.weights)
    assert not field.weights[1].any() and initial == field.array_bytes
    field.update(pos, next_pos, dirs[0], [0.3, 0], True, observed=True)
    assert field.weights[1].any() and np.array_equal(weights[0], field.weights[0])
    return dict(
        seed=seed,
        angle_error=err(hidden),
        frozen_error=err(frozen),
        correct=scores[0],
        shuffled=scores[1],
        p95_ms=float(np.percentile(costs, 95)),
        bytes=initial,
    )


if __name__ == "__main__":
    rows = [assay(seed) for seed in range(11401, 11425)]
    Path("measurements/clean_body.json").write_text(json.dumps(rows, indent=2))
    print({k: np.mean([r[k] for r in rows]) for k in rows[0]})
