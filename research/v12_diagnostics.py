"""Post-seal interface diagnostics: need-dependent goals and silent sensory twins."""

import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from loki_memory.journey import LokiV12
from .v12_life import Body


def check(seed):
    w = Body(seed)
    rng = np.random.default_rng(seed)
    m = LokiV12(str(seed), w.widths, w.motors, place_dim=2)
    for _ in range(1600):
        a = rng.integers(len(w.commands))
        m.observe(w.step(a), w.commands[a], place=w.place())
    goal_hits = []
    latency = []
    for original, location in enumerate((1, 5)):
        need = np.zeros(3)
        need[np.flatnonzero(w.perm == original)[0]] = 0.9
        t = perf_counter_ns()
        routes = m.journeys(need)
        latency.append((perf_counter_ns() - t) / 1e6)
        cue = m.goal_example(routes[0].journey_id)["cue"] if routes else np.zeros(2)
        angle = location * 2 * np.pi / 9
        goal_hits.append(
            bool(np.linalg.norm(cue - [np.cos(angle), np.sin(angle)]) < 0.1)
        )
    t = perf_counter_ns()
    m.foresee(w.commands[rng.integers(len(w.commands), size=32)])
    rollout_ms = (perf_counter_ns() - t) / 1e6

    # Same hall and executed action; the preceding sensory landmark determines the result.
    twin = LokiV12(str(seed), [2, 8], [2])
    landmarks = rng.random((4, 8), dtype="f4")
    hall = np.full(8, 0.5, dtype="f4")
    correct = []
    for trial in range(160):
        side = int(rng.integers(2))
        u = np.eye(2, dtype="f4")
        twin.observe(np.r_[[0.4, 0.4], landmarks[side]], u[1])
        twin.observe(np.r_[[0.4, 0.4], hall], u[0])
        if trial >= 96:
            estimate = twin.foresee([u[0]]).bedrock_change
            correct.append(
                float(estimate[side] > estimate[1 - side])
                + 0.5 * float(estimate[side] == estimate[1 - side])
            )
        end = np.array([0.4, 0.4])
        end[side] += 0.3
        twin.observe(np.r_[end, landmarks[2 + side]], u[0])
    return dict(
        seed=seed,
        goals=goal_hits,
        journeys_ms=latency,
        foresee32_ms=rollout_ms,
        twin=float(np.mean(correct)),
    )


if __name__ == "__main__":
    rows = [check(i) for i in range(18101, 18125)]
    Path("measurements/v12_diagnostics.json").write_text(json.dumps(rows, indent=1))
