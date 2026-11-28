"""Paired synthetic contact assay. Run as python -m research.observed_probe."""

import json
import platform
import time
from pathlib import Path

import numpy as np
from research.observed_outcomes import ObservedOutcomes


def sustained(correct):
    for i in range(3, len(correct)):
        if all(correct[i - 3 : i + 1]):
            return i + 1
    return len(correct) + 1  # finite follow-up; explicit censor code


def life(seed):
    r = np.random.default_rng(seed)
    motion = np.linalg.qr(r.normal(size=(4, 4)))[0][:2].astype("f")
    mark = np.linalg.qr(r.normal(size=(3, 3)))[0][0].astype("f")
    other_mark = np.roll(mark, 1)
    good = seed % 2
    base = ObservedOutcomes(2, 4, 3, 1)
    for i in range(16):
        # Independent of the target wiring: both actions see both outcomes.
        delta = 0.15 * (1 if (i // 2) % 2 else -1)
        base.calibrate(
            i % 2,
            motion[i % 2] + r.normal(0, 0.03, 4),
            delta * mark + r.normal(0, 0.005, 3),
            np.array([delta]),
        )
    fallback = r.integers(2, size=32)
    noise = r.normal(0, 0.01, 32)
    observed_motion_noise = r.normal(0, 0.03, (8, 4))
    visible_noise = r.normal(0, 0.005, (8, 3))
    rows = {}
    import copy

    for mode in ("none", "valid", "wrong", "ambiguous", "new_appearance"):
        m = copy.deepcopy(base)
        if mode != "none":
            for i in range(8):
                a = i % 2
                centre = motion.mean(0) if mode == "ambiguous" else motion[a]
                appearance = other_mark if mode == "new_appearance" else mark
                demo_good = good ^ (mode == "wrong")
                d = -0.15 if a == demo_good else 0.15
                m.observe(
                    centre + observed_motion_noise[i], d * appearance + visible_noise[i]
                )
        pre = m.report()
        correct = []
        for t in range(32):
            a = m.propose(int(fallback[t]), np.array([0.5]), True)
            ok = a == (good ^ (t >= 16))
            correct.append(bool(ok))
            m.confirm(a, np.array([(-0.15 if ok else 0.15) + noise[t]]))
        rows[mode] = dict(
            first=correct[0],
            first_four=sustained(correct[:16]),
            switch_four=sustained(correct[16:]),
            actions_correct=correct,
            pre_own=int(pre["experienced"].sum()),
            demo_events=int(pre["observed"].sum()),
        )
    return rows


def main():
    raw = {s: life(s) for s in range(32000, 32200)}
    summary = {}
    for k in next(iter(raw.values())):
        rows = [v[k] for v in raw.values()]
        stats = {}
        for name in ("first_four", "switch_four"):
            values = [v[name] for v in rows]
            stats[name] = dict(
                median=float(np.median(values)),
                p90=float(np.percentile(values, 90)),
                censored=sum(i == 17 for i in values),
            )
        summary[k] = dict(
            first=np.mean([v["first"] for v in rows]),
            **stats,
            pre_own_max=max(v["pre_own"] for v in rows),
            demo_accepted=np.mean([v["demo_events"] for v in rows]),
        )
    m = ObservedOutcomes(90, 4, 3, 2)
    d = np.array([0.5, 0.5], np.float32)
    latency = {}
    for name, due in [("held", False), ("reconsider", True)]:
        samples = []
        for i in range(5100):
            start = time.perf_counter_ns()
            m.propose(0, d, due)
            if i >= 100:
                samples.append(time.perf_counter_ns() - start)
        latency[name + "_p95_us"] = np.percentile(samples, 95) / 1000
    out = dict(
        summary=summary,
        seeds=raw,
        latency=latency,
        device=platform.processor(),
        numpy=np.__version__,
        array_bytes=sum(
            v.nbytes for v in vars(m).values() if isinstance(v, np.ndarray)
        ),
        physical_joules=None,
    )
    Path("measurements/observed_transfer.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n"
    )
    print(summary, latency, out["array_bytes"])


if __name__ == "__main__":
    main()
