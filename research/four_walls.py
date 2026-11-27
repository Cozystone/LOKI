"""Paired held-out organ tests. Context labels and channel permutations live in this judge."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np

from loki_memory.walls import (
    ContextWitness,
    DelayedEvidence,
    PopulationPrior,
    LagSignature,
)


def context_trial(seed, rare=False, calibrated=False):
    rng = np.random.default_rng(seed)
    n = 16
    textures = rng.uniform(0.1, 0.9, (n, 12))
    angles = np.arange(n) * 2 * np.pi / n
    pos = 0.5 + 0.42 * np.column_stack((np.cos(angles), np.sin(angles)))
    layouts = [rng.permutation(n) for _ in range(3)]
    shocks = rng.integers(n, size=(3, 2))

    def stream(world, places, quiet=False):
        records = []
        for i in places:
            i = i % n
            r = (i == shocks[world]).astype(float)
            records.append(
                (
                    np.clip(
                        textures[layouts[world][i]] + rng.normal(0, 0.014, 12), 0, 1
                    ),
                    pos[i],
                    np.zeros(2) if quiet else r,
                )
            )
        return records

    s = ContextWitness(rare=rare, calibrated=calibrated)
    times = []
    sleep = []

    def pass_stream(data, train=False):
        s.cut()
        reads = []
        for row in data:
            t = perf_counter_ns()
            reads.append(s.observe(*row))
            times.append((perf_counter_ns() - t) / 1e6)
        if train:
            t = perf_counter_ns()
            identity = s.consolidate()
            sleep.append((perf_counter_ns() - t) / 1e6)
            return identity
        return reads

    route = list(range(n + 1)) + list(range(n - 1, -1, -1))
    ta = pass_stream(stream(0, route), True)

    def recall(world, label):
        correct = []
        rel = []
        for i in range(n):
            j = (i + 1) % n
            out = s.recall_edge(
                label, textures[layouts[world][i]], (pos[j] - pos[i] + 1) / 2
            )
            if out is None:
                correct.append(False)
                continue
            key, r = out
            correct.append(
                np.linalg.norm(textures - key, axis=1).argmin() == layouts[world][j]
            )
            if (j == shocks[world]).any():
                rel.extend((r[j == shocks[world]] > 0.5).tolist())
        return [float(np.mean(correct)), float(np.mean(rel)) if rel else 0.0]

    base = recall(0, ta)
    tb = pass_stream(stream(1, route), True)
    a_after_b = recall(0, ta)
    accepted = []
    steps = []
    false = []
    first = []
    quiet = []
    appearance = []
    for k in range(16):
        idx = (k + np.arange(13) * (1 if k % 2 else -1)) % n
        for world, label in enumerate((ta, tb)):
            data = stream(world, idx)
            reads = pass_stream(data)
            accepted.append(reads[-1][0] == label and ta != tb)
            steps.append(
                next(
                    (i for i, (c, _) in enumerate(reads) if c == label and ta != tb),
                    len(reads),
                )
            )
            quiet.append(
                pass_stream(stream(world, idx, True))[-1][0] == label and ta != tb
            )
            # Identical texture sets on both sides. Nearest-appearance tie picks the first.
            costs = [
                np.min(np.linalg.norm(textures[p] - data[-1][0], axis=1))
                for p in layouts[:2]
            ]
            appearance.append(int(np.argmin(costs)) == world)
        unknown = pass_stream(stream(2, idx))
        false.append(any(c >= 0 for c, _ in unknown))
        first.append(unknown[0][0] >= 0)
    # Same cut and exact familiar stream presented under an unseen judge identity.
    indist = pass_stream(stream(0, np.arange(13)))
    return dict(
        known=float(np.mean(accepted)),
        appearance=float(np.mean(appearance)),
        no_relief=float(np.mean(quiet)),
        steps_median=float(np.median(steps)),
        unseen_any=float(np.mean(false)),
        first_false=float(np.mean(first)),
        identical_false=indist[-1][0] >= 0,
        base=base,
        returned=a_after_b,
        distinct=ta != tb,
        tick_p95_ms=float(np.percentile(times, 95)),
        commit_max_ms=max(sleep),
        bytes=len(s.state()),
        digest=s.sha(),
    )


def credit_trial(seed):
    rng = np.random.default_rng(seed)
    out = {}
    for sign in (-1, 1):
        learners = [DelayedEvidence(), DelayedEvidence()]
        short = 0
        flags = rng.integers(2, size=32)
        randomized = rng.permutation(flags)
        for i, follow in enumerate(flags):
            delay = rng.integers(40, 121)
            # Both levels can have spontaneous restoration or harm. No other reward.
            delta = sign * (float(follow) - 0.5) * 0.45 + rng.normal(0, 0.23)
            delta = np.clip(delta, -1, 1)
            for learner, flag in zip(learners, (follow, randomized[i])):
                learner.cut()
                learner.observe(flag, 0.5, 0)
                for _ in range(delay - 1):
                    learner.observe(None, 0.5, 0)
                learner.observe(None, 0.5, delta)
            if delay <= 30:
                short += (follow - 0.5) * delta
        out[str(sign)] = dict(
            correct=bool(sign * learners[0].weight.mean() > 0),
            shuffled=bool(sign * learners[1].weight.mean() > 0),
            short=float(np.sign(short)),
        )
    return out


def prior_trial(seed):
    rng = np.random.default_rng(seed)
    p = rng.beta(3, 2, size=64)
    prior = PopulationPrior()
    for prob in p:
        prior.add_body(rng.binomial(32, prob), 32)
    out = {}
    for mode in ("matched", "reversed"):
        probs = rng.beta(3, 2, 128)
        if mode == "reversed":
            probs = 1 - probs
        local = rng.binomial(12, probs)
        dense = rng.binomial(64 * 12, probs)
        guesses = [
            (local + 1) / 14,
            np.full(128, prior.estimate(0, 0)),
            np.array([prior.estimate(x, 12) for x in local]),
            (dense + 1) / (64 * 12 + 2),
        ]
        out[mode] = [
            dict(
                sign=float(np.mean((v > 0.5) == (probs > 0.5))),
                mse=float(np.mean((v - probs) ** 2)),
            )
            for v in guesses
        ]
    return out


def align_trial(seed):
    rng = np.random.default_rng(seed)

    def run(rates):
        perm = rng.permutation(len(rates))
        bodies = []
        for coeff in (rates, rates[perm]):
            tau = LagSignature(len(coeff))
            x = rng.normal(size=len(coeff))
            for _ in range(1200):
                x = x * coeff + rng.normal(size=len(coeff)) * np.sqrt(1 - coeff**2)
                tau.observe((1 + np.tanh(x)) / 2)
            bodies.append(tau)
        mapping = bodies[0].match(bodies[1])
        return float(np.mean(perm[mapping] == np.arange(len(rates))))

    return dict(
        asymmetric=run(np.linspace(-0.85, 0.85, 8)), symmetric=run(np.full(8, 0.25))
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    rows = [
        dict(
            seed=s,
            context=context_trial(s),
            credit=credit_trial(s),
            population=prior_trial(s),
            align=align_trial(s),
        )
        for s in (range(28201, 28225) if args.sealed else [28100])
    ]
    path = Path("measurements") / (
        "walls_sealed.json" if args.sealed else "walls_dev.json"
    )
    path.write_text(json.dumps(rows, indent=2))
    print(rows[0])
