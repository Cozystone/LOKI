"""A bounded stochastic predictive-state filter; simulator state never enters this module."""

import os
from pathlib import Path
import tempfile

import numpy as np


class BeliefModel:
    def __init__(self, revision=16):
        self.emission = np.zeros((96, 24), np.float32)
        self.count = np.zeros((4, 96, 96), np.float32)
        self.n = 0
        self.revision = revision
        self.theta = np.zeros((4, 53, 24), np.float32)
        self.precision = np.repeat((np.eye(53, dtype=np.float32) * 8)[None], 4, axis=0)
        self.projection = (
            np.random.default_rng(11).normal(0, 1.8, (28, 24)).astype("f4")
        )
        if revision >= 11:
            self.projection.fill(0)
            self.projection[np.arange(24), np.arange(24)] = np.random.default_rng(
                11
            ).uniform(2, 6, 24)
        self.attempts = np.zeros((4, 96), np.float32)
        if revision >= 14:
            self.axis = np.ones(24, dtype="f4") / np.sqrt(np.float32(24))
            self.lag = np.zeros(3, dtype="f4")
            self.previous_result = np.zeros(24, dtype="f4")
        self.splits = 0
        self.cross = np.zeros((4, 2), np.float32)

    def features(self, x, last):
        v = np.r_[x, last]
        return np.r_[1.0, v, np.sin(v @ self.projection)].astype("f4")

    def emission_cost(self, x):
        return np.mean((self.emission[: self.n] - x) ** 2, axis=1)

    def slot(self, x):
        d = self.emission_cost(x)
        if self.n < 96 and (not self.n or d.min() > 1e-9):
            i = self.n
            self.n += 1
            self.emission[i] = x
            return i
        return int(d.argmin())

    def fit(self, x, a, y, last=None, topology=True, adapt=True, incoming=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if (
            x.shape != (24,)
            or y.shape != (24,)
            or not (np.isfinite(x).all() and np.isfinite(y).all())
        ):
            raise ValueError("expected two finite 24-component observations")
        if not isinstance(a, (int, np.integer)) or a not in range(4):
            raise ValueError("action must be in range(4)")
        last = np.zeros(4) if last is None else last
        if adapt:
            h = self.features(x, last)
            p = self.precision[a]
            v = p @ h
            gain = v / (1 + h @ v)
            residual = y - x - h @ self.theta[a]
            if self.revision >= 14:
                projection = float(self.axis @ residual)
                if self.lag[2] and np.array_equal(
                    self.previous_result, np.asarray(x, dtype="f4")
                ):
                    self.lag[0] += 0.01 * (projection * self.lag[1] - self.lag[0])
                self.axis += 0.02 * projection * (residual - projection * self.axis)
                self.axis /= np.linalg.norm(self.axis)
                self.lag[1:] = projection, 1
                self.previous_result[:] = y
            self.theta[a] += np.outer(gain, residual)
            p -= np.outer(gain, v)
        if topology:
            i = self.slot(x)
            j = self.slot(y)
            if self.revision >= 2 and incoming is not None:
                u, motor = incoming
                same = np.flatnonzero(self.emission_cost(x) < 1e-9)
                if len(same):
                    candidates = self.count[motor, u, same]
                    i = int(same[candidates.argmax()])
                    row = self.count[a, i]
                    if row.sum() and not row[j] and self.n < 96:
                        existing = [k for k in same if self.count[a, k, j] > 0]
                        if existing:
                            clone = existing[0]
                        else:
                            clone = self.n
                            self.n += 1
                            self.splits += 1
                            self.emission[clone] = x
                        self.count[motor, u, i] = max(0, self.count[motor, u, i] - 1)
                        self.count[motor, u, clone] += 1
                        i = clone
            if self.revision >= 7:
                candidates = np.flatnonzero(self.emission_cost(y) < 1e-9)
                if len(candidates):
                    support = self.count[a, i, candidates]
                    if support.max() > 0:
                        j = int(candidates[support.argmax()])
            self.count[a, i, j] += 1
            return i, j

    def probabilities(self):
        c = self.count[:, : self.n, : self.n]
        if self.revision >= 8:
            c = c.copy()
            present = (c > 0).astype("f4")
            for matrix in present:
                np.fill_diagonal(matrix, 0)
            for a in range(4):
                missing = c[a].sum(1) == 0
                if not missing.any():
                    continue
                confidence = []
                for other in present:
                    candidates = other.T * (present[a].sum(1) > 0)[:, None]
                    confidence.append(
                        float(
                            np.sum(candidates * present[a]) / max(4.0, candidates.sum())
                        )
                    )
                best = int(np.argmax(confidence))
                if confidence[best] > 0.75:
                    c[a, missing] = present[best].T[missing] * confidence[best]
        denom = c.sum(2, keepdims=True)
        return np.where(
            denom > 0,
            c / np.where(denom > 0, denom, 1),
            np.eye(self.n, dtype="f4")[None],
        )

    def infer(self, x, prior=None):
        if not self.n:
            raise ValueError(
                "belief inference requires at least one observed transition"
            )
        cost = self.emission_cost(x)
        scale = 0.000015
        likelihood = np.exp(-(cost - cost.min()) / scale)
        posterior = (
            likelihood
            if prior is None
            else likelihood * (0.998 * prior + 0.002 / self.n)
        )
        return posterior / posterior.sum()

    def forecast(self, x, a, last, belief=None):
        local = x + self.features(x, last) @ self.theta[a]
        if self.revision >= 5 and self.cross[a, 1] > 0:
            gain = np.clip(self.cross[a, 0] / self.cross[a, 1], 0.05, 0.5)
            local = x + gain * (local - x)
        if belief is None or not self.n:
            return local.clip(0, 1)
        trans = self.probabilities()[a]
        prediction = (belief @ trans) @ self.emission[: self.n]
        evidence = belief @ self.count[a, : self.n, : self.n].sum(1)
        w = evidence / (8 + evidence) * np.exp(-self.emission_cost(x).min() / 0.001)
        return np.clip(local * (1 - w) + prediction * w, 0, 1)

    def imagine(self, observation, commands, previous):
        """Marginal mean over two persistent latent disturbances, one row per future tick."""
        paths = np.repeat(np.asarray(observation, dtype=float)[None], 2, axis=0)
        spread = (
            self.axis * np.sqrt(max(0.0, self.lag[0])) if self.revision >= 14 else 0.0
        )
        result = []
        for a in commands:
            for branch, sign in enumerate((-1, 1)):
                x = paths[branch]
                posterior = self.infer(x) if self.n else None
                paths[branch] = np.clip(
                    self.forecast(x, a, previous, posterior) + sign * spread, 0, 1
                )
            result.append(paths.mean(axis=0))
            previous = np.eye(4)[a]
        return np.asarray(result).reshape(-1, 24)

    def policy(self, goal):
        t = self.probabilities()
        terminal = self.infer(goal)
        risk = (
            np.exp(np.sum(t * np.log(t + 1e-20), axis=2)) if self.revision >= 9 else 1.0
        )
        if self.revision >= 16:
            risk = risk / (1 + 0.12 * self.attempts[:, : self.n])
        v = terminal.copy()
        for _ in range(32):
            q = 0.96 * risk * (t @ v)
            v = np.maximum(terminal, q.max(axis=0))
        return q, t

    def choose(self, b, q, t, active=True):
        utilities = q @ b
        if active and np.max(b) < 0.85:
            # Mutual information between initial hypothesis and observable next view.
            groups = np.unique(
                self.emission[: self.n].round(4), axis=0, return_inverse=True
            )[1]
            for a in range(4):
                outcomes = np.zeros((self.n, groups.max() + 1))
                for j, g in enumerate(groups):
                    outcomes[:, g] += t[a, :, j]
                mix = b @ outcomes
                info = np.sum(
                    b[:, None] * outcomes * np.log((outcomes + 1e-15) / (mix + 1e-15))
                )
                utilities[a] += 0.08 * info
        if self.revision >= 6 and active:
            utilities -= (
                0.16
                * (max(0.05, utilities.max()) if self.revision >= 10 else 1.0)
                * np.log1p(self.attempts[:, : self.n] @ b)
            )
        action = int(np.argmax(utilities))
        if self.revision >= 6 and active:
            self.attempts[action, : self.n] += b
        return action

    def compact_step(self, belief, goal):
        """Choose one action and predict its successor with bounded planning scratch."""
        from .belief_compact import decide

        if self.revision != 16:
            raise ValueError("compact controller implements revision 16")
        belief = np.asarray(belief)
        if belief.shape != (self.n,):
            raise ValueError("belief width must match the active hypotheses")
        return decide(
            self.count,
            self.emission,
            self.attempts,
            belief,
            self.infer(np.asarray(goal, dtype=np.float32)),
        )

    def save(self, path):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = f.name
        try:
            with open(name, "wb") as f:
                np.savez(
                    f,
                    **(
                        {
                            k: getattr(self, k)
                            for k in ("axis", "lag", "previous_result")
                        }
                        if self.revision >= 14
                        else {}
                    ),
                    emission=self.emission,
                    count=self.count,
                    theta=self.theta,
                    precision=self.precision,
                    projection=self.projection,
                    cross=self.cross,
                    meta=np.array([self.n, self.revision, self.splits], "<i8"),
                )

            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            m = cls(int(z["meta"][1]))
            for k in ("emission", "count", "theta", "precision", "projection", "cross"):
                getattr(m, k)[:] = z[k]
            if m.revision >= 14:
                for k in ("axis", "lag", "previous_result"):
                    getattr(m, k)[:] = z[k]
            m.n, m.revision, m.splits = map(int, z["meta"])
            return m

    def source(self, x, a, result, appearance=True):
        """Infer the preceding hypothesis using independent evidence from a completed action."""
        cost = self.emission_cost(x)
        if self.revision >= 15 and appearance:
            candidates = self.emission[: self.n]
            gain = np.clip(
                np.median((x + 0.02) / (candidates + 0.02), axis=1), 0.75, 1.3
            )
            squared = (x - gain[:, None] * candidates) ** 2
            ordered = np.sort(squared, axis=1)
            cost = ordered[:, :18].mean(axis=1) + 0.012 * (gain - 1) ** 2
        prior = np.exp(-(cost - cost.min()) / 0.015) if appearance else np.ones(self.n)
        prior *= self.count[a, : self.n, : self.n].sum(1) > 0
        future = self.emission_cost(result)
        likelihood = self.probabilities()[a] @ np.exp(-(future - future.min()) / 0.0005)
        posterior = prior * (likelihood + 1e-8)
        return posterior / posterior.sum() if posterior.sum() else self.infer(x)

    def repair(self, x, a, y):
        if self.revision >= 4 and self.n:
            b = self.infer(x)
            # If the correctable source is unique, discount its conflicting law.
            i = int(np.argmax(b))
            j = int(np.argmin(self.emission_cost(y)))
            if b[i] > 0.8:
                row = self.count[a, i]
                evidence = row[j] / max(1.0, row.sum())
                row *= 0.1 + 0.9 * evidence
        return self.fit(x, a, y, adapt=False)

    def explore(self, x, rng):
        b = self.infer(x)
        exposure = self.count[:, : self.n, : self.n].sum(2) @ b
        return int(rng.choice(np.flatnonzero(exposure <= exposure.min() + 1e-6)))
