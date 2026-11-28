"""CPU event-side adapter. Create one instance per continuously learning individual."""

import numpy as np


class ObservedOutcomes:
    def __init__(self, actions, motion_width, appearance_width, channels):
        def zeros(*shape):
            return np.zeros(shape, np.float32)

        self.motion = zeros(actions, motion_width)
        self.motion_n = zeros(actions)
        self.gram = np.eye(appearance_width, dtype=np.float32) * 0.01
        self.cross = zeros(appearance_width, channels)
        self.decoder = zeros(appearance_width, channels)
        self.seen = zeros(actions, channels)
        self.seen_n = zeros(actions)
        self.own = zeros(actions, channels)
        self.own_n = zeros(actions)
        self.agreement = 0.5

    def calibrate(self, executed, motion, visible_change, own_change):
        """Embodiment calibration. This sample need not come from a lever."""
        n = self.motion_n[executed] = min(self.motion_n[executed] + 1, 1024)
        self.motion[executed] += (motion - self.motion[executed]) / n
        self.gram *= 0.99
        self.cross *= 0.99
        self.gram += np.outer(visible_change, visible_change)
        self.cross += np.outer(visible_change, own_change)
        self.decoder[:] = np.linalg.solve(self.gram, self.cross)

    def observe(self, motion, visible_change):
        """External event only. No actor command, interoception or provenance flag."""
        distances = np.linalg.norm(self.motion - motion, axis=1)
        distances[self.motion_n == 0] = np.inf
        if len(distances) < 2:
            return False
        first, second = np.partition(distances, 1)[:2]
        if not np.isfinite(second) or second <= 0 or first >= 0.65 * second:
            return False
        action = distances.argmin()
        n = self.seen_n[action] = min(self.seen_n[action] + 1, 32)
        self.seen[action] += (visible_change @ self.decoder - self.seen[action]) / n
        return True

    def confirm(self, executed, own_change):
        """Write only the outcome of the observer's actual executed action."""
        if self.seen_n[executed]:
            pred = self.seen[executed]
            error = np.linalg.norm(pred - own_change)
            scale = np.linalg.norm(pred) + np.linalg.norm(own_change)
            match = 1 - error / max(scale, 1e-6)
            self.agreement = 0.25 * self.agreement + 0.75 * float(np.clip(match, 0, 1))
        n = self.own_n[executed]
        self.own[executed] += (1 if n == 0 else 0.85) * (
            own_change - self.own[executed]
        )
        self.own_n[executed] = min(n + 1, 1024)

    def propose(self, pick, deficits, due, allowed=None):
        if not due:
            return pick
        change = self.seen * (0.25 * self.agreement)
        direct = self.own_n > 0
        change[direct] = self.own[direct]
        costs = np.square(np.clip(deficits + change, 0, 1)).mean(axis=1)
        if allowed is not None:
            costs = np.where(allowed, costs, np.inf)
        best = int(costs.argmin())
        return best if costs[best] < costs[pick] else pick

    def report(self):
        return dict(
            observed=self.seen_n.copy(),
            experienced=self.own_n.copy(),
            agreement=self.agreement,
        )
