"""I0 follow-up to the pinned v12 model. Import explicitly to opt in."""

import json
import numpy as np
from .journey import LokiV12, _body


class LokiV121(LokiV12):
    scales = np.array([0, 0.125, 0.25, 0.5, 1], dtype="f4")

    def __init__(
        self,
        contract_signature,
        widths,
        motors,
        budget=262144,
        place_dim=None,
        revision=121,
    ):
        if revision != 121:
            raise ValueError("not a v12.1 contract")
        b = int(widths[0])
        n = min(sum(widths), max(192, b + 1))
        m = min(sum(motors), 24)
        # Reserve for the maximum possible number of places before sizing the v12 field.
        reserve = 4 * (6 * n + 64 * b * (m + 3)) + 32
        super().__init__(
            contract_signature, widths, motors, budget - reserve, place_dim
        )
        self.budget = budget
        self.config.update(budget=budget, revision=121)
        self.header = json.dumps(
            self.config, separators=(",", ":"), ensure_ascii=False
        ).encode()
        s = len(self.visits)
        old = self.matrix
        self.matrix = np.zeros(old.size + 6 * n + s * b * (m + 3), np.float32)
        self.matrix[: old.size] = old
        for name, arr in vars(self).copy().items():
            if isinstance(arr, np.ndarray) and np.shares_memory(arr, old):
                offset = (arr.ctypes.data - old.ctypes.data) // 4
                setattr(
                    self,
                    name,
                    self.matrix[offset : offset + arr.size].reshape(arr.shape),
                )
        pos = old.size

        def take(shape):
            nonlocal pos
            size = int(np.prod(shape))
            out = self.matrix[pos : pos + size].reshape(shape)
            pos += size
            return out

        self.loss = take((5, n))
        self.earned = take((n,))
        self.positive_motor = take((s, b, m))
        self.examples = take((s, b))
        self.headroom = take((s, b))
        self.misses = take((s, b))
        assert pos == self.matrix.size and 12 + len(self.header) + 4 * pos <= budget

    def _scale(self):
        return self.scales[self.loss.argmin(axis=0)]

    def forecast(self, motor_onehot):
        raw, _ = _body(self.x, self._motor(motor_onehot), self.w)
        return self._out(self.x + self._scale() * (raw - self.x))

    def confidence(self):
        return self._out(
            self.earned * self.seen * np.exp(-self.ctrl[3] / 0.01 - self.ctrl[8])
        )

    def advance(self, motor):
        raw, _ = _body(self.x, self._motor(motor), self.w)
        blended = self.x + self._scale() * (raw - self.x)
        super().advance(motor)
        self.x[:] = blended
        return self._out(blended)

    def observe(self, values, motor_onehot=None, visible=None, place=None):
        previous = int(self.ctrl[1])
        live = self.ctrl[2] == 1 and motor_onehot is not None
        if live:
            u = self._motor(motor_onehot)
            y, mask = self._sense(values, visible)
            valid = mask & (self.seen > 0)
            raw, _ = _body(self.x, u, self.w)
            delta = raw - self.x
            actual = y - self.x
            errors = np.abs(self.scales[:, None] * delta - actual)
            error = np.abs(self._scale() * delta - actual)
            before = self.x[: self.bed].copy()
        super().observe(values, motor_onehot, visible, place)
        if not live:
            return
        self.loss[:, valid] += 0.015 * (errors[:, valid] - self.loss[:, valid])
        self.earned[valid] += 0.015 * ((error[valid] <= 0.05) - self.earned[valid])
        self.misses *= 0.995
        j = int(self.ctrl[1])
        if previous != j or j < 0:
            return
        for h in np.flatnonzero(valid[: self.bed]):
            if actual[h] > 0:
                self.examples[j, h] += 1
                self.positive_motor[j, h] += (
                    u - self.positive_motor[j, h]
                ) / self.examples[j, h]
                self.headroom[j, h] = max(before[h], self.headroom[j, h])
                self.misses[j, h] = 0
            elif self.examples[j, h] >= 2 and before[h] <= self.headroom[j, h] + 1e-6:
                # Fractional support for this exact combination under the factors
                # observed when replenishment succeeded at this place.
                support = float(np.prod(self.positive_motor[j, h][u > 0]))
                self.misses[j, h] = min(6.0, self.misses[j, h] + support)

    def next_resource(self, channel):
        direct = super().next_resource(channel)
        n = len(direct)
        old = int(self.last[channel])
        if old < 0:
            return direct
        sites = np.flatnonzero(self.hits[:n, channel] > 0)
        if len(sites) < 2:
            return direct
        transitions = self.order[channel, :n, :n]
        incoming = transitions.sum(axis=0)
        # Soft structural prior: a rarely observed incoming edge may complete
        # an approximately one-to-one cycle. Conflicting observed edges dominate.
        prior = np.zeros(n)
        prior[sites] = 1 / (1 + incoming[sites])
        prior[old] = 0
        prior /= prior.sum()
        row = transitions[old] + prior
        return row / row.sum()

    def resource_at(self, channel, ahead=0):
        estimate = self.resource_belief(channel)
        p = estimate["mass"]
        old = int(self.last[channel])
        if old >= 0:
            visits = self.order[channel, old, : len(p)]
            if visits.sum():
                interval = (
                    visits @ self.duration[channel, old, : len(p)]
                ) / visits.sum()
                phase = (self.since[channel] + ahead - interval) / max(
                    1.0, 0.1 * interval
                )
                switch = 1 / (1 + np.exp(-np.clip(phase, -40, 40)))
            else:
                switch = 0.0
            p *= switch
            p[old] += 1 - switch
        if old >= 0:
            if not p.any():
                p[old] = 1
            keep = np.exp(-self.misses[: len(p), channel] / (1 + max(0, ahead)))
            rejected = float(p @ (1 - keep))
            p *= keep
            alternatives = self.next_resource(channel) * keep
            if alternatives.sum() > 0:
                p += rejected * alternatives / alternatives.sum()
        estimate["unsupported_mass"] = max(0.0, 1 - float(p.sum()))
        return estimate

    def cut(self):
        super().cut()
        if hasattr(self, "misses"):
            self.misses.fill(0)

    @classmethod
    def load(cls, path, binding):
        m = super().load(path, binding)
        # Without an active source the private snapshot starts a new evidence chain.
        if m.ctrl[1] < 0:
            m.misses.fill(0)
            if not m.place_dim:
                for arr in (m.examples, m.positive_motor, m.headroom):
                    arr.fill(0)
        return m
