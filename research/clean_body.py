"""Per-body CPU prototype. Input/output roles only; no world or policy."""

import numpy as np


class TurningCue:
    """Transport one tracked visual bearing using measured self-turn."""

    def __init__(self):
        self.vector = np.zeros(2)
        self.last = np.zeros(2)
        self.cross = self.power = 0.0
        self.reliability = 0.0

    def cut(self):
        self.vector.fill(0)
        self.last.fill(0)
        self.reliability = 0.0

    def update(self, eye, turn):
        """eye is a body-frame vector of length <=1; zero means unobserved."""
        eye = np.asarray(eye, dtype=float)
        strength = np.linalg.norm(eye)
        old = np.linalg.norm(self.last)
        if old and strength and abs(turn) > 1e-5:
            delta = np.arctan2(
                self.last[0] * eye[1] - self.last[1] * eye[0], self.last @ eye
            )
            weight = old * strength
            self.cross = 0.99 * self.cross + weight * delta * turn
            self.power = 0.99 * self.power + weight * turn * turn
        if strength:
            self.vector[:] = eye / strength
            self.reliability = min(strength, 1.0)
        else:
            angle = self.cross / max(self.power, 1e-8) * turn
            c, s = np.cos(angle), np.sin(angle)
            x, y = self.vector
            self.vector[:] = c * x - s * y, s * x + c * y
            self.reliability *= 0.995 * (
                self.power / (self.power + 0.0001) if turn else 1.0
            )
        self.last[:] = eye
        return self.vector.copy() * self.reliability


class OutcomeField:
    """One instance per body and motor wiring. Separate direct and observed fits."""

    def __init__(self, cue_width, motor_width, internal_width):
        self.channels = internal_width
        self.width = (1 + cue_width) * (1 + motor_width)
        self.ends = np.array([1, 4, 8, 16, 32])
        self.lengths = np.diff(np.r_[0, self.ends])
        self.history = np.zeros((2, 32, self.width), np.float32)
        self.weights = np.zeros(
            (2, 5, self.width, internal_width + cue_width), np.float32
        )
        self.precision = np.repeat(
            np.eye(self.width, dtype=np.float32)[None], 2, axis=0
        )
        self.cursor = [0, 0]

    def features(self, cue, motor):
        return np.outer(np.r_[1.0, cue], np.r_[1.0, motor]).ravel().astype(np.float32)

    def cut(self, *, observed=None):
        streams = (0, 1) if observed is None else (int(observed),)
        for k in streams:
            self.history[k].fill(0)
            self.cursor[k] = 0

    def update(self, cue_before, cue_after, motor, change, onset, *, observed=False):
        """Once per tick of this source. change is measured deficit delta, or a
        separately calibrated visible proxy on the observed stream. Cut instead
        of updating across discontinuities, including gaps in demonstrations."""
        k = int(observed)
        ring = self.history[k]
        idx = self.cursor[k]
        ring[idx] = self.features(cue_before, motor) if onset else 0.0
        # Age zero covers the outcome of the currently executed command.
        ordered = ring[(idx - np.arange(32)) % 32]
        design = np.add.reduceat(ordered, np.r_[0, self.ends[:-1]], axis=0)
        target = np.r_[change, np.subtract(cue_after, cue_before)]
        weights = self.weights[k]
        error = target - np.einsum("ij,ijk->k", design, weights)
        weights += 0.32 * design[:, :, None] * error / (1.0 + np.sum(design * design))
        matured = ordered[-1]
        p = self.precision[k]
        v = p @ matured
        p -= np.outer(v, v) / (1.0 + matured @ v)
        self.cursor[k] = (idx + 1) % 32

    def read(self, cue, motor, *, observed=False):
        """Integrated change over 32 ticks conditional on no additional onset.
        Return body delta, cue delta, and design leverage (lower = more sampled)."""
        k = int(observed)
        f = self.features(cue, motor)
        out = self.lengths @ np.einsum("j,ijk->ik", f, self.weights[k])
        return (
            out[: self.channels],
            out[self.channels :],
            float(f @ self.precision[k] @ f),
        )

    @property
    def array_bytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray))


class SensoryPlane:
    """Learn one rotation-sensitive plane at rest; fixed axes preserve old keys."""

    def __init__(self, width):
        self.moment = np.zeros((width, width))
        self.mean = np.zeros(width)
        self.basis = np.zeros((width, 2))
        self.pairs = 0
        self.fitted = False

    def pair(self, previous, current, turn):
        if self.fitted:
            raise RuntimeError("changing axes requires migrating downstream memories")
        self.pairs += 1
        center = (previous + current) * 0.5
        self.mean += (center - self.mean) / self.pairs
        # Remove static channel offsets. Only self-turn-correlated changes add.
        cross = np.outer(previous - self.mean, current - self.mean)
        self.moment += turn * (cross - cross.T)

    def consolidate(self):
        if not np.any(self.moment):
            return False
        self.basis[:] = np.linalg.svd(self.moment)[0][:, :2]
        self.fitted = True
        return True

    def encode(self, observed):
        if not self.fitted:
            raise RuntimeError("calibrate before use")
        raw = (observed - self.mean) @ self.basis
        return raw / max(1.0, np.linalg.norm(raw))
