"""Fixed-state two-action sensory reconstruction. No dependency on the world."""
from pathlib import Path
import os
import tempfile

import numpy as np


class RecurrenceMemory:
    window = 32
    harmonics = 5
    max_period = 32

    def __init__(self, width):
        if not isinstance(width, (int, np.integer)) or width < 1:
            raise ValueError("width must be a positive integer")
        candidates = [(p, b) for p in range(2, self.max_period + 1) for b in range(p)]
        self.period, self.stride = np.array(candidates, dtype=np.uint8).T.copy()
        self.phase = np.zeros_like(self.period)
        self.history = np.zeros((self.window, width), np.float32)
        self.address = np.zeros((self.window, len(candidates)), np.uint8)
        self.error = np.zeros(len(candidates), np.float32)
        self.support = np.zeros_like(self.error)
        self.weights = np.zeros((2 * self.harmonics + 1, width), np.float32)
        self.orders = np.arange(1, self.harmonics + 1, dtype=np.float32)
        self.slot = self.used = self.winner = 0

    def _features(self, phase):
        angles = np.asarray(phase, dtype=np.float32)[..., None] * self.orders
        angles *= np.float32(2 * np.pi / self.period[self.winner])
        return np.concatenate((np.ones_like(angles[..., :1]), np.sin(angles), np.cos(angles)), axis=-1)

    def step(self, action, reading=None):
        """Execute one known action. Return the prediction made before the new reading.

        reading is the resulting sensor vector in [0,1], or None on sensor dropout.
        """
        if action not in (0, 1):
            raise ValueError("expected one of two action IDs")
        if reading is not None:
            reading = np.asarray(reading, dtype=np.float32)
            if reading.shape != self.history.shape[1:] or not np.isfinite(reading).all() or np.any((reading < 0) | (reading > 1)):
                raise ValueError("invalid sensor vector")
        self.phase += 1 if action == 0 else self.stride
        self.phase %= self.period
        feature = self._features(self.phase[self.winner])
        raw = feature @ self.weights
        prediction = raw.clip(0, 1)
        if reading is None:
            return prediction

        if self.used:
            matches = self.address[:self.used] == self.phase
            distance = ((self.history[:self.used] - reading) ** 2).mean(axis=1)
            self.error *= 0.995
            self.error += np.einsum("lh,l->h", matches, distance)
            self.support *= 0.995
            self.support += matches.sum(axis=0)
        self.history[self.slot] = reading
        self.address[self.slot] = self.phase
        self.slot = (self.slot + 1) % self.window
        self.used = min(self.window, self.used + 1)
        score = (self.error + 0.02) / (self.support + 2) + 1e-6 * self.period
        best = int(score.argmin())
        if best != self.winner:
            self.winner = best
            design = self._features(self.address[:self.used, best])
            gram = design.T @ design
            gram.flat[::len(gram) + 1] += 0.05
            self.weights[:] = np.linalg.solve(gram, design.T @ self.history[:self.used])
        else:
            self.weights += np.outer(feature, reading - raw) * (0.45 / (feature @ feature))
        return prediction

    def _arrays(self):
        return {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}

    @property
    def state_bytes(self):
        return sum(array.nbytes for array in self._arrays().values()) + 3 * 8

    def save(self, path):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            try:
                np.savez(f, version=1, cursors=[self.slot, self.used, self.winner], **self._arrays())
                f.flush()
                os.fsync(f.fileno())
            except BaseException:
                f.close()
                Path(f.name).unlink(missing_ok=True)
                raise
        try:
            os.replace(f.name, path)
        finally:
            Path(f.name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("unsupported checkpoint")
            model = cls(data["history"].shape[1])
            for name, dest in model._arrays().items():
                src = data[name]
                if src.dtype != dest.dtype or src.shape != dest.shape:
                    raise ValueError(f"invalid checkpoint array {name}")
                dest[:] = src
            model.slot, model.used, model.winner = map(int, data["cursors"])
            if not (0 <= model.slot < model.window and 0 <= model.used <= model.window and 0 <= model.winner < len(model.period)):
                raise ValueError("invalid checkpoint cursor")
        return model
