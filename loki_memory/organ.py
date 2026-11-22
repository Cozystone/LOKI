"""Continuous residual field. All learned and recurrent state lives in matrix."""
from pathlib import Path
import os
import tempfile

import numpy as np


class LokiMemory:
    motor_width = 8
    motor_features = 40

    def __init__(self, width, budget=262144, *, sleep=True):
        if not isinstance(width, (int, np.integer)) or width < 1:
            raise ValueError("width must be a positive integer")
        self.width, self.budget, self.sleep = int(width), int(budget), bool(sleep)
        # A fixed area-average front end for very wide numeric vectors. Small
        # feature streams (including the multimodal contract) keep every channel.
        self.group = (width + 2047) // 2048 if width > 4096 else 1
        self.features = (width + self.group - 1) // self.group
        for k in range(64, 7, -2):
            rows = self.features + k + 2 * self.motor_features + 7
            if 4 * k * rows <= self.budget:
                break
        else:
            raise ValueError("insufficient budget for eight latent coordinates")
        self.matrix = np.zeros((rows, k), np.float32)
        self._precision[:] = np.eye(k, dtype=np.float32)

    @property
    def state_bytes(self):
        return self.matrix.nbytes

    @property
    def _decoder(self):
        return self.matrix[:self.features]

    @property
    def _precision(self):
        return self.matrix[self.features:self.features + self.matrix.shape[1]]

    @property
    def _turn(self):
        return self.matrix[-7 - 2*self.motor_features:-7 - self.motor_features]

    @property
    def _shift(self):
        return self.matrix[-7-self.motor_features:-7]

    @property
    def _h(self):
        return self.matrix[-1]

    @property
    def _traces(self):
        return self.matrix[-5:-2, :self.motor_width]

    @property
    def _control(self):
        return self.matrix[-2]

    def _sense(self, frame, mask=None):
        x = np.asarray(frame)
        if x.shape != (self.width,):
            raise ValueError("wrong sensor shape")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != x.shape or not mask.any():
                raise ValueError("expected a nonempty boolean component mask")
            # Hidden sentinels (including NaNs) never enter the encoder.
            seen = x[mask]
        else:
            seen = x
        if not np.isfinite(seen).all() or np.any((seen < 0) | (seen > 1)):
            raise ValueError("normalize available sensor components to [0,1]")
        x = np.where(mask, x, 0) if mask is not None else x
        x = x.astype(np.float32, copy=False)
        if self.group == 1:
            return x - 0.5, mask
        starts = np.arange(0, self.width, self.group)
        sums = np.add.reduceat(x, starts)
        if mask is None:
            sizes = np.minimum(self.group, self.width - starts)
            return (sums / sizes).astype(np.float32) - 0.5, None
        counts = np.add.reduceat(mask, starts, dtype=np.int32)
        return (sums / np.maximum(counts, 1)).astype(np.float32) - 0.5, counts > 0

    @staticmethod
    def _bounded(z):
        return z * min(1.0, 3.0 / max(1e-9, float(np.linalg.norm(z))))

    def _settle(self, x, visible, prior):
        # Conjugate-gradient relaxation of a strictly convex quadratic.
        w = self._decoder if visible is None else self._decoder[visible]
        y = x if visible is None else x[visible]
        z = prior.copy()
        ridge = max(1e-6, len(y) * 2e-5)
        r = w.T @ (y - w @ z)
        p = r.copy()
        rr = float(r @ r)
        for _ in range(8):
            if rr < 1e-14:
                break
            wp = w @ p
            alpha = rr / max(1e-18, float(wp @ wp + ridge * (p @ p)))
            z += alpha * p
            r -= alpha * (w.T @ wp + ridge * p)
            new = float(r @ r)
            p = r + (new / rr) * p
            rr = new
        return self._bounded(z)

    def _key(self, error):
        # A fixed signed Fourier transform spreads innovations over all planes.
        i = np.arange(len(error), dtype=np.uint32)
        mixed = i * np.uint32(2654435761)
        sign = 1.0 - 2.0 * ((mixed ^ (mixed >> 13)) & 1).astype(np.float32)
        k = len(self._h)
        spectrum = np.fft.rfft(error * sign, n=max(k, len(error)))
        pick = np.linspace(1, len(spectrum)-1, k//2, dtype=int)
        wave = spectrum[pick]
        return np.stack((wave.real, wave.imag), axis=1).astype(np.float32).ravel()

    def _drive(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if not len(a) or not np.isfinite(a).all():
            raise ValueError("expected a finite motor command")
        freq = np.arange(1, self.motor_width//2 + 1, dtype=np.float32)
        # Same smooth map for every motor value. No ID-indexed parameters.
        phase = (a + 1) @ np.sin(np.outer(np.arange(1, len(a)+1), freq) * 1.618)
        return np.concatenate((np.cos(phase), np.sin(phase))).astype(np.float32)

    def _transport(self, h, traces, action):
        u = self._drive(action)
        decay = np.array([0.7, 0.97, 0.996], np.float32)[:, None]
        traces = decay * traces + (1 - decay) * u
        drive = np.concatenate((u, traces.ravel(), u * np.tanh(3*h[:self.motor_width])))
        angle = drive @ self._turn[:, ::2]
        co, si = np.cos(angle), np.sin(angle)
        pair = h.reshape(-1, 2)
        rotated = np.stack((co*pair[:, 0] - si*pair[:, 1], si*pair[:, 0] + co*pair[:, 1]), axis=1)
        return self._bounded(rotated.ravel() + drive @ self._shift), traces, drive, rotated

    def _render(self, h):
        values = (0.5 + self._decoder @ h).clip(0, 1)
        return values if self.group == 1 else np.repeat(values, self.group)[:self.width]

    def observe(self, frame, action=None):
        x, _ = self._sense(frame)
        prior = self._h.copy()
        has_motion = action is not None and bool(self._control[0])
        if has_motion:
            prior, traces, u, rotated = self._transport(prior, self._traces, action)
        settled = self._settle(x, None, prior)
        innovation = x - self._decoder @ settled
        precision = self._precision
        key = precision @ self._key(innovation)
        strength = min(1.0, float(np.sqrt(np.mean(innovation**2))) / 0.08)
        z = self._bounded(settled + strength * key / max(1e-8, float(np.linalg.norm(key))))
        # Predict dreams before writing; correct their displacement after writing.
        clock = int(self._control[2])
        dream = None
        if self.sleep and clock % 8 == 7:
            phase = (np.arange(len(z), dtype=np.float32) + 1) * (clock + 1) * 0.618034
            probe = np.sin(phase)
            dream = probe - precision @ probe
            teacher = self._decoder @ dream
        leverage = precision @ z
        denominator = 0.08 + float(z @ leverage)
        gain = leverage / denominator
        self._decoder[:] += np.outer(x - self._decoder @ z, gain)
        precision[:] -= np.outer(gain, leverage)
        # A small floor restores plasticity as evidence ages.
        precision[:] *= 0.999
        precision.flat[::len(z)+1] += 0.001
        if dream is not None:
            self._decoder[:] += np.outer(teacher - self._decoder @ dream, dream) * (0.7 / (0.1 + dream @ dream))
        if has_motion:
            if self._control[1]:
                error = z - prior
                rate = 0.2 / (0.5 + u @ u)
                tangent = rotated[:, 0] * error[1::2] - rotated[:, 1] * error[::2]
                self._turn[:, ::2] += rate * np.outer(u, tangent)
                self._shift[:] += rate * np.outer(u, error)
            self._traces[:] = traces
        self._h[:] = z
        self._control[:3] = 1, 1, (clock + 1) % 8192

    def cut(self):
        self._h.fill(0)
        self._traces.fill(0)
        self._control[:2] = 0

    def imagine(self, actions=()):
        h, traces = self._h, self._traces
        for action in actions:
            h, traces, _, _ = self._transport(h, traces, action)
        return self._render(h)

    def advance(self, action):
        h, traces, _, _ = self._transport(self._h, self._traces, action)
        self._h[:] = h
        self._traces[:] = traces
        self._control[:2] = 1, 0
        return self._render(h)

    def recall(self, cue, visible, actions=()):
        x, mask = self._sense(cue, visible)
        h = self._settle(x, mask, self._h)
        traces = self._traces
        for action in actions:
            h, traces, _, _ = self._transport(h, traces, action)
        return self._render(h)

    def save(self, path, *, binding=""):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            name = tmp.name
        try:
            with open(name, "wb") as out:
                np.savez(out, version=2, config=[self.width, self.budget, self.sleep], binding=binding, field=self.matrix)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, binding=""):
        with np.load(path, allow_pickle=False) as data:
            if "version" not in data or int(data["version"]) != 2 or data["binding"].item() != binding:
                raise ValueError("expected a v2 field for this body wiring")
            width, budget, sleep = map(int, data["config"])
            model = cls(width, budget, sleep=bool(sleep))
            field = data["field"]
            if field.shape != model.matrix.shape or field.dtype != np.float32 or not np.isfinite(field).all():
                raise ValueError("invalid field matrix")
            model.matrix[:] = field
            return model
