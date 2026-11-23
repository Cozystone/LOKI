"""Small reverse sensory field. Observation and imagined past use distinct APIs."""
import os
from pathlib import Path
import tempfile

import numpy as np

from .organ import LokiMemory


class ReverseMemory:
    """16×16 RGB retina, two normalized body channels, one persistent float32 slab."""

    layout = (
        ("visual", (24, 768)), ("visual_p", (24, 24)),
        ("return_map", (2, 24, 20)), ("return_p", (2, 24, 24)),
        ("motor_map", (24, 12)), ("motor_p", (24, 24)),
        ("omega", (51, 12)), ("path_omega", (12, 4)),
        ("trace", (2, 8)), ("mass", (2,)), ("levels", (2,)), ("valid", (2,)),
    )
    ridge = .1

    def __init__(self, budget=262144):
        cells = sum(int(np.prod(shape)) for _, shape in self.layout)
        self.core = LokiMemory(768, budget - 4*cells - 256)
        rows, columns = self.core.matrix.shape
        self.matrix = np.zeros((rows + (cells+columns-1)//columns, columns), np.float32)
        self.matrix[:rows] = self.core.matrix
        self.core.matrix = self.matrix[:rows]
        self.budget = budget
        field = self.matrix[rows:].ravel()
        offset = 0
        for name, shape in self.layout:
            end = offset + int(np.prod(shape))
            setattr(self, name, field[offset:end].reshape(shape))
            offset = end
        rng = np.random.default_rng(601)
        self.omega[:] = rng.normal(0, 2., self.omega.shape)
        self.path_omega[:] = rng.normal(0, 2., self.path_omega.shape)
        for p in (self.visual_p, self.motor_p, *self.return_p):
            np.fill_diagonal(p, 1/self.ridge)

    @property
    def state_bytes(self):
        return self.matrix.nbytes

    @staticmethod
    def _input(value, shape):
        a = np.asarray(value, dtype=np.float32)
        if a.shape != shape or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
            raise ValueError(f"expected normalized finite input of shape {shape}")
        return a

    @staticmethod
    def _pool(x, size):
        span = 16//size
        return x.reshape(size, span, size, span, 3).mean(axis=(1,3)).ravel()

    def _wave(self, cue, motor=None):
        phase = (cue-.5) @ self.omega[:48]
        if motor is not None:
            phase += motor @ self.omega[48:]
        return np.concatenate((np.cos(phase), np.sin(phase))) / np.float32(np.sqrt(12))

    @staticmethod
    def _fit(weights, precision, key, target, strength=1.):
        v = precision @ key
        gain = v * (strength / (1 + strength*max(0., float(key @ v))))
        weights += np.outer(gain, target-key @ weights)
        precision -= np.outer(gain, v)

    def cut(self):
        self.core.cut()
        for state in (self.trace, self.mass, self.levels, self.valid):
            state.fill(0)

    def _transport(self, delta, relief):
        angle = delta @ self.path_omega
        co, si = np.cos(angle), np.sin(angle)
        real, imag = self.trace[:, :4].copy(), self.trace[:, 4:].copy()
        self.trace[:, :4] = .90*(real*co + imag*si) + relief[:, None]
        self.trace[:, 4:] = .90*(imag*co - real*si)
        self.mass[:] = .90*self.mass + relief

    def observe(self, frame, levels, action=None):
        """An absent frame advances time but never supplies a visual learning target."""
        body = self._input(levels, (2,))
        x = None if frame is None else self._input(frame, (16,16,3))
        motor = None if action is None else self.core._motor(action)
        if self.valid[0] and motor is None:
            raise ValueError("supply the executed action or cut before a new sequence")
        old = self.core.imagine().reshape(16,16,3)
        if self.valid[0]:
            mk = self._wave(self._pool(old,4), motor)
            delta = mk @ self.motor_map
            self._transport(delta, np.maximum(body-self.levels, 0))
            if x is not None and self.valid[1]:
                observed_delta = self._pool(x,2)-self._pool(old,2)
                self._fit(self.motor_map, self.motor_p, mk, observed_delta)
                key = self._wave(self._pool(x,4))
                self._fit(self.visual, self.visual_p, key, old.ravel()-.5)
                for j, mass in enumerate(self.mass):
                    if mass > .001:
                        target = np.r_[-observed_delta, self.trace[j]/mass]
                        self._fit(self.return_map[j], self.return_p[j], key, target, min(1., float(mass)))
        if x is not None:
            self.core.observe(x.ravel(), action)
        elif motor is not None and self.valid[0]:
            self.core.advance(action)
        self.levels[:] = body
        self.valid[:] = (bool(self.valid[0]) or x is not None, x is not None)

    def recall(self, cue):
        """Return native RGB intensities of the inferred immediately preceding observation."""
        x = self._input(cue, (16,16,3))
        key = self._wave(self._pool(x,4))
        return (.5 + key @ self.visual).reshape(16,16,3).clip(0,1)

    def return_scores(self, cue, actions, glimpses=None):
        """Rank candidate moves by their resemblance to the recalled predecessor."""
        x = self._input(cue, (16,16,3))
        past = self.recall(x)
        actions = tuple(actions)
        if not actions:
            raise ValueError("at least one candidate is required")
        if glimpses is not None:
            if len(glimpses)!=len(actions):
                raise ValueError("one real observation per candidate is required")
            return np.array([-np.mean((self._input(g,(16,16,3))-past)**2) for g in glimpses])
        cue4 = self._pool(x,4)
        target = self._pool(past,2)-self._pool(x,2)
        return np.array([-np.mean((self._wave(cue4,self.core._motor(a)) @ self.motor_map-target)**2)
                         for a in actions])

    def pull(self, cue, levels, actions, *, base=None, use_path=True, use_sight=True):
        """Add recall-based differences to host action scores without advancing memory."""
        x = self._input(cue, (16,16,3))
        need = (1-self._input(levels, (2,)))**2
        cue4 = self._pool(x,4)
        key = self._wave(cue4)
        evidence = np.clip(1-self.ridge*((self.return_p @ key)*key).sum(axis=1),0,1)
        impression = (need*evidence) @ (key @ self.return_map)
        scores = []
        for action in actions:
            step = self._wave(cue4, self.core._motor(action)) @ self.motor_map
            phase = step @ self.path_omega
            displacement = np.r_[np.cos(phase)-1, np.sin(phase)]
            sight = step @ impression[:12] if use_sight else 0.
            path = displacement @ impression[12:] / 4 if use_path else 0.
            scores.append(sight + path)
        if not scores:
            raise ValueError("at least one candidate action is required")
        scores = np.asarray(scores, np.float32)
        scores -= scores.mean()
        if base is not None:
            host = np.asarray(base, dtype=np.float32)
            if host.shape != scores.shape or not np.isfinite(host).all():
                raise ValueError("invalid host action scores")
            scores += host
        return scores

    def save(self, path, *, binding=""):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temp = Path(tmp.name)
        try:
            with temp.open("wb") as out:
                np.savez(out, version=6, budget=self.budget, binding=binding, field=self.matrix)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp,path)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, binding=""):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"])!=6 or data["binding"].item()!=binding:
                raise ValueError("incompatible reverse memory")
            model = cls(int(data["budget"]))
            field = data["field"]
            if field.shape!=model.matrix.shape or field.dtype!=np.float32 or not np.isfinite(field).all():
                raise ValueError("invalid reverse field")
            model.matrix[:] = field
        return model
