"""Per-body relative place; motion is measured over the interval ending now."""
import numpy as np


class PlaceCue:
    def __init__(self):
        self.weights = np.zeros((27, 2))
        self.precision = np.eye(27) * 1e4
        self.cut()

    def cut(self):
        """Also call on target switches, rebirth and body handoff."""
        self.place = None
        self.visible = False

    def update(self, eye, depth, motion):
        """Return unit bearing or zero. Magnitude is not confidence.
        eye: associated bearing vector; depth: positive radial sensor reading.
        motion: three locally scaled achieved displacements, never commands.
        Missing depth or bearing makes the current observation unavailable.
        """
        observed = None
        if eye is not None and depth is not None and np.isfinite(depth) and depth > 0:
            eye = np.asarray(eye, dtype=float)
            length = np.linalg.norm(eye)
            if np.isfinite(length) and length > 0:
                observed = eye * (depth / length)
        u = np.asarray(motion, dtype=float)
        if u.shape != (3,) or not np.all(np.isfinite(u)):
            self.cut()
        if self.place is not None:
            a, b, c = u
            drive = [a, b, c, a*a, b*b, c*c, a*b, a*c, b*c]
            design = np.outer(drive, [1., *self.place]).ravel()
            prediction = self.place + design @ self.weights
            if observed is not None and self.visible:
                v = self.precision @ design
                gain = v / (1. + design @ v)
                self.weights += np.outer(gain, observed - prediction)
                self.precision -= np.outer(gain, v)
            self.place = prediction
        if observed is not None:
            self.place = observed
        self.visible = observed is not None
        if self.place is None:
            return np.zeros(2)
        return self.place / max(np.linalg.norm(self.place), 1e-12)
