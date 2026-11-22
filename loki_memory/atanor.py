"""ATANOR raw-contract front end without checkpoints, sensor labels or Torch dependencies."""
import json
from math import prod

import numpy as np
from .organ import LokiMemory


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def pool_image(frame, visible=None, stride=8):
    """Normalized HWC image to area means and an availability mask per feature."""
    h, w, c = frame.shape
    mask = np.ones(frame.shape, bool) if visible is None else visible
    pad = ((0, -h % stride), (0, -w % stride), (0, 0))
    grid = ((h+stride-1)//stride, stride, (w+stride-1)//stride, stride, c)
    values = np.pad(np.where(mask, frame, 0), pad).reshape(grid).sum(axis=(1, 3))
    counts = np.pad(mask, pad).reshape(grid).sum(axis=(1, 3))
    return (values / np.maximum(counts, 1)).astype(np.float32), counts > 0


class AtanorMemory:
    def __init__(self, contract, budget=262144):
        self.channels = tuple((s.kind, tuple(s.shape)) for s in contract.sensors)
        self.motors = tuple(contract.motors)
        widths = [((s[0]+7)//8)*((s[1]+7)//8)*s[2] if k == "image" else prod(s) for k,s in self.channels]
        self.memory = LokiMemory(sum(widths), budget)

    def features(self, raw, body=0):
        parts = []
        if len(raw) != len(self.channels):
            raise ValueError("unexpected raw channel count")
        for value, (kind, shape) in zip(raw, self.channels):
            x = array(value[body])
            if x.shape != shape:
                raise ValueError("unexpected raw shape")
            if kind == "image":
                x = pool_image(x / 255)[0]
            else:
                x = 0.5 + 0.5 * x / (1 + np.abs(x))
            parts.append(x.ravel())
        return np.concatenate(parts)

    def motor(self, action):
        if hasattr(action, "item"):
            action = action.item()
        if not isinstance(action, (int, np.integer)) or not 0 <= action < prod(self.motors):
            raise ValueError("invalid contract motor")
        values = []
        for radix in reversed(self.motors):
            action, digit = divmod(action, radix)
            values.append(digit / max(1, radix-1))
        return np.array(values[::-1], np.float32)

    def step(self, raw, previous_action=None, *, body=0, cut=False):
        command = self.motor(previous_action) if previous_action is not None and not cut else None
        if cut:
            self.memory.cut()
        if raw is None:
            return self.memory.advance(command) if command is not None else self.memory.imagine()
        self.memory.observe(self.features(raw, body), command)
        return self.memory.imagine()

    def _binding(self, wiring_id):
        if not isinstance(wiring_id, str) or not wiring_id:
            raise ValueError("supply a stable body wiring ID")
        return json.dumps((self.channels, self.motors, wiring_id))

    def save(self, path, wiring_id):
        self.memory.save(path, binding=self._binding(wiring_id))

    def restore(self, path, wiring_id):
        loaded = LokiMemory.load(path, binding=self._binding(wiring_id))
        if loaded.budget != self.memory.budget or loaded.width != self.memory.width:
            raise ValueError("incompatible memory dimensions")
        self.memory = loaded
