"""ATANOR raw-contract front end without checkpoints, sensor labels or Torch dependencies."""
import json
from math import prod

import numpy as np
from .organ import LokiMemory
from .transport import complete_pixels


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
        grid = None
        if self.channels and self.channels[-1][0]=="image" and sum(widths)<=2048:
            h,w,c = self.channels[-1][1]
            if w>8:
                grid = ((h+7)//8,(w+7)//8,c)
        self.memory = LokiMemory(sum(widths), budget, grid_shape=grid)

    def _bundle(self, raw, body, available):
        parts = []
        masks = []
        if len(raw) != len(self.channels) or (available is not None and len(available) != len(raw)):
            raise ValueError("unexpected raw channel count")
        for i, (value, (kind, shape)) in enumerate(zip(raw, self.channels)):
            if value is None or (available is not None and not available[i]):
                size = ((shape[0]+7)//8)*((shape[1]+7)//8)*shape[2] if kind=="image" else prod(shape)
                parts.append(np.zeros(size,np.float32))
                masks.append(np.zeros(size,bool))
                continue
            x = array(value[body])
            if x.shape != shape:
                raise ValueError("unexpected raw shape")
            if kind == "image":
                x = pool_image(x / 255)[0]
            else:
                x = 0.5 + 0.5 * x / (1 + np.abs(x))
            parts.append(x.ravel())
            masks.append(np.ones(x.size,bool))
        return np.concatenate(parts), np.concatenate(masks)

    def features(self, raw, body=0):
        values, mask = self._bundle(raw, body, None)
        return np.where(mask, values, np.nan)

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

    def step(self, raw, previous_action=None, *, body=0, cut=False, available=None):
        command = self.motor(previous_action) if previous_action is not None and not cut else None
        if cut:
            self.memory.cut()
        if raw is not None:
            features, mask = self._bundle(raw, body, available)
            if mask.any():
                self.memory.observe(features, command, visible=None if mask.all() else mask)
                return self.memory.imagine()
        return self.memory.advance(command) if command is not None else self.memory.imagine()

    def _binding(self, wiring_id):
        if not isinstance(wiring_id, str) or not wiring_id:
            raise ValueError("supply a stable body wiring ID")
        return json.dumps((self.channels, self.motors, wiring_id))

    def save(self, path, wiring_id):
        self.memory.save(path, binding=self._binding(wiring_id))

    def restore(self, path, wiring_id):
        loaded = LokiMemory.load(path, binding=self._binding(wiring_id))
        if (loaded.budget != self.memory.budget or loaded.width != self.memory.width
                or loaded.grid_shape != self.memory.grid_shape):
            raise ValueError("incompatible memory dimensions")
        self.memory = loaded
