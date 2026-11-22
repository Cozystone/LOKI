"""LOKI streaming associative memory. Only NumPy is required by the organ."""
from pathlib import Path
import os
import tempfile
import numpy as np


class LokiMemory:
    """Store bounded sensory episodes and observed action transitions.

    Input order defines neighboring samples in a tile. Input values are [0,1].
    Each tile borrows a shared unit-RMS waveform under a quantized affine map.
    """
    edges = 8
    overhead = 128  # reserve for scalar configuration and cursors in the byte budget

    def __init__(self, width, budget=262144, tile=64, atoms=32):
        if width < 1 or tile < 1 or not 0 <= atoms <= 255:
            raise ValueError("invalid memory dimensions")
        self.width, self.budget, self.tile = int(width), int(budget), int(tile)
        self.tiles = (self.width + self.tile - 1) // self.tile
        fixed = self.overhead + atoms * (4 * tile + 4)
        per_slot = self.tiles * (3 if atoms else 1) + self.edges * 10 + 6
        self.capacity = min(32767, (budget - fixed) // per_slot)
        if self.capacity < 2:
            raise ValueError("budget cannot hold two experiences at this width")
        shape = (self.capacity, self.tiles)
        self.center = np.zeros(shape, np.uint8)
        self.index = np.zeros(shape if atoms else (0, 0), np.uint8)
        self.gain = np.zeros_like(self.index, dtype=np.int8)
        self.atoms = np.zeros((atoms, tile), np.float32)
        self.refs = np.zeros(atoms, np.int32)
        links = (self.capacity, self.edges)
        self.actions = np.full(links, -1, np.int32)
        self.targets = np.full(links, -1, np.int16)
        self.counts = np.zeros(links, np.float32)
        self.strength = np.zeros(self.capacity, np.float32)
        self.age = np.zeros(self.capacity, np.uint16)
        self.used = 0
        self.current = -1
        self.grounded = False

    def _arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    @property
    def state_bytes(self):
        return self.overhead + sum(v.nbytes for v in self._arrays().values())

    def _tiles(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.width,) or not np.isfinite(frame).all() or np.any((frame < 0) | (frame > 1)):
            raise ValueError("expected a finite sensor vector in [0,1]")
        if self.width % self.tile:
            frame = np.pad(frame, (0, self.tiles * self.tile - self.width), mode="edge")
        return frame.reshape(self.tiles, self.tile)

    def _project(self, blocks, learn):
        mean = blocks.mean(axis=1)
        residual = blocks - mean[:, None]
        dot = (residual @ self.atoms.T) / self.tile
        if learn and len(self.atoms):
            energy = (residual ** 2).mean(axis=1)
            # Reuse unreferenced atoms only. All existing episode decoders stay valid.
            free = np.flatnonzero(self.refs == 0)
            for j in free[:8]:
                unexplained = energy - np.square(dot).max(axis=1)
                k = int(unexplained.argmax())
                if unexplained[k] < 0.0003:
                    break
                self.atoms[j] = residual[k] / np.sqrt(energy[k])
                dot[:, j] = residual @ self.atoms[j] / self.tile
        return mean, dot

    def _distances(self, mean, dot, columns):
        # Expand squared error algebraically. The cue energy is common to all
        # candidates. Work scales with the encoded table, without decoding it.
        centers = self.center[:self.used, columns].astype(np.float32) / 255
        score = (centers - mean) ** 2
        if len(self.atoms):
            gain = self.gain[:self.used, columns].astype(np.float32) / 254
            ids = self.index[:self.used, columns]
            score += gain * (gain - 2 * dot[np.arange(len(mean)), ids])
        return score.mean(axis=1)

    def _render(self, slot):
        if slot < 0:
            return np.full(self.width, 0.5, np.float32)
        center = self.center[slot].astype(np.float32) / 255
        if len(self.atoms):
            gain = self.gain[slot].astype(np.float32) / 254
            decoded = center[:, None] + gain[:, None] * self.atoms[self.index[slot]]
        else:
            decoded = np.repeat(center, self.tile)
        return decoded.ravel()[:self.width].clip(0, 1)

    def locate(self, cue, visible):
        """Find the sensory address using visible whole tiles only."""
        mask = np.asarray(visible)
        if mask.dtype != bool or mask.shape != (self.tiles,) or not mask.any():
            raise ValueError("visible must select at least one whole tile")
        cue = np.asarray(cue)
        if cue.shape != (self.width,):
            raise ValueError("wrong cue width")
        columns = np.flatnonzero(mask)
        locations = (columns[:, None] * self.tile + np.arange(self.tile)).clip(max=self.width - 1)
        blocks = cue[locations].astype(np.float32)
        if not np.isfinite(blocks).all() or np.any((blocks < 0) | (blocks > 1)):
            raise ValueError("invalid visible samples")
        if not self.used:
            return -1
        mean, dot = self._project(blocks, False)
        return int(self._distances(mean, dot, columns).argmin())

    def recall(self, cue, visible):
        """Reconstruct from a partial cue without changing the live address."""
        return self._render(self.locate(cue, visible))

    def cut(self):
        """Break the causal chain while retaining consolidated experience."""
        self.current = -1
        self.grounded = False

    @staticmethod
    def _check_action(action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action <= 2147483647:
            raise ValueError("expected a nonnegative int32 motor ID")

    def _successor(self, slot, action):
        if slot < 0:
            return -1
        counts = np.where(self.actions[slot] == action, self.counts[slot], 0)
        return int(self.targets[slot, counts.argmax()]) if counts.max() > 0 else slot

    def imagine(self, actions, *, start=None):
        """Roll out action IDs without writing to the live state."""
        slot = self.current if start is None else int(start)
        if not -1 <= slot < self.used:
            raise ValueError("invalid start address")
        for action in actions:
            self._check_action(action)
            slot = self._successor(slot, action)
        return self._render(slot)

    def advance(self, action):
        """Carry the live address across a missing observation."""
        self._check_action(action)
        self.current = self._successor(self.current, action)
        self.grounded = False
        return self._render(self.current)

    def _replace(self):
        if self.used < self.capacity:
            slot = self.used
            self.used += 1
            return slot
        slot = int(np.lexsort((-self.age.astype(np.int32), self.strength))[0])
        if len(self.atoms):
            self.refs -= np.bincount(self.index[slot], minlength=len(self.atoms)).astype(np.int32)
        invalid = self.targets == slot
        invalid[slot] = True
        self.targets[invalid] = self.actions[invalid] = -1
        self.counts[invalid] = 0
        self.strength[slot] = 0
        if self.current == slot:
            self.cut()
        return slot

    def observe(self, frame, action=None):
        """Observe resulting sensation. action is the motor ID since the last frame."""
        if action is not None:
            self._check_action(action)
        blocks = self._tiles(frame)
        mean, dot = self._project(blocks, True)
        if len(self.atoms):
            ids = np.abs(dot).argmax(axis=1)
            coef = dot[np.arange(self.tiles), ids]
            best_fit = -np.square(coef).mean()
        else:
            best_fit = 0.0
        slot = -1
        if self.used:
            distance = self._distances(mean, dot, slice(None))
            near = int(distance.argmin())
            # Admit only differences above our own codec's distortion floor.
            if distance[near] - best_fit <= 0.0005:
                slot = near
        if slot < 0:
            slot = self._replace()
            self.center[slot] = np.rint(mean * 255)
            if len(self.atoms):
                self.index[slot] = ids
                self.gain[slot] = np.rint(coef * 254).clip(-127, 127)
                self.refs += np.bincount(ids, minlength=len(self.atoms)).astype(np.int32)
        prev = self.current
        if self.grounded and prev >= 0 and action is not None:
            hits = (self.actions[prev] == action) & (self.targets[prev] == slot)
            edge = int(hits.argmax()) if hits.any() else int(self.counts[prev].argmin())
            if not hits.any():
                self.counts[prev, edge] = 0
            self.counts[prev] *= 0.99
            self.actions[prev, edge], self.targets[prev, edge] = action, slot
            self.counts[prev, edge] += 1
        self.strength *= 0.995
        self.strength[slot] += 1
        self.age += (self.age < 65535).astype(np.uint16)
        self.age[slot] = 0
        self.current = slot
        self.grounded = True
        return slot

    def save(self, path, *, binding=""):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            name = tmp.name
        try:
            with open(name, "wb") as out:
                np.savez(out, binding=np.asarray(binding), config=[1, self.width, self.budget, self.tile, len(self.atoms), self.used, self.current, self.grounded], **self._arrays())
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, binding=""):
        with np.load(path, allow_pickle=False) as data:
            if str(data["binding"].item()) != binding:
                raise ValueError("checkpoint belongs to a different body binding")
            version, width, budget, tile, atoms, used, current, grounded = map(int, data["config"])
            if version != 1:
                raise ValueError("unsupported LOKI checkpoint")
            mem = cls(width, budget, tile, atoms)
            if not 0 <= used <= mem.capacity or not -1 <= current < used:
                raise ValueError("invalid LOKI checkpoint cursor")
            for key, array in mem._arrays().items():
                saved = data[key]
                if saved.shape != array.shape or saved.dtype != array.dtype:
                    raise ValueError(f"invalid LOKI checkpoint array {key}")
                array[:] = saved
            mem.used, mem.current, mem.grounded = used, current, bool(grounded)
        return mem
