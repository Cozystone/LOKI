"""Continuous contextual address for a completed, observed action transition."""
from ctypes import CDLL, c_void_p, c_int
from pathlib import Path
import sys
import numpy as np

from .portable import PortableField, PortableMemory, _Window, _input

_LIB = Path(__file__).with_name("_phase8.dll" if sys.platform == "win32" else "_phase8.so")
_native = None
if _LIB.exists():
    _native = CDLL(str(_LIB)).phase8
    _native.argtypes = [c_void_p] * 5
    _native.restype = c_int


class Transition:
    """Rows are normalized sensation before, executed action, sensation after."""
    def __init__(self, frame, history):
        self.frame = frame
        self.history = history


class ContextField(PortableField):
    def __init__(self, blob=None):
        old_size = PortableField.size * 4
        extra = 64 * (32 + 64)
        if blob is not None and len(blob) not in (old_size, old_size + 4 * extra):
            raise ValueError("unexpected contextual field size")
        super().__init__(None if blob is None else blob[:old_size])
        projection = (np.random.default_rng(801).normal(size=(64, 32)).astype("<f4")
                      if blob is None or len(blob) == old_size else
                      np.frombuffer(blob[old_size:old_size+8192], dtype="<f4").reshape(64, 32))
        if blob is not None and len(blob) > old_size:
            complement = np.frombuffer(blob[old_size+8192:], dtype="<f4").reshape(64,64)
        else:
            # Protect at most half the modes so a mature v7 field still admits writes.
            values, axes = np.linalg.eigh(self.precision)
            used = axes[:, :32][:, values[:32] < 9.99]
            complement = np.eye(64, dtype="<f4") - used @ used.T
        self.data = np.concatenate((self.data, projection.ravel(), complement.ravel()))
        self.weights = self.data[:64*776].reshape(64, 776)
        self.precision = self.data[64*776:64*840].reshape(64, 64)
        self.omega = self.data[64*840:PortableField.size].reshape(51, 32)
        self.binding = self.data[PortableField.size:PortableField.size+2048].reshape(64, 32)
        self.complement = self.data[-4096:].reshape(64,64)

    def key(self, cue):
        if not isinstance(cue, Transition):
            return super().key(cue)
        frame = np.ascontiguousarray(cue.frame, dtype=np.float32)
        history = np.ascontiguousarray(cue.history, dtype=np.float32)
        if frame.shape != (16, 16, 3) or history.shape != (3, 8):
            raise ValueError("expected 16x16 RGB and three rows of eight channels")
        if _native is not None:
            key = np.empty(64, np.float32)
            if not _native(frame.ctypes.data, history.ctypes.data, self.omega.ctypes.data,
                           self.binding.ctypes.data, key.ctypes.data):
                raise ValueError("expected finite normalized observations")
            return self._project(key)
        _input(frame, frame.shape)
        _input(history, history.shape)
        tiles = frame.reshape(4,4,4,4,3).mean(axis=(1,3))
        mean = tiles.mean(axis=(0,1))
        contrast = tiles - mean
        visual = np.r_[contrast.ravel() / (.10 + np.sqrt(np.mean(contrast**2))), (mean-.5)*5]
        effect = np.outer(2*history[1]-1, history[2]-history[0]).ravel()
        phase = .14*(visual @ self.omega) + 2.5*(effect @ self.binding)
        return self._project(np.r_[np.cos(phase), np.sin(phase)])

    def _project(self, key):
        key = self.complement @ key
        length = np.linalg.norm(key)
        if length < 1e-6:
            raise ValueError("transition has no representable contextual direction")
        return key / length


class _ContextWindow(_Window):
    def observe_transition(self, frame, history, signals=None):
        """Learn the preceding anchor under this completed transition; cut at episode boundaries."""
        self._check()
        cue = Transition(frame, history)
        key = self._field.key(cue)
        signals = np.zeros(8, np.float32) if signals is None else _input(signals,(8,))
        updated = self.past is not None
        if updated:
            self.dirty = True
            self._field.write_key(key, self.past)
        self.past = np.r_[np.asarray(frame,dtype=np.float32).ravel(), signals]
        return updated


class ContextMemory(PortableMemory):
    field_type = ContextField
    _window = staticmethod(_ContextWindow)

    @property
    def state_bytes(self):
        return super().state_bytes + 3*8*4
