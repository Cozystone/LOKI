"""Live body adapter for v8. Only complete actions in the active session are learned."""
from dataclasses import dataclass

import numpy as np

from .contextual import ContextMemory, Transition
from .portable import _Window, _input


@dataclass(frozen=True)
class _Pending:
    ticket: object
    anchor: np.ndarray
    action: np.ndarray


class _BodyWindow(_Window):
    def begin_action(self, frame, before, action):
        """Capture the command actually executed and its pre-action observation."""
        self._check()
        if self.past is not None:
            raise RuntimeError("finish the pending action or cut the stream")
        image = _input(frame, (16, 16, 3))
        signals = _input(before, (8,))
        command = _input(action, (8,)).copy()
        ticket = object()
        self.past = _Pending(ticket, np.concatenate((image.ravel(), signals)), command)
        return ticket

    def finish_action(self, ticket, frame, after):
        """Ignore late, duplicate or foreign callbacks without consuming a current action."""
        self._check()
        pending = self.past
        if pending is None or pending.ticket is not ticket:
            return False
        signals = _input(after, (8,))
        history = np.stack((pending.anchor[768:], pending.action, signals))
        key = self._field.key(Transition(frame, history))
        self.dirty = True
        self._field.write_key(key, pending.anchor)
        self.past = None
        return True

    def observe(self, frame, signals=None):
        self._check()
        raise TypeError("live body writes require begin_action and finish_action")


class BodyMemory(ContextMemory):
    """Same checkpoint/recall as v8; a process-local ticket binds each physical action.

    Transfer, cut and resume discard outstanding tickets. A wake transaction publishes
    its pending state only on commit, exactly like the v7 visual anchor.
    """
    _window = staticmethod(_BodyWindow)

    @property
    def state_bytes(self):
        # Learned field plus one anchor (image and before channels), and one command.
        return self._field.data.nbytes + (776 + 8)*4

    def begin_action(self, frame, before, action):
        with self.wake() as window:
            return window.begin_action(frame, before, action)

    def finish_action(self, ticket, frame, after):
        with self.wake() as window:
            return window.finish_action(ticket, frame, after)
