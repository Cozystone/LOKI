"""Context-specific numerical parity and transactional migration regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from loki_memory import contextual
from loki_memory.contextual import ContextMemory, ContextField, Transition
from loki_memory.portable import PortableMemory


class ContextTests(unittest.TestCase):
    def test_keys(self):
        rng = np.random.default_rng(85)
        cue = Transition(rng.random((16,16,3),dtype=np.float32),
                         rng.random((3,8),dtype=np.float32))
        field = ContextField()
        key = field.key(cue)
        with patch.object(contextual, "_native", None):
            np.testing.assert_allclose(key,field.key(cue),atol=2e-6)
        cue.history[0,1] = np.nan
        with self.assertRaises(ValueError):
            field.key(cue)

    def test_rollback_and_transfer(self):
        with TemporaryDirectory() as d:
            path = Path(d)/"volume"
            m = PortableMemory.create(path,"A")
            f = np.full((16,16,3),.4,np.float32)
            h = np.full((3,8),.6,np.float32)
            m.observe(f)
            m.observe(1-f)
            v = ContextMemory.resume(path,m.identity,"A")
            c = Transition(f,h)
            initial = v.recall(c)[0]
            with self.assertRaises(ArithmeticError), v.wake() as w:
                w.observe(1-f)
                w.observe_transition(f,h)
                raise ArithmeticError()
            np.testing.assert_array_equal(initial,v.recall(c)[0])
            with v.wake() as w:
                self.assertFalse(w.observe(1-f))
                self.assertTrue(w.observe_transition(f,h))
            ref = v.recall(c)[0]
            self.assertFalse(np.array_equal(ref,initial))
            n = ContextMemory.resume(path,v.transfer("B"),"B")
            np.testing.assert_array_equal(n.recall(c)[0],ref)
            with self.assertRaises(RuntimeError):
                v.recall(c)
            with n.wake() as w:
                self.assertFalse(w.observe_transition(f,h))
            self.assertLessEqual(n.state_bytes,256*1024)
