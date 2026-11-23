import tempfile
import unittest
from pathlib import Path
import numpy as np
from loki_memory import GraphMemory, BodyMemory


class PersistentGraph(unittest.TestCase):
    def test_migration_rollback_then_reuse_ticket(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"life"
            m=BodyMemory.create(path,"A")
            m=GraphMemory.resume(path,m.identity,"A")
            f=np.ones((16,16,3),np.float32)
            s=np.full(8,.5,np.float32)
            t=m.begin_action(f,s,s)
            with self.assertRaises(ArithmeticError),m.wake() as w:
                self.assertTrue(w.finish_action(t,f,s))
                raise ArithmeticError
            self.assertIsNone(m.episode(1))
            self.assertTrue(m.finish_action(t,f,s))
            self.assertEqual(m.episode(1)["host"],"A")
            with self.assertRaises(ValueError):
                m.event(2)
            self.assertLessEqual(m.state_bytes,262144)
    def test_swapped_responses_require_independent_witness(self):
        with tempfile.TemporaryDirectory() as d:
            m=GraphMemory.create(Path(d)/"m","A")
            f=np.full((16,16,3),.5,np.float32)
            s=np.full(8,.5,np.float32)
            for y in (s,s+.2,s,s-.2):
                x=s if m.episode(1) is None else m.episode(int(m._field.clock[0]))["after"][12:]
                self.assertTrue(m.finish_action(m.begin_action(f,x,s),f,y))
            for prev,other in ((1,4),(3,2)):
                right=m.episode(prev+1)["after"]
                wrong=m.episode(other)["after"]
                self.assertEqual(m.witnessed(f,"A",prev,s,right)["id"],prev+1)
                self.assertIsNone(m.witnessed(f,"A",prev,s,wrong))
