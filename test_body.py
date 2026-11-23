import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from loki_memory import BodyMemory, Transition


class BodyBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"field.db"
        self.m = BodyMemory.create(self.path,"A")
        self.f = np.full((16,16,3),.3,np.float32)
        self.s = np.linspace(0,1,8,dtype=np.float32)

    def begin(self,m):
        return m.begin_action(self.f,self.s,1-self.s)

    def test_order_cut_duplicate(self):
        m=self.m
        late=self.begin(m)
        with self.assertRaises(RuntimeError):
            self.begin(m)
        m.cut()
        fresh=self.begin(m)
        self.assertFalse(m.finish_action(late,self.f,self.s))
        with self.assertRaises(ValueError):
            m.finish_action(fresh,self.f,self.s[:4])
        self.assertTrue(m.finish_action(fresh,self.f,self.s))
        self.assertFalse(m.finish_action(fresh,self.f,self.s))
        self.assertLessEqual(m.state_bytes,262144)

    def test_wake_rollback_and_input_ownership(self):
        m=self.m
        original=self.f.copy()
        ticket=self.begin(m)
        self.f.fill(.8)
        cue=Transition(self.f,np.stack((self.s,1-self.s,self.s)))
        before=m.recall(cue)
        with self.assertRaises(ArithmeticError),m.wake() as w:
            self.assertTrue(w.finish_action(ticket,self.f,self.s))
            raise ArithmeticError
        np.testing.assert_array_equal(before[0],m.recall(cue)[0])
        self.assertTrue(m.finish_action(ticket,self.f,self.s))
        self.assertLess(np.mean((m.recall(cue)[0]-original)**2),.01)
        with self.assertRaises(TypeError):
            m.observe(self.f)
        with m.wake() as w:
            self.assertFalse(hasattr(w,"observe_transition"))

    def test_restart_and_process_exit(self):
        m=self.m
        self.assertTrue(m.finish_action(self.begin(m),1-self.f,self.s))
        cue=Transition(1-self.f,np.stack((self.s,1-self.s,self.s)))
        old=m.recall(cue)
        stale=self.begin(m)
        code="""
import os,sys,numpy as np
from loki_memory import BodyMemory
m=BodyMemory.resume(sys.argv[1],sys.argv[2],'A')
f=np.ones((16,16,3),np.float32)
s=np.ones(8,np.float32)
with m.wake() as w:
    t=w.begin_action(f,s,s)
    assert w.finish_action(t,f,s)
    os._exit(27)
"""
        run=subprocess.run([sys.executable,"-B","-c",code,str(self.path),m.identity])
        self.assertEqual(run.returncode,27)
        restored=BodyMemory.resume(self.path,m.identity,"A")
        for a,b in zip(old,restored.recall(cue)):
            np.testing.assert_array_equal(a,b)
        self.assertFalse(restored.finish_action(stale,self.f,self.s))
        with self.assertRaises(RuntimeError):
            m.finish_action(stale,self.f,self.s)
