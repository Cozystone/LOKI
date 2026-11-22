import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from loki_memory import LokiMemory
from loki_memory.atanor import AtanorMemory


class FieldTest(unittest.TestCase):
    def test_single_matrix_and_budget(self):
        for width in (1,64,96,4096,65536,122880):
            m = LokiMemory(width)
            self.assertEqual([k for k,v in vars(m).items() if isinstance(v,np.ndarray)],["matrix"])
            self.assertLessEqual(m.state_bytes,262144)
            x = np.full(width,.6,np.float32)
            shape = m.matrix.shape
            for a in range(3):
                m.observe(x,a)
            self.assertEqual(m.matrix.shape,shape)
            self.assertEqual(m.imagine([5,1]).shape,(width,))

    def test_mask_and_nondestructive_imagination(self):
        rng = np.random.default_rng(900)
        for n in (97, 4201):
            m = LokiMemory(n)
            x = rng.random(n,dtype=np.float32)
            mask = np.arange(n)%4==1
            m.observe(x)
            state = m.matrix.copy()
            a = m.recall(x,mask,[2,3])
            b = m.recall(np.where(mask,x,np.nan),mask,[2,3])
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(state,m.matrix)
            m.observe(np.where(mask,x,np.nan),2,visible=mask)
            self.assertTrue(np.isfinite(m.matrix).all())
            current = m.matrix.copy()
            with self.assertRaises(ValueError):
                m.observe(np.full(n,np.nan),4)
            np.testing.assert_array_equal(current,m.matrix)

    def test_cut_blackout_and_resume(self):
        rng = np.random.default_rng(6)
        m = LokiMemory(32)
        for t in range(20):
            m.observe(rng.random(32,dtype=np.float32),t%3)
        law = m._coupling.copy()
        prediction = m.imagine([[.1,-.2]])
        np.testing.assert_array_equal(prediction,m.advance([.1,-.2]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"field.npz"
            m.save(path,binding="w")
            with self.assertRaises(ValueError):
                LokiMemory.load(path,binding="q")
            loaded = LokiMemory.load(path,binding="w")
            for t in range(16):
                x = rng.random(32,dtype=np.float32)
                m.observe(x,t%3)
                loaded.observe(x,t%3)
                np.testing.assert_array_equal(m.matrix,loaded.matrix)
                if t==0:
                    np.testing.assert_array_equal(m._coupling,law)
        m.cut()
        law = m._coupling.copy()
        m.observe(x,2)
        np.testing.assert_array_equal(m._coupling,law)

    def test_contract_partial_sensor_blackout(self):
        c = SimpleNamespace(motors=[3,5],sensors=[SimpleNamespace(kind="image",shape=(7,11,3)),
                                                 SimpleNamespace(kind="audio",shape=(4,))])
        adapter = AtanorMemory(c)
        rgb = np.ones((1,7,11,3),np.float32)*128
        audio = np.zeros((1,4),np.float32)
        adapter.step([rgb,audio],cut=True)
        partial = adapter.step([None,audio],11)
        self.assertEqual(partial.shape,(10,))
        self.assertTrue(np.isfinite(adapter.step(None,3)).all())
        with self.assertRaises(ValueError):
            adapter.step([rgb,audio],15)


if __name__=="__main__":
    unittest.main()
