import tempfile
import unittest
from pathlib import Path
import numpy as np
from loki_memory.lifelong import LifelongMemory


class BoundedMemoryTests(unittest.TestCase):
    def test_snapshots_resume_across_forced_prototype_merges(self):
        rng=np.random.default_rng(10001)
        m=LifelongMemory()
        x=rng.random((240,12));a=rng.integers(4,size=240);y=rng.random((240,8))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"memory"
            m.save(path);size=path.stat().st_size
            m.learn(x[:120],a[:120],y[:120]);m.save(path)
            resumed=m.load(path)
            for obj in (m,resumed):
                obj.learn(x[120:],a[120:],y[120:])
            for left,right in zip(m.arrays,resumed.arrays):
                np.testing.assert_array_equal(left,right)
            resumed.save(path)
            self.assertEqual(size,path.stat().st_size)
            self.assertLess(size*2,8*1024**2)
            self.assertEqual(resumed.mass.sum(),240)
            self.assertEqual(resumed.count.sum(),240)
            before=path.read_bytes()
            path.write_bytes(before[:-4])
            with self.assertRaises(ValueError):m.load(path)

    def test_inferred_path_uses_observed_directed_links(self):
        m=LifelongMemory();p,q,r=np.eye(24,dtype='f4')[:3]
        m.transition(p,2,q);m.transition(q,1,r)
        self.assertEqual(m.plan(p,r),[2,1]);self.assertEqual(m.plan(r,p),[])
