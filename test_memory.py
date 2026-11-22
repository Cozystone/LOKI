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

    def test_paired_effect_swap_budget_and_resume(self):
        m = LokiMemory(96,grid_shape=(4,8,3))
        rng = np.random.default_rng(45)
        for i in range(12):
            m.observe(rng.random(96,dtype=np.float32),i%5)
        a,b = [0,1,0,2],[4,1,4,2]
        x,y = rng.random((2,96),dtype=np.float32)
        np.testing.assert_array_equal(m.predict_effect(a,b),m.imagine(a)-m.imagine(b))
        m.learn_effect(a,b,x,y)
        state = m.matrix.copy()
        predicted = m.predict_effect(a,b)
        self.assertGreater(np.linalg.norm(predicted),0.)
        np.testing.assert_allclose(predicted,-m.predict_effect(b,a),atol=1e-7)
        np.testing.assert_array_equal(m.predict_effect(a,a),np.zeros(96))
        np.testing.assert_array_equal(m.matrix,state)
        self.assertLessEqual(m.state_bytes,256*1024)
        self.assertEqual([k for k,v in vars(m).items() if isinstance(v,np.ndarray)],["matrix"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"v4.npz"
            m.save(path)
            restored = LokiMemory.load(path)
            np.testing.assert_array_equal(predicted,restored.predict_effect(a,b))
            # Version 3 ended in row padding where the v4 readout now begins.
            old = state[:m._v3_rows].copy()
            old.ravel()[m._layout["effect_weights"][0]:] = 0
            np.savez(path,version=3,config=[96,262144,1],binding="",grid=m.grid_shape,field=old)
            restored = LokiMemory.load(path)
            np.testing.assert_array_equal(restored.imagine(a),m.imagine(a))
            self.assertEqual(float(restored._part("effect_count")[0]),0.)

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

    def test_chart_rollout_resume_and_missing_observations(self):
        rng = np.random.default_rng(42)
        grid = (4,8,3)
        model = LokiMemory(100,grid_shape=grid)
        for i in range(20):
            model.observe(rng.random(100,dtype=np.float32),i%4)
        state = model.matrix.copy()
        plan = [0,3,1,1,0,2]*4
        expected = model.imagine(plan)
        np.testing.assert_array_equal(model.matrix,state)
        for a in plan[:5]:
            model.advance(a)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"grid.npz"
            model.save(path)
            restored = LokiMemory.load(path)
            for a in plan[5:]:
                result = restored.advance(a)
        np.testing.assert_array_equal(result,expected)
        old_chart = model._part("chart").copy()
        old_retina = model._part("retina").copy()
        expected_affine = model._affine_step(model._part("affine"),1)
        cue = np.full(100,np.nan)
        cue[0] = .7
        model.observe(cue,1,visible=np.isfinite(cue))
        np.testing.assert_array_equal(old_chart,model._part("chart"))
        np.testing.assert_array_equal(old_retina,model._part("retina"))
        np.testing.assert_array_equal(expected_affine,model._part("affine"))

    def test_delta_is_independent_of_live_state(self):
        rng = np.random.default_rng(81)
        m = LokiMemory(360)
        base = rng.uniform(-.35,.35,m.width).astype(np.float32)
        for t in range(55):
            m.observe(.5+np.sin(t*.3)*base,t%3)
        mask = np.arange(m.width)%3==0
        cue = np.where(mask,base,np.nan)
        predicted = m.predict_change(cue,mask)
        for a in (0,1,1,2):
            m.advance(a)
        m.cut()
        np.testing.assert_array_equal(predicted,m.predict_change(cue,mask))
        self.assertLess(np.mean((predicted[~mask]-base[~mask])**2),np.mean(base[~mask]**2))

    def test_all_visible_and_implicit_masks_match(self):
        rng = np.random.default_rng(8)
        models = [LokiMemory(96,grid_shape=(4,8,3)) for _ in range(2)]
        for t in range(7):
            x = rng.random(96,dtype=np.float32)
            models[0].observe(x,t)
            models[1].observe(x,t,visible=np.ones(96,bool))
        np.testing.assert_array_equal(models[0].matrix,models[1].matrix)


if __name__=="__main__":
    unittest.main()
