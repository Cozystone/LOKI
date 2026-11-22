import tempfile
import unittest
from pathlib import Path
import numpy as np
from loki_memory import CorticalMemory,LokiMemory,LongTermCortex


class CortexTest(unittest.TestCase):
    def test_salience_and_immutable_queries(self):
        c = LongTermCortex(96)
        rng = np.random.default_rng(30)
        x = rng.uniform(.2,.8,96).astype(np.float32)
        y = 1-x
        self.assertFalse(c.imprint(x,1,x,x))
        self.assertTrue(c.imprint(x,1,y,x,integrity_change=.3))
        saved = {k:v.copy() for k,v in vars(c).items() if isinstance(v,np.ndarray)}
        estimate = c.recall_episode(x,1,x)
        self.assertLess(np.mean((estimate-y)**2),.02*np.mean((y-x)**2))
        for k,v in saved.items():
            np.testing.assert_array_equal(getattr(c,k),v)
        self.assertEqual(c.sleep(0),0)
        self.assertFalse(c.global_live[0])
        self.assertGreater(c.sleep(),0)

    def test_save_resume_and_no_phantom_transition(self):
        m = CorticalMemory(LokiMemory(13),LongTermCortex(13))
        x = np.linspace(.1,.9,13,dtype=np.float32)
        m.observe(x)
        m.observe(1-x,2)
        m.cortex.sleep()
        m.observe(x,1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"both.zip"
            m.save(path,binding="test")
            with self.assertRaises(ValueError):
                m.load(path,binding="other")
            copy = m.load(path,binding="test")
        np.testing.assert_array_equal(m.predict(0),copy.predict(0))
        np.testing.assert_array_equal(m.recall_episode(1),copy.recall_episode(1))
        for model in (m,copy):
            model.advance(1)
            np.testing.assert_array_equal(model.recall_episode(1),model.core.imagine([1]))
            self.assertFalse(model.observe(x,2))
            model.cut()
            self.assertFalse(model.observe(1-x,1))
        np.testing.assert_array_equal(m.cortex.weights,copy.cortex.weights)
        old = m.cortex.global_gram.copy()
        with self.assertRaises(ValueError):
            m.observe(x,1,visible=np.ones(13,int))
        np.testing.assert_array_equal(old,m.cortex.global_gram)

    def test_bounded_large_sensor(self):
        m = LongTermCortex(4100,4,modes=0)
        x = np.linspace(0.,1.,4100,dtype=np.float32)
        initial = m.state_bytes
        for i in range(8):
            m.imprint(x,i%4,1-x,x)
        m.sleep()
        self.assertEqual(m.predict(x,2,x).shape,x.shape)
        self.assertEqual(m.state_bytes,initial)

    def test_wave_off_and_episode_controls(self):
        from research.v5 import Trial
        trial = Trial(LokiMemory(12),[16])
        x = np.linspace(.1,.9,12,dtype=np.float32)
        trial.core.observe(x)
        trial.train(x,1,1-x,x,.2)
        trial.sleep()
        wave_on,wave_off = trial.fields["r16"],trial.fields["wave_off"]
        np.testing.assert_array_equal(wave_on.global_projection,wave_off.global_projection)
        result = trial.predictions(x,1,episodic=True)
        np.testing.assert_array_equal(result["fifo_r16"],result["priority_fifo"])
        self.assertLess(np.mean((result["r16"]-(1-x))**2),.005)
