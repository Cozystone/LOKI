import tempfile
import unittest
from pathlib import Path
import numpy as np
from loki_memory.journey_121 import LokiV121


class MissEvidenceTests(unittest.TestCase):
    def test_misses_require_visible_matched_opportunities(self):
        m = LokiV121("fixture", [1, 1], [2], place_dim=1)
        yes, no = np.eye(2, dtype="f4")
        for _ in range(5):
            m.observe([0.2, 0.5], no, place=[0])
            m.observe([0.4, 0.5], yes, place=[0])
        initial = m.misses.copy()
        m.observe([0.18, 0.5], no, place=[0])
        np.testing.assert_equal(initial, m.misses)
        m.observe([0.18, 0.5], yes, [False, True], place=[0])
        np.testing.assert_equal(initial, m.misses)
        m.observe([1.0, 0.5], no, place=[0])
        m.observe([1.0, 0.5], yes, place=[0])
        np.testing.assert_equal(initial, m.misses)
        m.observe([0.18, 0.5], no, place=[0])
        m.observe([0.18, 0.5], yes, place=[0])
        self.assertGreater(m.misses.sum(), 0)
        self.assertGreater(m.resource_at(0)["unsupported_mass"], 0)
        m.cut()
        self.assertEqual(m.misses.sum(), 0)

    def test_fixed_checkpoint_and_readonly_forecast(self):
        m = LokiV121("real", [3, 41, 96], [5, 6, 3], place_dim=30)
        u = np.concatenate([np.eye(k)[0] for k in m.motors])
        m.observe(np.full(140, 0.5), place=np.zeros(30))
        m.observe(np.full(140, 0.51), u, place=np.zeros(30))
        before = m.matrix.copy()
        m.forecast(u)
        m.resource_at(0)
        m.confidence()
        np.testing.assert_equal(m.matrix, before)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "checkpoint"
            m.save(p, False)
            restored = LokiV121.load(p, m.config)
            np.testing.assert_equal(m.matrix, restored.matrix)
            self.assertLessEqual(p.stat().st_size, 262144)
            m.save(p)
            self.assertEqual(LokiV121.load(p, m.config).misses.sum(), 0)
