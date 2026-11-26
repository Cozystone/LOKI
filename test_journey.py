import tempfile
import unittest
from pathlib import Path
import numpy as np
from loki_memory.journey import LokiV12
from research.v12_capacity import stream


class JourneyContractTests(unittest.TestCase):
    def test_batch_matches_one_life(self):
        reference = LokiV12("capacity", [1, 1], [2], place_dim=1)
        native = LokiV12(**reference.config)
        stream(native._arrays(), 0, 96)
        for tick in range(96):
            p = (tick % 8) / 7
            reference.observe(
                [(tick % 32) / 31, p],
                None if tick == 0 else [tick % 2, 1 - tick % 2],
                place=[p],
            )
        np.testing.assert_allclose(reference.matrix, native.matrix, atol=2e-5)

    def test_readonly_binding_and_private_cut(self):
        m = LokiV12("body", [1, 2], [2, 3], place_dim=1)
        u = [1, 0, 0, 1, 0]
        for t in range(50):
            m.observe([t % 2, 0.3, 0.8], u, place=[t % 2])
        before = m.matrix.copy()
        self.assertEqual(m.foresee([u] * 32).source, "recalled")
        np.testing.assert_equal(m.imagine([u]), m.forecast(u))
        for route in m.journeys([0.8]):
            self.assertEqual(route.source, "recalled")
            self.assertEqual(m.goal_example(route.journey_id)["source"], "recalled")
        np.testing.assert_equal(m.matrix, before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            m.save(path, False)
            restored = LokiV12.load(path, m.config)
            np.testing.assert_equal(m.matrix, restored.matrix)
            m.save(path)
            restored = LokiV12.load(path, m.config)
            self.assertTrue((restored.x == 0).all())
            saved = restored.count.copy()
            restored.observe([0.4, 0.7, 0.7], u, place=[0])
            np.testing.assert_equal(saved, restored.count)
            np.testing.assert_equal(
                LokiV12.load(path, dict(m.config, contract_signature="else")).count, 0
            )
            self.assertLessEqual(path.stat().st_size, 262144)
        np.testing.assert_equal(m.matrix, before)

    def test_missing_channels_and_anonymous_wide_body(self):
        m = LokiV12("wide", [2, 1, 510], [8, 9, 12])
        u = np.concatenate([np.eye(k)[-1] for k in m.motors])
        raw = np.full(m.width, 0.4)
        mask = np.arange(m.width) % 5 != 0
        raw[~mask] = np.nan
        m.observe(raw, u, mask)
        self.assertTrue(np.isfinite(m.forecast(u)).all())
        self.assertEqual(m.forecast(u).shape, (513,))
        with self.assertRaises(ValueError):
            m.observe(raw, u, np.ones(m.width, bool))
        weights = m.w.copy()
        m.advance(u)
        m.observe(raw, u, mask)
        np.testing.assert_array_equal(weights, m.w)
        m.cut()
        self.assertEqual(m.belief().confidence, 0)
