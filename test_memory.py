"""Regression tests for address reuse, imagined transitions, and sparse cues."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from loki_memory import LokiMemory
from loki_memory.atanor import AtanorImageMemory


class MemoryTest(unittest.TestCase):
    def test_body_wiring_and_black_frame(self):
        contract = SimpleNamespace(motors=[3, 5], sensors=[SimpleNamespace(kind="image", shape=(4, 6, 3))])
        observer = AtanorImageMemory(contract)
        black = [np.zeros((1, 4, 6, 3), np.uint8)]
        observer.step(black)
        self.assertEqual(observer.memory.used, 0)
        observer.step(black, available=True)
        self.assertEqual(observer.memory.used, 1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "body.npz"
            observer.save(path, "body-1")
            with self.assertRaises(ValueError):
                observer.restore(path, "body-2")
            observer.restore(path, "body-1")
        np.testing.assert_array_equal(observer.step(black, 14), black[0][0])

    def test_blackout_cannot_teach_an_edge(self):
        mem = LokiMemory(100)
        first = np.full(100, 0.15, np.float32)
        mem.observe(first)
        mem.advance(10)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "paused.npz"
            mem.save(path)
            mem = LokiMemory.load(path)
        mem.observe(1 - first, 33)
        self.assertFalse((mem.actions == 33).any())
        mem.observe(first, 33)
        self.assertTrue((mem.actions == 33).any())
        mem.cut()
        mem.observe(first, 39)
        self.assertFalse((mem.actions == 39).any())

    def test_hidden_values_and_padding(self):
        rng = np.random.default_rng(20)
        mem = LokiMemory(97, tile=16)
        frame = rng.random(97, dtype=np.float32)
        mem.observe(frame)
        visible = np.array([1, 0, 0, 1, 0, 0, 1], bool)
        a = mem.recall(frame, visible)
        frame[~np.repeat(visible, 16)[:97]] = np.nan
        np.testing.assert_array_equal(a, mem.recall(frame, visible))

    def test_reused_address_removes_stale_edges_and_refs(self):
        mem = LokiMemory(128, budget=650, tile=8, atoms=4)
        self.assertEqual(mem.capacity, 2)
        rng = np.random.default_rng(33)
        frames = rng.random((5, 128), dtype=np.float32)
        mem.observe(frames[0])
        old = mem.imagine(())
        mem.observe(frames[1], 555)
        np.testing.assert_array_equal(mem.imagine((), start=0), old)
        for frame in frames[2:]:
            mem.observe(frame, 42)
            expected = np.bincount(mem.index[:mem.used].ravel(), minlength=len(mem.refs))
            np.testing.assert_array_equal(expected, mem.refs)
        self.assertFalse((mem.actions == 555).any())
        self.assertLessEqual(mem.state_bytes, 650)


if __name__ == "__main__":
    unittest.main()
