import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from loki_memory.trace_store import TraceStore
from loki_memory.trace_atlas import TraceAtlas
from loki_memory.trace_volume import TraceVolume


def put(store, i, t, body=0, live=True):
    key = np.tile((i % 4) / 3, (1, 32))
    return int(
        store.write(
            key, [[i, 0]], np.zeros((1, 16)), [[1]], [0], t, [live], body=[body]
        )[0]
    )


class TraceTests(unittest.TestCase):
    def test_pressure_bad_handles_checksum(self):
        store = TraceStore(1, 2, bodies=1)
        a, b = put(store, 0, 0), put(store, 1, 1)
        self.assertEqual(put(store, 2, 2), -1)
        self.assertEqual(int(store.meta[1]), 1)
        store.link([10**10], [b], [True])
        self.assertTrue((store.nbr == -1).all())
        state = store.state()
        self.assertEqual(TraceStore.load(state).state(), state)
        with self.assertRaises(ValueError):
            TraceStore.load(state[:-1] + bytes([state[-1] ^ 1]))

    def test_chronology(self):
        s = TraceAtlas(1, 16, bodies=2)
        path = [put(s, i, t) for t, i in enumerate([0, 1, 1, 2, 3, 1, 4])]
        self.assertEqual(s.recall_episode(0, 2), [(path[0], 0, 0), (path[1], 1, 2)])
        self.assertEqual([i for i, *_ in s.recall_episode(0, 5, 2)], [path[4], path[5]])
        key = np.ones((1, 32))
        ctx = np.zeros((1, 16))
        ids, _, conf, _ = s.cue(key, [[3, 0]], ctx, body=[1], landmark=True)
        self.assertTrue(s.anchor(ids, conf, [[3, 0]], body=[1]).item())
        put(s, 7, 6, 1)
        self.assertEqual(len(s.recall_episode(1, 6)), 1)
        self.assertEqual(len(s.recall_episode(0, 6)), 6)
        put(s, 7, 7, live=False)
        b = put(s, 3, 8)
        self.assertEqual(s.recall_episode(0, 8), [(b, 8, 8)])
        s.cut()
        ids, _, conf, _ = s.cue(np.full((1, 32), 2 / 3), [[2, 0]], ctx, landmark=True)
        self.assertTrue(s.anchor(ids, conf, [[2, 0]]).item())
        new = put(s, 8, 9)
        self.assertEqual(s.recall_episode(0, 9), [(new, 9, 9)])
        for t in range(10, 40):
            put(s, 8 + t % 2, t)
        self.assertEqual(s.recall_episode(0, 1), [])
        self.assertEqual(
            TraceAtlas.load(s.state()).recall_episode(0, 39), s.recall_episode(0, 39)
        )

    def test_transfer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mind.sqlite"
            lap = TraceVolume.create(path, "laptop", TraceAtlas(1, 64, bodies=1))
            a = put(lap.store, 0, 0)
            b = put(lap.store, 1, 1)
            lap.store.link([a], [b], [True])
            identity = lap.transfer("glasses")
            with self.assertRaises(RuntimeError):
                lap.checkpoint()
            eye = TraceVolume.resume(path, identity, "glasses")
            q = np.ones((1, 32)) / 3
            ctx = np.zeros((1, 16))
            ids, _, conf, _ = eye.store.cue(q, [[0, 0]], ctx, landmark=True)
            self.assertTrue(eye.store.anchor(ids, conf, [[0, 0]]).item())
            c = put(eye.store, 3, 2)
            eye.store.link([b], [c], [True])
            self.assertFalse((eye.store.nbr[b] == c).any())
            eye.transfer("laptop")
            restored = TraceVolume.resume(path, identity, "laptop")
            self.assertEqual(restored.store.recall_episode(0, 2), [(c, 2, 2)])
            self.assertEqual(
                restored.store.recall_episode(0, 1), [(a, 0, 0), (b, 1, 1)]
            )

    def test_atlas_window_merge_and_stale_event(self):
        s = TraceAtlas(1, 8192)
        key = np.zeros((1, 32))
        ctx = np.zeros((1, 16))
        a, b = [
            s.write(key, [[p, 0]], ctx, [[1]], [0], i, [True]).item()
            for i, p in enumerate((0, 0.70))
        ]
        self.assertNotEqual(a, b)
        report = s.consolidate()
        self.assertEqual(report["merged"], 1)
        self.assertFalse(s.read([b])["valid"].item())
        self.assertEqual(s.recall_episode(0, 1), [])
        self.assertEqual(s.recall_episode(0, 0), [(a, 0, 0)])
        self.assertEqual(s.cue(key, [[0, 0]], ctx)[0][0, 0].item(), a)

    def test_indistinguishable_cues_and_multi_chart(self):
        s = TraceAtlas(1, 128)
        rng = np.random.default_rng(36)
        key = rng.random((96, 32), dtype="f4")
        ctx = rng.random((96, 16), dtype="f4")
        pos = rng.random((96, 2), dtype="f4") * 20
        for i in range(96):
            if i == 48:
                s.cut()
            s.write(
                key[i : i + 1], pos[i : i + 1], ctx[i : i + 1], [[1]], [0], i, [True]
            )
        for i in range(0, 96, 3):
            args = s._inputs(key[i : i + 1], pos[i : i + 1], ctx[i : i + 1], None)
            indexed = s._cpu_query(*args, False, 8)
            dense = TraceStore._cpu_query(s, *args, False, 8)
            torch.testing.assert_close(indexed[0], dense[0])
        # Two externally labelled worlds with identical evidence yield identical reports.
        q = s.cue(key[:1], pos[:1], ctx[:1])
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(q, s.cue(key[:1], pos[:1], ctx[:1])))
        )

    def test_interrupted_save(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.bin"
            s = TraceStore(1, 4)
            put(s, 0, 0)
            s.save(path)
            code = "import sys,os;from loki_memory.trace_store import TraceStore;s=TraceStore(1,4);os.replace=lambda *_: os._exit(73);s.save(sys.argv[1])"
            proc = subprocess.run([sys.executable, "-c", code, str(path)])
            self.assertEqual(proc.returncode, 73)
            self.assertEqual(TraceStore.load(path.read_bytes()).sha(), s.sha())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_reference(self):
        host = TraceStore(1, 16, bodies=4)
        gpu = TraceStore.load(host.state(), device="cuda")
        for s in (gpu, host):
            put(s, 0, 0)
            k = torch.zeros(4, 32)
            p = torch.zeros(4, 2)
            ctx = torch.zeros(4, 16)
            ids, _, conf, _ = s.cue(k, p, ctx, landmark=True)
            self.assertTrue(s.anchor(ids, conf, p).all())
            ids = s.write(
                k, p + 2, ctx, torch.ones(4, 1), torch.zeros(4), 1, [1, 1, 0, 1]
            )
            self.assertEqual(ids.tolist(), [1, 1, -1, 1])
            s.link([0] * 4, ids, [1] * 4)
        self.assertEqual(gpu.state(), host.state())
