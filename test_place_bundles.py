from types import SimpleNamespace
import unittest
import numpy as np
import torch
from loki_memory.place_bundles import PlaceBundles
from research.r2_places import encode, inputs, replay


class TestPlaceBundles(unittest.TestCase):
    def test_adult_import_and_state_roundtrip(self):
        source = PlaceBundles()
        data = inputs(29000, 0, True)
        replay(source, data, 0)
        b = source.sleep()
        snapshot = source.going_snapshot(b)
        self.assertEqual(snapshot["used"], 64)
        restarted = PlaceBundles()
        restarted.import_going(SimpleNamespace(**snapshot, mended=True))
        torch.testing.assert_close(source.going(b)[1], restarted.going(0)[1])
        with self.assertRaises(ValueError):
            restarted.import_going(SimpleNamespace(**snapshot))
        size = restarted.state_bytes
        replay(restarted, inputs(29001, 1, True), 1)
        self.assertGreater(int(restarted.generation.sum()), 0)
        self.assertEqual(restarted.state_bytes, size)
        restored = PlaceBundles()
        restored.load_state_dict(restarted.state_dict())
        self.assertEqual(restarted.identify(), restored.identify())
        self.assertEqual(int(restored.sleep()), int(restarted.sleep()))
        for k, v in restored.state_dict().items():
            torch.testing.assert_close(v, restarted.state_dict()[k])

    def test_recycle_clears_both_edge_ends_and_relief(self):
        s = PlaceBundles(slots=2)
        for i in (0, 1):
            s._recruit(torch.from_numpy(encode([i, 0])))
        s.T.fill_(1)
        s.edges.fill_(True)
        s.rates.fill_(0.4)
        s.prev[0] = 0
        j = s._recruit(torch.from_numpy(encode([3, 0])))
        self.assertFalse(s.T[j].any() or s.T[:, j].any())
        self.assertFalse(
            s.edges[:, j].any() or s.edges[:, :, j].any() or s.rates[:, j].any()
        )
        self.assertEqual(s.generation[j], 1)
        self.assertNotEqual(s.prev[0], j)

    def test_native_path_scale_and_death(self):
        xy = np.array([[-6.0, 10.0], [8.0, -11.0]])
        code = torch.tensor(np.stack(list(map(encode, xy))))
        torch.testing.assert_close(
            PlaceBundles.odometry(code), torch.tensor(xy).float()
        )
        s = PlaceBundles(bodies=2)
        s.observe(code, torch.ones(2, 2), torch.zeros(2), [True, False])
        self.assertEqual(s.prev[1], -1)
        self.assertFalse(s.walk.any())
        self.assertFalse(s.going_snapshot(-1)["R"].any())
