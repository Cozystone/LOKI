import unittest
import numpy as np
from research.observed_outcomes import ObservedOutcomes


class ObserverContract(unittest.TestCase):
    def test_provenance_and_executor(self):
        o = ObservedOutcomes(2, 2, 1, 1)
        for a in (0, 1):
            for sign in (-1.0, 1.0):
                o.calibrate(a, np.eye(2)[a], np.array([sign]), np.array([sign]))
        self.assertTrue(o.observe(np.array([0.0, 1.0]), np.array([-0.2])))
        self.assertFalse(o.observe(np.array([0.5, 0.5]), np.array([-0.2])))
        self.assertEqual(o.report()["experienced"].sum(), 0)
        self.assertEqual(o.propose(0, np.array([0.5]), False), 0)
        self.assertEqual(o.propose(0, np.array([0.5]), True, [True, False]), 0)
        self.assertEqual(o.propose(0, np.array([0.5]), True), 1)
        seen = o.seen.copy()
        o.confirm(1, np.array([0.2]))
        np.testing.assert_array_equal(o.seen, seen)
        self.assertEqual(o.propose(0, np.array([0.5]), True), 0)
