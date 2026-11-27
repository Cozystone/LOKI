import unittest
import numpy as np
from loki_memory.walls import (
    ContextWitness,
    RouteWitness,
    DelayedEvidence,
    PopulationPrior,
    LagSignature,
)


class WallContracts(unittest.TestCase):
    def test_queries_cannot_rewrite_long_term_banks(self):
        s = RouteWitness()
        x = np.linspace(0, 1, 12)
        for i in range(12):
            s.observe(np.roll(x, i), [0.5, 0.5], [0, 0.1])
        self.assertEqual(s.consolidate(), 0)
        before = [a.copy() for a in (s.edge.count, s.edge.edges, s.ends, s.count)]
        size = len(s.state())
        for i in range(80):
            s.observe(np.roll(x, -i), [0.1, 0.3], [0.8, 0])
        for old, new in zip(before, (s.edge.count, s.edge.edges, s.ends, s.count)):
            np.testing.assert_array_equal(old, new)
        s.cut()
        self.assertEqual(s.observe(x, [0, 0], [0, 0]), (-1, 0.0))
        self.assertEqual(len(s.state()), size)

    def test_saturated_bank_refuses_write(self):
        s = ContextWitness(banks=1)
        for _ in range(8):
            s.observe(np.zeros(12), [0, 0], [0, 0])
        s.consolidate()
        old = s.edges.copy()
        for _ in range(8):
            s.observe(np.ones(12), [0, 0], [1, 1])
        self.assertEqual(s.consolidate(), -1)
        np.testing.assert_array_equal(s.edges, old)

    def test_bounded_statistics(self):
        d = DelayedEvidence()
        p = PopulationPrior()
        l = LagSignature(3)
        lengths = [len(x.state()) for x in (d, p, l)]
        for _ in range(80):
            d.observe_levels(1, 0.5, 0.5, 0.51)
            p.add_body(2, 3)
            l.observe([0.2, 0.3, 0.5])
        self.assertEqual(lengths, [len(x.state()) for x in (d, p, l)])
        d.cut()
        self.assertFalse(d.trace.any())
        with self.assertRaises(ValueError):
            d.observe(1, 1, 0.1)
