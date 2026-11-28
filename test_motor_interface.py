import unittest
from types import SimpleNamespace

import torch

from research.motor_interface import MotorProposal, rebirth_with_timer_cut


class MotorInterfaceTests(unittest.TestCase):
    def test_cold_start_and_tied_evidence_preserve_choice(self):
        codes = torch.cartesian_prod(torch.arange(2), torch.arange(2))
        probe = MotorProposal(codes, 2)
        action = torch.tensor([2, 3])
        deficits = torch.ones(2, 2)
        due = torch.ones(2, dtype=torch.bool)
        self.assertTrue(torch.equal(probe.propose(action, deficits, due, None), action))
        probe.visited[:] = True
        self.assertTrue(torch.equal(probe.propose(action, deficits, due, None), action))

    def test_delayed_device_effects_stop_at_rebirth(self):
        body = SimpleNamespace(
            dead=torch.tensor([True, False]),
            busy=torch.tensor([11, 7]),
            rise_in=torch.tensor([3, 2]),
            fall_in=torch.tensor([9, 8]),
        )
        rebirth_with_timer_cut(lambda b: b.dead.zero_(), body)
        self.assertEqual(body.busy.tolist(), [0, 7])
        self.assertEqual(body.rise_in.tolist(), [-1, 2])
        self.assertEqual(body.fall_in.tolist(), [-1, 8])

    def test_executor_constraints_and_cut(self):
        codes = torch.cartesian_prod(torch.arange(2), torch.arange(2))
        p = MotorProposal(codes, 2)
        p.visited[:] = True
        p.count.fill_(10)
        p.total[1, 3] = -5
        a = torch.tensor([0, 0])
        d = torch.ones(2, 2)
        self.assertEqual(
            p.propose(a, d, torch.tensor([True, False]), None).tolist(), [1, 0]
        )
        mask = torch.tensor([True, False, False, False])
        self.assertEqual(
            p.propose(a, d, torch.tensor([True, True]), mask).tolist(), [0, 0]
        )
        p.visited[1::2] = False
        self.assertEqual(
            p.propose(a, d, torch.tensor([True, True]), None).tolist(), [0, 0]
        )
        p.record(torch.tensor([1, 2]), torch.ones(2, 2), torch.tensor([True, False]))
        self.assertFalse(p.history[0].any())
        self.assertFalse(p.previous[0].any())
        self.assertTrue(p.history[1].any())


if __name__ == "__main__":
    unittest.main()
