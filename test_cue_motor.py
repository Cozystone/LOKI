import unittest
import torch
from research.cue_motor import CueMotorProposal


class CueMotorTests(unittest.TestCase):
    def test_cut_cancels_one_pending_outcome_and_preserves_other_body(self):
        org = CueMotorProposal(2, 3)
        action = torch.tensor([0, 0])
        d = torch.ones(2, 2)
        cue = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        for tick in range(21):
            cut = torch.tensor([tick == 10, False])
            org.record(
                action,
                d,
                d if tick == 0 else d * 0.75,
                cut,
                cue if tick == 0 else cue * 0,
            )
        self.assertEqual(float(org.count[0].sum()), 0.0)
        self.assertGreater(float(org.count[1].sum()), 0.0)
        self.assertAlmostEqual(float(org.total[1].sum()), -0.5, places=5)

    def test_no_evidence_and_mask_preserve_original(self):
        org = CueMotorProposal(1, 3)
        org.count.fill_(4)
        org.mass2.fill_(4)
        org.total[:, :, 1] = -2
        d = torch.ones(1, 2)
        p = torch.tensor([0])
        due = torch.tensor([True])
        cue = torch.tensor([[1.0, 0.0]])
        self.assertEqual(int(org.propose(p, d, due, None, cue)), 1)
        for flag, allowed, k in [
            (due, None, cue * 0),
            (~due, None, cue),
            (due, torch.tensor([True, False, False]), cue),
        ]:
            self.assertEqual(int(org.propose(p, d, flag, allowed, k)), 0)
