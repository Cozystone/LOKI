"""Matched synthetic cue/action contract check. Run as python -m research.cue_motor_probe."""

import json
import argparse
import numpy as np
import torch
from research.cue_motor import CueMotorProposal


def run(replication):
    torch.set_num_threads(1)
    rng = [
        np.random.default_rng(s)
        for s in (range(5134, 5158) if replication else range(5110, 5134))
    ]
    flip = torch.tensor([int(r.integers(2)) for r in rng])
    rows = torch.arange(len(rng))
    due = torch.ones_like(rows, dtype=torch.bool)
    streams = {
        name: CueMotorProposal(len(rows), 3, decay=decay, effective_support=replication)
        for name, decay in [
            ("true", "tick"),
            ("shuffled", "tick"),
            ("lever_tick", "tick"),
            ("lever_experience", "experience"),
        ]
    }
    if replication:
        streams["original"] = CueMotorProposal(len(rows), 3, effective_support=False)
    summaries = []
    outcomes = []
    for trial in range(160):
        side, act, shuffled = torch.from_numpy(
            np.stack([r.integers(2, size=3) for r in rng])
        ).T

        def key(s):
            return torch.stack((torch.zeros_like(s), s * 2 - 1), 1).float()

        cues = dict(
            original=key(side),
            true=key(side),
            shuffled=key(shuffled),
            lever_tick=key(side * 0),
            lever_experience=key(side * 0),
        )
        for name, organ in streams.items():
            is_lever = name.startswith("lever")
            correct = (flip ^ int(trial >= 80)) if is_lever else (side ^ flip)
            d = torch.full((len(rows), 2), 0.55)
            picked = organ.propose(
                act, d, due, None, key(side * 0 if is_lever else side)
            )
            outcomes.append((name, trial, (picked == correct).tolist()))
            empty = organ.propose(act, d, due, None, torch.zeros_like(cues[name]))
            assert torch.equal(empty, act)
            for tick in range(22):
                after = d.clone()
                if tick == 8:
                    after[act == correct] -= 0.25
                if tick == 14:
                    after[act != correct] += 0.12
                command = act if tick == 0 else torch.full_like(act, 2)
                cue = cues[name] if tick == 0 else torch.zeros_like(cues[name])
                organ.record(command, d, after, due & False, cue)
                d = after
        if trial in (79, 99, 119, 159):
            for name in streams:
                a = np.array(
                    [
                        v
                        for n, t, v in outcomes
                        if n == name and trial - 19 <= t <= trial
                    ]
                )
                summaries.append(
                    dict(
                        name=name,
                        end=trial + 1,
                        accuracy=float(a.mean()),
                        per_body=a.mean(0).tolist(),
                    )
                )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication", action="store_true")
    print(json.dumps(run(parser.parse_args().replication), indent=2))
