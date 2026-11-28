"""Optional LOKI proposal, evaluated in the pinned upstream lever_world harness."""

import argparse
import hashlib
import random
import numpy as np
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


class MotorProposal:
    """Pooled onset-conditioned body changes, with separate eligibility per body."""

    def __init__(self, codes, bodies):
        groups = torch.cat([F.one_hot(col).float() for col in codes.T], dim=1)
        self.groups = groups
        self.previous = torch.zeros(bodies, groups.shape[1], device=codes.device)
        self.history = self.previous[:, None].repeat(1, 21, 1)
        self.count = torch.zeros_like(self.history[0])
        self.total = self.count[..., None].repeat(1, 1, 2)
        self.visited = torch.zeros(len(codes), device=codes.device, dtype=torch.bool)
        self.changed = 0
        self.opportunities = 0

    def propose(self, action, d, due, allowed):
        # Lag zero is the change on the same tick as the motor onset.
        mean = self.total / self.count.clamp_min(1)[..., None]
        effect = self.groups @ mean[1:].sum(0)
        utility = -(d[:, None, :] + effect).clamp(0, 1).square().mean(-1)
        eligible = self.visited if allowed is None else self.visited & allowed
        best = utility.masked_fill(~eligible, -torch.inf).argmax(1)
        improve = (
            utility.gather(1, best[:, None]) > utility.gather(1, action[:, None]) + 1e-6
        )
        switch = due & improve[:, 0] & eligible[best]
        self.changed += int(switch.sum())
        self.opportunities += int(due.sum())
        return torch.where(switch, best, action)

    def record(self, action, delta, cut):
        now = self.groups[action]
        onset = (now - self.previous).clamp_min(0)
        self.history = torch.cat([onset[:, None], self.history[:, :-1]], 1)
        self.history[cut] = (
            0  # A reset or a terminal transition supplies no delayed target.
        )
        self.previous = now.clone()
        self.previous[cut] = 0
        self.count.mul_(0.995).add_(self.history.sum(0))
        self.total.mul_(0.995).add_(torch.einsum("ntg,nc->tgc", self.history, delta))
        self.visited[action] = True


def rebirth_with_timer_cut(rebirth, body, *args):
    """Upstream rebirth clears dead flags, so retain the mask before calling it."""
    cut = body.dead.clone()
    rebirth(body, *args)
    body.busy[cut] = 0
    body.rise_in[cut] = -1
    body.fall_in[cut] = -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("being")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=9781)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--active", action="store_true")
    mode.add_argument(
        "--native",
        action="store_true",
        help="shared life-boundary fix only, no motor hook",
    )
    mode.add_argument(
        "--at-choice",
        action="store_true",
        help="use the optional pinned pre-credit chooser hook",
    )
    ap.add_argument(
        "--audit",
        action="store_true",
        help="three short phases on deterministic single-thread CPU",
    )
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from tools import lever_world as lv

    original_being, original_rebirth = lv.being, lv.rebirth
    holder = {}
    if args.audit:
        from tools import s4_kin, need_fork

        lv.DEV = s4_kin.DEV = need_fork.DEV = "cpu"
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        digest = hashlib.sha256()
        step = lv.LeverBody.step

        def audited_step(body, act, dt=None):
            result = step(body, act, dt)
            for tensor in (act, body.w.store, body.w.dead, body.w.pos):
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return result

        lv.LeverBody.step = audited_step

    def build(body, n, path):
        organs, mind = original_being(body, n, path)
        probe = MotorProposal(torch.tensor(body.combos, device=body.dev), n)
        holder["probe"] = probe
        if args.native:
            return organs, mind
        choose, after = mind.choose, mind.after

        def deficits(x):
            return 1 - x[:, [mind.going.ins[2], mind.going.ins[1]]].clamp(0, 1)

        if args.at_choice:
            import inspect

            if "motor_candidate" not in inspect.getsource(type(mind).choose):
                raise RuntimeError("apply research/motor_candidate.patch first")
            mind.motor_candidate = lambda x, pick, due, allowed: probe.propose(
                pick, deficits(x), due, allowed
            )

        def select(x):
            action = choose(x)
            if args.active:
                pick = probe.propose(
                    action,
                    deficits(x),
                    mind.again & (mind.src != 4),
                    getattr(mind, "allowed", None),
                )
                mind.cur = torch.where(pick != action, pick, mind.cur)
                return pick
            return action

        def observe(x, a, y, died, already):
            probe.record(a, deficits(y) - deficits(x), died | already)
            return after(x, a, y, died, already)

        mind.choose, mind.after = select, observe
        return organs, mind

    def rebirth(body, *rest):
        rebirth_with_timer_cut(original_rebirth, body, *rest)

    lv.being, lv.rebirth = build, rebirth
    sys.argv = [
        "lever_world",
        args.output,
        "--being",
        args.being,
        "--bodies",
        "8",
        "--seed",
        str(args.seed),
        "--eye",
        "1.1",
        "--t1",
        "240",
        "--t2",
        "720",
        "--t3",
        "1200",
    ]
    if args.audit:
        sys.argv += ["--bodies", "2", "--t1", "40", "--t2", "120", "--t3", "200"]
    lv.main()
    out = Path(args.output)
    data = json.loads(out.read_text())
    data["motor_probe"] = dict(
        active=args.active or args.at_choice,
        at_choice=args.at_choice,
        native=args.native,
        changed=holder["probe"].changed,
        due=holder["probe"].opportunities,
    )
    if args.audit:
        data["trajectory_sha256"] = digest.hexdigest()
    out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
