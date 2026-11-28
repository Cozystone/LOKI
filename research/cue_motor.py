"""Per-body recall of measured interoceptive change after a movement onset."""

import math
import torch


class CueMotorProposal:
    def __init__(
        self, bodies, actions, device="cpu", decay="tick", effective_support=True
    ):
        if decay not in ("tick", "experience"):
            raise ValueError("decay must be tick or experience")
        self.decay = decay
        self.effective_support = effective_support
        self.bodies = torch.arange(bodies, device=device)
        angle = torch.arange(8, device=device) * (math.tau / 8)
        self.centres = torch.stack((angle.cos(), angle.sin()), 1)
        self.count = torch.zeros(bodies, 8, actions, device=device)
        self.mass2 = torch.zeros_like(self.count)
        self.total = torch.zeros_like(self.count[..., None].expand(-1, -1, -1, 2))
        self.keys = self.count.new_zeros(21, bodies, 8)
        self.actions = torch.zeros(21, bodies, device=device, dtype=torch.long)
        self.start = self.total.new_zeros(21, bodies, 2)
        self.previous = torch.full_like(self.bodies, -1)
        self.clock = 0

    def _key(self, cue):
        # Direction in the eye's frame. Vector length carries observational support.
        magnitude = cue.norm(dim=1, keepdim=True)
        direction = cue / magnitude.clamp_min(1e-6)
        weights = (direction @ self.centres.T).clamp_min(0).pow(4)
        return (
            magnitude.clamp(max=1)
            * weights
            / weights.sum(1, keepdim=True).clamp_min(1e-6)
        )

    @torch.no_grad()
    def propose(self, pick, deficits, due, allowed, cue):
        key = self._key(cue)
        count = torch.einsum("bk,bka->ba", key, self.count)
        change = (
            torch.einsum("bk,bkad->bad", key, self.total)
            / count.clamp_min(1e-6)[..., None]
        )
        utility = -(deficits[:, None] + change).clamp(0, 1).square().mean(-1)
        supported = count >= 1
        if self.effective_support:
            nearest = key.argmax(1)
            mass = self.count[self.bodies, nearest]
            squares = self.mass2[self.bodies, nearest]
            supported = (count >= 0.05) & (mass.square() >= 2 * squares) & (squares > 0)
        if allowed is not None:
            supported &= allowed
        best = utility.masked_fill(~supported, -torch.inf).argmax(1)
        rows = self.bodies
        switch = due & supported[rows, pick] & supported[rows, best]
        switch &= utility[rows, best] > utility[rows, pick] + 1e-6
        return torch.where(switch, best, pick)

    @torch.no_grad()
    def record(self, executed, before, after, cut, cue_before):
        """Call once per tick. Both deficit tensors are post-organ, in [0,1]."""
        slot = self.clock
        self.keys[slot] = self._key(cue_before) * (executed != self.previous)[:, None]
        self.keys[:, cut] = 0
        self.actions[slot] = executed
        # Exclude same-tick effects and read only after 20 subsequent ticks.
        self.start[slot] = after
        self.previous = torch.where(cut, -1, executed)
        self.clock = (slot + 1) % len(self.keys)
        key = self.keys[self.clock]
        action = self.actions[self.clock]
        delta = after - self.start[self.clock]
        rows = self.bodies
        if self.decay == "tick":
            self.count.mul_(0.995)
            self.mass2.mul_(0.995**2)
            self.total.mul_(0.995)
        else:
            fade = 0.995**key
            self.count[rows, :, action] *= fade
            self.mass2[rows, :, action] *= fade.square()
            self.total[rows, :, action] *= fade[..., None]
        self.mass2[rows, :, action] += key.square()
        self.count[rows, :, action] += key
        self.total[rows, :, action] += key[..., None] * delta[:, None]
