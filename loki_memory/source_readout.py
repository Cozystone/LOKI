"""One body's bounded, advisory source readout. All positions are endogenous odometry."""
import torch
from torch import nn


class SourceReadout(nn.Module):
    def __init__(self, candidates=2, channels=2, contexts=2, cue_width=4):
        super().__init__()
        for name, value in dict(
            centers=torch.zeros(contexts,cue_width), used=torch.zeros(contexts,dtype=torch.bool),
            spatial=torch.ones(contexts+1,candidates), body=torch.full((contexts+1,),.04),
            positions=torch.zeros(candidates,2), changes=torch.zeros(candidates,channels),
            attempts=torch.zeros(candidates), context=torch.zeros((),dtype=torch.long),
        ).items():
            self.register_buffer(name,value)

    @torch.no_grad()
    def offer(self,cue,positions,changes):
        """New comparison at one decision boundary; stable row = position source.
        changes[i] is the gut's predicted deficit change for journey i.
        Keep this slate intact during verification, including after failed attempts.
        """
        self.positions.copy_(positions)
        self.changes.copy_(changes)
        self.attempts.zero_()
        self.context.zero_()
        if len(self.centers):
            distance=(self.centers-cue).square().mean(1).masked_fill(~self.used,float('inf'))
            i=int(distance.argmin())
            empty=torch.where(~self.used)[0]
            if distance[i]>.04 and len(empty):
                i=int(empty[0])
                self.centers[i].copy_(cue)
                self.used[i]=True
            if (self.centers[i]-cue).square().mean()<=.04:
                self.context.fill_(i+1)

    def advice(self,deficits):
        i=int(self.context)
        # Explicit engineering prior: one-cell error scale, .2 deficit scale.
        spatial_support=1/(1+self.spatial[i])
        gut_support=1/(1+self.body[i]/.04)
        after=(deficits+self.changes).clamp(0,1)
        benefit=deficits.square().mean()-after.square().mean(1)
        support=spatial_support/(1+self.attempts)
        return support*gut_support*benefit, support, gut_support

    @torch.no_grad()
    def verify(self, own_position=None, comparable=None, attempted=None,
               before=None, after=None, adequate_miss=False):
        """comparable must refer to the SAME verified event; default no location credit.
        attempted is an executed journey index. An adequate miss applies only locally.
        No evidence => no update.
        """
        ctx=int(self.context)
        for i in ([0,ctx] if ctx else [0]):
            if own_position is not None and comparable is not None:
                error=(self.positions-own_position).square().sum(1)
                self.spatial[i].add_(.2*comparable*(error-self.spatial[i]))
            if attempted is not None and before is not None and after is not None:
                residual=((before+self.changes[attempted]).clamp(0,1)-after).square().mean()
                self.body[i].lerp_(residual,.2)
        if adequate_miss and attempted is not None:
            self.attempts[attempted]+=1
