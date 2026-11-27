"""Event-level somatic forecasts. One instance follows one body's experience."""
import torch
from torch import nn


class SomaticReadout(nn.Module):
    def __init__(self, cue_width, channels, slots=32, events=128):
        super().__init__()
        for prefix, n in (('fast',slots), ('past',events)):
            self.register_buffer(prefix+'_cue', torch.zeros(n,cue_width))
            self.register_buffer(prefix+'_action',torch.full((n,),-1,dtype=torch.long))
            self.register_buffer(prefix+'_delta',torch.zeros(n,channels))
        self.register_buffer('visits',torch.zeros(slots))
        self.register_buffer('error',torch.zeros(slots))
        self.register_buffer('serial',torch.zeros((),dtype=torch.long))

    def _distance(self,cues,cue):
        return (cues-cue).square().mean(-1)

    @torch.no_grad()
    def remember(self,cue,action,before,after):
        """Call once when the executed chunk ends, including for an unrewarding end."""
        change = after-before
        distance = self._distance(self.fast_cue,cue).masked_fill(self.fast_action!=action,float('inf'))
        i = int(distance.argmin())
        if distance[i]>.04:
            i = int(self.visits.argmin())
            self.visits[i]=0
            self.fast_action[i]=action
            self.fast_cue[i].copy_(cue)
            self.fast_delta[i].copy_(change)
            self.error[i]=0
        residual = change-self.fast_delta[i]
        self.error[i].lerp_(residual.abs().mean(),.6)
        self.fast_delta[i].add_(residual,alpha=.35)
        self.visits[i].add_(1).clamp_max_(1024)
        j = int(self.serial)%len(self.past_action)
        self.past_cue[j].copy_(cue)
        self.past_action[j]=action
        self.past_delta[j].copy_(change)
        self.serial.add_(1)

    @torch.no_grad()
    def choose(self,cue,actions,deficits,mode='adaptive',generator=None):
        distance = self._distance(self.fast_cue,cue).expand(len(actions),-1)
        distance = distance.masked_fill(self.fast_action!=actions[:,None],float('inf'))
        dist,idx = distance.min(1)
        support = (1-torch.exp(-self.visits[idx]/3))*torch.exp(-20*dist-8*self.error[idx])
        confidence = float(support.min())
        called = mode=='slow' or (mode=='adaptive' and confidence<.65)
        if called:
            dist = self._distance(self.past_cue,cue)
            age = (self.serial-1-torch.arange(len(dist),device=cue.device))%len(dist)
            weights = torch.exp(-20*dist-age/16)*(self.past_action==actions[:,None])
            delta = (weights@self.past_delta)/weights.sum(1,keepdim=True).clamp_min(1e-12)
        else:
            delta = self.fast_delta[idx]*dist.isfinite()[:,None]
        predicted = (deficits+delta).clamp(0,1)
        cost = predicted.square().mean(1)
        ties = torch.where(cost==cost.min())[0]
        index = ties[torch.randint(len(ties),(1,),generator=generator)]
        return int(actions[index]),confidence,called
