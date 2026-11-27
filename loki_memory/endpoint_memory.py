"""Fixed-space, event-terminated endpoint associations for an ATANOR-side probe."""
import torch
from torch import nn


class EndpointMemory(nn.Module):
    def __init__(self, width=30, senses=4, channels=2, capacity=128, window=320):
        super().__init__()
        width += senses
        shapes = dict(keys=(capacity,width), occupied=(capacity,), trace=(window,width),
                      moves=(window,), latency=(capacity,channels), motor=(capacity,channels),
                      amount=(capacity,channels), drift=(channels,), count=(), cursor=())
        for name, shape in shapes.items():
            self.register_buffer(name, torch.zeros(shape, dtype=torch.long if name in
                                 ('moves','motor','count','cursor') else torch.float32))
        self.latency.fill_(float('inf'))

    @torch.no_grad()
    def observe(self, sense, path, action, before, after):
        """Append a real transition. A positive channel recovery terminates the trace."""
        i = int(self.cursor)
        self.trace[i].copy_(torch.cat((sense,path)))
        self.moves[i] = action
        self.cursor.fill_((i+1) % len(self.moves))
        self.count.clamp_max_(len(self.moves)-1).add_(1)
        change = after-before
        self.drift.copy_(torch.where(change >= 0, .95*self.drift+.05*change, self.drift))
        return (-change).clamp_min(0)

    @torch.no_grad()
    def consolidate(self, relief):
        """Replay once before resuming observations, only for an experienced recovery."""
        if not relief.any():
            return
        for delay in range(1,int(self.count)+1):
            j = (int(self.cursor)-delay) % len(self.moves)
            key = self.trace[j]
            distance = (self.keys-key).square().mean(1)
            known = distance.masked_fill(self.occupied==0,float('inf'))
            k = int(known.argmin())
            if known[k] > 1e-7:
                empty = torch.where(self.occupied==0)[0]
                if not len(empty):
                    continue
                k = int(empty[0])
                self.keys[k].copy_(key)
                self.occupied[k] = 1
            update = (relief>0) & (delay <= self.latency[k])
            self.latency[k,update] = delay
            self.motor[k,update] = self.moves[j]
            self.amount[k,update] = relief[update]
        self.count.zero_()

    @torch.no_grad()
    def choose(self, sense, path, deficits):
        key = torch.cat((sense,path))
        k = ((self.keys-key).square().mean(1).masked_fill(self.occupied==0,float('inf'))).argmin()
        times = self.latency[k]
        supported = times.isfinite()
        if not supported.any():
            return 0
        # Compare counterfactual recovery against no recovery at the same arrival time.
        need = (deficits+self.drift*times.nan_to_num(posinf=0)).clamp(0,1)
        improvement = need.square()-(need-self.amount[k]).clamp_min(0).square()
        channel = (improvement/times).masked_fill(~supported,-1).argmax()
        return int(self.motor[k,channel])
