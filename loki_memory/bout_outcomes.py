"""Estimate bodily changes after an anonymous executed onset over competing delays."""
import numpy as np


class BoutOutcomes:
    def __init__(self, motors, span=20):
        self.history=np.full(span,-1,dtype=int)
        self.mean=np.zeros((span,motors))
        self.seen=np.zeros_like(self.mean)
        self.baseline=0.

    def observe(self,onset,change):
        self.history[1:]=self.history[:-1].copy()
        self.history[0]=onset
        lags=np.flatnonzero(self.history>=0)
        actions=self.history[lags]
        residual=change-self.mean[lags,actions]
        self.mean[lags,actions]+=.24*residual
        self.seen[lags,actions]+=1
        self.baseline+=.015*(change-self.baseline)

    def forecast(self,current_only=False):
        # Select the delay with strongest supported variation across executed motors.
        support=self.seen/(self.seen+3)
        effect=(self.mean-self.baseline)*support
        spread=effect.var(axis=1)
        lag=0 if current_only else int(spread.argmax())
        return effect[lag],support[lag],lag
