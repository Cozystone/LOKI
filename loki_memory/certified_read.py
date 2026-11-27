"""Reuse an episodic read only while an input-space bound certifies its identity."""
import numpy as np


class CertifiedRead:
    def __init__(self, capacity, width, actions, channels):
        self.keys = np.zeros((capacity,width))
        self.outcomes = np.zeros((capacity,actions,channels))
        self.counts = np.zeros((capacity,actions),dtype=np.int32)
        self.used = 0
        self.anchor = np.zeros(width)
        self.index = -1
        self.radius = 0.
        self.reads = 0

    def learn(self, cue, action, change):
        matches = np.flatnonzero(np.all(self.keys[:self.used]==cue, axis=1))
        if len(matches):
            i = matches[0]
        elif self.used < len(self.keys):
            i = self.used
            self.used += 1
            self.keys[i] = cue
        else:
            return False
        self.counts[i,action] += 1
        self.outcomes[i,action] += (change-self.outcomes[i,action])/self.counts[i,action]
        self.index = -1
        return True

    def choose(self, cue, deficits, mode):
        if not self.used:
            raise ValueError("empty memory")
        full = mode=='full' or self.index<0
        if not full:
            delta = cue-self.anchor
            if mode=='sample':
                delta = delta[::8]
            full = np.dot(delta,delta)>=self.radius**2
        if full:
            self.reads += 1
            delta = self.keys[:self.used]-cue
            distances = np.einsum('ij,ij->i',delta,delta)
            self.index = int(distances.argmin())
            if mode!='full':
                pair = np.sqrt(np.partition(distances,1)[:2]) if self.used>1 else [0,np.inf]
                self.radius = max(0.,(max(pair)-min(pair))*.5-1e-10)
                self.anchor[:] = cue
        after = np.clip(deficits+self.outcomes[self.index],0.,1.)
        return int(np.sum(after*after,axis=1).argmin())
