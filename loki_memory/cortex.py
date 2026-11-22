"""Fixed-size associative field of action-conditioned changes."""
from pathlib import Path
import os
import tempfile

import numpy as np


class LongTermCortex:
    """Single-writer RAM field. Only consolidated coefficients answer queries."""

    rank = 40

    def __init__(self, width, regions=16, *, seed=59, selective=True, modes=512):
        if not isinstance(width, (int, np.integer)) or width < 1:
            raise ValueError("expected positive sensor width")
        if not isinstance(regions, int) or regions < 2 or regions > 4096 or regions & (regions-1):
            raise ValueError("regions must be a power of two between 2 and 4096")
        self.width, self.regions, self.seed = int(width), regions, int(seed)
        self.selective = bool(selective)
        self.group = (self.width + 191) // 192
        self.features = (self.width + self.group - 1) // self.group
        rng = np.random.default_rng(self.seed)
        bits = regions.bit_length()-1
        self.projection = rng.normal(size=(self.features, 8+bits)).astype(np.float32)
        self.projection /= np.sqrt(self.features)
        self.waves = rng.normal(size=(11,8)).astype(np.float32)
        self.gram = np.zeros((regions,self.rank,self.rank),np.float32)
        self.cross = np.zeros((regions,self.rank,self.features),np.float32)
        self.weights = np.zeros_like(self.cross)
        self.support = np.zeros(regions,np.float32)
        self.skill = np.zeros((regions,2),np.float32)
        if modes not in (0,512):
            raise ValueError("supported mode counts are 0 and 512")
        self.modes = modes
        if modes:
            self.phase_projection = (rng.standard_normal((self.features+3,modes+16),
                                                        dtype=np.float32)*5)
            self.hologram = np.zeros((2*modes,self.features+32),np.float32)
        self.activity = np.zeros(2,np.float32)
        self.dirty = np.zeros(regions,bool)

    @property
    def state_bytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x,np.ndarray))

    def _frame(self,x):
        x = np.asarray(x,dtype=np.float32)
        if x.shape!=(self.width,) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
            raise ValueError("expected finite normalized sensor vector")
        return x

    def _pool(self,x):
        if self.group==1:
            return x
        starts = np.arange(0,self.width,self.group)
        return (np.add.reduceat(x,starts)/np.minimum(self.group,self.width-starts)).astype(np.float32)

    def _key(self,x,action):
        from .organ import LokiMemory
        projected = self._pool(x-.5) @ self.projection
        latent = projected[:8]*3
        angles = projected[8:]
        bits = 1 << np.arange(len(angles))
        region = int((angles>=0) @ bits)
        weak = int(np.argmin(np.abs(angles)))
        neighbor = region ^ int(bits[weak])
        mix = .5*np.exp(-12*abs(float(angles[weak])))
        motor = LokiMemory._motor(action)
        phase = np.r_[latent,motor] @ self.waves
        drive = np.r_[1.,motor,motor[0]*motor[1]]
        phi = np.r_[np.outer(drive,np.r_[1.,latent[:6]]).ravel(),
                    latent[6:],np.sin(phase[:2]),np.cos(phase[:1])].astype(np.float32)
        return phi,(region,neighbor),(1-mix,mix)

    def _read(self,x,baseline,key):
        phi,regions,mix = key
        result = baseline.copy()
        for region,weight in zip(regions,mix):
            n = self.support[region]
            if n>0:
                delta = phi @ self.weights[region]
                if self.group>1:
                    delta = delta.repeat(self.group)[:self.width]
                credit,energy = self.skill[region]
                trust = np.clip(credit/(energy+.001),0,1) if energy>0 else .5
                result += weight*trust*n/(n+12)*(x+delta-baseline)
        return result.clip(0,1)

    def _wave(self,x,action):
        from .organ import LokiMemory
        angles = np.r_[self._pool(x-.5),LokiMemory._motor(action)] @ self.phase_projection
        co,si = np.cos(angles),np.sin(angles)
        n = self.modes
        return np.r_[co[:n],si[:n]], np.r_[co[n:],si[n:]]

    def _episode(self,x,action,law):
        wave,tag = self._wave(x,action)
        read = wave @ self.hologram / self.modes
        similarity = float(tag @ read[self.features:])/16
        certainty = np.clip((similarity-.45)/.45,0.,1.)
        delta = read[:self.features]
        scale = 1/(1+min(float(self.activity[1]),2*self.modes)/(2*self.modes))
        if self.group>1:
            delta = delta.repeat(self.group)[:self.width]
        output = (law + certainty*(x+scale*delta-law)).clip(0,1)
        return output,wave,tag,read

    def predict(self,cue,action,baseline):
        x = self._frame(cue)
        law = self._read(x,self._frame(baseline),self._key(x,action))
        return self._episode(x,action,law)[0] if self.modes and self.activity[1] else law

    def imprint(self,cue,action,outcome,baseline,*,integrity_change=0.):
        """Return whether this measured transition contributed to the sufficient statistics."""
        x,y,b = self._frame(cue),self._frame(outcome),self._frame(baseline)
        if not np.isfinite(integrity_change) or abs(integrity_change)>1:
            raise ValueError("expected normalized signed integrity change")
        key = self._key(x,action)
        law = self._read(x,b,key)
        error = float(np.sqrt(np.mean((y-law)**2)))
        salient = abs(integrity_change)
        strong = salient>=.08 or error>max(.12,3*self.activity[0])
        self.activity[0] += .015*(error-self.activity[0])
        phi,regions,mix = key
        for region,weight in zip(regions,mix):
            if self.support[region]:
                diff = x + (phi @ self.weights[region]).repeat(self.group)[:self.width] - b
                evidence = np.array([np.mean(diff*(y-b)),np.mean(diff*diff)])
                self.skill[region] *= 1-.005*weight
                self.skill[region] += weight*evidence
        if self.modes and (strong or not self.selective):
            _,wave,tag,read = self._episode(x,action,law)
            innovation = np.r_[self._pool(y-x),tag]-read
            self.hologram += np.outer(wave,innovation)  # decoder divides by the squared norm of the wave
            self.activity[1] = min(2*self.modes,self.activity[1]+1)
        if self.selective and error<.025 and salient<.08:
            return False
        phi,regions,mix = key
        gram = np.outer(phi,phi)
        cross = np.outer(phi,self._pool(y-x))
        for region,weight in zip(regions,mix):
            if self.gram[region,0,0]>8192:
                self.gram[region] *= .5
                self.cross[region] *= .5
            gain = weight*(1+min(3.,salient*8))
            self.gram[region] += gain*gram
            self.cross[region] += gain*cross
            self.dirty[region] = True
        return True

    def sleep(self,max_regions=None):
        """Publish ridge solutions, optionally limiting work per idle interval."""
        if max_regions is not None and (not isinstance(max_regions,int) or max_regions<0):
            raise ValueError("expected a nonnegative idle work limit")
        dirty = np.flatnonzero(self.dirty)
        for region in dirty[:max_regions]:
            gram = self.gram[region].astype(np.float64)
            gram.flat[::self.rank+1] += .35
            self.weights[region] = np.linalg.solve(gram,self.cross[region])
            self.support[region] = self.gram[region,0,0]
            self.dirty[region] = False
        return len(dirty[:max_regions])

    def save(self,path,*,binding=""):
        arrays = {k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
            name = tmp.name
        try:
            with open(name,"wb") as f:
                np.savez(f,version=2,config=[self.width,self.regions,self.seed,int(self.selective),self.modes],
                         binding=binding,**arrays)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name,path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls,path,*,binding=""):
        with np.load(path,allow_pickle=False) as data:
            if int(data["version"])!=2 or data["binding"].item()!=binding:
                raise ValueError("incompatible cortex or wiring")
            w,b,s,sel,modes = map(int,data["config"])
            cortex = cls(w,b,seed=s,selective=bool(sel),modes=modes)
            for k,v in vars(cortex).items():
                if isinstance(v,np.ndarray):
                    src = data[k]
                    if src.shape!=v.shape or src.dtype!=v.dtype or not np.isfinite(src).all():
                        raise ValueError("invalid cortical field")
                    v[:] = src
            return cortex


class CorticalMemory:
    """One-step closed-loop interface. Only real consecutive observations are imprinted."""

    def __init__(self,core,cortex):
        if core.width!=cortex.width:
            raise ValueError("different sensor widths")
        self.core,self.cortex = core,cortex
        self.observed = False

    def predict(self,action):
        baseline = self.core.imagine([action])
        if not self.observed:
            return baseline
        return self.cortex.predict(self.core.imagine(),action,baseline)

    def predict_effect(self,a,b):
        return self.predict(a)-self.predict(b)

    def observe(self,frame,action=None,*,visible=None,integrity_change=0.):
        written = False
        complete = visible is None or np.all(visible)
        if self.observed and complete and action is not None:
            written = self.cortex.imprint(self.core.imagine(),action,frame,self.core.imagine([action]),
                                          integrity_change=integrity_change)
        self.core.observe(frame,action,visible=visible)
        self.observed = bool(complete)
        return written

    def cut(self):
        self.core.cut()
        self.observed = False

    def advance(self,action):
        self.observed = False
        return self.core.advance(action)
