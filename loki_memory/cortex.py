"""Fixed-size associative field of action-conditioned changes."""
from pathlib import Path
import os
import tempfile
import zipfile

import numpy as np

from .organ import LokiMemory


class LongTermCortex:
    """Single-writer RAM field with consolidated laws and online episodic waves."""

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
        # Draw the same basis in the wave-off control so the law features are paired.
        phase_projection = rng.standard_normal((self.features+3,528),dtype=np.float32)
        if modes:
            self.phase_projection = phase_projection*5
            self.hologram = np.zeros((2*modes,self.features+32),np.float32)
        self.global_projection = rng.normal(size=(self.features,18)).astype(np.float32)
        self.global_projection /= np.sqrt(self.features)
        self.global_gram = np.zeros((125,125),np.float32)
        self.global_cross = np.zeros((125,self.features),np.float32)
        self.global_weights = np.zeros_like(self.global_cross)
        self.global_dirty = np.zeros(1,bool)
        self.global_live = np.zeros(1,bool)
        self.credits = np.zeros((regions,2,2),np.float32)
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

    def _global_key(self,x,action,b):
        motor = LokiMemory._motor(action)
        drive = np.r_[1.,motor,motor[0]*motor[1]]
        cue = self._pool(x-.5) @ self.global_projection
        trend = self._pool(b-x) @ self.global_projection[:,:6]
        return np.outer(drive,np.r_[1.,cue*4,trend*4]).astype(np.float32).ravel()

    def _mix(self,base,expert,key,index,truth=None):
        _,regions,weights = key
        diff = expert-base
        credit = 0.
        for r,w in zip(regions,weights):
            score = self.credits[r,index]
            credit += w*np.clip(score[0]/(.003+score[1]),0,1)
            if truth is not None:
                score *= 1-.01*w
                score += w*np.array([np.mean(diff*(truth-base)),np.mean(diff**2)])
        return base+credit*diff

    def _laws(self,x,action,b,key,truth=None):
        local = self._read(x,b,key)
        g = self._global_key(x,action,b)
        if self.global_live[0]:
            correction = (g@self.global_weights).repeat(self.group)[:self.width]
            local = self._mix(local,(b+correction).clip(0,1),key,0,truth)
        return local,g

    def _query(self,cue,action,baseline,episodic):
        x,b = self._frame(cue),self._frame(baseline)
        key = self._key(x,action)
        law,_ = self._laws(x,action,b,key)
        if self.modes and self.activity[1]:
            event = self._episode(x,action,law)[0]
            return event if episodic else self._mix(law,event,key,1)
        return law

    def predict(self,cue,action,baseline):
        return self._query(cue,action,baseline,False)

    def recall_episode(self,cue,action,baseline):
        """Query the salient past consequence; fall back to the law on weak resonance."""
        return self._query(cue,action,baseline,True)

    def imprint(self,cue,action,outcome,baseline,*,integrity_change=0.):
        """Return whether this measured transition contributed to the sufficient statistics."""
        x,y,b = self._frame(cue),self._frame(outcome),self._frame(baseline)
        if not np.isfinite(integrity_change) or abs(integrity_change)>1:
            raise ValueError("expected normalized signed integrity change")
        key = self._key(x,action)
        law,g = self._laws(x,action,b,key,y)
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
        packet = None
        if self.modes and (self.activity[1] or strong or not self.selective):
            packet = self._episode(x,action,law)
            self._mix(law,packet[0],key,1,y)
        if self.modes and (strong or not self.selective):
            _,wave,tag,read = packet
            matched = np.clip((float(tag@read[self.features:])/16-.45)/.45,0,1)
            innovation = np.r_[self._pool(y-x),tag]-matched*read
            self.hologram += np.outer(wave,innovation)  # decoder divides by the squared norm of the wave
            self.activity[1] = min(2*self.modes,self.activity[1]+1)
        if self.selective and error<.025 and salient<.08:
            return False
        if self.global_gram[0,0]>16384:
            self.global_gram *= .5
            self.global_cross *= .5
        self.global_gram += np.outer(g,g)
        self.global_cross += np.outer(g,self._pool(y-b))
        self.global_dirty[:] = True
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
        if self.global_dirty[0] and max_regions!=0:
            gram = self.global_gram.astype(np.float64)
            gram.flat[::len(gram)+1] += .5
            self.global_weights[:] = np.linalg.solve(gram,self.global_cross)
            self.global_dirty[:] = False
            self.global_live[:] = True
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
                np.savez(f,version=3,config=[self.width,self.regions,self.seed,int(self.selective),self.modes],
                         binding=binding,**arrays)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name,path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls,path,*,binding=""):
        with np.load(path,allow_pickle=False) as data:
            if int(data["version"])!=3 or data["binding"].item()!=binding:
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

    def recall_episode(self,action):
        baseline = self.core.imagine([action])
        if not self.observed:
            return baseline
        return self.cortex.recall_episode(self.core.imagine(),action,baseline)

    def observe(self,frame,action=None,*,visible=None,integrity_change=0.):
        written = False
        if visible is not None:
            # Validate even an all-true mask before the cortex can be mutated.
            self.core._sense(frame,visible)
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

    def save(self,path,*,binding=""):
        """Atomically publish one consistent core/field checkpoint between steps."""
        path = Path(path)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            folder = Path(tmp)
            self.core.save(folder/"core.npz",binding=binding)
            self.cortex.save(folder/"cortex.npz",binding=binding)
            archive = folder/"memory.zip"
            with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_STORED) as out:
                for name in ("core.npz","cortex.npz"):
                    out.write(folder/name,name)
                out.writestr("observed",str(int(self.observed)))
            with archive.open("rb+") as out:
                os.fsync(out.fileno())
            os.replace(archive,path)

    @classmethod
    def load(cls,path,*,binding=""):
        with zipfile.ZipFile(path) as src:
            with src.open("core.npz") as stream:
                core = LokiMemory.load(stream,binding=binding)
            with src.open("cortex.npz") as stream:
                cortex = LongTermCortex.load(stream,binding=binding)
            flag = src.read("observed")
            if flag not in (b"0",b"1"):
                raise ValueError("invalid observation boundary")
        model = cls(core,cortex)
        model.observed = flag==b"1"
        return model
