"""Bounded temporal superposition over completed live-body transitions."""
import hashlib
import numpy as np

from .body import BodyMemory, _BodyWindow
from .contextual import ContextField
from .portable import _input, serialized


def latent(frame, signals):
    pixels = _input(frame,(16,16,3)).reshape(2,8,2,8,3).mean(axis=(1,3))
    return np.concatenate((pixels.ravel(), _input(signals,(8,))))


def _source(body):
    bits = np.unpackbits(np.frombuffer(hashlib.sha256(body.encode()).digest()[:1],np.uint8))
    return bits[:6].astype(np.float32)


class EpisodeField(ContextField):
    base = 246144
    extra = 90*54*2 + 37*20*4 + 8

    def __init__(self, blob=None):
        extended = blob is not None and len(blob)==self.base+self.extra
        super().__init__(blob[:self.base] if extended else blob)
        base = self.data
        self.data = np.zeros((self.base+self.extra)//4, dtype="<f4")
        self.data[:base.size] = base
        for name in ("weights","precision","omega","binding","complement"):
            view = getattr(self,name)
            offset = (view.ctypes.data-base.ctypes.data)//4
            setattr(self,name,self.data[offset:offset+view.size].reshape(view.shape))
        self.tail = self.data.view(np.uint8)[self.base:]
        if extended:
            self.tail[:] = np.frombuffer(blob[self.base:],dtype=np.uint8)
        self.trace = np.frombuffer(self.tail,"<f2",90*54).reshape(90,54)
        self.dynamics = np.frombuffer(self.tail,"<f4",37*20,90*54*2).reshape(37,20)
        self.clock = np.frombuffer(self.tail,"<u8",1,self.extra-8)

    @staticmethod
    def phase(t):
        # Stateless frequencies, no per-event addresses saved.
        j = np.arange(1,46,dtype=np.float64)
        theta = (np.sin(j*78.233)*43758.5453 % 1)*(2*np.pi*t)
        return np.r_[np.cos(theta),np.sin(theta)].astype(np.float32)/np.sqrt(np.float32(45))

    def record(self, before, action, after, body):
        t=int(self.clock[0])+1
        k=self.phase(t)
        target=np.r_[before,action,after,_source(body)]-.5
        w=self.trace.astype(np.float32)*.99
        self.trace[:]=w+np.outer(k,target-k@w)
        x=self.features(before,action)
        error=after-before-x@self.dynamics
        self.dynamics+=.4*np.outer(x,error)/(x@x+1e-6)
        self.clock[0]=t

    @staticmethod
    def features(state, action):
        return np.r_[state-.5,action-.5,(state[:8]-.5)*(action-.5),np.float32(1)]

    def event(self,t):
        if not isinstance(t,(int,np.integer)) or not 1<=t<=int(self.clock[0]):
            raise ValueError("expected a previously completed action number")
        row=np.clip(.5+self.phase(t)@self.trace.astype(np.float32),0,1)
        return row[:20],row[20:28],row[28:48],row[48:]

    def rollout(self,state,actions):
        state=_input(state,(20,)).copy()
        for action in actions:
            x=self.features(state,_input(action,(8,)))
            state=np.clip(state+x@self.dynamics,0,1)
        return state


class _EpisodeWindow(_BodyWindow):
    def _learn(self,key,pending,frame,signals):
        super()._learn(key,pending,frame,signals)
        before=latent(pending.anchor[:768].reshape(16,16,3),pending.anchor[768:])
        self._field.record(before,pending.action,latent(frame,signals),self.body)


class EpisodeMemory(BodyMemory):
    field_type=EpisodeField

    def _window(self,field,past):
        w=_EpisodeWindow(field,past)
        w.body=self.body
        return w

    @serialized
    def _read(self,fn,*args):
        from .portable import transaction
        with transaction(self.path) as db:
            self._check(db)
            return fn(*args)

    def event(self,t):
        return self._read(lambda: self._field.event(t))

    def rollout(self,state,actions):
        return self._read(lambda: self._field.rollout(state,actions))

    def occurrences(self,after,body,tolerance=.08,window=96):
        after=_input(after,(20,))
        def find():
            last=int(self._field.clock[0])
            matches=[]
            for t in range(max(1,last-window+1),last+1):
                _,_,seen,source=self._field.event(t)
                if np.array_equal(source>.5,_source(body)>.5) and np.mean((seen-after)**2)<=tolerance**2:
                    matches.append(t)
            return matches
        return self._read(find)

    def affordance(self,state,probes):
        """Action-effect profile in sensory coordinates, without class labels."""
        state=_input(state,(20,))
        return self._read(lambda: np.stack([self._field.rollout(state,[a])-state for a in probes]))
