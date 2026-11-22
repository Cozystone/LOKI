"""Fixed-budget sensory attractor and driven latent dynamics."""
from pathlib import Path
import os
import tempfile
import numpy as np
from . import transport


class LokiMemory:
    def __init__(self, width, budget=262144, *, sleep=True, grid_shape=None):
        if not isinstance(width,(int,np.integer)) or width < 1:
            raise ValueError("width must be a positive integer")
        self.width, self.budget, self.sleep = int(width), int(budget), bool(sleep)
        self.group = (width+2047)//2048
        f = self.features = (width+self.group-1)//self.group
        r = self.rank = min(f, 20 if f<=256 else 6)
        self.grid_shape = () if grid_shape is None else tuple(grid_shape)
        if self.grid_shape and (len(self.grid_shape)!=3
                or any(not isinstance(v,(int,np.integer)) or v<1 for v in self.grid_shape)
                or self.grid_shape[1]<2 or np.prod(self.grid_shape)>width or self.group!=1):
            raise ValueError("grid_shape must describe an unpooled HWC sensor suffix")
        self.bands = 12
        n = (5+2*self.bands)*3
        m = 4*r+4
        # Only immutable shape/offset metadata is external to the one numerical field.
        for k in range(64,7,-2):
            shapes = (("decoder",(f,k)),("precision",(k,k)),("h",(k,)),
                      ("synthesis",(r,f)),("read_precision",(r,r)),
                      ("force",(n+1,r)),("force_precision",(n+1,n+1)),
                      ("coupling",(m,r)),("motion_precision",(m,m)),
                      ("traces",(5+2*self.bands,3)),("anchor",(f,)),("mean",(f,)),
                      ("z",(r,)),("control",(6,)))
            if f>256:
                shapes += (("variance",(f,)),("samples",(1,)))
            if self.grid_shape:
                height,columns,channels = self.grid_shape
                bands = min(64,4*columns)
                shapes += (("flow",(16,3)),("flow_precision",(16,16)),("origin",(1,)),
                           ("affine",(3,)),("retina",self.grid_shape),
                           ("chart",(1+2*bands,1+height*channels)))
            cells = sum(int(np.prod(shape)) for _,shape in shapes)
            rows = (cells+k-1)//k
            if 4*rows*k<=budget:
                break
        else:
            raise ValueError("insufficient budget for this field")
        self.matrix = np.zeros((rows,k),np.float32)
        self._layout = {}
        offset = 0
        for name,shape in shapes:
            end = offset + int(np.prod(shape))
            self._layout[name] = (offset,end,shape)
            offset = end
        for name,scale in (("precision",1),("read_precision",20),("force_precision",2),("motion_precision",3)):
            a = self._part(name)
            np.fill_diagonal(a,scale)

        if self.grid_shape:
            np.fill_diagonal(self._part("flow_precision"),20)

    @property
    def state_bytes(self):
        return self.matrix.nbytes

    def _part(self,name):
        start,end,shape = self._layout[name]
        return self.matrix.reshape(-1)[start:end].reshape(shape)

    @property
    def _decoder(self):
        return self._part("decoder")

    @property
    def _h(self):
        return self._part("h")

    @property
    def _coupling(self):
        return self._part("coupling")

    def _project(self,x):
        # A stationary signed sketch: learning the decoder cannot rotate the
        # coordinates in which a physical transition was measured.
        i = np.arange(self.features,dtype=np.uint32)
        h = (i+np.uint32(1))*np.uint32(2654435761)
        h ^= h>>16
        bins = (h%self.rank).astype(np.intp)
        sign = 1-2*((h>>17)&1).astype(np.float32)
        return (np.bincount(bins,weights=x*sign,minlength=self.rank) /
                np.sqrt(self.features/self.rank)).astype(np.float32)

    @staticmethod
    def _motor(action):
        a = np.asarray(action,dtype=np.float32).ravel()
        if not len(a) or not np.isfinite(a).all():
            raise ValueError("expected finite motor input")
        if len(a)==1:
            return np.cos((a[0]+.5)*np.arange(1,4)*1.137).astype(np.float32)
        if len(a)==3:
            return np.tanh(a)
        return np.tanh(a @ np.cos(np.outer(np.arange(1,len(a)+1),np.arange(1,4))*1.137)).astype(np.float32)

    def _tick(self,traces,u):
        next_t = np.empty_like(traces)
        decay = np.exp(-1/np.array([1,4,16,64,256],np.float32))[:,None]
        next_t[:5] = decay*traces[:5] + (1-decay)*u
        angle = 2*np.pi*np.arange(1,self.bands+1,dtype=np.float32)[:,None]/512
        co,si = np.cos(angle)*np.exp(-1/384),np.sin(angle)*np.exp(-1/384)
        re,im = traces[5::2],traces[6::2]
        next_t[5::2] = co*re-si*im+u/32
        next_t[6::2] = si*re+co*im
        return next_t

    def _forced(self,traces):
        coefficients = self._part("force")
        return traces.ravel() @ coefficients[:-1]+coefficients[-1]

    @staticmethod
    def _regress(weights,precision,x,error):
        v = precision @ x
        gain = v / (1 + max(0.,float(x @ v)))
        weights += np.outer(gain,error)
        precision -= np.outer(gain,v)

    def _operator(self,u):
        r = self.rank
        terms = self._coupling[:4*r].reshape(4,r,r)
        delta = terms[0]+np.tensordot(u,terms[1:],axes=1)
        identity = np.eye(r,dtype=np.float32)
        skew = .5*(delta-delta.T)
        spin = np.linalg.solve(identity-.5*skew,identity+.5*skew)
        deformation = spin.T @ (identity+delta)
        eig, basis = np.linalg.eigh(.5*(deformation + deformation.T))
        stable = spin @ ((basis * eig.clip(0,0.995)) @ basis.T)
        bias = self._coupling[-1] + u @ self._coupling[-4:-1]
        return stable,bias

    def _next(self,x,z,traces,action):
        u = self._motor(action)
        following = self._tick(traces,u)
        old_force = self._forced(traces)
        operator,bias = self._operator(u)
        predicted = (z-old_force) @ operator + bias + self._forced(following)
        mean = self._part("mean")
        synthesis = self._part("synthesis")
        residual = x - mean - (z-self._project(mean)) @ synthesis
        level = mean + (predicted-self._project(mean)) @ synthesis
        return (level + .82*residual).clip(-.5,.5),predicted,following

    def _learn_transition(self,old,x,previous,following,u):
        a,b = self._project(old), self._project(x)
        dz = b-a
        read = self._part("synthesis")
        target = x-old
        if self.features>256:
            target = x-self._part("mean")
            dz = self._project(target)
            self._part("variance")[:] += target**2
            self._part("samples")[:] += 1
        self._regress(read,self._part("read_precision"),dz,target-dz@read)
        q = a-self._forced(previous)
        qnext = b-self._forced(following)
        phi = np.concatenate((q,(u[:,None]*q).ravel(),u,[1])).astype(np.float32)
        self._regress(self._coupling,self._part("motion_precision"),phi,qnext-q-phi@self._coupling)
        slow = np.append(following.ravel(),np.float32(1))
        self._regress(self._part("force"),self._part("force_precision"),slow,b-self._forced(following))

    def _write(self,x,mask):
        w, p = self._decoder,self._part("precision")
        z = self._settle(x,mask,np.zeros_like(self._h))
        residual = x-w@z
        if mask is not None:
            residual[~mask] = 0
        count = self.features if mask is None else int(mask.sum())
        key = p@self._key(residual)
        strength = min(1.,float(np.linalg.norm(residual))/np.sqrt(count)/.08)
        unexplained = min(1.,float(residual@residual)/max(1e-8,float(x[mask]@x[mask] if mask is not None else x@x)))
        novel = key*(strength/max(1e-8,float(np.linalg.norm(key))))
        z = self._bounded((1-unexplained)*z + novel)
        error = x-w@z
        if mask is not None:
            error[~mask] = 0
        v = p@z
        gain = z/(.04+float(z@z))
        if self.sleep:
            # Correct only interference perpendicular to the new activation:
            # the response to the present write stays exactly the same.
            free = p@v
            direction = free*((z@gain)/max(1e-8,float(z@free)))
            if z@free>1e-6 and np.linalg.norm(direction)<=2*np.linalg.norm(gain):
                gain = direction
        w += np.outer(error,gain)
        remaining = error*(1-z@gain)
        improvement = max(0.,float(error@error-remaining@remaining)/count)
        lock = min(1.,improvement/.007)**2
        used = np.clip(1-p.trace()/len(z),0,1)
        history = self._part("control")
        typical = float(history[3])
        exceptional = np.clip((improvement/(.001+typical)-1.5)/2,0,1)
        lock *= max(exceptional,1/(1+(used/.38)**10))
        history[3] += .035*(improvement-history[3])
        p -= (lock/(1e-5+max(0.,float(z@v)))) * np.outer(v,v)
        p *= .9999995
        p.flat[::len(z)+1] += .0000005
        self._h[:] = z

    def _flow_key(self,action):
        a = np.asarray(action,dtype=np.float32).ravel()
        u = self._motor(action)
        v = a if len(a)==3 else u
        phase = v[:,None]*np.pi*np.array([1,2],np.float32)
        return np.r_[1.,u,np.cos(phase).ravel(),np.sin(phase).ravel()].astype(np.float32)

    def _affine_step(self, affine, action):
        u = self._flow_key(action)
        dx, ds, uncertainty = u@self._part("flow")
        return np.array([affine[0]+np.exp(np.clip(affine[1],-1.8,1.8))*dx,
                         np.clip(affine[1]+ds,-1.8,1.8),affine[2]+np.clip(uncertainty,0.,.5)],np.float32)

    def _grid(self,x):
        return x[-int(np.prod(self.grid_shape)):].reshape(self.grid_shape)

    def _image(self,x,source,affine):
        self._grid(x)[:] = transport.render(source,self._grid(self._part("mean")),
                 self._part("chart"),float(self._part("origin")[0]),*affine)
        return x

    def _see_grid(self,x,action,mask):
        flags = self._part("control")
        origin = self._part("origin")
        chart = self._part("chart")
        affine = self._part("affine")
        retinal = self._part("retina")
        seen = None if mask is None else self._grid(mask)
        if seen is not None and not seen.any():
            if flags[0] and action is not None:
                affine[:] = self._affine_step(affine,action)
            self._image(x,retinal,affine)
            return
        complete = seen is None or seen.all()
        if flags[0] and action is not None:
            moved = self._affine_step(affine,action)
            if flags[1] and complete:
                delta, confidence = transport.register(retinal,self._grid(x))
                h,w = self.grid_shape[:2]
                zoom = np.exp(delta[1])
                cols = (np.arange(w)-(w-1)/2)*zoom+(w-1)/2+delta[0]
                rows = (np.arange(h)-(h-1)/2)*zoom+(h-1)/2
                shared = (cols>=0)&(cols<=w-1)
                truth = self._grid(x)[:,shared]
                warp = transport.sample(retinal,cols,rows)[:,shared]
                baseline = self._grid(self._part("mean")).mean(axis=1,keepdims=True)
                risk = np.mean((warp-truth)**2)/(.001+np.mean((truth-baseline)**2))
                delta = np.r_[delta,np.float32(min(.5,.6*risk))]
                u = self._flow_key(action)
                amp = np.sqrt(confidence)
                self._regress(self._part("flow"),self._part("flow_precision"),u*amp,
                              (delta-u@self._part("flow"))*amp)
                moved = self._affine_step(affine,action)
            origin[:] = (origin+moved[0])%(8*self.grid_shape[1])
        else:
            moved = np.zeros(3)
        retinal[:] = self._grid(x)
        if complete:
            transport.deposit(chart,retinal,float(origin[0]),moved[1])
        affine.fill(0)

    def observe(self,frame,action=None,*,visible=None):
        x,mask = self._sense(frame,visible)
        if visible is not None and np.all(visible):
            mask = None
        u = None if action is None else self._motor(action)
        self._write(x,mask)
        if mask is not None:
            x = np.where(mask,x,self._decoder@self._h).clip(-.5,.5)
        if self.grid_shape:
            self._see_grid(x,action,mask)
        control = self._part("control")
        anchor,traces = self._part("anchor"),self._part("traces")
        if u is not None and control[0]:
            following = self._tick(traces,u)
            if control[1] and mask is None:
                self._learn_transition(anchor,x,traces,following,u)
            traces[:] = following
        mean = self._part("mean")
        mean += (.025 if control[0] else 1.)*(x-mean)
        anchor[:] = x
        self._part("z")[:] = self._project(x)
        control[:2] = 1,mask is None

    def _output(self,x):
        values = (.5+x).clip(0,1)
        return np.repeat(values,self.group)[:self.width] if self.group!=1 else values.copy()

    def imagine(self,actions=()):
        x,z,traces = self._part("anchor"),self._part("z"),self._part("traces")
        affine = self._part("affine") if self.grid_shape else None
        for action in actions:
            x,z,traces = self._next(x,z,traces,action)
            if self.grid_shape:
                affine = self._affine_step(affine,action)
                self._image(x,self._part("retina"),affine)
        return self._output(x)

    def advance(self,action):
        x,z,t = self._next(self._part("anchor"),self._part("z"),self._part("traces"),action)
        if self.grid_shape:
            affine = self._part("affine")
            affine[:] = self._affine_step(affine,action)
            self._image(x,self._part("retina"),affine)
        self._part("anchor")[:] = x
        self._part("z")[:] = z
        self._part("traces")[:] = t
        self._part("control")[:2] = 1,0
        return self._output(x)

    def cut(self):
        if self.grid_shape:
            for name in ("retina","affine","origin","chart"):
                self._part(name).fill(0)
        for name in ("anchor","z","traces","control","h"):
            self._part(name).fill(0)

    def recall(self,cue,visible,actions=(),*,carry=True):
        x,mask = self._sense(cue,visible)
        h = self._settle(x,mask,self._h if carry else np.zeros_like(self._h))
        level = self._decoder@h
        if self.grid_shape:
            self._grid(level)[:] = transport.complete_pixels(self._grid(level)+.5,self._grid(x)+.5,self._grid(mask))-.5
        level[mask] = x[mask]
        traces = self._part("traces")
        if not carry:
            traces = np.zeros_like(traces)
        z = self._project(level)
        if self.grid_shape:
            source = self._grid(level).copy()
            affine = np.zeros(3,np.float32)
        for a in actions:
            level,z,traces = self._next(level,z,traces,a)
            if self.grid_shape:
                affine = self._affine_step(affine,a)
                if carry:
                    self._image(level,source,affine)
                else:
                    self._grid(level)[:] = transport.render(source,self._grid(self._part("mean")),
                        np.zeros_like(self._part("chart")),0.,*affine)
        return self._output(level)

    def predict_change(self,delta,visible):
        """Reconstruct a signed cross-sensor change without any temporal prior."""
        delta, visible = np.asarray(delta), np.asarray(visible)
        if delta.shape!=(self.width,) or visible.shape!=delta.shape or visible.dtype!=bool or not visible.any():
            raise ValueError("expected a component mask and signed cue")
        x,mask = self._sense(.5+.5*delta,visible)
        read = self._part("synthesis")
        design = read[:,mask]
        if self.features>256:
            n = max(1.,float(self._part("samples")[0]))
            precision = self._part("read_precision")
            covariance = np.linalg.inv(precision.astype(float))/n
            explained = ((covariance@read)*read).sum(axis=0)
            variance = self._part("variance")/n
            noise = np.maximum(variance-explained,.25*variance+.0002)
            weighted = design/noise[mask]
            gram = n*precision + weighted@design.T
            z = np.linalg.solve(gram,weighted@(2*x[mask]))
            # Small adaptation samples constrain the reliability of a marginal
            # regression on untrained combinations of sensory components.
            z *= n/(n+12*self.rank)
            return np.repeat((z@read).clip(-1,1),self.group)[:self.width]
        gram = design@design.T
        gram.flat[::len(gram)+1] += .002
        z = np.linalg.solve(gram,design@(2*x[mask]))
        return np.repeat((z@read).clip(-1,1),self.group)[:self.width]

    def _sense(self, frame, mask=None):
        x = np.asarray(frame)
        if x.shape != (self.width,):
            raise ValueError("wrong sensor shape")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != x.shape or not mask.any():
                raise ValueError("expected a nonempty boolean component mask")
            # Hidden sentinels (including NaNs) never enter the encoder.
            seen = x[mask]
        else:
            seen = x
        if not np.isfinite(seen).all() or np.any((seen < 0) | (seen > 1)):
            raise ValueError("normalize available sensor components to [0,1]")
        x = np.where(mask, x, 0) if mask is not None else x
        x = x.astype(np.float32, copy=False)
        if self.group == 1:
            return x - 0.5, mask
        starts = np.arange(0, self.width, self.group)
        sums = np.add.reduceat(x, starts)
        if mask is None:
            sizes = np.minimum(self.group, self.width - starts)
            return (sums / sizes).astype(np.float32) - 0.5, None
        counts = np.add.reduceat(mask, starts, dtype=np.int32)
        return (sums / np.maximum(counts, 1)).astype(np.float32) - 0.5, counts > 0

    @staticmethod
    def _bounded(z):
        return z * min(1.0, 3.0 / max(1e-9, float(np.linalg.norm(z))))

    def _settle(self, x, visible, prior):
        # Conjugate-gradient relaxation of a strictly convex quadratic.
        w = self._decoder if visible is None else self._decoder[visible]
        y = x if visible is None else x[visible]
        z = prior.copy()
        ridge = max(1e-6, len(y) * 2e-5)
        r = w.T @ (y - w @ z)
        p = r.copy()
        rr = float(r @ r)
        for _ in range(8):
            if rr < 1e-14:
                break
            wp = w @ p
            alpha = rr / max(1e-18, float(wp @ wp + ridge * (p @ p)))
            z += alpha * p
            r -= alpha * (w.T @ wp + ridge * p)
            new = float(r @ r)
            p = r + (new / rr) * p
            rr = new
        return self._bounded(z)

    def _key(self, error):
        # A fixed signed Fourier transform spreads innovations over all planes.
        i = np.arange(len(error), dtype=np.uint32)
        mixed = i * np.uint32(2654435761)
        sign = 1.0 - 2.0 * ((mixed ^ (mixed >> 13)) & 1).astype(np.float32)
        k = len(self._h)
        spectrum = np.fft.rfft(error * sign, n=max(k, len(error)))
        pick = np.linspace(1, len(spectrum)-1, k//2, dtype=int)
        wave = spectrum[pick]
        return np.stack((wave.real, wave.imag), axis=1).astype(np.float32).ravel()


    def save(self, path, *, binding=""):
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            name = tmp.name
        try:
            with open(name, "wb") as out:
                np.savez(out, version=3, config=[self.width, self.budget, self.sleep], binding=binding, grid=self.grid_shape, field=self.matrix)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, binding=""):
        with np.load(path, allow_pickle=False) as data:
            if "version" not in data or int(data["version"]) != 3 or data["binding"].item() != binding:
                raise ValueError("expected a v3 field for this body wiring")
            width, budget, sleep = map(int, data["config"])
            model = cls(width, budget, sleep=bool(sleep), grid_shape=tuple(map(int,data["grid"])) if "grid" in data else None)
            field = data["field"]
            if field.shape != model.matrix.shape or field.dtype != np.float32 or not np.isfinite(field).all():
                raise ValueError("invalid field matrix")
            model.matrix[:] = field
            return model
