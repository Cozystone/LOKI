"""Continuous affine motion on a declared rectangular sensor lattice."""
import numpy as np


def sample(grid, columns, rows):
    h,w = grid.shape[:2]
    x,y = columns.clip(0,w-1),rows.clip(0,h-1)
    i,j = x.astype(int),y.astype(int)
    dx,dy = (x-i)[None,:,None],(y-j)[:,None,None]
    line = (1-dx)*grid[:,i]+dx*grid[:,np.minimum(i+1,w-1)]
    return ((1-dy)*line[j]+dy*line[np.minimum(j+1,h-1)]).astype(np.float32)


def register(before, after):
    h,w = before.shape[:2]
    if np.array_equal(before,after):
        return np.zeros(2,np.float32),1.
    # Compare only shared support. The overlap and velocity penalties break
    # ambiguous alignments on uniform or nearly uniform surfaces.
    x,y = np.arange(w)-(w-1)/2,np.arange(h)-(h-1)/2
    shifts = np.arange(-max(1,w//3),max(1,w//3)+.25,.5)
    best = np.inf
    identity = np.mean((after-before)**2)
    result = (0.,0.)
    for scale in (-.13,0.,.13):
        cols = x[None,:]*np.exp(scale)+(w-1)/2+shifts[:,None]
        inside = (cols>=0)&(cols<=w-1)
        pred = sample(before,cols.ravel(),y*np.exp(scale)+(h-1)/2)
        err = ((pred.reshape(h,len(shifts),w,-1)-after[:,None])**2).mean(axis=(0,3))
        loss = ((err*inside).sum(axis=1)/np.maximum(1,inside.sum(axis=1))
                +.00015*(np.abs(shifts)/w+abs(scale))+.003*identity*(~inside).mean(axis=1))
        i = int(loss.argmin())
        if loss[i]<best:
            best = loss[i]
            result = (shifts[i],scale)
    return np.array(result,np.float32),float(np.clip(1-best/(identity+.00001),.05,1))


def harmonics(columns,width,bands):
    phase = np.outer(columns,np.arange(1,bands+1))*2*np.pi/(8*width)
    return np.column_stack((np.ones(len(columns)),np.cos(phase),np.sin(phase))).astype(np.float32)


def deposit(chart,grid,origin,log_scale):
    h,w,c = grid.shape
    coords = np.arange(w)+origin
    basis = harmonics(coords,w,(len(chart)-1)//2)
    signals = grid.transpose(1,0,2).reshape(w,h*c)
    chart *= .95*np.exp(-3*abs(log_scale))
    chart[:,:-1] += basis.T @ signals
    chart[:,-1] += basis.sum(axis=0)


def render(anchor, mean, chart, origin, shift, stretch, risk):
    h,w,c = anchor.shape
    s = np.exp(np.clip(stretch,-1.8,1.8))
    x = (np.arange(w)-(w-1)/2)*s+(w-1)/2+shift
    y = (np.arange(h)-(h-1)/2)*s+(h-1)/2
    image = sample(anchor,x,y)
    bands = (len(chart)-1)//2
    taper = 2*(1-np.arange(1,bands+1)/(bands+1))
    basis = harmonics(x+origin,w,bands)*np.r_[1,taper,taper]
    integrated = basis@chart
    density = np.maximum(0,integrated[:,-1])
    back = mean.mean(axis=1)[:,None,:]
    history = integrated[:,:-1].reshape(w,h,c).transpose(1,0,2)
    history = (history+back*bands*.35)/(density[None,:,None]+bands*.35)
    history = sample(history,np.arange(w),y)
    distance = np.maximum(np.maximum(-x,x-w+1),0)
    cover = np.exp(-1.2*distance)[None,:,None]
    confidence = np.exp(-min(6.,risk))
    return (confidence*(cover*image+(1-cover)*history)+(1-confidence)*mean).clip(-.5,.5)


def complete_pixels(prior, cue, visible):
    """Interpolate the visible residual along both native sensor axes."""
    prior, cue, visible = np.asarray(prior), np.asarray(cue), np.asarray(visible)
    if prior.shape != cue.shape or visible.shape != cue.shape or visible.dtype != bool:
        raise ValueError("expected aligned HWC prediction, raw cue and mask")
    residual = np.where(visible,cue-prior,0)
    total = np.zeros_like(residual)
    weight = np.full_like(residual,.04)
    for axis in (0,1):
        shape = [1,1,1]
        n = shape[axis] = cue.shape[axis]
        index = np.arange(n).reshape(shape)
        left = np.maximum.accumulate(np.where(visible,index,-n),axis=axis)
        right = np.flip(np.minimum.accumulate(np.flip(np.where(visible,index,2*n),axis=axis),axis=axis),axis=axis)
        for near in (left,right):
            valid = (near>=0)&(near<n)
            w = valid/(1+np.abs(index-near))**1.5
            total += w*np.take_along_axis(residual,near.clip(0,n-1),axis=axis)
            weight += w
    return np.where(visible,cue,(prior+total/weight).clip(0,1))


