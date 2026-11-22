"""Signed, action-conditioned sensory effects shared across retinal columns."""
import numpy as np
from . import transport


def path(field, actions):
    """Integrate apparent heading in sensor columns; motor coordinates stay anonymous."""
    phase = float(field._part("affine")[0])
    drives, headings = [], []
    for a in actions:
        phase += float(field._flow_key(a) @ field._part("flow")[:,0])
        drives.append(field._motor(a))
        headings.append(phase)
    return np.asarray(drives),np.asarray(headings),phase


def features(field, left, right):
    height,width,channels = field.grid_shape
    size = field._part("effect_weights").shape[0]
    u,p,a = path(field,left)
    v,q,b = path(field,right)
    if len(u)!=len(v):
        raise ValueError("paired plans must have equal duration")
    if not len(u) or np.array_equal(u,v):
        return np.zeros((width,size),np.float32)
    angle = (p+q-a-b)*.5*(np.pi/2)/width
    delta = u-v
    motor = np.concatenate((delta.sum(0),np.cos(angle)@delta,np.sin(angle)@delta,
                            np.linspace(-1,1,len(delta))@delta))
    scale = float(np.linalg.norm(delta,axis=1).sum())
    sign = np.sign(motor[np.argmax(np.abs(motor))])
    if not sign:
        sign = np.sign(delta.ravel()[np.flatnonzero(delta)[0]])
    motion = sign*motor/max(1.,scale)
    gate = sign*np.tanh(scale/5)
    anchor = field._part("retina")
    mean = field._grid(field._part("mean"))
    chart = field._part("chart")
    phase = field._part("origin")[0]
    turn = .5*(a+b)
    facing = transport.render(anchor,mean,chart,phase,turn,0.,0.)
    def columns(view):
        return view.transpose(1,0,2).reshape(width,-1)
    col = np.arange(width)
    center = columns(facing)
    left_view = center[(col-2).clip(0,width-1)]
    right_view = center[(col+2).clip(0,width-1)]
    tiled = lambda vector: np.broadcast_to(vector,(width,len(vector)))
    context = np.column_stack((center,left_view,right_view,columns(anchor),
                 tiled(center.mean(0)),tiled(columns(anchor).mean(0)),tiled(motion),
                 (col+.5)/width-.5,tiled(np.array([np.sin(turn/width),np.cos(turn/width),np.tanh(scale/16)]))))
    # Fixed random features are defined by the algorithm and regenerated per call.
    # No retained generator state, feature bank of past inputs, or trained side array.
    rng = np.random.default_rng(8407)
    direct = np.column_stack((np.ones(width),center,left_view,right_view,tiled(motion)))
    count = min(size//2,direct.shape[1])
    projection = rng.standard_normal((context.shape[1],size-count),dtype=np.float32)
    bias = rng.uniform(-1,1,size-count)
    nonlinear = np.tanh(context@projection*.65+bias)
    return (gate*np.column_stack((direct[:,:count],nonlinear))).astype(np.float32)
