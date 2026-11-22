"""Export training-only matrix and a self-contained, read-only evaluation fixture."""
import numpy as np
import torch
from .reference import ROOT, setup
from .dark import compound
from .v4_forward import plans,simulate
from loki_memory import LokiMemory


def export(model,seed):
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    (ROOT/"checkpoints").mkdir(exist_ok=True)
    (ROOT/"examples").mkdir(exist_ok=True)
    model.cut()
    model.save(ROOT/"checkpoints/v4-room-grid.npz")
    frames,actions = compound(seed,128,5)
    world = World(1,make_room(seed),device="cpu",seed=seed,drain=0)
    for a in actions:
        world.step(torch.tensor([(int(a)+1)%5]))
    np.testing.assert_array_equal(world.picture(8,4).numpy().ravel(),frames[-1])
    cmd = plans(np.random.default_rng(seed+17000),1,[4],[0])[:,0]
    endpoints = [simulate(world,arm[None])[0,-1] for arm in cmd]
    np.savez_compressed(ROOT/"examples/forward_idle.npz",prefix=frames,observed_actions=actions,
                        plans=cmd,endpoints=endpoints,seed=seed)


if __name__=="__main__":
    from pathlib import Path
    import subprocess
    setup()
    torch.set_num_threads(1)
    directory = Path(subprocess.check_output(["git","rev-parse","--absolute-git-dir"],text=True).strip())
    export(LokiMemory.load(directory/"v4-training/v4.npz"),5101)
