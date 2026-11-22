"""Pair physically different hidden motions with identical published observations/actions."""
from copy import deepcopy
import numpy as np
import torch
from loki_memory import LokiMemory
from .reference import mse, setup, write


def main():
    setup()
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    torch.set_num_threads(1)
    rows = []
    for seed in range(2001,2013):
        rng = np.random.default_rng(seed)
        worlds = [World(1,make_room(seed),device="cpu",seed=seed,drain=0) for _ in range(2)]
        m = LokiMemory(96)
        for i in range(128):
            a = int(rng.integers(2))
            for w in worlds:
                w.step(torch.tensor([2+a]))
            m.observe(worlds[0].picture(8,4).numpy().ravel(),a)
        other = deepcopy(m)
        bounds, observed = [], []
        for t in range(20):
            a = int(rng.integers(2))
            for w in worlds:
                w.step(torch.tensor([2+a]))
            if t%4==3:
                worlds[1].step(torch.tensor([1]))
            pred = m.advance(a)
            np.testing.assert_array_equal(pred,other.advance(a))
            y,z = [w.picture(8,4).numpy().ravel() for w in worlds]
            bounds.append(mse(y,z)/4)
            observed.append((mse(pred,y)+mse(pred,z))/2)
        rows.append(dict(seed=seed,pair_error_lower_bound=float(np.mean(bounds)),v2_mse=float(np.mean(observed))))
    write("ambiguity",dict(rows=rows,mean_bound=float(np.mean([r["pair_error_lower_bound"] for r in rows]))))


if __name__=="__main__":
    main()
