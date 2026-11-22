import numpy as np
import torch
from loki_memory import LokiMemory
from loki_memory.atanor import AtanorMemory, pool_image
from .reference import mse, setup, write


def lift(grid):
    return grid.repeat(8, axis=0).repeat(8, axis=1).ravel()


def room(seed, old, fifo_type):
    from atanor_os.emerge.room_body import RoomBody
    from atanor_os.emerge.room_vary import condition
    motor, judge = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    torch.manual_seed(seed)
    world = RoomBody(1, *condition(seed), seed=seed, device="cpu")
    adapter = AtanorMemory(world.c)
    multi = adapter.memory
    actions = motor.integers(0, np.prod(world.c.motors), size=160)
    images, stream = [], []
    for t in range(161):
        if t:
            world.step(torch.tensor([actions[t-1]]))
        raw = world.raw()
        stream.append(adapter.features(raw))
        images.append(raw[-1][0].numpy().copy())
    shape = images[0].shape
    grid = pool_image(images[0])[0].shape
    nvis = int(np.prod(grid))
    nbody = multi.width-nvis
    varying = np.ptp(stream,axis=0)[:nbody] > 1e-6
    vmask = np.arange(multi.width) >= nbody
    v1multi = old(multi.width)
    v1image = old(images[0].size)
    visual = LokiMemory(nvis)
    fifo = fifo_type(shape)
    stats = {}
    def add(key, a, b):
        stats.setdefault(key, []).append(mse(a,b))
    for i, target in enumerate(stream):
        frame = images[i].astype(np.float32)/255
        cmd = int(actions[i-1]) if i else None
        if i >= 41:
            vis = np.zeros(v1image.tiles, bool)
            vis[judge.choice(len(vis), len(vis)//4, replace=False)] = True
            mask = np.repeat(vis, 64)
            partial = np.where(mask, frame.ravel(), np.nan)
            small, avail = pool_image(partial.reshape(shape),mask.reshape(shape))
            rendered = visual.recall(small.ravel(),avail.ravel()).reshape(grid)
            for name, estimate in (("v2", lift(rendered)), ("cue_pool",lift(np.where(avail,small,.5))),
                                   ("v1",v1image.recall(partial, vis)), ("fifo",fifo.recall(partial,mask))):
                add("raw_hidden_"+name,estimate[~mask],frame.ravel()[~mask])
            for name, estimate in (("v2",lift(visual.imagine([cmd]).reshape(grid))),
                                   ("v1",v1image.imagine([cmd])),("last",images[i-1].ravel()/255)):
                add("raw_next_"+name,estimate,frame.ravel())
            # v1 publicly decodes each live address; this judge extends its
            # nearest-neighbor readout to exactly the same component mask as v2.
            candidates = np.stack([v1multi.imagine((),start=s) for s in range(v1multi.used)])
            for mask_name, known in (("body_cue",~vmask),("vision_cue",vmask)):
                cue = np.where(known,target,np.nan)
                pred = multi.recall(cue,known)
                standalone = multi.recall(cue,known,carry=False)
                baseline = candidates[np.square(candidates[:,known]-target[known]).mean(axis=1).argmin()]
                for label, val in (("v1",baseline),("v2",pred),("cue_alone",standalone),("prior_alone",multi.imagine())):
                    add(mask_name+"_"+label,val[~known], target[~known])
                    if mask_name=="vision_cue":
                        add("vision_cue_dynamic_"+label,val[:nbody][varying],target[:nbody][varying])
            for horizon in (1,8):
                end = i-1+horizon
                if end > 160:
                    continue
                plan = actions[i-1:end]
                for name, pred in (("v2",multi.imagine([adapter.motor(a) for a in plan])),
                                   ("v1",v1multi.imagine(plan)),("last",stream[i-1])):
                    add(f"future{horizon}_vision_{name}",pred[nbody:],stream[end][nbody:])
                    add(f"future{horizon}_body_{name}",pred[:nbody],stream[end][:nbody])
                    add(f"future{horizon}_dynamic_{name}",pred[:nbody][varying],stream[end][:nbody][varying])
        multi.observe(target,adapter.motor(cmd) if i else None)
        v1multi.observe(target,cmd)
        visual.observe(target[nbody:],cmd)
        v1image.observe(frame.ravel(),cmd)
        fifo.observe(frame.ravel())
    return dict(seed=seed, state_bytes=multi.state_bytes, dynamic_body_channels=int(varying.sum()),
                mse={k:float(np.mean(v)) for k,v in stats.items()})


def main():
    old, _, _, _, fifo = setup()
    torch.set_num_threads(1)
    result = {}
    for group,seeds,needed in (("original",range(401,409),7),("fresh",range(1201,1213),10),
                               ("sealed",range(1801,1813),10)):
        rows = [room(s,old,fifo) for s in seeds]
        means = {k:float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]}
        gates = {}
        for metric,controls in (("raw_hidden",("v1","fifo")),("raw_next",("v1","last"))):
            gates[metric] = {}
            for c in controls:
                k, b = metric+"_v2",metric+"_"+c
                wins = sum(r["mse"][k] < r["mse"][b] for r in rows)
                gates[metric][c] = dict(wins=wins,passed=wins>=needed and means[k]<=.9*means[b])
        result[group] = dict(rows=rows,mean_mse=means,gates=gates)
    write("rooms",result)


if __name__ == "__main__":
    main()
