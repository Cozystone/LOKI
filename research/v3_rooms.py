"""Held-out RoomBody trajectories with independently scored static, action and delta queries."""
import numpy as np
import torch
from loki_memory import LokiMemory
from loki_memory.atanor import AtanorMemory, pool_image
from .reference import mse, setup, write
from .rooms import lift
from .v3_common import compare, prior_version


def room(seed, v2, fifo_type):
    from atanor_os.emerge.room_body import RoomBody
    from atanor_os.emerge.room_vary import condition
    motor, judge, pairs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    torch.manual_seed(seed)
    world = RoomBody(1, *condition(seed), seed=seed, device="cpu")
    adapter = AtanorMemory(world.c)
    commands = motor.integers(np.prod(world.c.motors), size=160)
    images, stream = [], []
    for i in range(161):
        if i:
            world.step(torch.tensor([commands[i-1]]))
        raw = world.raw()
        stream.append(adapter.features(raw))
        images.append(raw[-1][0].numpy().copy())
    grid = pool_image(images[0])[0].shape
    nvis = int(np.prod(grid))
    width = len(stream[0])
    visual = {"v2":v2(nvis), "v3":LokiMemory(nvis)}
    joint = {"v2":v2(width), "v3":adapter.memory}
    fifo = fifo_type(images[0].shape)
    scores = {}

    def score(key, name, y, target):
        scores.setdefault(key, {}).setdefault(name, []).append(mse(y, target))

    for i, (frame, full) in enumerate(zip(images, stream)):
        x = frame.astype(np.float32)/255
        cmd = int(commands[i-1]) if i else None
        if i > 40:
            tiles = np.zeros(x.size//64, bool)
            tiles[judge.choice(len(tiles), len(tiles)//4, replace=False)] = True
            known = np.repeat(tiles, 64)
            partial = np.where(known, x.ravel(), np.nan)
            small, present = pool_image(partial.reshape(x.shape), known.reshape(x.shape))
            for name, model in visual.items():
                y = model.recall(small.ravel(), present.ravel()).reshape(grid)
                score("hidden", name, lift(y)[~known], x.ravel()[~known])
                score("next", name, lift(model.imagine([cmd]).reshape(grid)), x.ravel())
            score("hidden", "fifo", fifo.recall(partial, known)[~known], x.ravel()[~known])
            score("hidden", "cue", lift(np.where(present, small, .5))[~known], x.ravel()[~known])
            score("next", "last", images[i-1].ravel()/255, x.ravel())
        for model in visual.values():
            model.observe(full[-nvis:], cmd)
        fifo.observe(x.ravel())
        if i <= 40:
            for model in joint.values():
                model.observe(full, adapter.motor(cmd) if i else None)

    vision = np.arange(width) >= width-nvis
    for _ in range(120):
        a = int(pairs.integers(41,161))
        b = int(pairs.choice([j for j in range(41,161) if abs(j-a)>=16]))
        delta = stream[b]-stream[a]
        for metric, known in (("body_to_vision", ~vision), ("vision_to_body", vision)):
            cue = np.where(known, delta, np.nan)
            pred = joint["v3"].predict_change(cue, known)
            score(metric,"v3",pred[~known],delta[~known])
            for name, model in joint.items():
                results = [model.recall(np.where(known,stream[j],np.nan),known,carry=False) for j in (a,b)]
                label = "v2" if name=="v2" else "v3_static"
                score(metric,label,(results[1]-results[0])[~known],delta[~known])
            score(metric,"zero",0,delta[~known])
    return dict(seed=seed,**{k:{n:float(np.mean(v)) for n,v in by.items()} for k,by in scores.items()})


def run(seeds=range(3301,3313), file="v3_rooms"):
    *_, fifo = setup()
    torch.set_num_threads(1)
    rows = [room(s, prior_version(), fifo) for s in seeds]
    controls = dict(hidden=("v2","fifo"),next=("v2","last"),
                    body_to_vision=("v2","zero"),vision_to_body=("v2","zero"))
    write(file, dict(rows=rows, gates={k:compare(rows,k,c,.1,10) for k,c in controls.items()}))


if __name__=="__main__":
    run()
