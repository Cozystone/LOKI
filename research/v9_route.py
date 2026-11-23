import tempfile
from pathlib import Path
import json
import numpy as np
from loki_memory import GraphMemory
from .v6_maze import Maze
from .reference import ROOT


def run(seed):
    with tempfile.TemporaryDirectory() as d:
        m=GraphMemory.create(Path(d)/"route","maze")
        actions=np.eye(8,dtype="f4")[:4]
        signals=np.zeros(8,"f4")
        rng=np.random.default_rng(seed)
        for training in (True,False):
            world=Maze(seed+5000 if training else seed)
            p=world.start
            m.cut()
            with m.wake() as w:
                for j in range(96 if training else len(world.guide)):
                    a=rng.choice(world.actions(p)) if training else world.guide[j]
                    q=world.step(p,a)
                    assert w.finish_action(w.begin_action(world.view(p),signals,actions[a]),world.view(q),signals)
                    p=q
        plan=m.reversal_plan(96+len(world.guide))
        hits=[]
        for turn in (0,2):
            q=(*p[:2],(p[2]+turn)%4)
            for cmd in plan or []:
                q=world.step(q,int(np.argmax(cmd)))
                if q[:2]==world.start[:2]: break
            hits.append(q[:2]==world.start[:2])
        return hits


if __name__=="__main__":
    rows=[run(s) for s in range(7201,7225)]
    print(np.sum(rows,axis=0))
    (ROOT/"measurements/v9_route.json").write_text(json.dumps(rows)+"\n")
