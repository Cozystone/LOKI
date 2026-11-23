"""Physical return in a first-person 3D maze. Positions are evaluator-only."""
from collections import defaultdict
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from loki_memory import ReverseMemory
from .reference import ROOT, write


DIRECTIONS = ((0,-1),(1,0),(0,1),(-1,0))
FORWARD,BACK,LEFT,RIGHT = range(4)
INVERSE = (BACK,FORWARD,RIGHT,LEFT)
LEVELS = np.array([.35,.8],np.float32)


class Maze:
    def __init__(self,seed):
        rng = np.random.default_rng(seed)
        self.edges = defaultdict(set)
        # A randomized spanning tree with junctions, reproducible independently of placement.
        visited = {(0,0)}
        frontier = []
        def expand(p):
            for dx,dy in DIRECTIONS:
                q = p[0]+dx,p[1]+dy
                if 0<=q[0]<5 and 0<=q[1]<5 and q not in visited:
                    frontier.append((p,q))
        expand((0,0))
        while frontier:
            p,q = frontier.pop(int(rng.integers(len(frontier))))
            if q not in visited:
                visited.add(q)
                self.edges[p].add(q)
                self.edges[q].add(p)
                expand(q)
        paths=[]
        def visit(path):
            if len(path)>=6 and sum(len(self.edges[p])>=3 for p in path[1:-1])>=2:
                paths.append(path)
            if len(path)<9:
                for q in sorted(self.edges[path[-1]]):
                    if q not in path:
                        visit(path+[q])
        for start in sorted(visited):
            visit([start])
        if not paths:
            raise ValueError("no route meets preregistered geometry")
        path=paths[int(rng.integers(len(paths)))]
        self.floor=np.zeros((11,11),bool)
        for (x,y),neighbors in self.edges.items():
            self.floor[2*y+1,2*x+1]=True
            for xx,yy in neighbors:
                self.floor[y+yy+1,x+xx+1]=True
        self.palette=rng.uniform(.22,.88,(6,3))
        self.material=rng.integers(0,6,(11,11))
        self.start=(*path[0],DIRECTIONS.index((path[1][0]-path[0][0],path[1][1]-path[0][1])))
        self.guide=[]
        pose=self.start
        for q in path[1:]:
            heading=DIRECTIONS.index((q[0]-pose[0],q[1]-pose[1]))
            while pose[2]!=heading:
                a=RIGHT if (heading-pose[2])%4<=2 else LEFT
                self.guide.append(a)
                pose=self.step(pose,a)
            self.guide.append(FORWARD)
            pose=self.step(pose,FORWARD)
        self.view=lru_cache(maxsize=100)(self._render)

    def step(self,pose,a):
        x,y,h=pose
        if a in (LEFT,RIGHT):
            return x,y,(h+(1 if a==RIGHT else -1))%4
        dx,dy=DIRECTIONS[h]
        sign=1 if a==FORWARD else -1
        q=x+sign*dx,y+sign*dy
        return (*q,h) if q in self.edges[x,y] else pose

    def actions(self,p):
        return [a for a in range(4) if self.step(p,a)!=p]

    def _render(self,p):
        # Pinhole rays. Intersect textured vertical walls, the floor and the ceiling.
        x,y,h=p
        origin=np.array([2*x+1.5,2*y+1.5])
        forward=np.array(DIRECTIONS[h])
        right=np.array(DIRECTIONS[(h+1)%4])
        horizontal=forward + np.linspace(-.9,.9,16)[:,None]*right
        vz=np.linspace(.72,-.72,16)
        plane=(.8/np.abs(vz))[:,None]
        image=np.empty((16,16,3),np.float32)
        for col,ray in enumerate(horizontal):
            cell=origin.astype(int)
            step=np.sign(ray).astype(int)
            with np.errstate(divide="ignore"):
                stride=np.abs(1/ray)
            boundary=.5*stride
            while self.floor[cell[1],cell[0]]:
                axis=int(np.argmin(boundary))
                depth=boundary[axis]
                boundary[axis]+=stride[axis]
                cell[axis]+=step[axis]
            point=origin+ray*depth
            wall=depth<=plane[:,0]
            z=.8+depth*vz
            brick=(np.floor(point[1-axis]*4+np.floor(z*5)*.5)%3==0)
            pigment=self.palette[self.material[cell[1],cell[0]]]
            image[wall,col]=pigment*(.63+.18*axis+.12*brick[wall,None])/(1+.035*depth)
            on_plane=origin+plane*ray
            checker=np.floor(on_plane*2).sum(axis=1)%2
            neutral=np.where(vz[:,None]>0,[[.59,.62,.66]],[[.35,.31,.25]])
            surface=neutral*(.75+.17*checker[:,None])
            image[~wall,col]=surface[~wall]
        return image


def prepare(world,seed):
    model=ReverseMemory()
    warm=Maze(seed+5000)
    rng=np.random.default_rng(seed+8000)
    p=warm.start
    model.observe(warm.view(p),LEVELS)
    for _ in range(96):
        a=int(rng.choice(warm.actions(p)))
        p=warm.step(p,a)
        model.observe(warm.view(p),LEVELS,a)
    clean=ReverseMemory()
    clean.motor_map[:]=model.motor_map
    clean.motor_p[:]=model.motor_p
    model=clean
    p=world.start
    poses=[p]
    model.observe(world.view(p),LEVELS)
    for a in world.guide:
        p=world.step(p,a)
        poses.append(p)
        model.observe(world.view(p),LEVELS,a)
    model.cut()
    shuffled=ReverseMemory()
    offset=int(rng.integers(1,len(poses)-1))
    for i in range(len(poses)-1):
        shuffled.cut()
        shuffled.observe(world.view(poses[(i+offset)%(len(poses)-1)]),LEVELS)
        shuffled.observe(world.view(poses[i+1]),LEVELS,world.guide[i])
    shuffled.cut()
    return model,shuffled,poses


def best(values,rng):
    v=np.asarray(values)
    return int(rng.choice(np.flatnonzero(v>=v.max()-1e-9)))


def voyage(world,poses,model,shuffled,mode,seed,flipped=False):
    p=poses[-1]
    if flipped:
        p=(*p[:2],(p[2]+2)%4)
    rng=np.random.default_rng(seed)
    track=[p]
    scanning=0
    for i in range(2*len(world.guide)):
        acts=world.actions(p)
        if mode=="random":
            a=int(rng.choice(acts))
        elif mode=="oracle":
            a=INVERSE[world.guide[-i-1]]
        elif mode=="model":
            a=acts[best(model.return_scores(world.view(p),acts),rng)]
        else:
            glimpses=[]
            for candidate in acts:
                trial=world.step(p,candidate)
                glimpses.append(world.view(trial))
                assert world.step(trial,INVERSE[candidate])==p
                scanning+=2
            if mode=="copy":
                scores=[-np.mean((im-world.view(p))**2) for im in glimpses]
            else:
                organ=shuffled if mode=="shuffled" else model
                scores=organ.return_scores(world.view(p),acts,glimpses)
            a=acts[best(scores,rng)]
        p=world.step(p,a)
        track.append(p)
        if p[:2]==poses[0][:2]:
            break
    return dict(success=int(p[:2]==poses[0][:2]),decisions=len(track)-1,
                scan_steps=scanning,first_inverse=int(track[1]==poses[-2])),track


def plate(world,guide,tracks,model):
    from .v6 import display
    image=Image.new("RGB",(980,430),"#faf8f2")
    draw=ImageDraw.Draw(image)
    draw.text((16,12),"First-person 3D return   /   preregistered maze 7201",fill="#273238")
    views=(world.view(guide[0]),world.view(guide[-2]),world.view(guide[-1]),model.recall(world.view(guide[-1])))
    for i,(im,label) in enumerate(zip(views,("Departure","Previous view (ground truth)","Current view at turnaround","Raw LOKI recall"))):
        image.paste(display(im).resize((160,160),Image.Resampling.NEAREST),(16+i*239,40))
        draw.text((16+i*239,208),label,fill="#273238")
    for i,(label,steps) in enumerate([("Guided outward route",guide),*tracks.items()]):
        left=16+i*239
        scale=14
        for y,x in np.ndindex(world.floor.shape):
            color="#edece5" if world.floor[y,x] else "#505967"
            draw.rectangle((left+x*scale,235+y*scale,left+(x+1)*scale-1,235+(y+1)*scale-1),fill=color)
        xy=[(left+(2*x+1.5)*scale,235+(2*y+1.5)*scale) for x,y,_ in steps]
        draw.line(xy,fill="#b45028",width=3)
        for p,color in ((guide[0],"#167053"),(guide[-1],"#ad3e36")):
            x,y,_=p
            xx,yy=left+(2*x+1.5)*scale,235+(2*y+1.5)*scale
            draw.ellipse((xx-4,yy-4,xx+4,yy+4),fill=color)
        draw.text((left,398),label,fill="#273238")
    image.save(ROOT/"examples/v6-maze.png")


def measure():
    rows=[]
    modes=("model","scan","shuffled","copy")
    for seed in range(7201,7225):
        world=Maze(seed)
        model,shuffled,poses=prepare(world,seed)
        for flipped in (False,True):
            scores={}
            tracks={}
            for mode in (*modes,*(() if flipped else ("oracle",))):
                scores[mode],track=voyage(world,poses,model,shuffled,mode,seed,flipped)
                tracks[mode]=track
            random=[voyage(world,poses,model,shuffled,"random",seed*100+i,flipped)[0] for i in range(64)]
            scores["random"]={k:float(np.mean([r[k] for r in random])) for k in random[0]}
            rows.append(dict(seed=seed,flipped=flipped,guide_len=len(world.guide),scores=scores))
            if seed==7201 and not flipped:
                plate(world,poses,{"Predicted moves":tracks["model"],"Physical scan":tracks["scan"],
                                   "Shuffled memory":tracks["shuffled"]},model)
    summary={}
    for flipped in (False,True):
        g=[r["scores"] for r in rows if r["flipped"]==flipped]
        summary[str(flipped)]={m:{k:float(np.mean([r[m][k] for r in g])) for k in g[0][m]} for m in g[0]}
    s=summary["False"]
    return dict(rows=rows,summary=summary,passed=s["model"]["success"]>=.75 and s["model"]["success"]-s["random"]["success"]>=.2)


if __name__=="__main__":
    result=measure()
    write("v6_maze",result)
    print(result["summary"])
