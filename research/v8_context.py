"""Paired v7/v8 round trip. Only the evaluator holds earlier target images."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from loki_memory.contextual import ContextMemory, Transition, _native
from loki_memory.portable import PortableMemory
from .v7_roundtrip import scene, camera
from .reference import ROOT


def run(seed, root, shared_motor=False):
    rng = np.random.default_rng(seed)
    looks = [scene(rng) for _ in range(8)]
    targets = np.stack([scene(rng) for _ in range(16)])
    old = [scene(rng) for _ in range(8)]
    old_targets = np.stack([scene(rng) for _ in old])
    laws = rng.uniform(-.32, .32, (32,8))
    motors = rng.permutation(8)
    if shared_motor:
        motors[:] = motors[0]
    def event(i, abroad):
        pair = i % 8
        polarity = rng.choice([-1, 1])
        command = np.zeros(8)
        command[motors[pair]] = polarity * rng.uniform(.91, 1)
        before = rng.uniform(.38,.62,8)
        after = before + command[motors[pair]]*laws[i] + rng.normal(0,.009,8)
        sensor = np.stack([before,(command+1)/2,after]).astype(np.float32)
        frame = camera(looks[pair],rng) if abroad else (
            looks[pair]+rng.normal(0,.006,looks[pair].shape)).clip(0,1).astype(np.float32)
        return frame, sensor
    first = np.arange(8) + 8*rng.integers(0,2,8)
    second = (first+8)%16
    train = {i:event(i,i in second) for i in range(16)}
    queries = {i:event(i,i in first) for i in range(16)}
    neg = [event(i,i%2==0) for i in range(16,32)]
    def trial(cls, path):
        ctx = cls is ContextMemory
        # Start from a genuine image-only database in both arms.
        m = PortableMemory.create(path,"A")
        for cue,target in zip(old,old_targets):
            m.cut()
            m.observe(target)
            m.observe(cue)
        before = [m.recall(x) for x in old]
        m = cls.resume(path,m.identity,"A")
        assert all(all(np.array_equal(a,b) for a,b in zip(pred,m.recall(x)))
                   for pred,x in zip(before,old))
        def cue(frame,h):
            return Transition(frame,h) if ctx else frame
        def learn(indices):
            for i in indices:
                frame,h = train[i]
                m.cut()
                with m.wake() as w:
                    w.observe(targets[i])
                    if ctx:
                        w.observe_transition(frame,h)
                    else:
                        w.observe(frame)
        def move(body, sample):
            c = cue(*sample)
            prev = m.recall(c)
            ident = m.transfer(body)
            nxt = cls.resume(path,ident,body)
            assert all(np.array_equal(a,b) for a,b in zip(prev,nxt.recall(c)))
            return nxt
        def score(indices, shuffle=False):
            rows = []
            for i in indices:
                frame,h = queries[i]
                if shuffle:
                    h = queries[(i+8)%16][1]
                pred,_,s = m.recall(cue(frame,h))
                errors = np.mean((targets-pred)**2,axis=(1,2,3))
                ok = errors.argmin()==i
                rows.append([ok,ok and s>=.65,s>=.65,errors[i]])
            return np.mean(rows,axis=0).tolist()
        learn(first)
        m = move("B",train[first[0]])
        outbound = score(first)
        learn(second)
        m = move("A",train[second[0]])
        legacy = np.stack([m.recall(x)[0] for x in old])
        distances = ((legacy[:,None]-old_targets)**2).mean(axis=(2,3,4))
        return dict(out=outbound,back=score(second),retained=score(first),
                    shuffled=score(range(16),True),
                    false_accept=np.mean([m.recall(cue(*n))[2]>=.65 for n in neg]),
                    legacy_top1=np.mean(distances.argmin(1)==np.arange(8)),
                    legacy_mse=distances.diagonal().mean().item(),state_bytes=m.state_bytes)
    return {cls.__name__:trial(cls, root/f"{seed}-{cls.__name__}") for cls in (PortableMemory,ContextMemory)}


def latency(root):
    rng = np.random.default_rng(8150)
    frames = rng.random((32,16,16,3),dtype=np.float32)
    histories = rng.random((32,3,8),dtype=np.float32)
    machines = [c.create(root/c.__name__,"A") for c in (PortableMemory,ContextMemory)]
    records = [[[],[]],[[],[]]]
    for batch in range(24):
        for idx in ([0,1] if batch%2 else [1,0]):
            with machines[idx].wake() as w:
                for f,h in zip(frames,histories):
                    t = perf_counter_ns()
                    if idx:
                        w.observe_transition(f,h)
                    else:
                        w.observe(f)
                    mid = perf_counter_ns()
                    w.recall(Transition(f,h) if idx else f)
                    end = perf_counter_ns()
                    if batch>=4:
                        records[idx][0].append((mid-t)/1000)
                        records[idx][1].append((end-mid)/1000)
    stats = {m.__class__.__name__:dict(zip(["write_us","read_us"],
            [np.percentile(r,[50,95]).tolist() for r in row])) for m,row in zip(machines,records)}
    batches = [[],[]]
    for batch in range(44):
        for idx in ([0,1] if batch%2 else [1,0]):
            start = perf_counter_ns()
            with machines[idx].wake() as w:
                for f,h in zip(frames[:16],histories[:16]):
                    if idx:
                        w.observe_transition(f,h)
                    else:
                        w.observe(f)
            if batch>=4:
                batches[idx].append((perf_counter_ns()-start)/16000)
    for m,r in zip(machines,batches):
        stats[type(m).__name__]["commit16_us_per_tick"] = np.percentile(r,[50,95]).tolist()
    return stats


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--final",action="store_true")
    args=parser.parse_args()
    with TemporaryDirectory() as d:
        seeds=range(8801,8825) if args.final else range(8101,8105)
        rows={s:run(s,Path(d)) for s in seeds}
        speed=latency(Path(d))
        stress={s:run(s,Path(d),True) for s in range(8901,8913)} if args.final else {}
    means={c:{k:np.mean([v[c][k] for v in rows.values()],axis=0).tolist()
              for k in rows[seeds[0]][c]} for c in rows[seeds[0]]}
    result=dict(native=_native is not None,columns=["top1","accepted_correct","accepted","mse"],
                means=means,latency=speed,rows=rows,shared_motor_rows=stress)
    print(json.dumps([means,speed],indent=1))
    if args.final:
        (ROOT/"measurements/v8_context.json").write_text(json.dumps(result,indent=2)+"\n")
