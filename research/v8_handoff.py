"""Paired boundary experiment against the original v8 raw writer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from loki_memory import BodyMemory, ContextMemory, Transition
from loki_memory.portable import transaction
from .reference import ROOT


def field(memory):
    with transaction(memory.path) as db:
        return db.execute("SELECT field FROM loki7 WHERE id=?", (memory.identity,)).fetchone()[0]


def identical(a, b):
    return all(np.array_equal(x, y) for x,y in zip(a,b))


def trial(seed, root):
    rng = np.random.default_rng(seed)
    frames = rng.random((6,16,16,3),dtype=np.float32)
    h, delayed, local = rng.random((3,3,8),dtype=np.float32)
    oldcue = Transition(frames[1], h)
    newcue = Transition(frames[5], local)
    raw = ContextMemory.create(root/"raw","laptop")
    live = BodyMemory.create(root/"live","laptop")
    with raw.wake() as w:
        w.observe(frames[0], h[0])
        w.observe_transition(frames[1], h)
    def start(m, i, history):
        return m.begin_action(frames[i],history[0],history[1])
    assert live.finish_action(start(live,0,h),frames[1],h[2])
    prior = field(raw)
    assert prior == field(live)
    recall = live.recall(oldcue)
    late = start(live,2,delayed)

    def switch(m, host):
        return type(m).resume(m.path,m.transfer(host),host)

    raw = switch(raw,"glasses")
    live = switch(live,"glasses")
    retained = field(live)==prior and identical(recall,live.recall(oldcue))
    # Replay v8 with destination's first sample as the final row of the foreign history.
    mixed = np.stack((delayed[0],delayed[1],local[0]))
    with raw.wake() as w:
        w.observe_transition(frames[4],mixed,local[0])
    cold = field(raw)!=prior
    with raw.wake() as w:
        w.observe_transition(frames[4],mixed,local[0])
    primed = field(raw)!=prior
    assert not live.finish_action(late,frames[4],local[0])
    fresh = start(live,4,local)
    assert not live.finish_action(late,frames[4],local[0])
    protected = field(live)==prior
    assert live.finish_action(fresh,frames[5],local[2])
    # Independent uninterrupted v8 writes only the two legitimate episodes.
    clean = ContextMemory.create(root/"clean","ref")
    for i,hist in ((0,h),(4,local)):
        clean.cut()
        with clean.wake() as w:
            w.observe(frames[i],hist[0])
            w.observe_transition(frames[i+1],hist)
    live = switch(live,"laptop")
    assert not live.finish_action(late,frames[3],delayed[2])
    equivalent = field(clean)==field(live) and all(
        identical(live.recall(cue),clean.recall(cue)) for cue in (oldcue,newcue))
    return dict(raw_cold_changed=cold,raw_primed_changed=primed,
                old_recall_preserved=retained,unwanted_write_prevented=protected,
                both_match_clean_after_return=equivalent)


if __name__=="__main__":
    rows={}
    for seed in range(9501,9525):
        with TemporaryDirectory() as d:
            rows[seed]=trial(seed,Path(d))
    counts={k:sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    (ROOT/"measurements/v8_handoff.json").write_text(json.dumps(dict(counts=counts,rows=rows),indent=2)+"\n")
    print(counts)
    assert list(counts.values())==[0,24,24,24,24]
