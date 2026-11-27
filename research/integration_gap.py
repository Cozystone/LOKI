"""python -m research.integration_gap"""
import itertools
import json
from pathlib import Path

import numpy as np

from loki_memory.bout_outcomes import BoutOutcomes


def life(seed, integrated, distinct, calibrated, cost, current=False, prospective=False, walk_return=False):
    rng=np.random.default_rng(seed)
    wire=rng.permutation(5)
    # Ordinals and the world's causal wiring are harness-private.
    true=int(seed%2)+1
    delay=(8,14)[(seed//2)%2]
    model=BoutOutcomes(5)
    pending=np.zeros(21)
    d=.55
    background=0.
    clock=0
    rows=[]
    for t in range(480):
        phase=t//160
        right=true if phase<2 else 3-true
        pred,support,lag=model.forecast(current)
        ids=wire if integrated else wire[:4]
        estimate=pred[ids].copy()
        if not distinct:
            estimate[:4]=pred[wire[0]]
        if integrated:
            # External place organ always promises a large bodily improvement.
            # Fade that claim toward its actually measured motor outcome.
            trust=1-support[wire[4]] if calibrated else 1.
            estimate[-1]=trust*(-.19)+(1-trust)*estimate[-1]
        # Fixed paired exploration noise; this is NOT epistemic agency.
        u,proposal,jitter=rng.random(),rng.random(),rng.random(5)
        durations=int(rng.integers(4,12))
        at_arrival = d + lag*background if prospective else d
        expected=np.clip(at_arrival+estimate,0,1)
        arm=int(np.argmin(expected**2+1e-9*jitter[:len(ids)]))
        if u<.16:
            arm=int(proposal*len(ids))
        motor=wire[arm]
        observed=motor if distinct or arm==4 else wire[0]
        if phase==0:
            relief=.12 if arm==3 else 0.
        else:
            relief=.13 if arm==right else (.032 if arm==3 else 0.)
        if walk_return and phase==2:
            relief=.13 if arm==4 else 0.
        pending[(clock+delay)%len(pending)]+=relief
        saturated=0
        for beat in range(durations):
            old=d
            i=clock%len(pending)
            d=float(np.clip(d+.006+(.008 if cost and arm==4 else 0)-pending[i],0,1))
            pending[i]=0
            model.observe(observed if beat==0 else -1,d-old)
            saturated+=int(d==1 and old<1)
            if d>old and d<1:
                background+=.05*(d-old-background)
            clock+=1
        rows.append((arm,lag,saturated))
    result=[]
    for phase in range(3):
        part=rows[phase*160:(phase+1)*160]
        correct=3 if phase==0 else (true if phase==1 else 3-true)
        if walk_return and phase==2:
            correct=4
        hit=np.array([r[0]==correct for r in part])
        first=next((t+10 for t in range(151) if sum(hit[t:t+10])>=7),None)
        result.append(dict(tail=float(hit[-40:].mean()),correct=int(hit.sum()),
                           walk=sum(r[0]==4 for r in part),saturations=sum(r[2] for r in part),
                           seven_of_ten=first,lag_accuracy=float(np.mean([r[1]==delay for r in part[-40:]]))))
    return result


def probe(seed):
    # Same immediate deficits, novelty, action costs and neutral mean prediction.
    # Two supplied hypotheses differ only on the diagnostic intervention.
    rng=np.random.default_rng(seed)
    direction=int(rng.integers(2))
    models=np.full((2,2),.5)
    models[:,direction]=[.2,.8]
    neutral=int(rng.integers(2))
    selected=int(models.var(axis=0).argmax())
    return dict(diagnostic=direction,neutral=neutral,disagreement=selected,
                reversed=int(models[:,::-1].var(axis=0).argmax()))


if __name__=='__main__':
    rows=[]
    for s in range(96100,96124):
        cases={'isolated':life(s,False,True,False,False),
               'current':life(s,True,True,True,False,True)}
        for g,q,c in itertools.product((0,1),repeat=3):
            cases[f'{g}{q}{c}']=life(s,True,g,q,c)
        rows.append(dict(seed=s,cases=cases,probe=probe(s)))
    for k in rows[0]['cases']:
        a=[r['cases'][k] for r in rows]
        print(k,[sum(p[i]['tail']>=.7 for p in a) for i in (1,2)],
              sum(min(p[1]['tail'],p[2]['tail'])>=.7 for p in a),
              round(np.mean([p[2]['walk'] for p in a]),1))
    print('neutral',sum(r['probe']['neutral']==r['probe']['diagnostic'] for r in rows))
    Path('measurements/integration_gap.json').write_text(json.dumps(rows,indent=2)+'\n')
