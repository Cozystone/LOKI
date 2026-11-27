"""Registered follow-up, keeping the original result file intact."""
import itertools
import json
from pathlib import Path

import numpy as np

from research.integration_gap import life


def diagnostic(rng):
    # Likelihood of two anonymous observations under each hypothesis for each intervention.
    # Each costs the same now; at the next choice different body acts may be needed.
    informative=int(rng.integers(2))
    likelihood=np.full((2,2,2),.5)
    likelihood[informative]=np.eye(2)
    def costs(lik, future):
        # Joint weights include equal prior for each latent explanation.
        # For each possible observation take the best future bodily outcome.
        harms=np.array([[.1,.8],[.8,.1]])**2 if future else np.zeros((2,2))
        return np.array([(w.T@harms*.5).min(axis=1).sum() for w in lik])
    active=costs(likelihood,True)
    inert=costs(likelihood,False)
    return dict(informative=informative,selected=int(active.argmin()),
                reversed=int(costs(likelihood[::-1],True).argmin()),
                no_future_tie=bool(inert[0]==inert[1]),costs=active.tolist())


if __name__=='__main__':
    rows=[]
    for s in range(96300,96324):
        cases={}
        for prospective in (False,True):
            for g,q,c in itertools.product((0,1),repeat=3):
                cases[f'{int(prospective)}-{g}{q}{c}']=life(s,True,g,q,c,prospective=prospective)
            cases[f'{int(prospective)}-isolated']=life(s,False,True,False,False,prospective=prospective)
        rows.append(dict(seed=s,cases=cases,diagnostic=diagnostic(np.random.default_rng(s))))
    for k in rows[0]['cases']:
        cases=[r['cases'][k] for r in rows]
        print(k,[sum(x[i]['tail']>=.7 for x in cases) for i in (1,2)],
              sum(min(x[1]['tail'],x[2]['tail'])>=.7 for x in cases))
    Path('measurements/integration_revision.json').write_text(json.dumps(rows,indent=2)+'\n')
