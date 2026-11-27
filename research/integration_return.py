from pathlib import Path
import json
from research.integration_gap import life
if __name__=='__main__':
    rows=[dict(seed=s,normal=life(s,True,True,True,False,prospective=True),
               return_walk=life(s,True,True,True,False,prospective=True,walk_return=True),
               zero_lag=life(s,True,True,True,False,current=True,prospective=True))
          for s in range(96500,96524)]
    print({k:sum(min(x[k][1]['tail'],x[k][2]['tail'])>=.7 for x in rows)
           for k in ('normal','return_walk','zero_lag')})
    Path('measurements/integration_return.json').write_text(json.dumps(rows,indent=2)+'\n')
