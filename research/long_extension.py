"""Execute separately after the registered comparison."""
import json
from pathlib import Path

import torch

from research.long_endpoint import run

if __name__ == '__main__':
    torch.set_num_threads(1)
    rows = [dict(seed=s, **run(s,True,extended=True)) for s in range(8500,8524)]
    Path('measurements/long_extension.json').write_text(json.dumps(rows,indent=2)+'\n')
    for model in ('one','20','shuffled','end'):
        swap = 'swap' if model=='end' else model+'_swap'
        print(model,sum(r['trials'][model]['arrived'] and r['trials'][swap]['arrived'] for r in rows))
    print('future',sum(r['trials']['future']['first'] for r in rows),'dead',
          [sum(r['trials'][k]['dead'] for r in rows) for k in ('future','delayed')])
    print('bytes',rows[0]['bytes'],'p95 range',
          [(min(r[k] for r in rows),max(r[k] for r in rows)) for k in ('full_p95','sleep_p95')])
