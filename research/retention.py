import numpy as np
from loki_memory import LokiMemory
from .reference import mse, setup, write


def run():
    old, _, scenes, _, _ = setup()
    rows = []
    for seed in [522, *range(1601, 1613), *range(1901,1913)]:
        rng = np.random.default_rng(seed)
        d, tile = 65536, 64
        # Match v1 limits() exactly, including the preceding independent-noise test.
        for _ in range(13):
            rng.random(d, dtype=np.float32)
        waves = rng.normal(size=(8, tile)).astype(np.float32)
        waves -= waves.mean(axis=1, keepdims=True)
        waves /= np.sqrt((waves**2).mean(axis=1, keepdims=True))
        first = scenes(rng, 1, waves)[0]
        visible = np.arange(d//tile) % 4 == 0
        mask = np.repeat(visible, tile)
        cue = np.where(mask, first, np.nan)
        models = [old(d), LokiMemory(d), LokiMemory(d, sleep=False)]
        def judge():
            return {k: mse(m.recall(cue, v)[~mask], first[~mask]) for k,m,v in
                    zip(("v1", "v2", "no_sleep"), models, (visible, mask, mask))}
        for m in models:
            m.observe(first)
        before = judge()
        for i in range(240):
            frame = scenes(rng, 1, waves)[0]
            if i == 119:
                middle = frame.copy()
            for m in models:
                m.observe(frame, 1)
        after = judge()
        other = {tag:{k:mse(m.recall(np.where(mask,x,np.nan),v)[~mask],x[~mask]) for k,m,v in
                      zip(("v1","v2","no_sleep"),models,(visible,mask,mask))} for tag,x in (("middle",middle),("last",frame))}
        rows.append(dict(seed=seed, before=before, after=after, other=other,
                         untrained=mse(first[~mask], 0.5), state_bytes=models[1].state_bytes))
    summary = {}
    for group, subset in (("fresh",rows[1:13]),("sealed",rows[13:])):
        means = {k: float(np.mean([r["after"][k] for r in subset])) for k in rows[0]["after"]}
        wins = sum(r["after"]["v2"] < r["after"]["v1"] for r in subset)
        gate = rows[0]["after"]["v2"] <= .010 and means["v2"] <= min(.010, .5*means["v1"]) and wins >= 10
        summary[group] = dict(means=means,wins=wins,gate=gate)
    write("retention", dict(rows=rows,summary=summary))


if __name__ == "__main__":
    run()
