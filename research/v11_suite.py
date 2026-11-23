"""Independent measurements on the same candidate, without exposing simulator state."""

import json, sys
import numpy as np
from .reference import ROOT
from .v11_world import Maze, descriptor
from .v11_identity import run as identity
from .v11_detour import run as detour
from .v11_alias import probe
from .v10_world import run as v10
from loki_memory.belief import BeliefModel


def extra(seed, rev):
    rng = np.random.default_rng(seed)
    w = Maze(seed)
    other = Maze(seed + 9900)

    def table(world):
        return {
            (*p, h): descriptor(world, (*p, h))
            for p in sorted(world.edges)
            for h in range(4)
        }

    ds = table(w)
    foreign = table(other)
    ps = list(ds)
    m = BeliefModel(rev)
    for i in range(1000):
        p = ps[rng.integers(len(ps))]
        a = rng.integers(4)
        y = ds[w.step(p, a)]
        m.fit(ds[p], a, y, adapt=i < 800)
        if i >= 800:
            delta = m.features(ds[p], np.zeros(4)) @ m.theta[a]
            m.cross[a] += [delta @ (y - ds[p]), delta @ delta]
    trans = m.probabilities()
    mse = []
    reach = []
    cf = []
    fps = list(foreign)
    for _ in range(16):
        p = fps[rng.integers(len(fps))]
        x = foreign[p]
        y = x.copy()
        b = m.infer(x)
        last = np.zeros(4)
        row = []
        target = foreign[fps[rng.integers(len(fps))]]
        go = m.infer(target)
        observed = []
        for a in range(4):
            prior = b @ trans[a]
            nxt = foreign[other.step(p, a)]
            cf.append(
                [
                    float(-sum(prior * np.log(prior + 1e-20))),
                    float(prior @ go - b @ go),
                    float(prior @ m.infer(nxt)),
                    float(np.mean((m.forecast(x, a, last, b) - nxt) ** 2)),
                ]
            )
        for j in range(4):
            a = int(rng.integers(4))
            p = other.step(p, a)
            y = m.forecast(y, a, last, b)
            b = b @ trans[a]
            last = np.eye(4)[a]
            row.append([np.mean((y - foreign[p]) ** 2), np.mean((x - foreign[p]) ** 2)])
            observed.append(float(np.mean((target - foreign[p]) ** 2) < 1e-9))
        reach.append([float(b @ go), observed[-1]])
        mse.append(row)
    # Kidnap after one chosen action. Same-room orientation/position and a different room.
    kid = []
    for mode in range(5):
        world = w
        p = ps[rng.integers(len(ps))]
        goal = w.start
        b = m.infer(ds[p])
        q, t = m.policy(ds[goal])
        m.attempts.fill(0)
        detection = reloc = None
        for j in range(32):
            a = m.choose(b, q, t)
            prior = b @ t[a]
            p = world.step(p, a)
            if j == 0:
                if mode < 3:
                    p = (*p[:2], (p[2] + (1, -1, 2)[mode]) % 4)
                elif mode == 3:
                    p = ps[rng.integers(len(ps))]
                else:
                    world = other
                    p = fps[rng.integers(len(fps))]
            evidence = (ds if mode < 4 else foreign)[p]
            if detection is None and prior @ m.infer(evidence) < 0.05:
                detection = j
            b = m.infer(evidence, prior)
            # Distinguish visual indistinguishability: unique exact emission needed.
            equals = [
                pose
                for pose, v in ds.items()
                if np.array_equal(v, m.emission[np.argmax(b)])
            ]
            if reloc is None and mode < 4 and equals == [p]:
                reloc = j
            if mode < 4 and p[:2] == goal[:2]:
                break
        kid.append([detection, reloc, int(mode < 4 and p[:2] == goal[:2])])
    return dict(
        multistep=np.mean(mse, axis=0).tolist(),
        candidate=np.mean(cf, 0).tolist(),
        endpoint_brier=float(np.mean([(p - y) ** 2 for p, y in reach])),
        kidnap=kid,
    )


if __name__ == "__main__":
    rev = int(sys.argv[1])
    start = 11101 + 100 * rev
    data = []
    for s in range(start, start + 24):
        base = v10(s)
        data.append(
            dict(
                seed=s,
                alias=probe(s, rev),
                identity=identity(s, rev),
                detour=detour(s, rev),
                v10=base,
                extra=extra(s, rev),
            )
        )
    (ROOT / f"measurements/v11_suite_{rev}.json").write_text(
        json.dumps(data, indent=1, default=float)
    )
