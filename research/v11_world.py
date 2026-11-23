import argparse
import copy
import json
import numpy as np
from .reference import ROOT
from .v10_world import Maze, descriptor
from loki_memory.belief import BeliefModel


def cyclic(seed):
    world = Maze(seed)
    rng = np.random.default_rng(seed + 71000)
    gaps = [
        (p, (p[0] + dx, p[1] + dy))
        for p in sorted(world.edges)
        for dx, dy in ((1, 0), (0, 1))
        if (p[0] + dx, p[1] + dy) in world.edges
        and (p[0] + dx, p[1] + dy) not in world.edges[p]
    ]
    for idx in rng.choice(len(gaps), 2, replace=False):
        p, q = gaps[idx]
        world.edges[p].add(q)
        world.edges[q].add(p)
        world.floor[p[1] + q[1] + 1, p[0] + q[0] + 1] = True
    world.palette = world.palette[:, [2, 0, 1]] * 0.72 + 0.1
    world.palette[:, 2] = 0.65 * world.palette[:, 2] + 0.35 * world.palette[:, 0]
    return world


def measure(seed, rev, acquisition=None, holdout=False, controls=False, compact=False):
    rng = np.random.default_rng(seed)

    def sample(world):
        poses = [(*xy, h) for xy in sorted(world.edges) for h in range(4)]
        return poses, {p: descriptor(world, p) for p in poses}

    factory = cyclic if holdout else Maze
    m = BeliefModel(rev)
    warm = factory(seed + 5000)
    ps, ds = sample(warm)
    for tick in range(600):
        p = ps[rng.integers(len(ps))]
        a = rng.integers(4)
        y = ds[warm.step(p, a)]
        if rev >= 5 and tick >= 480:
            delta = m.features(ds[p], np.zeros(4)) @ m.theta[a]
            m.cross[a] += [delta @ (y - ds[p]), delta @ delta]
        else:
            m.fit(ds[p], a, y, topology=False)
    w = factory(seed)
    ps, ds = sample(w)
    loss = []
    for p in ps:
        for a in range(4):
            y = ds[w.step(p, a)]
            loss.append(
                [
                    np.mean((z - y) ** 2)
                    for z in (m.forecast(ds[p], a, np.zeros(4)), ds[p])
                ]
            )
    observed = {}
    if controls:
        from loki_memory.lifelong import LifelongMemory

        baseline = LifelongMemory()
    stream_rng = np.random.default_rng(seed + 801)
    mode = acquisition or (
        "active" if rev >= 13 else "balanced" if rev == 12 else "random"
    )
    for tick in range(1000):
        p = ps[rng.integers(len(ps))]
        a = rng.integers(4)
        if mode == "balanced":
            if tick % 4 == 0:
                origin = ps[stream_rng.integers(len(ps))]
                actions = stream_rng.permutation(4)
            p = origin
            a = actions[tick % 4]
        if mode == "active" and m.n:
            a = m.explore(ds[p], stream_rng)
        if mode == "pairs":
            if tick % 2:
                p = target
            else:
                incoming = None
        target = w.step(p, a)
        observed[p, a] = target
        if controls:
            baseline.transition(ds[p], a, ds[target])
        i, _ = m.fit(
            ds[p],
            a,
            ds[target],
            adapt=False,
            incoming=incoming if mode == "pairs" else None,
        )
        incoming = (i, a)
    p = w.start
    guide = [p]
    incoming = None
    for a in w.guide:
        q = w.step(p, a)
        if controls:
            baseline.transition(ds[p], a, ds[q])
        observed[p, a] = q
        i, _ = m.fit(ds[p], a, ds[q], adapt=False, incoming=incoming)
        incoming = (i, a)
        p = q
        guide.append(p)

    def travel(p, g, mask=False, jump=None, block=None, closed=True):
        model = copy.deepcopy(m)
        current = ds[p].copy()
        if mask:
            current[:6] = 0
        if not compact:
            q, t = model.policy(ds[g])
        b = model.infer(current)
        trace = [p]
        masses = []
        bad = 0
        delay = None
        for j in range(32):
            if p[:2] == g[:2]:
                break
            if compact:
                a, prior = model.compact_step(b, ds[g])
            else:
                if rev >= 16 and j:
                    q, t = model.policy(ds[g])
                a = model.choose(b, q, t)
                prior = b @ t[a]
            nxt = w.step(p, a)
            if block and {p[:2], nxt[:2]} == block:
                nxt = p
            if jump is not None and j == 0:
                nxt = jump
            evidence = ds[nxt].copy()
            if mask and j == 0:
                evidence[:6] = 0
            mass = (
                float(prior @ model.infer(evidence))
                if mask or not closed
                else float(prior[int(model.emission_cost(evidence).argmin())])
            )
            masses.append(mass)
            if mass < (0.98 if rev >= 4 else 0.05):
                bad += 1
                if delay is None:
                    delay = j
                if rev >= 1 and closed:
                    model.repair(current, a, evidence)
                    if not compact:
                        q, t = model.policy(ds[g])
                    if len(prior) != model.n:
                        prior = np.pad(prior, (0, model.n - len(prior)))
            b = model.infer(evidence, prior) if closed else prior
            current = evidence
            p = nxt
            trace.append(p)
        return dict(
            goal=g,
            success=int(p[:2] == g[:2]),
            steps=len(trace) - 1,
            bad=bad,
            delay=delay,
            mass=float(np.mean(masses)) if masses else None,
            trace=trace,
        )

    starts = [(*p[:2], (p[2] + k) % 4) for k in (0, 1, -1, 2)] + [
        (*sorted(w.edges[p[:2]])[0], p[2]),
        p,
    ]
    out = [travel(s, w.start, mask=i == 5) for i, s in enumerate(starts)]
    unseen = [q for q in ps if q[:2] not in [r[:2] for r in guide]]
    pairs = [
        (a, b)
        for a in unseen
        for b in unseen
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2
    ]
    a, g = pairs[rng.integers(len(pairs))]
    if controls:

        def baseline_trial(random):
            pos = a
            for _ in range(32):
                if pos[:2] == g[:2]:
                    break
                if random:
                    action = rng.integers(4)
                else:
                    plan = baseline.plan(ds[pos], ds[g])
                    if not plan:
                        break
                    action = plan[0]
                pos = w.step(pos, action)
            return int(pos[:2] == g[:2])

        return dict(
            seed=seed,
            novel=travel(a, g)["success"],
            v10=baseline_trial(False),
            random=baseline_trial(True),
        )

    def supported(start, goal):
        frontier = {start}
        seen = {start}
        for tick in range(33):
            if any(x[:2] == goal[:2] for x in frontier):
                return tick
            frontier = {
                observed[x, act]
                for x in frontier
                for act in range(4)
                if (x, act) in observed
            } - seen
            seen.update(frontier)
        return None

    ceiling = [supported(s, w.start) for s in starts] + [supported(a, g)]
    novel = travel(a, g)
    kidnapped = [travel(p, w.start, jump=s) for s in starts[1:5]]
    return dict(
        seed=seed,
        ceiling=ceiling,
        transfer=np.mean(loss, 0).tolist(),
        occupied=m.n,
        recovery=[r["success"] for r in out],
        novel=novel,
        open_loop=travel(a, g, closed=False)["success"],
        kidnap=[(r["delay"], r["success"]) for r in kidnapped],
        rotated=out[3],
        # A spanning tree loses all paths between endpoints across a cut edge.
        tree_edges=sum(map(len, w.edges.values())) // 2,
        tree_vertices=len(w.edges),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rev", type=int)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    start = 11001 - args.rev * 100 if args.dev else 11101 + args.rev * 100
    rows = [measure(s, args.rev) for s in range(start, start + (1 if args.dev else 24))]
    print(
        np.sum([r["recovery"] for r in rows], 0),
        sum(r["novel"]["success"] for r in rows),
        np.mean([r["transfer"] for r in rows], 0),
    )
    if not args.dev:
        (ROOT / f"measurements/v11_world_{args.rev}.json").write_text(
            json.dumps(rows, indent=1)
        )
