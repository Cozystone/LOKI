"""Judge-side public A3 layout replay. No coordinates or world label cross the organ API."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import torch
from loki_memory.place_bundles import PlaceBundles
from research.walls_a3 import walk

GRID = [
    "################",
    "#.....####.....#",
    "#.....####.....#",
    "#.....####.....#",
    "#.....####.....#",
    "##..########..##",
    "##..########..##",
    "#..............#",
    "#..............#",
    "####..#..#..####",
    "####..#..#..####",
    "####..#..#..####",
    "#######..#######",
    "#######..#######",
    "################",
    "################",
]
START = (8, 13)
SITES = [[(1, 1), (14, 1)], [(14, 1), (1, 1)], [(5, 4), (10, 4)], [(1, 4), (14, 4)]]
CELLS = [(x, y) for y, row in enumerate(GRID) for x, ch in enumerate(row) if ch == "."]
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def encode(xy):
    # Exact five-scale PathOrgan formula applied by the simulated BODY to its integrated motion.
    q = np.asarray(xy) @ np.array([[1.0, 0.5, -0.5], [0, 0.8660254, 0.8660254]])
    scale = np.repeat([2, 4, 8, 16, 32], 3)
    phase = np.tile(q, 5) * 2 * np.pi / scale
    return np.r_[
        np.cos(phase) * scale / (2 * np.pi), np.sin(phase) * scale / (2 * np.pi)
    ].astype("f4")


def route(rng, world, tour, reverse):
    dest = list(SITES[world])
    if reverse:
        dest.reverse()
    pts = [START]
    for p in dest:
        pts.extend(walk(GRID, pts[-1], p)[1:])
    if tour:
        # A depth-first walk of all accessible locations from this spring.
        seen = set()

        def visit(at):
            seen.add(at)
            for j in rng.permutation(4):
                dx, dy = DIRS[j]
                q = at[0] + dx, at[1] + dy
                if q in seen or q not in CELLS:
                    continue
                pts.append(q)
                visit(q)
                pts.append(at)

        visit(pts[-1])
    return pts


def inputs(seed, world, tour=False, reverse=False):
    rng = np.random.default_rng(seed)
    path = route(rng, world, tour, reverse)
    pos = np.zeros(2)
    old = START
    rows = []
    for p in path:
        pos += np.subtract(p, old)
        old = p
        code = encode(pos) + rng.normal(0, 0.002, 30)
        change = rng.normal(0, 0.01, 2) + 0.8 * np.all(
            np.asarray(SITES[world]) == p, axis=1
        )
        # Equal nominal movement error distribution in the three geometrically identical rooms.
        rows.append(
            tuple(
                torch.tensor(np.asarray(a), dtype=torch.float32)
                for a in (code[None], change[None], [abs(rng.normal(0.015, 0.005))])
            )
        )
    return path, rows


def read(s, b, sites):
    _, r, _, pu = s.going(b)
    hits = []
    for k, p in enumerate(sites):
        j = int(r[:, k].argmax())
        hits.append(
            bool(r[j, k] > 0.15 and np.linalg.norm(pu[j].numpy() + START - p) <= 1)
        )
    return np.mean(hits)


def replay(s, data, world):
    path, rows = data
    s.cut()
    choices = []
    dedicated = [None, None]
    for i, (p, args) in enumerate(zip(path, rows)):
        b, _ = s.observe(*args)
        choices.append(b)
        for k, site in enumerate(SITES[world]):
            if dedicated[k] is not None or site not in path[: i + 1]:
                continue
            active = s.visits > 0
            near = (
                ((s.Pu - torch.tensor(np.subtract(site, START))).norm(dim=1) <= 1)
                & active
                & (s.fast[:, k] > 0.15)
            )

            # Must win cue assignment at the spring, but not at its doorway or corridor end.
            def unit(q):
                dist = (
                    (s.C - torch.from_numpy(encode(np.subtract(q, START))))
                    .square()
                    .sum(1)
                )
                return int(dist.masked_fill(~active, float("inf")).argmin())

            door = (2 if site[0] < 8 else 13, 5)
            corridor = (door[0], 8)
            j = unit(site)
            if near[j] and j not in (unit(door), unit(corridor)):
                dedicated[k] = i - path.index(site)
    return choices, dedicated


def trial(seed):
    data = [inputs(seed + w, w, True, bool(seed % 2)) for w in range(3)]
    probes = [inputs(seed + 121 + w, w, False, not bool(seed % 2)) for w in range(3)]

    def unseen_target(store, probe, w):
        ss = copy.deepcopy(store)
        ss.cut()
        for xy, args in zip(*probe):
            chosen, _ = ss.observe(*args)
            if xy in SITES[w]:
                other = 1 - SITES[w].index(xy)
                _, r, _, pu = ss.going(chosen)
                j = int(r[:, other].argmax())
                return float(
                    r[j, other] > 0.15
                    and np.linalg.norm(pu[j].numpy() + START - SITES[w][other]) <= 1
                )

    def run(pooled):
        s = PlaceBundles(pooled=pooled)
        labels = []
        near = []
        base = None
        for w in range(3):
            _, d = replay(s, data[w], w)
            near.extend(d)
            labels.append(s.sleep())
            if w == 0:
                base_unseen = unseen_target(s, probes[0], 0)
                alone = copy.deepcopy(s)
                choices, _ = replay(alone, probes[w], w)
                base = read(alone, choices[-1], SITES[w])
        choices = []
        scores = []
        for w in range(3):
            ss = copy.deepcopy(s)
            v, _ = replay(ss, probes[w], w)
            choices.append(v)
            scores.append(read(ss, v[-1], SITES[w]))
        first = []
        lat = []
        early = []
        correct = []
        for w in (0, 1):
            contact = min(probes[w][0].index(p) for p in SITES[w])
            wanted = labels[w]
            valid = wanted >= 0 and labels[0] != labels[1]
            first.append(contact)
            v = choices[w]
            early.append(bool(valid and v[8] == wanted))
            correct.append(bool(valid and v[-1] == wanted))
            lat.append(
                next(
                    (j for j, x in enumerate(v[contact:]) if x == wanted and valid),
                    None,
                )
            )
        # A fourth remapping must reuse existing capacity. Preserve the three-world judge above.
        stress = copy.deepcopy(s)
        fourth = inputs(seed + 403, 3, True)
        replay(stress, fourth, 3)
        fourth_id = stress.sleep()
        fourth_return = unseen_target(stress, probes[0], 0)
        # No reset hint between these two readouts. Return to birth through contiguous steps.
        continuous = copy.deepcopy(s)
        continuous.cut()
        outward = inputs(seed + 77, 0)[1]
        for args in outward:
            continuous.observe(*args)
        for args in outward[-2::-1]:
            continuous.observe(*args)
        initial = int(continuous.reading)
        check = []
        for args in probes[1][1]:
            check.append(continuous.observe(*args)[0])
        contact = min(probes[1][0].index(p) for p in SITES[1])
        post = next((i for i, c in enumerate(check[contact:]) if c == labels[1]), None)
        # Exact twin has identical history but the external judge assigns a new identity.
        twin_wrong = choices[0][-1] >= 0
        glitch = copy.deepcopy(s)
        glitch.cut()
        for args in outward:
            glitch.observe(*args)
        before = int(glitch.splits)
        p, change, err = outward[-1]
        glitch.observe(p, 1 - change.clamp(0, 1), err)
        return dict(
            heldout_site_baseline=base_unseen,
            heldout_site_return=unseen_target(s, probes[0], 0),
            fourth_bundle=fourth_id,
            evictions=int(stress.forgotten),
            after_fourth=fourth_return,
            unannounced_initial=initial,
            unannounced_post=post,
            exact_twin_wrong=twin_wrong,
            nuisance_splits=int(glitch.splits) - before,
            base=base,
            returned=scores[0],
            all_worlds=scores,
            labels=labels,
            dedicated=near,
            first_contact=first,
            early=early,
            final=correct,
            post_contact=lat,
            recycled=int(s.moves),
            bytes=s.state_bytes,
            active=int((s.visits > 0).sum()),
        )

    # Give the partition baseline the correct map even on the first beat.
    ps = []
    for w, n in enumerate((21, 21, 22)):
        s = PlaceBundles(slots=n, bundles=1, pooled=True)
        replay(s, data[w], w)
        b = s.sleep()
        ps.append(read(s, b, SITES[w]))
    return dict(seed=seed, partial=run(False), pooled=run(True), oracle_partition=ps)


def timing(n):
    s = PlaceBundles(bodies=n)
    rng = np.random.default_rng(29000)
    elapsed = []
    exports = []
    sleeps = []
    codes = torch.tensor(np.array([encode(np.subtract(p, START)) for p in CELLS]))
    for i in range(320):
        p = codes[rng.integers(len(codes), size=n)]
        r = torch.tensor(
            rng.choice([0.0, 0.8], size=(n, 2), p=[0.97, 0.03]), dtype=torch.float32
        )
        err = torch.full((n,), 0.01)
        t = perf_counter_ns()
        b, _ = s.observe(p, r, err)
        f = perf_counter_ns()
        s.going(b)
        g = perf_counter_ns()
        if i >= 32:
            elapsed.append((f - t) / 1e6)
            exports.append((g - f) / 1e6)
        if i % 40 == 39:
            start = perf_counter_ns()
            s.sleep()
            sleeps.append((perf_counter_ns() - start) / 1e6)
    return dict(
        p95_ms=float(np.percentile(elapsed, 95)),
        export_p95_ms=float(np.percentile(exports, 95)),
        sleep_max_ms=max(sleeps),
        bytes=s.state_bytes,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    results = [trial(s) for s in (range(29101, 29125) if args.sealed else [29000])]
    out = dict(rows=results, cpu1=timing(1), cpu64=timing(64))
    dest = Path("measurements") / ("r2_sealed.json" if args.sealed else "r2_dev.json")
    dest.write_text(json.dumps(out, indent=2))
    print(results[0], out["cpu64"])
