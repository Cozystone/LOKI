"""Change floor geometry as well as relief mapping; evaluator-only layout and route."""

import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np
import torch
from loki_memory.place_bundles import PlaceBundles
from research.r2_places import GRID, SITES, START, DIRS, encode, timing, walk


def generator(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    f = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "t_room"
    )
    env = {"np": np}
    exec(compile(ast.Module(body=[f], type_ignores=[]), str(path), "exec"), env)
    return env["t_room"]


def live(s, grid, start, sites, seed, tour):
    rng = np.random.default_rng(seed)
    path = [start]
    order = rng.permutation(2)
    for k in order:
        path.extend(walk(grid, path[-1], sites[k])[1:])
    if tour:
        visited = set()

        def visit(p):
            visited.add(p)
            for k in rng.permutation(4):
                dx, dy = DIRS[k]
                q = p[0] + dx, p[1] + dy
                if grid[q[1]][q[0]] != "#" and q not in visited:
                    path.append(q)
                    visit(q)
                    path.append(p)

        visit(path[-1])
    est = np.zeros(2)
    prev = start
    distance = []
    delay = []
    first = None
    s.cut()
    moves = []
    for p in path:
        moves.append((p, False))
        if rng.random() < 0.12:
            dx, dy = DIRS[rng.integers(4)]
            q = p[0] + dx, p[1] + dy
            if grid[q[1]][q[0]] == "#":
                moves.append((p, True))
            else:
                moves.extend([(q, False), (p, False)])
    for i, (p, blocked) in enumerate(moves):
        est += np.subtract(p, prev) + rng.normal(0, 0.002, 2)
        prev = p
        hit = np.all(np.asarray(sites) == p, axis=1)
        delta = 0.80 * hit + rng.normal(0, 0.008, 2)
        b, _ = s.observe(
            torch.from_numpy(encode(est)[None]),
            torch.tensor(delta[None]),
            [float(blocked)],
        )
        if hit.any() and first is None:
            other = 1 - int(hit.argmax())
            _, r, _, pu = s.going(b)
            j = int(r[:, other].argmax())
            first = bool(
                r[j, other] > 0.15
                and np.linalg.norm(pu[j].numpy() + start - sites[other]) <= 1
            )
        if hit.any() and int(hit.argmax()) not in delay:
            k = int(hit.argmax())
            delay.append(k)
            candidates = (s.fast[:, k] > 0.15) & (s.visits > 0)
            d = torch.linalg.norm(s.Pu - torch.tensor(np.subtract(p, start)), dim=1)
            distance.append(float(d.masked_fill(~candidates, float("inf")).min()))
    return first, distance


def trial(seed, generate, supported_motion=False):
    grid, _ = generate(seed)
    where = lambda s: next(
        (x, y) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch == s
    )
    a = (grid, where("s"), [where("w"), where("f")])
    s = PlaceBundles(supported_motion=supported_motion)
    live(s, *a, seed, True)
    bank_a = s.sleep()
    alone, _ = live(copy.deepcopy(s), *a, seed + 55, False)
    distances = []
    banks = [bank_a]
    for k in range(2):
        _, d = live(s, GRID, START, SITES[k], seed + 1 + k, True)
        distances += d
        banks.append(s.sleep())
    returned, _ = live(copy.deepcopy(s), *a, seed + 55, False)
    return dict(
        seed=seed,
        alone=alone,
        returned=returned,
        banks=banks,
        distances=distances,
        recycled=int(s.moves),
        forgotten=int(s.forgotten),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    generate = generator(args.source)
    rows = [trial(i, generate) for i in range(29201, 29225)]
    data = dict(rows=rows, cpu1=timing(1), cpu64=timing(64))
    Path("measurements/r2_geometry.json").write_text(
        json.dumps(data, indent=2, allow_nan=False)
    )
    print(
        sum(r["alone"] for r in rows),
        sum(r["returned"] for r in rows),
        data["cpu1"],
        data["cpu64"],
    )
