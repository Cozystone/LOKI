"""Layout-only replay. Extract the pure layout function without importing the life harness."""

import argparse
import ast
from collections import deque
from pathlib import Path
import json
import numpy as np
from loki_memory.walls import ContextWitness


def load_layout(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    fn = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "layout"
    )
    env = dict(np=np, S=16)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), env)
    return env["layout"]


def walk(grid, start, end):
    parents = {start: None}
    q = deque([start])
    while q:
        at = q.popleft()
        if at == end:
            break
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            xy = at[0] + dx, at[1] + dy
            if grid[xy[1]][xy[0]] != "#" and xy not in parents:
                parents[xy] = at
                q.append(xy)
    seq = [end]
    while seq[-1] != start:
        seq.append(parents[seq[-1]])
    return seq[::-1]


def measure(layout, rare=False, calibrated=False):
    grids = [layout(mirror=m)[0] for m in (False, True)]
    path = walk(grids[0], (8, 13), (1, 1)) + walk(grids[0], (1, 1), (14, 1))[1:]

    def rows(grid, trajectory=None):
        data = []
        for x, y in path if trajectory is None else trajectory:
            steps = []
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                k = 0
                while grid[y + (k + 1) * dy][x + (k + 1) * dx] != "#":
                    k += 1
                steps.append(k / 15)
            view = np.tile(steps, 3)
            data.append(
                (
                    view,
                    np.array([x, y]) / 15,
                    np.array([grid[y][x] == "w", grid[y][x] == "f"], float),
                )
            )
        return data

    first, second = map(rows, grids)
    contact = next(i for i, row in enumerate(first) if row[2].any())
    s = ContextWitness(rare=rare, calibrated=calibrated)
    for row in first:
        s.observe(*row)
    a = s.consolidate()

    def memory():
        hits = []
        for i, row in enumerate(first):
            if row[2].any():
                predicted = s.recall_edge(
                    a, first[i - 1][0], 0.5 + (row[1] - first[i - 1][1]) / 2
                )[1]
                hits.append(predicted.tolist())
        return hits

    before = memory()
    evidence = [s.observe(*row) for row in second]
    b = s.consolidate()

    def query(sequence):
        s.cut()
        return [s.observe(*row)[0] for row in sequence]

    reverse = walk(grids[0], (8, 13), (14, 1)) + walk(grids[0], (14, 1), (1, 1))[1:]
    nuisance = []
    for disturbances in (1, 2):
        seq = list(first)
        for i in (5, 10)[:disturbances]:
            x, p, r = seq[i]
            seq[i] = (x, p, np.ones_like(r))
        nuisance.append(query(seq)[-1])
    noisy = []
    if calibrated:
        for seed in range(28401, 28425):
            rng = np.random.default_rng(seed)
            for direction in (path, reverse):
                outputs = []
                for grid in grids:
                    trace = [
                        (np.clip(x + rng.normal(0, 0.01, x.shape), 0, 1), p, r)
                        for x, p, r in rows(grid, direction)
                    ]
                    outputs.append(query(trace)[-1])
                noisy.append(outputs)
    return dict(
        noisy_returns=noisy,
        returns=[query(first)[-1], query(second)[-1]],
        reversed_route=[query(rows(grid, reverse))[-1] for grid in grids],
        nuisance=nuisance,
        beats=len(path),
        first_contact=contact,
        first_bank=a,
        mirror_bank=b,
        approvals_before_contact=sum(k >= 0 for k, _ in evidence[:contact]),
        last_support=evidence[-1][1],
        split=bool(a != b),
        old_relief_before=before,
        old_relief_after=memory(),
        expected=[[1, 0], [0, 1]],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--rare", action="store_true")
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()
    result = measure(
        load_layout(args.source), args.rare or args.calibrated, args.calibrated
    )
    if args.rare or args.calibrated:
        from research.four_walls import context_trial

        result["fresh_ring"] = [
            context_trial(i, rare=True, calibrated=args.calibrated)
            for i in (range(28401, 28425) if args.calibrated else range(28301, 28325))
        ]
    name = (
        "walls_calibrated"
        if args.calibrated
        else "walls_rare"
        if args.rare
        else "walls_a3"
    )
    Path(f"measurements/{name}.json").write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "fresh_ring"})
