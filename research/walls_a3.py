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


def measure(layout):
    grids = [layout(mirror=m)[0] for m in (False, True)]
    path = walk(grids[0], (8, 13), (1, 1)) + walk(grids[0], (1, 1), (14, 1))[1:]

    def rows(grid):
        data = []
        for x, y in path:
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
    s = ContextWitness()
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
    return dict(
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
    result = measure(load_layout(ap.parse_args().source))
    Path("measurements/walls_a3.json").write_text(json.dumps(result, indent=2))
    print(result)
