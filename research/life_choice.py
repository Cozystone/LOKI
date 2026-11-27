"""Run from repository root with python -m research.life_choice."""
import copy
import json
import platform
import statistics
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from loki_memory.embodied_choice import EmbodiedChoice

START, GOAL = (1, 2), (5, 2)
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]
OLD = {START, GOAL, (1, 1), (5, 1)} | {(x, 0) for x in range(1, 6)}
OPEN = OLD | {(x, 2) for x in range(2, 5)}


def step(cell, move, floor):
    dx, dy = MOVES[move]
    candidate = cell[0] + dx, cell[1] + dy
    return candidate if candidate in floor else cell


def oracle(cell, floor):
    queue = deque([(cell, [])])
    seen = {cell}
    while queue:
        current, path = queue.popleft()
        if current == GOAL:
            return path[0]
        for move in range(4):
            target = step(current, move, floor)
            if target not in seen:
                seen.add(target)
                queue.append((target, path + [target]))
    raise ValueError("unreachable goal")


def life(seed):
    rng = np.random.default_rng(seed)
    wiring = rng.permutation(np.arange(90) % 4)
    sensors = rng.permutation(4)
    basis = rng.uniform(-0.13, 0.13, (2, 30))
    codes = {p: torch.tensor(np.array(p) @ basis, dtype=torch.float32)
             for p in OPEN | {(x, y) for x in range(7) for y in range(5)}}

    def see(p, floor):
        return torch.tensor([float(step(p, i, floor) != p) for i in sensors])

    memory = EmbodiedChoice(4, 1)
    neutral = torch.tensor([0.5])
    arena = {(x, y) for x in range(7) for y in range(5)}
    cell = START
    for _ in range(45):
        for action in rng.permutation(90):
            nxt = step(cell, wiring[action], arena)
            memory.experience(see(cell, arena), codes[cell], neutral, action, codes[nxt], neutral)
            cell = nxt
    for _ in range(12):
        cell = START
        while cell != GOAL:
            nxt = oracle(cell, OLD)
            move = next(i for i in range(4) if step(cell, i, OLD) == nxt)
            action = rng.choice(np.flatnonzero(wiring == move))
            after = torch.zeros(1) if nxt == GOAL else neutral
            memory.experience(see(cell, OLD), codes[cell], neutral, action, codes[nxt], after)
            cell = nxt

    results, elapsed = {}, []
    for condition, floor in [('script', OLD), ('oracle', OLD), ('shortcut', OPEN),
                              ('blocked', OLD), ('shuffle', OPEN), ('erase', OPEN)]:
        agent = copy.deepcopy(memory)
        if condition == 'shuffle':
            agent.effect.copy_(agent.effect[rng.permutation(90)])
        if condition == 'erase':
            agent.mass.zero_()
        cell = START
        first_direct = False
        for beat in range(18):
            before = torch.tensor([(beat + 1) / 18])
            sense = see(cell, floor)
            if condition == 'script':
                nxt = oracle(cell, floor)
            else:
                probe = codes[oracle(cell, floor)] if condition == 'oracle' else None
                start = time.perf_counter_ns()
                action = agent.choose(sense, codes[cell], before, probe)
                nxt = step(cell, wiring[action], floor)
                after = torch.zeros(1) if nxt == GOAL else torch.tensor([min(beat + 2, 18) / 18])
                agent.experience(sense, codes[cell], before, action, codes[nxt], after)
                elapsed.append((time.perf_counter_ns() - start) / 1e6)
            if beat == 0:
                first_direct = nxt == (2, 2)
            cell = nxt
            if cell == GOAL:
                break
        results[condition] = dict(refilled=cell == GOAL, beats=beat+1, first_direct=first_direct)
    return results, elapsed, sum(t.numel()*t.element_size() for t in memory.buffers())


if __name__ == '__main__':
    torch.set_num_threads(1)
    rows, timings = [], []
    for seed in range(8100, 8124):
        row, elapsed, size = life(seed)
        rows.append(row)
        timings.extend(elapsed)
    totals = {k: dict(refilled=sum(r[k]['refilled'] for r in rows),
                      first_direct=sum(r[k]['first_direct'] for r in rows),
                      median_beats=statistics.median(r[k]['beats'] for r in rows)) for k in rows[0]}
    result = dict(torch=torch.__version__, platform=platform.platform(), seeds=list(range(8100,8124)),
                  tensor_bytes=size, p95_ms=float(np.percentile(timings,95)), totals=totals, lives=rows)
    Path('measurements/life_choice.json').write_text(json.dumps(result, indent=2)+'\n')
    print(totals, size, result['p95_ms'])
