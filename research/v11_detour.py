"""A cyclic, unlabeled world with a genuine alternate route after closure."""

import json
import numpy as np
from .reference import ROOT
from loki_memory.belief import BeliefModel


def run(seed, rev):
    rng = np.random.default_rng(seed)
    m = BeliefModel(rev)
    obs = rng.random((8, 24), dtype="f4")
    wiring = rng.permutation(4)

    def move(s, a, blocked):
        motor = np.flatnonzero(wiring == a)[0]
        q = (s + (1 if motor == 0 else -1 if motor == 1 else 0)) % 8
        return s if blocked and {s, q} == {1, 2} else q

    for k in range(3):
        for s in rng.permutation(8):
            for a in range(4):
                m.fit(obs[s], a, obs[move(s, a, False)], adapt=False)

    def trial(correct):
        import copy

        world = copy.deepcopy(m)
        s = 0
        goal = 3
        q, t = world.policy(obs[goal])
        b = world.infer(obs[s])
        bad = 0
        for tick in range(32):
            a = world.choose(b, q, t)
            p = move(s, a, True)
            prior = b @ t[a]
            if correct and prior[world.emission_cost(obs[p]).argmin()] < (
                0.98 if rev >= 4 else 0.4
            ):
                bad += 1
                world.repair(obs[s], a, obs[p])
                q, t = world.policy(obs[goal])
            b = world.infer(obs[p], prior)
            s = p
            if s == goal:
                break
        return [int(s == goal), tick + 1, bad]

    return [trial(True), trial(False)]


if __name__ == "__main__":
    import sys

    rev = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    r = [run(s, rev) for s in range(11101 + 100 * rev, 11125 + 100 * rev)]
    (ROOT / f"measurements/v11_detour_{rev}.json").write_text(json.dumps(r))
    print(np.mean(r, 0))
