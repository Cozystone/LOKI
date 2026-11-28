"""Synthetic observer. Ground truth geometry lives here only."""
import json
from pathlib import Path

import numpy as np
from place_transport import PlaceCue


def rotate(p, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*p[0]-s*p[1], s*p[0]+c*p[1]])


def advance(p, u):
    return rotate(p - u[:2], -u[2])


def sensor(p, noise, exponent):
    return p / np.linalg.norm(p), np.linalg.norm(p)**exponent * noise


def error(p, q):
    if not np.linalg.norm(q):
        return 180.
    return float(np.degrees(np.arccos(np.clip(p @ q / np.linalg.norm(p)
                                              / np.linalg.norm(q), -1., 1.))))


def run(seed, sigma, exponent):
    rng = np.random.default_rng(seed)
    model = PlaceCue()
    for _ in range(200):
        p = rotate([rng.uniform(.3, 3.), 0.], rng.uniform(-.75, .75))
        u = rng.uniform(-.2, .2, 3)
        model.cut()
        for point, motion in [(p, np.zeros(3)), (advance(p, u), u)]:
            model.update(*sensor(point, 1 + sigma*rng.normal(), exponent), motion)
    trials = []
    for _ in range(32):
        p = rotate([rng.uniform(.4, 1.6), 0.], rng.uniform(-.75, .75))
        yaw = p.copy()
        view = sensor(p, 1 + sigma*rng.normal(), exponent)
        path = rng.uniform(-.07, .07, (16, 3))
        path[:, :2] += rng.uniform(-.15, .15, 2)
        wrong = rng.uniform(-.07, .07, (16, 3))
        wrong[:, :2] += rng.uniform(-.15, .15, 2)
        estimates = []
        for steps in [path, wrong]:
            model.cut()
            model.update(*view, np.zeros(3))
            for u in steps:
                q = model.update(None, None, u)
            estimates.append(q)
        for u in path:
            p = advance(p, u)
            yaw = rotate(yaw, -u[2])
        trials.append([error(p, estimates[0]), error(p, yaw), error(p, estimates[1])])
    return trials


if __name__ == '__main__':
    results = {}
    for name, sigma, power in [('clean', 0, 1), ('noise', .05, 1), ('squared', 0, 2)]:
        values = np.array([run(seed, sigma, power) for seed in range(2000, 2024)])
        results[name] = values.tolist()
        print(name, 'mean degrees', values.mean((0, 1)), 'within45', (values <= 45).mean((0, 1)))
    Path(__file__).with_name('place_trials.json').write_text(json.dumps(results))
