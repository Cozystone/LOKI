"""Prospective v13 tests. The judge keeps labels outside the store."""

import argparse
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from loki_memory.trace_store import TraceStore
from loki_memory.trace_atlas import TraceAtlas


def batch(store, key, pos, ctx, write=False):
    out = []
    for off in range(0, len(key), 64):
        args = (key[off : off + 64], pos[off : off + 64], ctx[off : off + 64])
        if write:
            size = len(args[0])
            ids = store.write(
                *args,
                np.zeros((size, 2), "f4"),
                np.zeros(size),
                int(store.meta[0]) + 1,
                np.ones(size, bool),
            )
        else:
            ids = store.cue(*args)[0][:, 0]
        out.extend(ids.tolist())
    return np.array(out)


def setup(key, pos, ctx, n=1024):
    s = TraceStore(2, n)
    batch(s, key[:1], pos[:1], ctx[:1], True)
    args = tuple(np.repeat(a[:1], 64, axis=0) for a in (key, pos, ctx))
    ids, _, conf, _ = s.cue(*args, landmark=True)
    assert s.anchor(ids, conf, args[1]).all()
    return s


def score(found, labels):
    # A collision only counts for the first owner. Rejection is always a miss.
    owner = {}
    for i, x in enumerate(labels):
        if x >= 0:
            owner.setdefault(x, i)
    return np.mean([owner.get(x, -1) == i for i, x in enumerate(found)])


def run(seed, worlds):
    rng = np.random.default_rng(seed)
    key = worlds.saturation(seed, 1024, D=32).codes
    # Odometry in a noiseless grid frame; no simulator IDs enter storage.
    delta = np.diff(
        np.column_stack((np.arange(1024) % 32, np.arange(1024) // 32)),
        axis=0,
        prepend=[[0, 0]],
    )
    pos = np.cumsum(delta, axis=0).astype("f4")

    def context(center):
        ema = np.full(16, 0.5)
        for obs in rng.normal(center, 0.06, (128, 16)):
            ema = 0.95 * ema + 0.05 * obs
        return np.tile(ema.astype("f4"), (1024, 1))

    a, b = context(0.22), context(0.77)
    noisy = lambda arr, sd: (arr + rng.normal(0, sd, arr.shape)).astype("f4")
    probe = (np.clip(noisy(key, 0.02), 0, 1), noisy(pos, 0.08), noisy(a, 0.01))
    s = setup(key, pos, a)
    labels = []
    cap = {}
    last = 0
    for n in (64, 128, 256, 512, 1024):
        labels.extend(batch(s, key[last:n], pos[last:n], a[last:n], True))
        cap[n] = score(batch(s, *(x[:n] for x in probe)), labels)
        last = n
    novel = tuple(x[:64] for x in probe)
    _, _, known, kn = s.cue(*novel)
    # Identical appearance & pose; only ambient sensory history changes.
    _, _, unk, un = s.cue(novel[0], novel[1], b[:64])
    auc = ((un[:, None] > kn).float() + 0.5 * (un[:, None] == kn)).mean().item()
    n = 144
    s = setup(key, pos, a)
    la = batch(s, key[:n], pos[:n], a[:n], True)
    for _ in range(2):
        batch(s, key[:n], pos[:n], a[:n], True)
    base = score(batch(s, *(x[:n] for x in probe)), la)
    lb = batch(s, key[:n], pos[:n], b[:n], True)
    for _ in range(2):
        batch(s, key[:n], pos[:n], b[:n], True)

    def both():
        return [
            score(batch(s, *(x[:n] for x in probe)), la),
            score(batch(s, probe[0][:n], probe[1][:n], b[:n]), lb),
        ]

    pre = both()

    def put(x, p, c):
        return int(
            s.write(
                x[None],
                np.asarray([p], "f4"),
                c[None],
                [[0, 0]],
                [0],
                int(s.meta[0]) + 1,
                [True],
            )[0]
        )

    dup = rng.random(32, dtype="f4")
    u = put(dup, [40, 40], a[0])
    put(dup, [40.70, 40], a[0])
    incidental = put(1 - dup, [50, 50], a[0])
    for _ in range(35):
        put(key[0], pos[0], a[0])
    sleep = s.consolidate()
    after = both()
    # Same appearance, same history, two coordinates; after rebirth this is ambiguous.
    twin = put(key[0], [9, 9], a[0])
    two = np.array([[0, 0], [9, 9]], "f4")
    ka = np.repeat(key[:1], 2, 0)
    ca = np.repeat(a[:1], 2, 0)
    observed = batch(s, ka, two, ca)
    twins = score(observed, [la[0], twin])
    s.cut([0])
    args = (key[:1], np.array([[-5, 7]], "f4"), a[:1])
    ids, _, conf, _ = s.cue(*args, landmark=True)
    refused = not bool(s.anchor(ids, conf, args[1]).item())
    q = pos[1:2] + [-5, 7]
    ids, _, conf, _ = s.cue(key[1:2], q, a[1:2], landmark=True)
    anchored = bool(s.anchor(ids, conf, q).item())
    err = float(np.linalg.norm(s.offset.numpy()[0] - [5, -7]))
    return dict(
        seed=seed,
        capacity=cap,
        baseline=base,
        before_sleep=pre,
        after_sleep=after,
        context_alias_accuracy=score(
            np.r_[
                batch(s, key[2:3], pos[2:3] + [-5, 7], a[2:3]),
                batch(s, key[2:3], pos[2:3] + [-5, 7], b[2:3]),
            ],
            [la[2], lb[2]],
        ),
        pos_alias=twins,
        cold_ambiguous_refused=refused,
        reanchor_steps=2 if anchored else None,
        translation_error=err,
        known_coverage=float((known >= 0.7).float().mean()),
        ood_high=float((unk >= 0.7).float().mean()),
        novelty_auc=auc,
        sleep=sleep,
        snapshot_bytes=len(s.state()),
    )


def episodes():
    s = TraceAtlas(2, 128)
    rng = np.random.default_rng(27109)
    x = rng.random((12, 32), dtype="f4")
    ids = [
        s.write(
            x[i : i + 1],
            [[i, 0]],
            np.zeros((1, 16)),
            [[i % 6 == 2, i % 6 == 4]],
            [0],
            i,
            [True],
        ).item()
        for i in range(12)
    ]
    for off in (0, 6):
        for _ in range(20):
            for path in ((0, 1, 2), (3, 1, 4)):
                s.link([-1], [ids[off + path[0]]], [False])  # independent episode
                for i, j in zip(path, path[1:]):
                    s.link([ids[off + i]], [ids[off + j]], [True])
    outcomes = [s.continuations(ids[i], ids[1])[0].tolist() for i in (0, 3)]
    sleep = s.consolidate()
    roles = s.role(ids)
    d = torch.cdist(roles[6:11], roles[:5])
    # Four structural categories: incoming arm, junction, and two relief types.
    category = np.array([0, 1, 2, 0, 3])
    picked = d.argmin(1).numpy()
    shuffled = np.mean(
        [np.mean(category[rng.permutation(5)][picked] == category) for _ in range(512)]
    )
    # Two walks terminating in different changes. Score value at each predecessor.
    for i, j in ((5, 2), (11, 4)):
        s.link([ids[i]], [ids[j]], [True])
    s.consolidate()
    directional = float(
        np.mean(
            [
                s.value[ids[5]].argmax().item() == 0,
                s.value[ids[11]].argmax().item() == 1,
            ]
        )
    )
    return dict(
        directional_value=directional,
        continuations=outcomes,
        expected=[[ids[2]], [ids[4]]],
        unordered_best=0.5,
        structural_role=float(np.mean(category[picked] == category)),
        shuffled=float(shuffled),
        sleep=sleep,
        junction_values=s.value[ids[1]].tolist(),
        source_values=s.value[ids[0]].tolist(),
    )


def saturation_fixture(seed, n, D):
    """Reproduce only the observation field consumed from the v11 generator."""
    rng = np.random.default_rng(seed + 600001)

    def texture():
        v = (rng.random(D) < 0.5).astype("f4")
        return (np.roll(v, 1) + 2 * v + np.roll(v, -1)) / 4

    codes = [texture() for _ in range(n)]
    seen = set()
    for i, row in enumerate(codes):
        while row.tobytes() in seen:
            row = texture()
        codes[i] = row
        seen.add(row.tobytes())
    return SimpleNamespace(codes=np.stack(codes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=Path)
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    worlds = SimpleNamespace(saturation=saturation_fixture)
    if args.worlds:
        spec = importlib.util.spec_from_file_location("v11_worlds", args.worlds)
        worlds = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worlds)
    torch.set_num_threads(1)
    rows = [
        run(i, worlds)
        for i in (range(26101, 26125) if args.sealed else range(26001, 26004))
    ]
    dest = Path("measurements") / ("v13_sealed.json" if args.sealed else "v13_dev.json")
    dest.write_text(json.dumps(dict(rows=rows, episodes=episodes()), indent=2))
    print(dest, rows[0], episodes())
