"""One live atlas for grounded and coordinate-free streams. Domain names are judge-only."""

import json
from pathlib import Path
import numpy as np
import torch
from loki_memory.trace_atlas import TraceAtlas


def check(seed):
    rng = np.random.default_rng(seed)
    s = TraceAtlas(3, 1024)
    position = np.array([[0.2, 0.7]], "f4")
    ambient = np.zeros((1, 16), "f4")
    zero = np.zeros((1, 32), "f4")

    def write(key, pos, context, clock):
        return s.write(key, pos, context, [[0, 0, 0]], [0], clock, [True]).item()

    origin = write(zero, position, ambient, 0)
    clock = 0
    cases = []
    for i in range(24):
        sound = np.abs(np.fft.rfft(rng.normal(size=62)))
        sound /= sound.max()
        # Slow unnamed telemetry channels at a sampled time.
        phase = rng.uniform(-np.pi, np.pi, 32)
        body = 0.5 + 0.45 * np.sin(phase + clock * 0.002)
        symbol = rng.integers(2, size=32)
        for domain, key in zip(("spectrum", "body", "binary"), (sound, body, symbol)):
            key = key[None].astype("f4")
            prefix = rng.random(16)
            contexts = [
                np.average(
                    rng.normal(base, 0.04, (16, 16)),
                    axis=0,
                    weights=0.9 ** np.arange(16)[::-1],
                )
                .clip(0, 1)[None]
                .astype("f4")
                for base in (prefix, 1 - prefix)
            ]
            for c in contexts:
                clock += 1
                token = write(key, None, c, clock)
                cases.append((domain, key, c, token, clock))
    clock += 1
    finish = write(zero, position, ambient, clock)
    blob = s.state()
    s = TraceAtlas.load(blob)
    rows = {}
    for name in ("spectrum", "body", "binary"):
        hits = []
        masked = []
        owned = set()
        for domain, key, c, token, _ in cases:
            if domain != name:
                continue
            for cue, acc in ((c, hits), (np.full_like(c, 0.5), masked)):
                query = np.clip(key + rng.normal(0, 0.01, key.shape), 0, 1)
                # Both members of a pair receive the identical masked cue (no noise advantage).
                if acc is masked:
                    query = key
                found = s.cue(query, None, cue)[0][0, 0].item()
                acc.append(token >= 0 and token not in owned and found == token)
            owned.add(token)
        rows[name] = dict(
            intact=float(np.mean(hits)), history_removed=float(np.mean(masked))
        )
    unknown = cases[0][3]
    missing = s.read([unknown])
    assert not missing["pos_valid"].item()
    assert not s.anchor([unknown], [1.0], [[0, 0]]).item()
    expected = (
        [(origin, 0, 0)]
        + [(t, when, when) for _, _, _, t, when in cases]
        + [(finish, clock, clock)]
    )
    return dict(
        seed=seed,
        rows=rows,
        dated_sequence=s.recall_episode(0, clock, clock + 1) == expected,
        events=len(expected),
        state_bytes=len(blob),
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = [check(seed) for seed in range(27601, 27613)]
    Path("measurements/v13_domains.json").write_text(json.dumps(results, indent=2))
    print(results[0])
