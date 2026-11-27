"""Smoke-test a separate upstream Going instance with detached R2 buffers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from loki_memory.place_bundles import PlaceBundles


def run(root):
    sys.path.insert(0, str(root.resolve()))
    from atanor_os.emerge.body_contract import PathOrgan
    from atanor_os.emerge.going import Going

    torch.manual_seed(29000)
    torch.set_num_threads(1)
    body = PathOrgan(1, "cpu")
    organ = PlaceBundles()
    max_error = 0.0
    for i in range(24):
        code = body(torch.tensor([[0.25, 0.02, 0.03]]))[:, :30]
        max_error = max(max_error, float((organ.odometry(code) - body.p).abs().max()))
        organ.observe(code, [[float(i == 6), float(i == 21)]], [0.01])
    context = organ.sleep()
    assert context >= 0
    shadow = Going(40, 3, 1, [0, 1], "cpu", chan="contract")
    state = organ.going_snapshot(context)
    for k in ("C", "Cn", "T", "R", "Vn", "Pu", "Pn", "used"):
        setattr(shadow, k, state[k])
    shadow.refresh()
    assert torch.isfinite(shadow.V).all() and shadow.V.shape == (64, 2)
    second = PlaceBundles()
    second.import_going(shadow)
    torch.testing.assert_close(second.going(0)[1], organ.going(context)[1])
    return dict(
        atanor_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        refresh_shape=list(shadow.V.shape),
        max_position_error=max_error,
        imported=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    r = run(ap.parse_args().root)
    Path("measurements/r2_bridge.json").write_text(json.dumps(r, indent=2))
    print(r)
