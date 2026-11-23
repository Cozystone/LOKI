"""Repeat the original v7 test unchanged, with no action history supplied."""
import json
from unittest.mock import patch

from loki_memory.contextual import ContextMemory
from . import v7_roundtrip
from .reference import ROOT


def main():
    rows = []
    for seed in range(7701, 7713):
        control = v7_roundtrip.run(seed)
        with patch.object(v7_roundtrip, "PortableMemory", ContextMemory):
            upgraded = v7_roundtrip.run(seed)
        metrics = ("outbound", "inbound", "retained", "novel_accept", "twin_accept")
        rows.append(dict(seed=seed, equal=all(control[k]==upgraded[k] for k in metrics),
                         control={k:control[k] for k in metrics},
                         upgraded={k:upgraded[k] for k in metrics}))
    (ROOT/"measurements/v8_legacy.json").write_text(json.dumps(rows,indent=2)+"\n")
    print("Matched seed rows:",sum(r["equal"] for r in rows),"/",len(rows))


if __name__=="__main__":
    main()
