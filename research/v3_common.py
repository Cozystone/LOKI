from pathlib import Path
import os
import subprocess
import numpy as np
from .reference import ROOT, module


def prior_version():
    path = Path(os.environ.get("LOKI_V2_REFERENCE",ROOT/".git/loki-v2"))
    pin = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    if pin!="0c987218f1d79c9112d355d79dad625d901f19fa":
        raise ValueError("expected pinned v2 comparison")
    return module("frozen_v2",path/"loki_memory/organ.py").LokiMemory


def compare(rows, metric, controls, minimum, wins_needed):
    values = np.array([r[metric]["v3"] for r in rows])
    result = {}
    for name in controls:
        baseline = np.array([r[metric][name] for r in rows])
        wins = int(np.sum(values<baseline))
        reduction = float(1-values.mean()/baseline.mean()) if baseline.mean()>0 else 0.
        result[name] = dict(v3=float(values.mean()),control=float(baseline.mean()),wins=wins,
                            reduction=reduction,passed=wins>=wins_needed and reduction>=minimum)
    return result
