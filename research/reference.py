"""Load pinned, external v1 controls only inside an experiment process."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / ".git/loki-v1"
ATANOR = ROOT / ".git/atanor-source"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def setup():
    for path, pin in ((V1, "fc2f85a89eb5a8ce6548f5e4293843333f4e02ea"), (ATANOR, "dc05860ae6837bcdd4de0be34e811bf0848e444f")):
        if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != pin:
            raise ValueError(f"wrong reference at {path}")
    sys.path.insert(0, str(ATANOR))
    old = module("v1organ", V1/"loki_memory/organ.py")
    recurrence = module("v1rec", V1/"loki_memory/recurrence.py")
    hd = module("hd", V1/"high_dim_experiment.py")
    with patch.dict(sys.modules, {"loki_memory.recurrence":recurrence, "high_dim_experiment":hd,
                                "loki_memory.atanor":old}):
        # The pinned body script imports the old adapter by name; only its FIFO
        # class is used by this judge.
        old.AtanorImageMemory = None
        dark = module("v1dark", V1/"experiment.py")
        body = module("v1body", V1/"body_experiment.py")
    return old.LokiMemory, recurrence.RecurrenceMemory, hd.scenes, dark, body.PooledFIFO


def mse(a, b):
    return float(np.mean(np.square(a-b)))


def write(name, data):
    directory = ROOT / "measurements"
    directory.mkdir(exist_ok=True)
    (directory/f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
