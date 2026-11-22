"""Re-run inherited contracts with frozen v3 and live v4 on the same episodes."""
import json
import numpy as np
import torch
from . import v3_heat, v3_retention
from .reference import ROOT,setup,write
from .v4_forward import frozen_v3


def main():
    torch.set_num_threads(1)
    _,_,scenes,_,_ = setup()
    v3 = frozen_v3()
    first = []
    for seed in (522,*range(3701,3713)):
        row = v3_retention.measure(seed,v3,scenes,False)
        score = row["t240_probe0"]
        first.append(dict(seed=seed,v3=score["v2"],v4=score["v3"],bytes=row["state_bytes"]))
    heat = []
    original = json.loads((ROOT/"measurements/v3_confirm_heat.json").read_text())
    for old in original["rows"]:
        row = v3_heat.world(old["seed"],v3)
        m = row["pulse_difference"]
        assert np.isclose(m["v2"],old["pulse_difference"]["v3"],atol=1e-7)
        heat.append(dict(seed=row["seed"],v4=m["v3"],v3=m["v2"],inner=m["inner"],last=m["last"],
                         original_v2=old["pulse_difference"]["v2"],sign=row["pulse_sign"]["v3"]))
    means = {k:float(np.mean([r[k] for r in heat])) for k in heat[0] if k!="seed"}
    first_mean = {k:float(np.mean([r[k] for r in first[1:]])) for k in ("v3","v4")}
    passed = all(r["v4"]<=min(.01,1.1*r["v3"]) for r in (first_mean,first[0]))
    heat_gate = means["sign"]>=.9 and means["v4"]<=1.1*means["v3"]
    for k in ("inner","last","original_v2"):
        heat_gate &= means["v4"]<=.5*means[k] and sum(r["v4"]<r[k] for r in heat)>=7
    write("v4_regression",dict(first=first,first_mean=first_mean,first_pass=passed,heat=heat,
                               heat_mean=means,heat_pass=bool(heat_gate)))


if __name__=="__main__":
    main()
