"""Train and test the trace ablation on the exact confirmation episodes and pairs."""
import torch
from .reference import setup,write
from .v3_common import prior_version
from .v3_heat import world


if __name__=="__main__":
    setup()
    torch.set_num_threads(1)
    rows = [world(seed,prior_version(),ablate=True) for seed in range(3801,3809)]
    write("v3_trace_ablation", [{k:v for k,v in row.items() if k!="trials"} for row in rows])
