"""Judge-only trajectory plots, shortest physical routes and paired cohort summaries."""

import argparse
from collections import deque
import json
import numpy as np
from .reference import ROOT
from .v11_world import Maze, cyclic


def read(name):
    return json.loads((ROOT / "measurements" / ("v11_" + name + ".json")).read_text())


def shortest(world, start, goal):
    queue = deque([(tuple(start), 0)])
    seen = {tuple(start)}
    while queue:
        pose, length = queue.popleft()
        if pose[:2] == tuple(goal[:2]):
            return length
        for action in range(4):
            target = world.step(pose, action)
            if target not in seen:
                seen.add(target)
                queue.append((target, length + 1))


def paired(rows):
    ix = np.random.default_rng(11).integers(len(rows), size=(20000, len(rows)))
    result = {}
    for name in ("v11", "single_path", "v9", "v10", "RLS", "recurrent", "matched_RLS"):
        if name not in rows[0]:
            continue
        diff = np.array([x["v11"][2] - x[name][2] for x in rows])
        result[name] = dict(
            mse=float(np.mean([x[name][2] for x in rows])),
            difference_ci95=np.quantile(diff[ix].mean(1), [0.025, 0.975]).tolist(),
        )
    return result


def summarize(world, suite, dynamics, changed=False):
    env = cyclic if changed else Maze
    successful = [(env(x["seed"]), x["novel"]) for x in world if x["novel"]["success"]]
    eff = [
        shortest(w, r["trace"][0], r["trace"][-1]) / r["steps"] for w, r in successful
    ]
    joint = np.mean(
        [s["entity" if changed else "identity"]["joint"] for s in suite], axis=0
    )
    return dict(
        seeds=[world[0]["seed"], world[-1]["seed"]],
        entity=joint.tolist(),
        visual=float(
            np.mean(
                [s["entity" if changed else "identity"]["visual"][0] for s in suite]
            )
        ),
        return_by_condition=np.sum([w["recovery"] for w in world], axis=0).tolist(),
        novel=sum(w["novel"]["success"] for w in world),
        open_loop=sum(w["open_loop"] for w in world),
        novel_mean_efficiency_on_success=float(np.mean(eff)),
        mean_surprise_trigger_count=float(np.mean([w["novel"]["bad"] for w in world])),
        mean_supported_outcome_mass=float(np.mean([w["novel"]["mass"] for w in world])),
        no_observed_path=sum(w["ceiling"][-1] is None for w in world),
        transfer=np.mean([w["transfer"] for w in world], axis=0).tolist(),
        active_passive=[
            float(np.mean([s["alias"][k] for s in suite])) for k in ("True", "False")
        ],
        detour=np.mean([s["detour"] for s in suite], axis=0).tolist(),
        dynamics=paired(dynamics),
    )


def report(rev):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    world = read(f"world_{rev}")
    suite = read(f"suite_{rev}")
    held = read("holdout" if rev == 13 else f"holdout_{rev}")
    result = dict(
        sealed=summarize(world, suite, read(f"dynamics_{rev}")),
        holdout=summarize(
            [x["world"] for x in held], held, [x["dynamics"] for x in held], True
        ),
        plant_replication=paired(
            [
                x["metrics"]
                for x in read("replication" if rev == 13 else f"replication_{rev}")
            ]
        ),
    )
    (ROOT / f"measurements/v11_summary_{rev}.json").write_text(
        json.dumps(result, indent=2)
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), layout="constrained")
    failures = [x for x in world if not x["novel"]["success"]]
    for ax, row in zip(axes, failures[:3]):
        w = Maze(row["seed"])
        trace = np.array(row["novel"]["trace"])
        for p, neighbors in w.edges.items():
            for q in neighbors:
                if p < q:
                    ax.plot([p[0], q[0]], [p[1], q[1]], color=".82", lw=3)
        ax.plot(trace[:, 0], trace[:, 1], color="#007B83", lw=2)
        ax.scatter(
            *trace[0, :2], s=100, facecolors="none", edgecolors="#00555D", label="Start"
        )
        ax.scatter(
            *trace[-1, :2], s=55, color="#BA3B17", marker="x", label="Last position"
        )
        if "goal" in row["novel"]:
            ax.scatter(
                *row["novel"]["goal"][:2],
                s=80,
                marker="*",
                color="purple",
                label="Goal",
            )
        ax.set(title=f"Seed {row['seed']} · {len(trace) - 1} actions", aspect="equal")
        ax.invert_yaxis()
        ax.axis("off")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7)
    fig.suptitle("Novel-goal failures · physical trajectories")
    fig.savefig(ROOT / f"examples/v11_failures_{rev}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("revision", type=int, default=16, nargs="?")
    report(parser.parse_args().revision)
