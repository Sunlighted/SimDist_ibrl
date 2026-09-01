"""Cache the adapt curves of the three matched-budget projects from wandb.

The Square baselines (dyna/ibrl/rlpd/iql) ran on the cluster and never landed in
a local exps/ dir, so wandb is the only source; caching keeps the plot scripts
offline and reproducible.

x-axis is target transitions for every arm, but the two families log it
differently: the model-based arms carry `adapt/transitions`, the model-free ones
count env steps in `other/step`. `_step` is wandb's own counter and is NOT the
budget -- using it would silently rescale the model-based arms by 1000x.

    python viz/fetch_wandb_curves.py
"""
import json
import os
import sys

import wandb

ENTITY = "cao1233"
PROJECTS = ["simdist_threading_3arm", "square_rotw1_6arm", "coffee_g5x_3arm"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
KEYS = ["score/score", "score/score_best"]


def series(run):
    """(x, {key: [values]}) for one run, dropping rows where score is missing."""
    xkey = "adapt/transitions" if "adapt/transitions" in run.summary else "other/step"
    want = [xkey] + [k for k in KEYS if k in run.summary]
    rows = [r for r in run.scan_history(keys=want) if r.get(xkey) is not None]
    rows = [r for r in rows if r.get("score/score") is not None]
    rows.sort(key=lambda r: r[xkey])
    return xkey, [int(r[xkey]) for r in rows], {k: [r.get(k) for r in rows] for k in want[1:]}


def main():
    api = wandb.Api(timeout=60)
    os.makedirs(OUT, exist_ok=True)
    for proj in PROJECTS:
        runs = list(api.runs(f"{ENTITY}/{proj}"))
        out = {}
        for i, r in enumerate(runs):
            xkey, xs, ys = series(r)
            out.setdefault(r.group or "(none)", {})[r.name] = {
                "xkey": xkey, "x": xs, **{k.split("/")[-1]: v for k, v in ys.items()}
            }
            print(f"  [{i+1}/{len(runs)}] {proj}/{r.group}/{r.name}  n={len(xs)} x={xkey}",
                  file=sys.stderr, flush=True)
        path = os.path.join(OUT, f"{proj}.json")
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"wrote {path}  ({sum(len(v) for v in out.values())} runs)", file=sys.stderr)


if __name__ == "__main__":
    main()
