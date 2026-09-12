"""Pull run histories out of W&B and cache them as JSON.

Every curve figure in the paper reads its data through here, so the network is
hit once per (project, group) and the plotting script stays re-runnable offline.

The env has no pandas, so nothing here returns a DataFrame -- histories come
back as plain lists and the plotting scripts use numpy.

    from wandb_fetch import fetch_group
    runs = fetch_group("simdist_threading_3arm", "plain_anch",
                       x="adapt/transitions", y=["score/score", "score/policy_only"])
"""
import json
import os

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ENTITY = "cao1233"


def _cache_path(project, group, x, ys):
    tag = "__".join([project, group, x] + list(ys)).replace("/", "-")
    return os.path.join(CACHE, tag + ".json")


def fetch_group(project, group, x, y, entity=ENTITY, refresh=False,
                state="finished", created_after=None):
    """Return {seed: {key: [...]}} for one W&B group, keyed by x and each y.

    `y` is a metric name or a list of them; every requested key comes back on
    the same row grid, so a figure that mixes two metrics (e.g. a proposal-only
    zero-shot point spliced onto a planner curve) cannot silently misalign them.

    `created_after` is an ISO date string used to disambiguate a group that was
    run more than once -- W&B keeps both batches under the same group name and
    there is nothing in the run name to tell them apart.
    """
    ys = [y] if isinstance(y, str) else list(y)
    path = _cache_path(project, group, x, ys)
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    import wandb
    api = wandb.Api(timeout=90)
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    out = {}
    for r in runs:
        if state is not None and r.state != state:
            continue
        if created_after is not None and str(r.created_at) < created_after:
            continue
        seed = r.config.get("seed")
        if seed is None:
            raise RuntimeError(f"run {r.id} has no seed in its config")
        if seed in out:
            raise RuntimeError(
                f"duplicate seed {seed} in {project}/{group} (runs collide even "
                f"after the created_after filter) -- tighten the filter")
        hist = list(r.scan_history(keys=[x] + ys))
        out[seed] = {k: [h[k] for h in hist] for k in [x] + ys}

    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    return out
