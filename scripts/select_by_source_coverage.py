"""Pick demos by how deeply they sit INSIDE stage-1's source distillation distribution.

`diag_demo_vs_source_coverage.py` found the one measure on which the LOST-ckpt
demo set (adapt 0.93) is an extreme outlier among 20 demo sets: the fraction of
its states that fall within the source set's own nearest-neighbour scale in the
FROZEN LATENT space -- 67.4% vs 32.6 +- 6.5% for everything else (z = +5.3).
Sign: closer to the source distribution -> better adapt.

But (a) the correlation across the other 19 sets is weak (rho = +0.38, ns) and
(b) `cover_lat%` is confounded with demo LENGTH (rho = +0.70), because
`source_diverse_v2` is a mixed 45%-success distribution full of slow /
exploratory trajectories, so long demos look source-like for free.

This turns the observation into an intervention: from ONE pool (one network, one
training stage) select 20 demos by `cover_lat%`, in two arms that separate the
confound --

  pure    the top-20 by cover_lat%                     (coverage, length free)
  lenmatch stratify by the pool's length quantiles and take the highest-coverage
          demo in each stratum                          (coverage, length pinned)

Compare against the 5 disjoint RANDOM 20-demo draws from the same pool
(adapt 0.455 .. 0.748, std 0.098). If either arm clears that band, "sit inside
the source distillation distribution" is causal; if both land inside it, the
original set's 67.4% is a property of its collecting network, not something
selection can buy.

Usage:
    python scripts/select_by_source_coverage.py --pool threading_d0_repro952_pool500 \
        --n 20 --out_prefix threading_d0_srccov
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np
import pyrallis
import torch

from rl.simdist_agent import SimDistAgent
from train_simdist import MainConfig

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"
SRC = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
_DATA = os.path.join(_REPO, "release/data/mimicgen")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def write_subset(pool, keys, idx, out, note):
    os.makedirs(out, exist_ok=True)
    with h5py.File(os.path.join(_DATA, pool, "dataset.hdf5"), "r") as src, \
         h5py.File(os.path.join(out, "dataset.hdf5"), "w") as hf:
        gg = hf.create_group("data")
        for i, j in enumerate(idx):
            gs = src[f"data/{keys[j]}"]
            gd = gg.create_group(f"demo_{i}")
            for ok in gs["obs"].keys():
                gd.create_dataset(f"obs/{ok}", data=gs[f"obs/{ok}"][:])
            for ok in gs.keys():
                if ok != "obs":
                    gd.create_dataset(ok, data=gs[ok][:])
            for ak, av in gs.attrs.items():
                gd.attrs[ak] = av
            gd.attrs["src_demo"] = keys[j]
        gg.attrs["num_demos"] = len(idx)
    cfg = json.load(open(os.path.join(_DATA, pool, "env_cfg.json")))
    cfg["composition"] = {"num_demos_total": len(idx), "selected_from": pool,
                          "selection": note, "src_demos": [keys[j] for j in idx]}
    json.dump(cfg, open(os.path.join(out, "env_cfg.json"), "w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--n_src_ep", type=int, default=3000)
    p.add_argument("--n_src_state", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")
    ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    ag.load_state_dict(sd); ag = ag.cuda(); ag.eval()

    rng = np.random.RandomState(args.seed)
    with h5py.File(os.path.join(_REPO, SRC), "r") as f:
        sk = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        pick = rng.choice(len(sk), size=min(args.n_src_ep, len(sk)), replace=False)
        S = np.concatenate([np.asarray(f[f"data/{sk[i]}"]["obs"]["state"]) for i in pick])
    if S.shape[0] > args.n_src_state:
        S = S[rng.choice(S.shape[0], args.n_src_state, replace=False)]
    src = torch.as_tensor(S, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        src_lat = torch.cat([ag.model.encode(src[i:i + 8192]) for i in range(0, src.shape[0], 8192)])
        sdl = src_lat.std(0) + 1e-6
        sub = src_lat[:4000]
        dd = torch.cdist(sub / sdl, sub / sdl); dd.fill_diagonal_(float("inf"))
        ref = float(dd.min(1).values.median())
    log(f"[source] {src.shape[0]} states, latent NN scale = {ref:.3f}")

    # per-demo latent coverage
    with h5py.File(os.path.join(_DATA, args.pool, "dataset.hdf5"), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        cov, lens = [], []
        with torch.no_grad():
            for k in keys:
                st = torch.as_tensor(np.asarray(f[f"data/{k}"]["obs"]["state"]),
                                     dtype=torch.float32, device="cuda")
                z = ag.model.encode(st)
                nn = torch.cdist(z / sdl, src_lat / sdl).min(1).values
                cov.append(float((nn < ref).float().mean()) * 100)
                lens.append(st.shape[0])
    cov = np.array(cov); lens = np.array(lens, dtype=float)
    log(f"[pool] {len(keys)} demos  cover_lat% {cov.mean():.1f}±{cov.std():.1f} "
        f"[{cov.min():.1f},{cov.max():.1f}]   len {lens.mean():.1f}±{lens.std():.1f}")

    # arm 1: pure top-n by coverage
    pure = np.argsort(-cov)[:args.n]
    # arm 2: length-stratified (pool's length quantiles), highest coverage per stratum
    order = np.argsort(lens)
    strata = np.array_split(order, args.n)
    lenm = np.array([st[np.argmax(cov[st])] for st in strata])

    # arm 3: SAME length stratification but a RANDOM pick inside each stratum --
    # separates "length-stratified diversity" from "source coverage". If this
    # matches lenmatch, the coverage term is doing nothing.
    r2 = np.random.RandomState(args.seed + 1)
    lenr = np.array([st[r2.randint(len(st))] for st in strata])

    for tag, idx in [("pure", np.sort(pure)), ("lenmatch", np.sort(lenm)),
                     ("lenrand", np.sort(lenr))]:
        out = os.path.join(_DATA, f"{args.out_prefix}_{tag}_n{args.n}")
        write_subset(args.pool, keys, idx, out,
                     {"pure": f"top-{args.n} by latent coverage of the source distillation set",
                      "lenmatch": "length-stratified, highest source-latent coverage per stratum",
                      "lenrand": "length-stratified, RANDOM pick per stratum (coverage ignored) "
                                 "-- control isolating stratified diversity"}[tag])
        log(f"  {tag:>9}: cover_lat% {cov[idx].mean():5.1f}±{cov[idx].std():4.1f}  "
            f"len {lens[idx].mean():6.1f}±{lens[idx].std():5.1f}  Σ={int(lens[idx].sum())}  -> {out}")
    log(f"\n  参照: 原始(丢失ckpt) cover_lat%=67.4  len=99.2 ; 5 组随机 cover≈32±7, adapt 0.455-0.748")
    os._exit(0)


if __name__ == "__main__":
    main()
