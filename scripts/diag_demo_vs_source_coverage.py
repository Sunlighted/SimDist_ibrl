"""Does a target demo set's overlap with the SOURCE distillation distribution
predict how well stage-3 adapts the dynamics?

Hypothesis (user, 2026-08-04): stage-1 distilled encoder+dynamics+policy on
`threading_d0_source_diverse_v2` (100k SOURCE trajectories, 800k steps), so the
dynamics head is most accurate on THAT distribution. Whether the new TARGET
demos land INSIDE or OUTSIDE that region should change what stage-3 has to do:

  inside  -> the dynamics already models the region; adaptation is a small
             correction, but the strong source prior may also resist it
  outside -> the dynamics must extrapolate/learn a new region, but there is no
             conflicting prior to overwrite

Both directions are plausible, so this measures and lets the sign decide. Every
previous measurement compared demos to OTHER DEMOS or to the collection
rollouts; none compared them to stage-1's own training distribution.

Measured per demo set, in BOTH spaces (the dynamics operates on z, so the latent
one is the mechanistically relevant one):

  d_src        median over demo states of the distance to the NEAREST source
               state, z-scored by the SOURCE std (so 1.0 = one source-sigma)
  d_src_lat    same after `encode()` (frozen stage-1 encoder)
  cover_frac   fraction of demo states closer than the source set's own median
               nearest-neighbour distance (= "inside the source density scale")

Usage:
    python scripts/diag_demo_vs_source_coverage.py --n_src_ep 3000
"""
import argparse
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

from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from rl.simdist_agent import SimDistAgent
from train_simdist import MainConfig

STAGE1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"
SRC = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
SETS = [  # label, dir, n, known adapt (25-50k window so every arm is comparable)
    ("orig (LOST ckpt)", "threading_d0_target_dynamics", 20, 0.932),
    ("matched n50", "threading_d0_target_expert_matched_n50", 50, 0.907),
    ("matched n20", "threading_d0_target_expert_matched_n20", 20, 0.879),
    ("prefix n20", "threading_d0_target_expert_n500", 20, 0.866),
    ("prefix n50", "threading_d0_target_expert_n500", 50, 0.756),
    ("snapB_160000", "threading_d0_snapB_160000", 20, 0.879),
    ("snapB_140000", "threading_d0_snapB_140000", 20, 0.873),
    ("snapB_400000", "threading_d0_snapB_400000", 20, 0.648),
    ("snapA_140000", "threading_d0_snapA_140000", 20, 0.858),
    ("snapA_200000", "threading_d0_snapA_200000", 20, 0.712),
    ("snapA_400000", "threading_d0_snapA_400000", 20, 0.436),
    ("rand20_g4", "threading_d0_rand20_g4", 20, 0.748),
    ("rand20_g1", "threading_d0_rand20_g1", 20, 0.578),
    ("rand20_g0", "threading_d0_rand20_g0", 20, 0.574),
    ("rand20_g3", "threading_d0_rand20_g3", 20, 0.517),
    ("rand20_g2", "threading_d0_rand20_g2", 20, 0.455),
    ("repro952_n20", "threading_d0_repro952_n20", 20, 0.630),
    ("sig01new n20", "threading_d0_target_sig01new", 20, 0.656),
    ("sig002_extra n20", "threading_d0_target_sig002_extra", 20, 0.620),
    ("sig01exp@002", "threading_d0_target_sig01expert_at002", 20, 0.330),
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_src_ep", type=int, default=3000, help="source episodes to sample")
    p.add_argument("--n_src_state", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(_REPO, STAGE1, "cfg.yaml")))
    sd = torch.load(os.path.join(_REPO, STAGE1, "latest.pt"), map_location="cuda")
    ag = SimDistAgent(1, (37,), None, 7, "", cfg.simdist, device="cuda")
    ag.load_state_dict(sd); ag = ag.cuda(); ag.eval()

    # ---- SOURCE distillation distribution ----
    rng = np.random.RandomState(args.seed)
    with h5py.File(os.path.join(_REPO, SRC), "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        pick = rng.choice(len(keys), size=min(args.n_src_ep, len(keys)), replace=False)
        S = np.concatenate([np.asarray(f[f"data/{keys[i]}"]["obs"]["state"]) for i in pick])
    if S.shape[0] > args.n_src_state:
        S = S[rng.choice(S.shape[0], args.n_src_state, replace=False)]
    src = torch.as_tensor(S, dtype=torch.float32, device="cuda")
    log(f"[source] {len(pick)} episodes -> {src.shape[0]} states (stage-1 distilled on this)")

    with torch.no_grad():
        src_lat = torch.cat([ag.model.encode(src[i:i + 8192]) for i in range(0, src.shape[0], 8192)])
        sd_s = src.std(0) + 1e-6
        sd_l = src_lat.std(0) + 1e-6
        # source's own NN scale (subsample to keep the pairwise cost sane)
        sub = src[rng.choice(src.shape[0], 4000, replace=False)]
        dd = torch.cdist(sub / sd_s, sub / sd_s); dd.fill_diagonal_(float("inf"))
        ref_s = float(dd.min(1).values.median())
        subl = src_lat[:4000]
        ddl = torch.cdist(subl / sd_l, subl / sd_l); ddl.fill_diagonal_(float("inf"))
        ref_l = float(ddl.min(1).values.median())
    log(f"[source] own NN scale: state={ref_s:.3f}  latent={ref_l:.3f}")

    def nn_to_src(X, base, s):
        out = []
        for i in range(0, X.shape[0], 2048):
            out.append(torch.cdist(X[i:i + 2048] / s, base / s).min(1).values)
        return torch.cat(out)

    rows = []
    log(f"\n{'set':>20}{'adapt':>7}{'d_src':>9}{'cover%':>8}{'d_src_lat':>11}{'cover_lat%':>11}")
    with torch.no_grad():
        for lab, path, n, ad in SETS:
            b = SequenceReplayBuffer(max_episodes=20000, device="cpu")
            load_demos_to_sequence_buffer(
                b, os.path.join(_REPO, "release/data/mimicgen", path, "dataset.hdf5"), num_data=n)
            D = torch.cat([torch.as_tensor(e["state"][:-1]) for e in b.episodes]).float().cuda()
            Dl = ag.model.encode(D)
            ds = nn_to_src(D, src, sd_s); dl = nn_to_src(Dl, src_lat, sd_l)
            r = (ad, float(ds.median()), float((ds < ref_s).float().mean()) * 100,
                 float(dl.median()), float((dl < ref_l).float().mean()) * 100)
            rows.append(r)
            log(f"{lab:>20}{ad:>7.3f}{r[1]:>9.3f}{r[2]:>8.1f}{r[3]:>11.3f}{r[4]:>11.1f}")

    from scipy import stats
    Y = np.array([r[0] for r in rows])
    from scipy.stats import t as tdist
    tc = tdist.ppf(1 - 0.05 / 2, len(Y) - 2); crit = tc / np.sqrt(len(Y) - 2 + tc ** 2)
    log(f"\nn={len(Y)}  单指标 p<0.05 需 |ρ|≥{crit:.2f}；4 重比较需 |ρ|≥"
        f"{tdist.ppf(1-0.0125/2,len(Y)-2)/np.sqrt(len(Y)-2+tdist.ppf(1-0.0125/2,len(Y)-2)**2):.2f}")
    for i, k in enumerate(["d_src", "cover%", "d_src_lat", "cover_lat%"], start=1):
        X = np.array([r[i] for r in rows])
        rho, pv = stats.spearmanr(X, Y)
        log(f"  {k:>12} vs adapt: Spearman ρ={rho:+.3f} p={pv:.4f}")
    os._exit(0)


if __name__ == "__main__":
    main()
