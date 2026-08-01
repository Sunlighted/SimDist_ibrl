"""Diagnostic: is the SOURCE dynamics model's error in the CONTACT phase
irreducible multimodality (obs != full MuJoCo state) or a learnable bias?

Decides whether a GENERATIVE dynamics model (flow/diffusion) could plausibly help
SimDist's contact-rich planning, BEFORE building one.

Three measurements, all split by phase (needle-ring distance):
  A) model 1-step residual RMS of the trained deterministic MLP (best.pt).
  B) k-NN ALEATORIC FLOOR: among near-identical (s,a) neighbors, the spread of
     s' — the observation-space stochasticity of p(s'|s,a), model-free.
  C) the decisive ratio  model_residual / knn_floor  per phase, PLUS the
     contact-vs-free spread ratio at a MATCHED (s,a) radius (controls for the
     smooth local variation that inflates any finite-radius neighbor spread).

Verdict:
  * contact ratio ~1 AND contact/free floor ratio >> 1  -> irreducible multimodal
    contact transition; a deterministic MLP MUST mode-average -> generative worth trying.
  * model_residual >> knn_floor -> transition ~deterministic given (s,a), MLP just
    underfits -> generative won't help (bias/capacity problem).

Also: bimodality coefficient of contact-phase residuals per state dim + saved
histograms (mode-averaging leaves a bimodal/heavy-tailed residual signature).
"""
import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.world_model import WorldModel, WorldModelConfig
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer

NEEDLE = slice(9, 12)
RING = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str,
                   default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--dyn_ckpt", type=str,
                   default="exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt")
    p.add_argument("--num_data", type=int, default=12000,
                   help="trajectories to load (density for the k-NN floor)")
    p.add_argument("--pool_cap", type=int, default=150000, help="max transitions in the k-NN pool")
    p.add_argument("--n_query", type=int, default=3000, help="query points per phase")
    p.add_argument("--k", type=int, default=8, help="neighbors per query")
    p.add_argument("--action_weight", type=float, default=1.0,
                   help="weight on the 7-dim action in the (s,a) metric (state is z-scored)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default="/tmp/claude-1003/-home-Chenyang-ibrl/d07c4310-7780-4dff-8283-c49ab8bfe257/scratchpad")
    return p.parse_args()


def build_flat(buf: SequenceReplayBuffer):
    """Flatten episodes into (s, a, s_next), SKIPPING the duplicated terminal
    transition (loader sets s_T := s_{T-1}, a zero-delta dummy)."""
    S, A, SN = [], [], []
    for ep in buf.episodes:
        s = ep["state"]  # [T+1, obs]  (torch)
        a = ep["action"]  # [T, act]
        T = a.shape[0]
        if T < 2:
            continue
        # usable transitions t = 0 .. T-2  (t=T-1 has the duplicated next-state)
        S.append(s[:T - 1])
        A.append(a[:T - 1])
        SN.append(s[1:T])
    S = torch.cat(S, 0).float().numpy()
    A = torch.cat(A, 0).float().numpy()
    SN = torch.cat(SN, 0).float().numpy()
    return S, A, SN


def phase_masks(S):
    d = np.linalg.norm(S[:, NEEDLE] - S[:, RING], axis=1)  # meters
    return {
        "free   (>=15cm)": d >= 0.15,
        "approach(5-15cm)": (d >= 0.05) & (d < 0.15),
        "contact(<5cm)  ": d < 0.05,
    }, d


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---- data ----
    buf = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(buf, args.data_path, num_data=args.num_data)
    S, A, SN = build_flat(buf)
    del buf
    log(f"[data] {S.shape[0]} usable transitions, obs={S.shape[1]} act={A.shape[1]}")

    masks, dist = phase_masks(S)
    for name, m in masks.items():
        log(f"  phase {name}: {m.sum():>8d}  ({100*m.mean():.1f}%)")

    # ---- model ----
    obs_dim, act_dim = S.shape[1], A.shape[1]
    wm_cfg = WorldModelConfig(encoder_type="identity", mlp_hidden_dim=512, num_layer=2,
                              layer_norm=1, horizon=5, rho=0.5, activation="relu")
    model = WorldModel(obs_dim, act_dim, wm_cfg).to(dev)
    sd = torch.load(args.dyn_ckpt, map_location=dev)
    model.load_state_dict(sd, strict=False)
    model.eval()
    mean = model.norm_state_mean.cpu().numpy()
    std = model.norm_state_std.cpu().numpy()
    log(f"[model] loaded {args.dyn_ckpt}; norm |mean|avg={np.abs(mean).mean():.3f} std avg={std.mean():.3f}")

    # ---- A) model 1-step residual (batched) ----
    resid = np.zeros_like(SN)  # pred - true, RAW units
    resid_norm = np.zeros_like(SN)  # in z-scored space
    bs = 8192
    with torch.no_grad():
        for i in range(0, S.shape[0], bs):
            s = torch.from_numpy(S[i:i + bs]).to(dev)
            a = torch.from_numpy(A[i:i + bs]).to(dev)
            z = model.encode(s)
            zp = model.next(z, a)
            pred = model.decode(zp)
            resid[i:i + bs] = (pred - torch.from_numpy(SN[i:i + bs]).to(dev)).cpu().numpy()
            resid_norm[i:i + bs] = ((pred - torch.from_numpy(SN[i:i + bs]).to(dev)) / model.norm_state_std).cpu().numpy()

    def needle_mm_rms(r, m):
        return float(np.sqrt((np.linalg.norm(r[m][:, NEEDLE], axis=1) ** 2).mean()) * 1000.0)

    def fullnorm_rms(r, m):
        return float(np.sqrt((r[m] ** 2).sum(1).mean()))

    log("\n==== A) MODEL 1-step residual (deterministic MLP best.pt) ====")
    res_needle_mm, res_full = {}, {}
    for name, m in masks.items():
        res_needle_mm[name] = needle_mm_rms(resid, m)
        res_full[name] = fullnorm_rms(resid_norm, m)
        log(f"  {name}: needle_RMS={res_needle_mm[name]:6.2f} mm | fullstate_normRMS={res_full[name]:.4f}")

    # ---- B) k-NN aleatoric floor: PER-PHASE self-pool + radius binning ----
    # For each phase, search neighbors WITHIN that phase's own transitions, then
    # report needle-next spread as a function of the (s,a) neighbor radius. As
    # radius->0 the smooth (s,a)->s' variation vanishes, so the small-radius
    # intercept is the ALEATORIC (obs-multimodality) floor. Comparing contact vs
    # free at the SAME tight radius isolates genuine multimodality from the fact
    # that contact is data-sparse (larger typical radius).
    zS = (S - mean) / std
    feat = np.concatenate([zS, args.action_weight * A], axis=1).astype(np.float32)

    def knn_within(pool_i, query_i):
        pf = torch.from_numpy(feat[pool_i]).to(dev)
        p_needle = torch.from_numpy(SN[pool_i][:, NEEDLE]).to(dev)
        radii, needle_std_mm = [], []
        qb = 128
        for i in range(0, query_i.shape[0], qb):
            qf = torch.from_numpy(feat[query_i[i:i + qb]]).to(dev)
            d = torch.cdist(qf, pf)
            vals, nn = torch.topk(d, args.k + 1, largest=False)  # +1 to drop self
            r = vals[:, args.k]
            nn = nn[:, 1:]
            nb = p_needle[nn]                                    # [qb,k,3] m
            ns = (nb - nb.mean(1, keepdim=True)).norm(dim=2)     # [qb,k]
            needle_std_mm.append((ns.pow(2).mean(1).sqrt() * 1000.0).cpu())
            radii.append(r.cpu())
        return torch.cat(radii).numpy(), torch.cat(needle_std_mm).numpy()

    log("\n==== B) k-NN ALEATORIC FLOOR (per-phase self-pool, spread of s' among near (s,a)) ====")
    per_phase = {}
    RADIUS_BINS = [0.0, 0.15, 0.30, 0.50, 0.80, 1.2, 10.0]
    for name, m in masks.items():
        cand = np.where(m)[0]
        if cand.shape[0] < 50:
            log(f"  {name}: too few ({cand.shape[0]}), skip"); continue
        pool_i = cand if cand.shape[0] <= args.pool_cap else np.random.choice(cand, args.pool_cap, replace=False)
        query_i = np.random.choice(cand, size=min(args.n_query, cand.shape[0]), replace=False)
        r, ns_mm = knn_within(pool_i, query_i)
        per_phase[name] = (r, ns_mm)
        log(f"  {name}: pool={pool_i.shape[0]:>7d}  median_radius={np.median(r):.3f}  "
            f"overall needle_next_std(median)={np.median(ns_mm):6.2f} mm")
        # radius-binned spread (the aleatoric floor = smallest-radius bin)
        line = "      by (s,a)-radius bin:  "
        for lo, hi in zip(RADIUS_BINS[:-1], RADIUS_BINS[1:]):
            sel = (r >= lo) & (r < hi)
            if sel.sum() >= 15:
                line += f"[{lo:.2f}-{hi:.2f}]={np.median(ns_mm[sel]):5.1f}mm(n{sel.sum()})  "
        log(line)

    # ---- C) decisive comparison ----
    log("\n==== C) VERDICT ====")
    log(f"  {'phase':18s} {'model_resid_mm':>14s}   1-step deterministic-MLP error")
    for name in masks:
        log(f"  {name:18s} {res_needle_mm[name]:14.2f}")
    # contact-vs-free aleatoric floor at a MATCHED tight radius bin
    log("\n  contact-vs-free needle-next spread at MATCHED (s,a) radius (isolates multimodality):")
    for lo, hi in zip(RADIUS_BINS[:-1], RADIUS_BINS[1:]):
        row = f"    radius[{lo:.2f}-{hi:.2f}]: "
        vals = {}
        for name in ("free   (>=15cm)", "contact(<5cm)  "):
            if name in per_phase:
                r, ns_mm = per_phase[name]
                sel = (r >= lo) & (r < hi)
                if sel.sum() >= 15:
                    vals[name] = float(np.median(ns_mm[sel]))
                    row += f"{name.strip()}={vals[name]:5.1f}mm(n{sel.sum()})  "
        if len(vals) == 2:
            row += f"=> contact/free = {vals['contact(<5cm)  '] / max(vals['free   (>=15cm)'], 1e-6):.2f}"
        if vals:
            log(row)
    log("    (contact/free >> 1 at small radius => genuine contact multimodality)")
    per_phase_data = per_phase  # for the histogram block below

    # ---- residual bimodality (contact phase) ----
    log("\n==== D) contact-phase residual bimodality (per state dim) ====")
    cmask = masks["contact(<5cm)  "]
    rc = resid_norm[cmask]  # normalized residual in contact
    from scipy import stats as sstats
    bc_dims = []
    for d_ in range(obs_dim):
        x = rc[:, d_]
        if x.std() < 1e-8:
            continue
        g1 = sstats.skew(x)
        g2 = sstats.kurtosis(x, fisher=True)  # excess
        n = len(x)
        # Sarle's bimodality coefficient (corrected); >0.555 suggests bimodal
        b = (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        bc_dims.append((d_, b, g1, g2, x.std()))
    bc_dims.sort(key=lambda t: -t[1])
    log(f"  {'dim':>4s} {'bimod_coef':>10s} {'skew':>7s} {'exkurt':>8s} {'norm_std':>9s}  (BC>0.555 ~ bimodal)")
    for d_, b, g1, g2, sd_ in bc_dims[:12]:
        flag = " <== " if b > 0.555 else ""
        tag = "NEEDLE" if 9 <= d_ < 12 else ("RING" if 23 <= d_ < 26 else "")
        log(f"  {d_:4d} {b:10.3f} {g1:7.2f} {g2:8.2f} {sd_:9.4f}  {tag}{flag}")
    n_bimodal = sum(1 for _, b, *_ in bc_dims if b > 0.555)
    log(f"  #dims with BC>0.555 (contact): {n_bimodal}/{obs_dim}")

    # ---- histograms ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    # top row: needle residual per axis (contact vs free), + needle delta
    for j, (ax, comp) in enumerate(zip(axes[0][:3], ["x", "y", "z"])):
        dcomp = 9 + j
        ax.hist(resid[cmask][:, dcomp] * 1000, bins=80, alpha=0.6, density=True, label="contact")
        ax.hist(resid[masks["free   (>=15cm)"]][:, dcomp] * 1000, bins=80, alpha=0.5, density=True, label="free")
        ax.set_title(f"needle-{comp} residual (mm)")
        ax.legend(); ax.set_yscale("log")
    # needle next-position spread among neighbors (contact)
    ax = axes[0][3]
    if "contact(<5cm)  " in per_phase_data:
        ax.hist(per_phase_data["contact(<5cm)  "][1], bins=60, alpha=0.7, label="contact")
    if "free   (>=15cm)" in per_phase_data:
        ax.hist(per_phase_data["free   (>=15cm)"][1], bins=60, alpha=0.6, label="free")
    ax.set_title("k-NN needle-next std (mm)"); ax.legend()
    # bottom row: top-4 bimodal residual dims (contact)
    for ax, (d_, b, *_ ) in zip(axes[1], bc_dims[:4]):
        ax.hist(rc[:, d_], bins=100, density=True)
        tag = "NEEDLE" if 9 <= d_ < 12 else ("RING" if 23 <= d_ < 26 else "")
        ax.set_title(f"dim {d_} {tag} resid (norm), BC={b:.2f}")
    fig.suptitle("dynamics contact-phase multimodality diagnostic")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "dyn_multimodality.png")
    fig.savefig(out, dpi=90)
    log(f"\n[fig] saved -> {out}")
    os._exit(0)


if __name__ == "__main__":
    main()
