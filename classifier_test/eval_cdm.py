"""Decisive comparison of per-transition gap signals.

Signals compared:
  - disc        : OLD H2O discriminator |log dyn-ratio|
  - dgi         : conditional-density two-head DGI ||E[Δs'|tgt]-E[Δs'|src]||
  - src_resid   : SOURCE-ONLY normalized residual ||Δs'_norm - μ_src(s,a)||
                  (the stable form of "how surprised is the source model" — the
                  user's improvement-1 signal; OOD-robust via broad source data)

Two regimes:
  (A) held-out anchor pairs: signal should RISE with ground-truth gap (Spearman),
      and be LOW on the s'_src reference.
  (B) flailing source rollouts (GT gap ≈ 0): signal should stay LOW. A detector
      that hallucinates here (like the discriminator) is the failure we fix.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cf_utils import REPO, counterfactual_step, get_sim_state, get_env
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from conditional_density import build_model
from common_utils import ibrl_utils as utils


def load_disc(p, dev):
    ck = torch.load(p, map_location=dev)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(dev)
    d.load_state_dict(ck["state_dict"]); d.eval(); return d


def load_cdm(p, dev):
    ck = torch.load(p, map_location=dev)
    m = build_model(ck["head"], ck["state_dim"], ck["action_dim"], ck["ds_dim"], hidden=ck["hidden"]).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def src_resid(cdm, s, a, sp):
    """||normalized Δs' - μ_src(s,a)|| — stable source-only surprise signal."""
    S, A, SP = (torch.as_tensor(x, dtype=torch.float32, device=cdm.s_mean.device) for x in (s, a, sp))
    with torch.no_grad():
        dsn = cdm.nz(SP - S, "ds")
        mu = cdm._mean(S, A, 0)  # source head mean in normalized Δs' space
    return (dsn - mu).norm(dim=-1).cpu().numpy()


def disc_score(disc, s, a, sp):
    S, A, SP = (torch.as_tensor(x, dtype=torch.float32, device=disc.s_mean.device) for x in (s, a, sp))
    with torch.no_grad():
        return np.abs(disc.log_dynamics_ratio(S, A, SP).cpu().numpy())


def dgi_score(cdm, s, a, sp):
    S, A, SP = (torch.as_tensor(x, dtype=torch.float32, device=cdm.s_mean.device) for x in (s, a, sp))
    with torch.no_grad():
        return cdm.dgi(S, A, SP).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--cdm", default=f"{os.path.dirname(__file__)}/cdm_gaussian.pt")
    p.add_argument("--cache", default=f"{os.path.dirname(__file__)}/cdm_data.npz")
    p.add_argument("--num_flail", type=int, default=6)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/eval_cdm.png")
    args = p.parse_args()
    dev = "cuda"

    disc = load_disc(args.disc, dev)
    cdm = load_cdm(args.cdm, dev)
    d = np.load(args.cache)

    # (A) held-out anchors
    va = d["a_va"]
    s, a, ssrc, stgt = d["a_s"][va], d["a_a"][va], d["a_src"][va], d["a_tgt"][va]
    gt = d["a_gap"][va]
    sig = {
        "disc": (disc_score(disc, s, a, stgt), disc_score(disc, s, a, ssrc)),
        "dgi": (dgi_score(cdm, s, a, stgt), dgi_score(cdm, s, a, ssrc)),
        "src_resid": (src_resid(cdm, s, a, stgt), src_resid(cdm, s, a, ssrc)),
    }
    print("=== (A) held-out anchors: Spearman(signal on s'_tgt, GT gap) ; "
          "mean on s'_tgt vs s'_src(ref) ===")
    for k, (vt, vs) in sig.items():
        print(f"  {k:10s} rho={spearmanr(vt, gt)[0]:+.3f} | "
              f"mean(s'_tgt)={vt.mean():.3f}  mean(s'_src ref)={vs.mean():.3f}")

    # (C) held-out REPLAY transitions — in-distribution suboptimal source data
    # (the user's exact scenario). These are real source-dynamics transitions;
    # the discriminator (trained on demo anchors only) sees them as OOD and
    # should hallucinate, while the conditional density (trained ON the replay)
    # should give a LOW score. No sim_state -> no per-row GT, but they are
    # source-generated so a faithful source model must rate them low.
    rva = d["r_va"]
    ridx = np.random.default_rng(0).choice(int(rva.sum()), size=min(4000, int(rva.sum())), replace=False)
    rs, ra, rns = d["r_s"][rva][ridx], d["r_a"][rva][ridx], d["r_ns"][rva][ridx]
    print("\n=== (C) held-out REPLAY (in-dist suboptimal source; true gap≈0) ===")
    for k, fn in [("disc", lambda S, A, SP: disc_score(disc, S, A, SP)),
                  ("dgi", lambda S, A, SP: dgi_score(cdm, S, A, SP)),
                  ("src_resid", lambda S, A, SP: src_resid(cdm, S, A, SP))]:
        v = fn(rs, ra, rns)
        hi = sig[k][0][gt > np.quantile(gt, 0.8)].mean()
        print(f"  {k:10s} replay mean={v.mean():.3f}  (high-gap-anchor mean={hi:.3f}; "
              f"want replay << high-gap)")

    # (B) flailing source rollouts (GT gap via counterfactual target step)
    tgt_env = get_env(REPO + "/exps/mimicgen/threading_ibrl_source_seed1/model0.pt", domain="target")
    print("\n=== (B) flailing source rollouts: signal mean (GT gap≈0 -> want LOW) ===")
    fl = flailing_with_sim(args.num_flail, args.horizon, args.seed, tgt_env)
    allgt = np.concatenate([f["gt"] for f in fl])
    print(f"  GT gap mean={allgt.mean():.4f} median={np.median(allgt):.4f}")
    flsig = {}
    for k, fn in [("disc", lambda S, A, SP: disc_score(disc, S, A, SP)),
                  ("dgi", lambda S, A, SP: dgi_score(cdm, S, A, SP)),
                  ("src_resid", lambda S, A, SP: src_resid(cdm, S, A, SP))]:
        v = np.concatenate([fn(f["s"], f["a"], f["sp"]) for f in fl])
        flsig[k] = v
        # ratio vs its own value on high-gap anchors = how well it separates
        hi_anchor = sig[k][0][gt > np.quantile(gt, 0.8)].mean()
        print(f"  {k:10s} flailing mean={v.mean():.3f}  (high-gap-anchor mean={hi_anchor:.3f}; "
              f"want flailing << high-gap)")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k in zip(axes, ["disc", "dgi", "src_resid"]):
        vt = sig[k][0]
        ax.scatter(gt, vt, s=8, alpha=0.4, color="C0", label="anchor (vs GT gap)")
        # flailing as a strip at gt≈0
        fv = flsig[k]
        ax.scatter(np.full_like(fv, allgt.mean()), fv, s=6, alpha=0.2, color="C3",
                   label="flailing (GT≈0)")
        ax.set_xlabel("GT gap"); ax.set_ylabel(f"{k} score")
        ax.set_title(f"{k}: anchor rho={spearmanr(vt, gt)[0]:.2f}\n"
                     f"flail mean {fv.mean():.2f} vs hi-anchor {vt[gt>np.quantile(gt,0.8)].mean():.2f}")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"\nsaved -> {args.out}")


def flailing_with_sim(num_traj, horizon, seed, tgt_env):
    """Flailing rollout capturing sim_state to compute the counterfactual GT gap."""
    src_env = get_env(REPO + "/exps/mimicgen/threading_ibrl_source_seed1/model0.pt",
                      domain="source", hand_init_sigma=0.1)
    import train_bc
    agent, _, _ = train_bc.load_model(REPO + "/exps/mimicgen/threading_flow_bc_seed1/model0.pt", "cuda")
    out = []
    with torch.no_grad(), utils.eval_mode(agent):
        for ti in range(num_traj):
            np.random.seed(seed + ti)
            obs, _ = src_env.reset()
            S, A, SIM = [], [], []
            for _ in range(horizon):
                SIM.append(get_sim_state(src_env)); S.append(obs["state"].cpu().numpy())
                act = (agent.act(obs, eval_mode=True).cpu() + 0.5 * torch.randn(7)).clamp(-1, 1)
                obs, r, term, _, _ = src_env.step(act); A.append(act.cpu().numpy())
                if term:
                    break
            S, A, SIM = np.asarray(S, np.float32), np.asarray(A, np.float32), np.asarray(SIM, np.float64)
            n = len(S) - 1
            stgt = np.stack([counterfactual_step(tgt_env, SIM[t], A[t])[0] for t in range(n)]).astype(np.float32)
            out.append(dict(s=S[:n], a=A[:n], sp=S[1:n + 1], gt=state_gap(S[1:n + 1], stgt)))
    return out


if __name__ == "__main__":
    main()
