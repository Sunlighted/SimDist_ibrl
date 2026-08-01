"""Train the residual-parameterized conditional-density model (二 + 四).

二 (zero-init residual): μ(c=1)=μ_base+Δμ, Δμ last layer zero-init + stronger
   weight decay -> DGI=||Δμ||→0 on OOD by construction (graceful degradation).
三 (counterfactual mixup): each anchor contributes (s,a,s'_src,c=0) AND
   (s,a,s'_tgt,c=1) — same (s,a), so the only thing separating them is Δs'.
四 (contrastive energy margin, GAP-AWARE): for anchors with a real gap, push
   E(real|c=0) - E(real|c=1) >= margin; gated by the ground-truth gap so we do
   NOT force separation in genuine no-gap (free-space) regions.

c=0 (base) is trained on the suboptimal replay + anchor s'_src; c=1 (base+res)
on anchor s'_tgt.
"""

import argparse
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from density_data import build_dataset, norm_stats
from conditional_density import build_model


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=f"{os.path.dirname(__file__)}/cdm_data.npz")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/cdm_gaussian_res.pt")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--target_frac", type=float, default=0.5)
    p.add_argument("--anchor_src_frac", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base_wd", type=float, default=1e-5)
    p.add_argument("--res_wd", type=float, default=1e-2,
                   help="stronger L2 on the target residual branch -> 0 on OOD")
    p.add_argument("--contrast", type=float, default=1.0, help="λ for the margin loss")
    p.add_argument("--margin", type=float, default=0.5, help="energy margin M")
    p.add_argument("--l2_delta", type=float, default=0.0,
                   help="penalty on ||Δμ(s,a)|| over the whole batch -> sparse gap "
                        "(Δμ defaults to 0 unless data/contrast pushes it up)")
    p.add_argument("--direct_gap", type=float, default=0.0,
                   help="λ for supervising the residual to regress the CLEAN sim2sim "
                        "gap vector (s'_tgt - s'_src)/σ on anchors (near-deterministic "
                        "physics -> clean target, not noise). Δμ learns the systematic "
                        "gap, ~0 on lo-gap anchors, ~0 off-support.")
    p.add_argument("--gap_gate_q", type=float, default=0.5,
                   help="apply margin only to anchors with GT gap above this quantile")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    d = build_dataset(cache=args.cache)
    ns = norm_stats(d)
    sdim, adim = d["r_s"].shape[1], d["r_a"].shape[1]

    m = build_model("gaussian_res", sdim, adim, sdim, hidden=args.hidden).to(dev)
    m.set_norm(**ns)
    opt = torch.optim.Adam([
        {"params": m.base_params(), "weight_decay": args.base_wd},
        {"params": m.residual_params(), "weight_decay": args.res_wd},
    ], lr=args.lr)

    # pools (train split)
    rep_s = to_t(d["r_s"][d["r_tr"]], dev); rep_a = to_t(d["r_a"][d["r_tr"]], dev)
    rep_ns = to_t(d["r_ns"][d["r_tr"]], dev)
    an_s = to_t(d["a_s"][d["a_tr"]], dev); an_a = to_t(d["a_a"][d["a_tr"]], dev)
    an_src = to_t(d["a_src"][d["a_tr"]], dev); an_tgt = to_t(d["a_tgt"][d["a_tr"]], dev)
    an_gap = to_t(d["a_gap"][d["a_tr"]], dev)
    gap_thresh = float(torch.quantile(an_gap, args.gap_gate_q))
    n_rep, n_an = len(rep_s), len(an_s)
    print(f"pools: replay {n_rep}, anchor {n_an}; gap-gate>{gap_thresh:.4f}")

    def ridx(n, nb):
        return torch.randint(0, n, (nb,), device=dev)

    for step in range(1, args.steps + 1):
        m.train()
        B = args.batch_size
        n_tgt = int(B * args.target_frac); n_src = B - n_tgt
        n_sa = int(n_src * args.anchor_src_frac); n_sr = n_src - n_sa
        it = ridx(n_an, n_tgt); ir = ridx(n_rep, n_sr); ia = ridx(n_an, n_sa)
        # c=1 target rows (an_src[it] is the paired counterfactual for direct_gap)
        s1, a1, sp1 = an_s[it], an_a[it], an_tgt[it]
        # c=0 source rows (replay + anchor s'_src)
        s0 = torch.cat([rep_s[ir], an_s[ia]], 0); a0 = torch.cat([rep_a[ir], an_a[ia]], 0)
        sp0 = torch.cat([rep_ns[ir], an_src[ia]], 0)

        s = torch.cat([s1, s0], 0); a = torch.cat([a1, a0], 0); sp = torch.cat([sp1, sp0], 0)
        is_t = torch.cat([torch.ones(n_tgt, device=dev), torch.zeros(n_src, device=dev)], 0)
        nll = m.loss(s, a, sp - s, is_t)

        # contrastive margin on target anchor rows WITH a real gap
        ct = torch.zeros((), device=dev)
        gap_it = an_gap[it]
        mask = gap_it > gap_thresh
        if args.contrast > 0 and mask.any():
            sm, am, spm = s1[mask], a1[mask], sp1[mask]
            e0 = m.energy(sm, am, spm, torch.zeros(mask.sum(), device=dev))  # under sim
            e1 = m.energy(sm, am, spm, torch.ones(mask.sum(), device=dev))   # under target
            ct = torch.clamp(args.margin - (e0 - e1), min=0).mean()

        # output sparsity on the residual (over the whole batch's (s,a))
        l2d = torch.zeros((), device=dev)
        if args.l2_delta > 0:
            l2d = m._delta(s, a).norm(dim=-1).mean()

        # direct supervision: residual regresses the clean sim2sim gap vector
        dg = torch.zeros((), device=dev)
        if args.direct_gap > 0:
            tgt_delta = m.nz(an_tgt[it] - an_s[it], "ds") - m.nz(an_src[it] - an_s[it], "ds")
            dg = ((m._delta(s1, a1) - tgt_delta) ** 2).sum(-1).mean()

        loss = nll + args.contrast * ct + args.l2_delta * l2d + args.direct_gap * dg
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                vs = to_t(d["a_s"][d["a_va"]], dev); va = to_t(d["a_a"][d["a_va"]], dev)
                dgi = m.dgi(vs, va).cpu().numpy()
            gt = d["a_gap"][d["a_va"]]
            rho = spearmanr(dgi, gt)[0]
            hi = dgi[gt > np.quantile(gt, 0.8)].mean(); lo = dgi[gt < np.quantile(gt, 0.5)].mean()
            print(f"[{step:6d}] nll {nll.item():.3f} ct {float(ct):.3f} l2d {float(l2d):.3f} | "
                  f"DGI hi/lo {hi:.3f}/{lo:.3f} | Spearman(DGI,GT)={rho:.3f}")

    torch.save({"state_dict": m.state_dict(), "head": "gaussian_res",
                "state_dim": sdim, "action_dim": adim, "ds_dim": sdim,
                "hidden": args.hidden}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
