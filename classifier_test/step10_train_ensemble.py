"""Step 10: train an ENSEMBLE of counterfactual dynamics discriminators.

Same counterfactual-pair training as step3, but N members with different
init/minibatch seeds (a deep ensemble; Lakshminarayanan et al.). At eval we
score with  ensemble_mean(|log dyn-ratio|) - kappa * ensemble_std, so that
OOD (s,a,s') — where members DISagree (high epistemic std) — get pushed DOWN,
while in-distribution transitions (members agree, low std) keep their score.

Episode split is FIXED across members (same train/val pairs); only the torch
seed varies (weight init + SGD minibatch order) → epistemic diversity. Member 0
reproduces the default single discriminator (discriminator_cf.pt) for a fair
single-vs-ensemble comparison.
"""

import argparse
import os

import numpy as np
import torch

from discriminator import DynamicsDiscriminator
from data_utils import episode_split


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_member(seed, data, args, device):
    """Train one discriminator with the given seed. Returns the module."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    s, a, s_src, s_tgt, tr, va = data

    all_s = np.concatenate([s[tr], s_src[tr], s_tgt[tr]], 0)
    d = DynamicsDiscriminator(s.shape[1], a.shape[1], args.hidden, bool(args.dropout)).to(device)
    d.set_norm(all_s.mean(0), all_s.std(0), a[tr].mean(0), a[tr].std(0))

    opt_sa = torch.optim.Adam(d.d_sa.parameters(), lr=args.lr)
    opt_sas = torch.optim.Adam(d.d_sas.parameters(), lr=args.lr)

    S, A, SRC, TGT = (to_t(x[tr], device) for x in (s, a, s_src, s_tgt))
    Sv, Av, SRCv, TGTv = (to_t(x[va], device) for x in (s, a, s_src, s_tgt))
    n = len(S)
    eps = 1e-8

    def losses(sb, ab, srcb, tgtb):
        if args.noise_std > 0:
            ns = args.noise_std
            sb = sb + torch.randn_like(sb) * ns
            ab = ab + torch.randn_like(ab) * ns
            srcb = srcb + torch.randn_like(srcb) * ns
            tgtb = tgtb + torch.randn_like(tgtb) * ns
        _, tgt_sas = d.prob_sas(sb, ab, tgtb)
        tgt_sa = d.prob_sa(sb, ab)
        _, src_sas = d.prob_sas(sb, ab, srcb)
        src_sa = d.prob_sa(sb, ab)
        dsa = (-torch.log(tgt_sa[:, 0] + eps) - torch.log(src_sa[:, 1] + eps)).mean()
        dsas = (-torch.log(tgt_sas[:, 0] + eps) - torch.log(src_sas[:, 1] + eps)).mean()
        return dsa, dsas

    for step in range(1, args.steps + 1):
        d.train()
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        dsa, dsas = losses(S[idx], A[idx], SRC[idx], TGT[idx])
        opt_sa.zero_grad(); dsa.backward(retain_graph=True)
        opt_sas.zero_grad(); dsas.backward()
        opt_sa.step(); opt_sas.step()

    d.eval()
    with torch.no_grad():
        _, tgt_sas = d.prob_sas(Sv, Av, TGTv)
        _, src_sas = d.prob_sas(Sv, Av, SRCv)
        acc = 0.5 * ((tgt_sas[:, 0] > 0.5).float().mean()
                     + (src_sas[:, 1] > 0.5).float().mean()).item()
    return d, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{os.path.dirname(__file__)}/pairs_demo.npz")
    p.add_argument("--out_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--n_members", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0,
                   help="FIXED across members so all see the same train/val pairs")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    z = np.load(args.pairs)
    s, a, s_src, s_tgt, ep = z["s"], z["a"], z["s_src"], z["s_tgt"], z["ep_id"]
    tr, va = episode_split(ep, args.val_ratio, args.split_seed)
    print(f"{len(s)} pairs, {len(np.unique(ep))} episodes; train {tr.sum()} / val {va.sum()}")
    data = (s, a, s_src, s_tgt, tr, va)

    for k in range(args.n_members):
        d, acc = train_member(seed=k, data=data, args=args, device=args.device)
        out = os.path.join(args.out_dir, f"disc_{k}.pt")
        torch.save({"state_dict": d.state_dict(), "state_dim": s.shape[1],
                    "action_dim": a.shape[1], "hidden": args.hidden,
                    "dropout": bool(args.dropout)}, out)
        print(f"  member {k} (seed {k}): val sas-acc {acc:.3f} -> {out}")
    print(f"saved {args.n_members} members to {args.out_dir}")


if __name__ == "__main__":
    main()
