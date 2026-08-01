"""Step ②: train an ENSEMBLE of multi-step (window) counterfactual discriminators.

Counterfactual pairs share (s0, a_seq); the source window (k-step source rollout)
is class 1, the target window (demo segment) is class 0. d_sa collapses to 0.5
(shared inputs); d_sas must key on the compounded k-step dynamics divergence.
5 members (different seeds) for the epistemic-disagreement OOD gate later.
"""

import argparse
import os

import numpy as np
import torch

from ms_discriminator import MultiStepDiscriminator
from data_utils import episode_split


def to_t(x, dev):
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def train_member(seed, data, args, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    s0, aseq, src, tgt, tr, va = data
    # normalization over the train split
    allw = np.concatenate([src[tr], tgt[tr]], 0)
    d = MultiStepDiscriminator(37, 7, args.k, args.hidden, bool(args.dropout)).to(dev)
    d.set_norm(s0[tr].mean(0), s0[tr].std(0), aseq[tr].mean(0), aseq[tr].std(0),
               allw.mean(0), allw.std(0))
    opt_sa = torch.optim.Adam(d.d_sa.parameters(), lr=args.lr)
    opt_sas = torch.optim.Adam(d.d_sas.parameters(), lr=args.lr)

    S0, ASEQ, SRC, TGT = (to_t(x[tr], dev) for x in (s0, aseq, src, tgt))
    S0v, ASEQv, SRCv, TGTv = (to_t(x[va], dev) for x in (s0, aseq, src, tgt))
    n = len(S0); eps = 1e-8

    def losses(s0b, ab, srcb, tgtb):
        if args.noise_std > 0:
            ns = args.noise_std
            s0b = s0b + torch.randn_like(s0b) * ns
            ab = ab + torch.randn_like(ab) * ns
            srcb = srcb + torch.randn_like(srcb) * ns
            tgtb = tgtb + torch.randn_like(tgtb) * ns
        _, tgt_sas = d.prob_sas(s0b, ab, tgtb); tgt_sa = d.prob_sa(s0b, ab)
        _, src_sas = d.prob_sas(s0b, ab, srcb); src_sa = d.prob_sa(s0b, ab)
        dsa = (-torch.log(tgt_sa[:, 0] + eps) - torch.log(src_sa[:, 1] + eps)).mean()
        dsas = (-torch.log(tgt_sas[:, 0] + eps) - torch.log(src_sas[:, 1] + eps)).mean()
        return dsa, dsas

    for step in range(1, args.steps + 1):
        d.train()
        idx = torch.randint(0, n, (args.batch_size,), device=dev)
        dsa, dsas = losses(S0[idx], ASEQ[idx], SRC[idx], TGT[idx])
        opt_sa.zero_grad(); dsa.backward(retain_graph=True)
        opt_sas.zero_grad(); dsas.backward()
        opt_sa.step(); opt_sas.step()

    d.eval()
    with torch.no_grad():
        _, tgt_sas = d.prob_sas(S0v, ASEQv, TGTv)
        _, src_sas = d.prob_sas(S0v, ASEQv, SRCv)
        acc = 0.5 * ((tgt_sas[:, 0] > 0.5).float().mean()
                     + (src_sas[:, 1] > 0.5).float().mean()).item()
    return d, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{os.path.dirname(__file__)}/pairs_multistep.npz")
    p.add_argument("--out_dir", default=f"{os.path.dirname(__file__)}/disc_ms")
    p.add_argument("--n_members", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    z = np.load(args.pairs)
    k = int(z["k"]); args.k = k
    s0 = z["s0"]; aseq = z["acts"].reshape(len(s0), -1)        # (N, k*7)
    src = z["src_win"].reshape(len(s0), -1)                    # (N, k*37)
    tgt = z["tgt_win"].reshape(len(s0), -1)
    ep = z["ep_id"]
    tr, va = episode_split(ep, args.val_ratio, args.split_seed)
    print(f"{len(s0)} window pairs (k={k}), train {tr.sum()} / val {va.sum()}")
    data = (s0, aseq, src, tgt, tr, va)

    for m in range(args.n_members):
        d, acc = train_member(m, data, args, args.device)
        out = os.path.join(args.out_dir, f"disc_{m}.pt")
        torch.save({"state_dict": d.state_dict(), "k": k, "hidden": args.hidden,
                    "dropout": bool(args.dropout)}, out)
        print(f"  member {m}: val sas-acc {acc:.3f} -> {out}")
    print(f"saved {args.n_members} members to {args.out_dir}")


if __name__ == "__main__":
    main()
