"""Step 3: train the dynamics discriminator on counterfactual pairs.

For each pair we make two examples sharing the SAME (s, a):
    (s, a, s'_tgt) -> target (class 0)
    (s, a, s'_src) -> source (class 1)
Because (s, a) is identical across the two classes, d_sa CANNOT separate them
(its loss should sit at log 2 ≈ 0.693, prob ≈ 0.5), forcing d_sas to key only
on the next state s' — i.e. the pure dynamics difference. This is the regime
H2O's ratio is designed for.
"""

import argparse
import os

import numpy as np
import torch

from discriminator import DynamicsDiscriminator
from data_utils import episode_split


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{os.path.dirname(__file__)}/pairs_demo.npz")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise_std", type=float, default=0.02,
                   help="small input noise; keep low so the sparse gap signal survives")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--zero_early", type=int, default=0,
                   help="for the first K steps of each episode, set s'_tgt := s'_src "
                        "so the pair has GT gap=0 -> teaches the discriminator to "
                        "output ~0 there (kills the start-transient false positive "
                        "while keeping those steps visible at test time)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    z = np.load(args.pairs)
    s, a, s_src, s_tgt, ep = z["s"], z["a"], z["s_src"], z["s_tgt"], z["ep_id"]
    print(f"{len(s)} pairs, {len(np.unique(ep))} episodes")

    if args.zero_early > 0:
        ts = z["t_in_ep"]
        early = ts < args.zero_early
        s_tgt = s_tgt.copy()
        s_tgt[early] = s_src[early]   # GT gap = 0 for the first K steps
        print(f"zero_early={args.zero_early}: forced GT gap=0 on {early.sum()} "
              f"start-transient pairs")

    tr, va = episode_split(ep, args.val_ratio, args.seed)
    print(f"train {tr.sum()} / val {va.sum()} pairs")

    # normalization over all states (s, s_src, s_tgt) and actions
    all_s = np.concatenate([s[tr], s_src[tr], s_tgt[tr]], 0)
    d = DynamicsDiscriminator(s.shape[1], a.shape[1], args.hidden, bool(args.dropout)).to(device)
    d.set_norm(all_s.mean(0), all_s.std(0), a[tr].mean(0), a[tr].std(0))

    opt_sa = torch.optim.Adam(d.d_sa.parameters(), lr=args.lr)
    opt_sas = torch.optim.Adam(d.d_sas.parameters(), lr=args.lr)

    S, A, SRC, TGT = (to_t(x[tr], device) for x in (s, a, s_src, s_tgt))
    Sv, Av, SRCv, TGTv = (to_t(x[va], device) for x in (s, a, s_src, s_tgt))
    n = len(S)

    def sample(nb):
        idx = torch.randint(0, n, (nb,), device=device)
        return S[idx], A[idx], SRC[idx], TGT[idx]

    def losses(sb, ab, srcb, tgtb):
        if args.noise_std > 0:
            ns = args.noise_std
            sb = sb + torch.randn_like(sb) * ns
            ab = ab + torch.randn_like(ab) * ns
            srcb = srcb + torch.randn_like(srcb) * ns
            tgtb = tgtb + torch.randn_like(tgtb) * ns
        eps = 1e-8
        # target = class 0, source = class 1; both share (sb, ab)
        _, tgt_sas = d.prob_sas(sb, ab, tgtb)
        tgt_sa = d.prob_sa(sb, ab)
        _, src_sas = d.prob_sas(sb, ab, srcb)
        src_sa = d.prob_sa(sb, ab)
        dsa = (-torch.log(tgt_sa[:, 0] + eps) - torch.log(src_sa[:, 1] + eps)).mean()
        dsas = (-torch.log(tgt_sas[:, 0] + eps) - torch.log(src_sas[:, 1] + eps)).mean()
        return dsa, dsas

    for step in range(1, args.steps + 1):
        d.train()
        sb, ab, srcb, tgtb = sample(args.batch_size)
        dsa, dsas = losses(sb, ab, srcb, tgtb)
        opt_sa.zero_grad(); dsa.backward(retain_graph=True)
        opt_sas.zero_grad(); dsas.backward()
        opt_sa.step(); opt_sas.step()

        if step % 1000 == 0 or step == 1:
            d.eval()
            with torch.no_grad():
                vdsa, vdsas = losses(Sv, Av, SRCv, TGTv)
                # accuracy of the joint classifier on val pairs
                _, tgt_sas = d.prob_sas(Sv, Av, TGTv)
                _, src_sas = d.prob_sas(Sv, Av, SRCv)
                acc = 0.5 * ((tgt_sas[:, 0] > 0.5).float().mean()
                             + (src_sas[:, 1] > 0.5).float().mean()).item()
            print(f"[{step:6d}] train dsa {dsa:.4f} dsas {dsas:.4f} | "
                  f"val dsa {vdsa:.4f} (log2={np.log(2):.3f}) dsas {vdsas:.4f} | val sas-acc {acc:.3f}")

    torch.save({"state_dict": d.state_dict(), "state_dim": s.shape[1],
                "action_dim": a.shape[1], "hidden": args.hidden,
                "dropout": bool(args.dropout)}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
