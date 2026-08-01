"""Train the conditional-density dynamics-gap model.

c=0 (source) batch is drawn from: the suboptimal replay AND the counterfactual
source-step anchors (s'_src). c=1 (target) batch is the target demos' real next
state (s'_tgt). A fixed per-batch source/target ratio keeps the tiny target set
from being drowned. CFG dropout replaces c with ∅ (zeros) at prob cfg_dropout;
since source >> target, ∅ ~ source, so c=1 retreats to source on unsupported
(OOD) inputs -> DGI -> 0 there.
"""

import argparse
import os

import numpy as np
import torch

from density_data import build_dataset, norm_stats
from conditional_density import build_model


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["gaussian", "diffusion"], default="gaussian")
    p.add_argument("--cache", default=f"{os.path.dirname(__file__)}/cdm_data.npz")
    p.add_argument("--out", default=None)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--target_frac", type=float, default=0.5,
                   help="fraction of each batch that is c=1 target rows")
    p.add_argument("--anchor_src_frac", type=float, default=0.3,
                   help="within the c=0 half, fraction drawn from cf anchors "
                        "(rest from replay); anchors share (s,a) with c=1")
    p.add_argument("--cfg_dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.out is None:
        args.out = f"{os.path.dirname(__file__)}/cdm_{args.head}.pt"

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device

    d = build_dataset(cache=args.cache)
    ns = norm_stats(d)
    sdim, adim = d["r_s"].shape[1], d["r_a"].shape[1]

    model = build_model(args.head, sdim, adim, sdim, hidden=args.hidden).to(dev)
    model.set_norm(**ns)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training pools (train split only)
    rep_s = to_t(d["r_s"][d["r_tr"]], dev); rep_a = to_t(d["r_a"][d["r_tr"]], dev)
    rep_ns = to_t(d["r_ns"][d["r_tr"]], dev)
    an_s = to_t(d["a_s"][d["a_tr"]], dev); an_a = to_t(d["a_a"][d["a_tr"]], dev)
    an_src = to_t(d["a_src"][d["a_tr"]], dev); an_tgt = to_t(d["a_tgt"][d["a_tr"]], dev)
    n_rep, n_an = len(rep_s), len(an_s)
    print(f"train pools: replay c0 {n_rep}, anchor {n_an}; head={args.head}")

    def ridx(n, nb):
        return torch.randint(0, n, (nb,), device=dev)

    for step in range(1, args.steps + 1):
        model.train()
        B = args.batch_size
        n_tgt = int(B * args.target_frac)
        n_src = B - n_tgt
        n_src_anchor = int(n_src * args.anchor_src_frac)
        n_src_replay = n_src - n_src_anchor

        # ---- c=1 target rows (from anchors' s'_tgt) ----
        it = ridx(n_an, n_tgt)
        s1, a1, sp1 = an_s[it], an_a[it], an_tgt[it]
        c1 = torch.ones(n_tgt, dtype=torch.long, device=dev)
        # ---- c=0 source rows: replay + anchor s'_src ----
        ir = ridx(n_rep, n_src_replay)
        ia = ridx(n_an, n_src_anchor)
        s0 = torch.cat([rep_s[ir], an_s[ia]], 0)
        a0 = torch.cat([rep_a[ir], an_a[ia]], 0)
        sp0 = torch.cat([rep_ns[ir], an_src[ia]], 0)
        c0 = torch.zeros(n_src, dtype=torch.long, device=dev)

        s = torch.cat([s1, s0], 0); a = torch.cat([a1, a0], 0)
        sp = torch.cat([sp1, sp0], 0); c = torch.cat([c1, c0], 0)
        ds = sp - s

        # CFG dropout -> ∅ (zero one-hot)
        oh = model.onehot(c, s.size(0), dev)
        drop = (torch.rand(s.size(0), device=dev) < args.cfg_dropout)
        oh[drop] = 0.0

        loss = model.loss(s, a, ds, oh)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 1000 == 0 or step == 1:
            model.eval()
            # quick val DGI on held-out anchors vs ground-truth gap
            with torch.no_grad():
                vs = to_t(d["a_s"][d["a_va"]], dev); va = to_t(d["a_a"][d["a_va"]], dev)
                vtgt = to_t(d["a_tgt"][d["a_va"]], dev)
                dgi = model.dgi(vs, va, vtgt).cpu().numpy()
            gt = d["a_gap"][d["a_va"]]
            from scipy.stats import spearmanr
            rho = spearmanr(dgi, gt)[0]
            print(f"[{step:6d}] loss {loss.item():.4f} | val DGI(tgt) mean {dgi.mean():.3f} "
                  f"| Spearman(DGI, GTgap)={rho:.3f}")

    torch.save({"state_dict": model.state_dict(), "head": args.head,
                "state_dim": sdim, "action_dim": adim, "ds_dim": sdim,
                "hidden": args.hidden}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
