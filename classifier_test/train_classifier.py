"""Train the H2O two-discriminator source-vs-target dynamics classifier.

Target ("real")  = release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5
Source ("sim")   = release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5

The source set is large (2000 demos) and the target set is tiny (20 demos),
so we sample equal-size minibatches from each domain (H2O does the same) to
keep the discriminator balanced. Inputs are standardized with stats pooled
over both domains, plus small gaussian input noise to curb overfitting.
"""

import argparse
import os

import numpy as np
import torch

from discriminator import DynamicsDiscriminator
from data_utils import load_transitions, episode_split, contact_mask

REPO = "/home/Chenyang/ibrl"
SOURCE_HDF5 = f"{REPO}/release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"
TARGET_HDF5 = f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def softmax_logits(d, s, a, ns):
    sa_logits, sas_logits = d.logits(s, a, ns)
    return torch.softmax(sa_logits, dim=1), torch.softmax(sas_logits, dim=1)


def evaluate_split(d, src, tgt, device):
    """Classification accuracy on held-out demo transitions (label 0=target)."""
    d.eval()
    with torch.no_grad():
        out = {}
        for name, (s, a, ns), lbl in [("source", src, 1), ("target", tgt, 0)]:
            if len(s) == 0:
                continue
            s, a, ns = to_t(s, device), to_t(a, device), to_t(ns, device)
            sa_prob, sas_prob = softmax_logits(d, s, a, ns)
            pred_sa = (sa_prob[:, 0] < sa_prob[:, 1]).long()  # 1 if source
            pred_sas = (sas_prob[:, 0] < sas_prob[:, 1]).long()
            out[f"{name}_acc_sa"] = (pred_sa == lbl).float().mean().item()
            out[f"{name}_acc_sas"] = (pred_sas == lbl).float().mean().item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_hdf5", default=SOURCE_HDF5)
    p.add_argument("--target_hdf5", default=TARGET_HDF5)
    p.add_argument("--source_max_ep", type=int, default=None,
                   help="cap source episodes (default: all 2000)")
    p.add_argument("--state_key", default="state")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/discriminator.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--contact_only", type=int, default=0,
                   help="train+val only on contact-phase transitions "
                        "(needle within --contact_thresh of the ring)")
    p.add_argument("--contact_thresh", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print("loading source ...")
    s_s, s_a, s_ns, s_ep = load_transitions(args.source_hdf5, args.state_key, args.source_max_ep)
    print(f"  source: {len(s_s)} transitions, {len(np.unique(s_ep))} episodes")
    print("loading target ...")
    t_s, t_a, t_ns, t_ep = load_transitions(args.target_hdf5, args.state_key)
    print(f"  target: {len(t_s)} transitions, {len(np.unique(t_ep))} episodes")

    if args.contact_only:
        sm, tm = contact_mask(s_s, args.contact_thresh), contact_mask(t_s, args.contact_thresh)
        s_s, s_a, s_ns, s_ep = s_s[sm], s_a[sm], s_ns[sm], s_ep[sm]
        t_s, t_a, t_ns, t_ep = t_s[tm], t_a[tm], t_ns[tm], t_ep[tm]
        print(f"  contact-only (thresh={args.contact_thresh}): "
              f"source {len(s_s)} / target {len(t_s)} transitions")

    # episode-level train/val split per domain
    s_tr, s_va = episode_split(s_ep, args.val_ratio, args.seed)
    t_tr, t_va = episode_split(t_ep, args.val_ratio, args.seed)

    src_tr = (s_s[s_tr], s_a[s_tr], s_ns[s_tr])
    tgt_tr = (t_s[t_tr], t_a[t_tr], t_ns[t_tr])
    src_va = (s_s[s_va], s_a[s_va], s_ns[s_va])
    tgt_va = (t_s[t_va], t_a[t_va], t_ns[t_va])
    print(f"  train: source {len(src_tr[0])} / target {len(tgt_tr[0])}")
    print(f"  val  : source {len(src_va[0])} / target {len(tgt_va[0])}")

    # normalization stats pooled over both domains (train split only)
    all_s = np.concatenate([src_tr[0], src_tr[2], tgt_tr[0], tgt_tr[2]], axis=0)
    all_a = np.concatenate([src_tr[1], tgt_tr[1]], axis=0)
    d = DynamicsDiscriminator(s_s.shape[1], s_a.shape[1], args.hidden, bool(args.dropout)).to(device)
    d.set_norm(all_s.mean(0), all_s.std(0), all_a.mean(0), all_a.std(0))

    opt_sa = torch.optim.Adam(d.d_sa.parameters(), lr=args.lr)
    opt_sas = torch.optim.Adam(d.d_sas.parameters(), lr=args.lr)

    src_t = [to_t(x, device) for x in src_tr]
    tgt_t = [to_t(x, device) for x in tgt_tr]
    n_src, n_tgt = len(src_t[0]), len(tgt_t[0])

    def sample(pool, n):
        idx = torch.randint(0, len(pool[0]), (n,), device=device)
        return pool[0][idx], pool[1][idx], pool[2][idx]

    print("training ...")
    for step in range(1, args.steps + 1):
        d.train()
        rs, ra, rns = sample(tgt_t, args.batch_size)   # real = target
        ss, sa, sns = sample(src_t, args.batch_size)   # sim  = source
        if args.noise_std > 0:
            ns = args.noise_std
            rs = rs + torch.randn_like(rs) * ns
            ra = ra + torch.randn_like(ra) * ns
            rns = rns + torch.randn_like(rns) * ns
            ss = ss + torch.randn_like(ss) * ns
            sa = sa + torch.randn_like(sa) * ns
            sns = sns + torch.randn_like(sns) * ns

        r_sa, r_sas = softmax_logits(d, rs, ra, rns)   # real -> class 0
        s_sa_p, s_sas_p = softmax_logits(d, ss, sa, sns)  # sim -> class 1
        eps = 1e-8
        dsa_loss = (-torch.log(r_sa[:, 0] + eps) - torch.log(s_sa_p[:, 1] + eps)).mean()
        dsas_loss = (-torch.log(r_sas[:, 0] + eps) - torch.log(s_sas_p[:, 1] + eps)).mean()

        opt_sa.zero_grad()
        dsa_loss.backward(retain_graph=True)
        opt_sas.zero_grad()
        dsas_loss.backward()
        opt_sa.step()
        opt_sas.step()

        if step % 1000 == 0 or step == 1:
            m = evaluate_split(d, src_va, tgt_va, device)
            print(f"[{step:6d}] dsa {dsa_loss.item():.4f} dsas {dsas_loss.item():.4f} | "
                  f"val sa(src/tgt) {m.get('source_acc_sa',0):.3f}/{m.get('target_acc_sa',0):.3f} "
                  f"sas {m.get('source_acc_sas',0):.3f}/{m.get('target_acc_sas',0):.3f}")

    torch.save({"state_dict": d.state_dict(),
                "state_dim": s_s.shape[1], "action_dim": s_a.shape[1],
                "hidden": args.hidden, "dropout": bool(args.dropout)}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
