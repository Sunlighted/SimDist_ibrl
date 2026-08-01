"""Score the trained discriminator on collected source/target rollout data.

Reports, for both the d_sa (state-action marginal) and the d_sas (joint)
classifiers:
  - per-domain accuracy and overall balanced accuracy (label 0=target,1=source)
and the dynamics log-ratio statistics (should be > 0 on target transitions,
< 0 on source transitions if the dynamics gap is captured).
"""

import argparse
import os

import numpy as np
import torch

from discriminator import DynamicsDiscriminator
from data_utils import contact_mask


def load_disc(path, device):
    ckpt = torch.load(path, map_location=device)
    d = DynamicsDiscriminator(ckpt["state_dim"], ckpt["action_dim"],
                              ckpt["hidden"], ckpt["dropout"]).to(device)
    d.load_state_dict(ckpt["state_dict"])
    d.eval()
    return d


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator.pt")
    p.add_argument("--data", default=f"{os.path.dirname(__file__)}/eval_transitions.npz")
    p.add_argument("--contact_only", type=int, default=0,
                   help="evaluate only on contact-phase transitions")
    p.add_argument("--contact_thresh", type=float, default=0.15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    d = load_disc(args.disc, args.device)
    z = np.load(args.data, allow_pickle=True)

    src = (z["src_s"], z["src_a"], z["src_ns"])
    tgt = (z["tgt_s"], z["tgt_a"], z["tgt_ns"])
    if args.contact_only:
        sm, tm = contact_mask(src[0], args.contact_thresh), contact_mask(tgt[0], args.contact_thresh)
        src = tuple(x[sm] for x in src)
        tgt = tuple(x[tm] for x in tgt)
        print(f"[contact-only thresh={args.contact_thresh}] "
              f"source {sm.sum()} / target {tm.sum()} transitions kept")

    domains = {  # name: (transitions, true_label)  label 0=target, 1=source
        "source": (src, 1),
        "target": (tgt, 0),
    }

    print(f"discriminator: {args.disc}")
    print(f"eval data    : {args.data} (weight={z['weight']})\n")

    accs_sa, accs_sas = {}, {}
    with torch.no_grad():
        for name, ((s, a, ns), lbl) in domains.items():
            s, a, ns = to_t(s, args.device), to_t(a, args.device), to_t(ns, args.device)
            sa_prob, sas_prob = d.prob_sas(s, a, ns)
            pred_sa = (sa_prob[:, 0] < sa_prob[:, 1]).long()    # 1 = source
            pred_sas = (sas_prob[:, 0] < sas_prob[:, 1]).long()
            acc_sa = (pred_sa == lbl).float().mean().item()
            acc_sas = (pred_sas == lbl).float().mean().item()
            accs_sa[name], accs_sas[name] = acc_sa, acc_sas
            log_r = d.log_dynamics_ratio(s, a, ns)
            print(f"[{name:6s}] n={len(s):6d} | "
                  f"acc d_sa={acc_sa:.3f}  d_sas={acc_sas:.3f} | "
                  f"P(target) d_sas mean={sas_prob[:,0].mean():.3f} | "
                  f"log dyn-ratio(tgt/src) mean={log_r.mean():.3f} "
                  f"median={log_r.median():.3f}")

    bal_sa = 0.5 * (accs_sa["source"] + accs_sa["target"])
    bal_sas = 0.5 * (accs_sas["source"] + accs_sas["target"])
    print(f"\nbalanced accuracy:  d_sa = {bal_sa:.3f}   d_sas = {bal_sas:.3f}")
    print("(d_sas = joint source/target dynamics classifier; d_sa = state-action only)")


if __name__ == "__main__":
    main()
