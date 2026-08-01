"""mechanism-2 precondition test: does an INDEPENDENT source-trained dynamics
ensemble's DISAGREEMENT localize the TARGET contact/gap region — so that
uncertainty-driven active collection at adapt would steer toward it?

Two things error-based steering already failed at (diag_priority_localization):
does DISAGREEMENT (epistemic, label-free) do better? Specifically test the §29
worry: the sim2sim gap is in P(s'|s,a), NOT in the state marginal d(s), so source
heads may AGREE on familiar-looking target states (all share the source bias) and
fail to flag the gap.

Measures, on TARGET transitions (source-ensemble is queried, target s' is truth):
  * disagreement = std over heads of predicted NEXT needle-position (mm) + full z-space.
  * true_gap_err = ||mean-head pred - true target s'|| needle mm  (what we WISH to target).
  * phase = needle-ring distance.
Reports: disagreement by phase (TARGET vs SOURCE data — does the gap raise it?);
Spearman(disagreement, true_gap_err); contact enrichment under prob ∝ disagree^α.
"""
import argparse, glob, os, sys
import numpy as np
import torch
from rl.world_model import WorldModel, WorldModelConfig
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer

NEEDLE = slice(9, 12); RING = slice(23, 26)
def log(*a): print(*a, file=sys.stderr, flush=True)


def load_flat(path, num_data, want_next=True):
    buf = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(buf, path, num_data=num_data)
    S, A, SN = [], [], []
    for ep in buf.episodes:
        s, a = ep["state"], ep["action"]; T = a.shape[0]
        if T < 2: continue
        S.append(s[:T-1]); A.append(a[:T-1]); SN.append(s[1:T])
    return torch.cat(S).float(), torch.cat(A).float(), torch.cat(SN).float()


def ensemble_stats(heads, S, A, SN, std_buf, dev):
    """Return per-transition (disagree_needle_mm, disagree_full, true_gap_needle_mm)."""
    n = S.shape[0]
    dis_n = np.zeros(n); dis_f = np.zeros(n); err_n = np.zeros(n)
    bs = 16384
    with torch.no_grad():
        for i in range(0, n, bs):
            s = S[i:i+bs].to(dev); a = A[i:i+bs].to(dev); sn = SN[i:i+bs].to(dev)
            preds = torch.stack([h.decode(h.next(h.encode(s), a)) for h in heads], 0)  # [K,b,obs]
            mean = preds.mean(0)
            # needle disagreement = RMS over heads of needle-pos deviation from head-mean, mm
            nd = (preds[:, :, NEEDLE] - mean[:, NEEDLE]).norm(dim=2)  # [K,b]
            dis_n[i:i+bs] = (nd.pow(2).mean(0).sqrt() * 1000).cpu().numpy()
            fd = ((preds - mean) / std_buf).norm(dim=2)  # [K,b] full z-space
            dis_f[i:i+bs] = fd.pow(2).mean(0).sqrt().cpu().numpy()
            err_n[i:i+bs] = ((mean[:, NEEDLE] - sn[:, NEEDLE]).norm(dim=1) * 1000).cpu().numpy()
    return dis_n, dis_f, err_n


def phase_of(S):
    d = (S[:, NEEDLE] - S[:, RING]).norm(dim=1).numpy()
    return d < 0.05, (d >= 0.05) & (d < 0.15), d >= 0.15  # contact, approach, free


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ens_glob", default="exps/mimicgen/threading_dyn_ens_seed*/best.pt")
    p.add_argument("--target_data", default="release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5")
    p.add_argument("--source_data", default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--num_data", type=int, default=15000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    ckpts = sorted(glob.glob(args.ens_glob))
    assert len(ckpts) >= 2, f"need >=2 heads, found {ckpts}"
    log(f"[ens] {len(ckpts)} heads: {ckpts}")
    heads = []
    for c in ckpts:
        wm = WorldModel(37, 7, WorldModelConfig(encoder_type="identity", mlp_hidden_dim=512,
                        num_layer=2, layer_norm=1, horizon=5, rho=0.5, activation="relu")).to(dev)
        wm.load_state_dict(torch.load(c, map_location=dev), strict=False); wm.eval()
        heads.append(wm)
    std_buf = heads[0].norm_state_std

    from scipy.stats import spearmanr
    for tag, path in [("TARGET", args.target_data), ("SOURCE", args.source_data)]:
        S, A, SN = load_flat(path, args.num_data)
        contact, approach, free = phase_of(S)
        dis_n, dis_f, err_n = ensemble_stats(heads, S, A, SN, std_buf, dev)
        base = contact.mean()
        log(f"\n===== {tag} data ({S.shape[0]} tr, contact base {100*base:.2f}%) =====")
        log(f"  {'phase':9s} {'disagree_needle_mm':>18s} {'disagree_full':>14s} {'true_gap_needle_mm':>18s}")
        for nm, m in [("free", free), ("approach", approach), ("contact", contact)]:
            log(f"  {nm:9s} {dis_n[m].mean():18.3f} {dis_f[m].mean():14.4f} {err_n[m].mean():18.3f}")
        if tag == "TARGET":
            log(f"  Spearman(disagree_needle, true_gap_err) = {spearmanr(dis_n, err_n).correlation:.3f}")
            log(f"  Spearman(disagree_full,   true_gap_err) = {spearmanr(dis_f, err_n).correlation:.3f}")
            # contact enrichment under disagreement-priority
            log("  contact fraction under prob ∝ disagree^α:")
            for label, sig in [("disagree_needle", dis_n), ("disagree_full", dis_f)]:
                row = f"    {label}: "
                for a_ in [0.0, 1.0, 2.0, 4.0]:
                    w = (sig + 1e-9) ** a_; w /= w.sum()
                    row += f"α{a_:.0f}={100*w[contact].sum():.2f}%(×{w[contact].sum()/max(base,1e-9):.1f})  "
                log(row)
    os._exit(0)


if __name__ == "__main__":
    main()
