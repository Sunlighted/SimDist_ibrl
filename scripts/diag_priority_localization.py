"""Precondition check for mechanism-1 (prioritized/hard-example dynamics training):
does the model's OWN prediction error actually localize the CONTACT phase (the
task-binding region), or does a task-agnostic residual upweight fast-moving
free-flight instead?

Under PER-style prioritized sampling  prob_i ∝ error_i^α , the expected fraction
of sampled transitions that are in contact is
    frac_contact(α) = Σ_{i∈contact} e_i^α / Σ_i e_i^α .
α=0 recovers the uniform base rate. We sweep α for two priority signals:
  * full  : full-state z-scored 1-step error  (the TASK-AGNOSTIC general choice)
  * needle: needle-position 1-step error in mm (the task-specific oracle)

If frac_contact stays ~base rate under the FULL signal even at large α, then the
general (task-agnostic) prioritization does NOT focus contact -> mechanism-1 is
dead in its general form (would upweight free flight). If the NEEDLE signal DOES
enrich contact, a task-aware version could work but is not general.
"""
import argparse, os, sys
import numpy as np
import torch
from rl.world_model import WorldModel, WorldModelConfig
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer

NEEDLE = slice(9, 12); RING = slice(23, 26)
def log(*a): print(*a, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--dyn_ckpt", default="exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt")
    p.add_argument("--num_data", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    buf = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(buf, args.data_path, num_data=args.num_data)
    # flat (s,a,s') skipping the duplicated terminal
    S, A, SN = [], [], []
    for ep in buf.episodes:
        s, a = ep["state"], ep["action"]; T = a.shape[0]
        if T < 2: continue
        S.append(s[:T - 1]); A.append(a[:T - 1]); SN.append(s[1:T])
    S = torch.cat(S).float(); A = torch.cat(A).float(); SN = torch.cat(SN).float()
    del buf
    obs_dim, act_dim = S.shape[1], A.shape[1]
    log(f"[data] {S.shape[0]} transitions")

    wm = WorldModel(obs_dim, act_dim, WorldModelConfig(
        encoder_type="identity", mlp_hidden_dim=512, num_layer=2, layer_norm=1,
        horizon=5, rho=0.5, activation="relu")).to(dev)
    wm.load_state_dict(torch.load(args.dyn_ckpt, map_location=dev), strict=False); wm.eval()
    std = wm.norm_state_std

    # per-transition 1-step errors
    e_full = np.zeros(S.shape[0], dtype=np.float64)   # full-state z-scored norm
    e_needle = np.zeros(S.shape[0], dtype=np.float64) # needle mm
    bs = 16384
    with torch.no_grad():
        for i in range(0, S.shape[0], bs):
            s = S[i:i+bs].to(dev); a = A[i:i+bs].to(dev); sn = SN[i:i+bs].to(dev)
            pred = wm.decode(wm.next(wm.encode(s), a))
            e_full[i:i+bs] = (((pred - sn) / std).norm(dim=1)).cpu().numpy()
            e_needle[i:i+bs] = ((pred[:, NEEDLE] - sn[:, NEEDLE]).norm(dim=1) * 1000).cpu().numpy()

    d = (S[:, NEEDLE] - S[:, RING]).norm(dim=1).numpy()
    contact = d < 0.05; approach = (d >= 0.05) & (d < 0.15); free = d >= 0.15
    base = contact.mean()
    log(f"\nbase rates: contact={100*base:.2f}%  approach={100*approach.mean():.1f}%  free={100*free.mean():.1f}%")
    log(f"mean 1-step error by phase:")
    for nm, m in [("free", free), ("approach", approach), ("contact", contact)]:
        log(f"  {nm:9s}: full_normErr={e_full[m].mean():.3f}  needle_mm={e_needle[m].mean():.2f}")

    from scipy.stats import spearmanr
    rho = spearmanr(e_full, e_needle).correlation
    log(f"\nSpearman(full_err, needle_err) = {rho:.3f}  (do the two priority signals even agree?)")

    def frac_under_alpha(err, alpha):
        w = err ** alpha
        w = w / w.sum()
        return {"contact": float(w[contact].sum()), "approach": float(w[approach].sum()), "free": float(w[free].sum())}

    log("\n==== expected CONTACT sample fraction under prioritized sampling prob ∝ err^α ====")
    for label, err in [("FULL (task-agnostic)", e_full), ("NEEDLE (task oracle)", e_needle)]:
        log(f"  priority = {label}")
        for a_ in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
            f = frac_under_alpha(err + 1e-9, a_)
            lift = f["contact"] / max(base, 1e-9)
            log(f"    α={a_:4.1f}: contact={100*f['contact']:5.2f}%  (lift×{lift:4.1f})   "
                f"approach={100*f['approach']:4.1f}%  free={100*f['free']:4.1f}%")

    # top-k% composition by each signal
    log("\n==== phase composition of the TOP-k% highest-error transitions ====")
    for label, err in [("FULL", e_full), ("NEEDLE", e_needle)]:
        log(f"  by {label}:")
        order = np.argsort(-err)
        for kpct in [1, 5, 10, 20]:
            k = int(len(err) * kpct / 100)
            top = order[:k]
            log(f"    top{kpct:2d}%: contact={100*contact[top].mean():5.2f}%  "
                f"approach={100*approach[top].mean():4.1f}%  free={100*free[top].mean():4.1f}%")
    os._exit(0)


if __name__ == "__main__":
    main()
