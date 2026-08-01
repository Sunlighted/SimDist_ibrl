"""Is the SOURCE dynamics model's CONTACT-phase error a single-step (hard-to-learn)
problem or a COMPOUNDING problem? Decides whether trajectory-level / joint H-step
dynamics (which only cuts compounding) can help the contact-limited ceiling.

For held-out H-step segments, compare per target-step:
  * teacher-forced 1-step error : decode(next(encode(s_t), a_t)) vs s_{t+1}
  * free-rolled error at depth t : roll from s_0 using own predictions vs s_{t+1}
binned by the TRUE phase (needle-ring distance) at the target step.

Verdict:
  contact: rolled ~ teacher-forced  -> single-step hardness -> trajectory-level WON'T help contact
  contact: rolled >> teacher-forced -> compounding          -> trajectory-level COULD help
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
    p.add_argument("--num_data", type=int, default=8000)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n_seg", type=int, default=40000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    buf = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(buf, args.data_path, num_data=args.num_data)
    obs_dim = buf.episodes[0]["state"].shape[1]; act_dim = buf.episodes[0]["action"].shape[1]

    wm = WorldModel(obs_dim, act_dim, WorldModelConfig(
        encoder_type="identity", mlp_hidden_dim=512, num_layer=2, layer_norm=1,
        horizon=args.horizon, rho=0.5, activation="relu")).to(dev)
    wm.load_state_dict(torch.load(args.dyn_ckpt, map_location=dev), strict=False); wm.eval()

    S, A = buf.sample(args.n_seg, args.horizon, dev)[:2]  # [B,H+1,obs], [B,H,act]
    H = args.horizon
    with torch.no_grad():
        # teacher-forced 1-step at each t
        tf_err = torch.zeros(S.shape[0], H, device=dev)
        for t in range(H):
            pred = wm.decode(wm.next(wm.encode(S[:, t]), A[:, t]))
            tf_err[:, t] = (pred[:, NEEDLE] - S[:, t + 1, NEEDLE]).norm(dim=1) * 1000
        # free rollout
        roll_err = torch.zeros(S.shape[0], H, device=dev)
        z = wm.encode(S[:, 0])
        for t in range(H):
            z = wm.next(z, A[:, t])
            pred = wm.decode(z)
            roll_err[:, t] = (pred[:, NEEDLE] - S[:, t + 1, NEEDLE]).norm(dim=1) * 1000
        # phase of the TRUE target state s_{t+1}
        dist = (S[:, 1:, NEEDLE] - S[:, 1:, RING]).norm(dim=2)  # [B,H] meters

    tf_err = tf_err.cpu().numpy(); roll_err = roll_err.cpu().numpy(); dist = dist.cpu().numpy()
    phases = {"free(>=15cm)": dist >= 0.15, "approach(5-15cm)": (dist >= 0.05) & (dist < 0.15),
              "contact(<5cm)": dist < 0.05}

    log(f"\n[data] {S.shape[0]} segments, H={H}\n")
    log(f"{'phase':18s} {'n':>7s} {'1step_TF':>9s} {'H-rolled':>9s} {'ratio':>6s}   (RMS needle mm)")
    for name, m in phases.items():
        n = int(m.sum())
        if n < 50:
            log(f"{name:18s} {n:>7d}  too few"); continue
        # teacher-forced error at these (step-agnostic) target states
        tf = float(np.sqrt((tf_err[m] ** 2).mean()))
        # rolled error at these target states (each carries its own rollout depth)
        rl = float(np.sqrt((roll_err[m] ** 2).mean()))
        log(f"{name:18s} {n:>7d} {tf:9.2f} {rl:9.2f} {rl/max(tf,1e-6):6.2f}")

    # contact error vs rollout depth (compounding curve within contact)
    log("\ncontact-phase error by rollout DEPTH t (compounding curve):")
    log(f"  {'depth t':>8s} {'n':>7s} {'1step_TF':>9s} {'rolled@t':>9s}")
    cm = phases["contact(<5cm)"]
    for t in range(H):
        sel = cm[:, t]
        n = int(sel.sum())
        if n < 30: continue
        tf = float(np.sqrt((tf_err[sel, t] ** 2).mean()))
        rl = float(np.sqrt((roll_err[sel, t] ** 2).mean()))
        log(f"  {t+1:8d} {n:>7d} {tf:9.2f} {rl:9.2f}")
    os._exit(0)


if __name__ == "__main__":
    main()
