"""Env-free autopsy of a SimDist world model on TARGET data: how accurate are the
DYNAMICS and VALUE heads on the target domain? (Phase-0 of the plateau diagnosis.)

Loads one or more world-model checkpoints (rebuilt from a stage-1 cfg.yaml for the
architecture) and, on the TARGET demos, reports:

  DYNAMICS: multi-step latent-consistency MSE  ||roll(z) - encode(s_t)||^2
            + decoded next-state MAE (raw units) ||decode(roll z) - s_t||
            per rollout step (incl. steps beyond H to see compounding).

  VALUE:    V_hat(encode(s)) vs the Monte-Carlo discounted return-to-go v2g(s);
            Spearman + Pearson + MSE + scale. Also V_hat(rolled z) — the latent the
            MPPI planner actually queries — vs v2g, to catch train/plan mismatch.

No env, no rollout — pure forward passes, fast. Compare a pre-adapt (stage-1)
ckpt against an adapted (e.g. 0.75) ckpt to see whether dynamics/value improved on
target.

Run:
  python scripts/diag_simdist_components.py \
      --stage1_cfg exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/cfg.yaml \
      --ckpts exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
              exps/mimicgen/threading_simdist_adapt_seed1/model0.pt \
      --target_demos release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5 \
      --num_demos 20 --eval_horizon 8
"""

import argparse
import os

import numpy as np
import torch
import pyrallis

from train_simdist import MainConfig
from rl.simdist_agent import SimDistAgent
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer


def _spearman(a, b):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def compute_v2g(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    """MC discounted return-to-go per STATE (self-contained, no buffer dependency).
    rewards r[0..T-1] over states s[0..T]: v2g[T]=0, v2g[t]=r[t]+disc*v2g[t+1]."""
    T = rewards.shape[0]
    v2g = torch.zeros(T + 1, dtype=torch.float32)
    for t in range(T - 1, -1, -1):
        v2g[t] = rewards[t] + discount * v2g[t + 1]
    return v2g


def build_agent(stage1_cfg_path, device="cuda"):
    cfg: MainConfig = pyrallis.load(MainConfig, open(stage1_cfg_path, "r"))  # type: ignore
    # state-only Threading: obs_dim 37, action_dim 7 (prop unused for state)
    agent = SimDistAgent(True, (37,), (9,), 7, cfg.rl_camera, cfg.simdist, device=device)
    return agent, cfg


@torch.no_grad()
def dynamics_report(agent, buffer, discount, eval_horizon, device="cuda"):
    """Roll the dynamics with the demos' OWN actions; per-step latent MSE +
    decoded-state MAE. Averages over all valid (episode, start) windows."""
    m = agent.model
    Hroll = eval_horizon
    lat_mse = np.zeros(Hroll)
    dec_mae = np.zeros(Hroll)
    cnt = np.zeros(Hroll)
    for ep in buffer.episodes:
        s = ep["state"].to(device)   # [T+1, obs]
        a = ep["action"].to(device)  # [T, act]
        T = a.shape[0]
        for start in range(0, T):  # every start
            hmax = min(Hroll, T - start)
            if hmax <= 0:
                continue
            z = m.encode(s[start].unsqueeze(0))
            for k in range(hmax):
                z = m.next(z, a[start + k].unsqueeze(0))
                z_enc = m.encode(s[start + k + 1].unsqueeze(0))
                lat_mse[k] += float(((z - z_enc) ** 2).mean())
                dec = m.decode(z)
                dec_mae[k] += float((dec - s[start + k + 1].unsqueeze(0)).abs().mean())
                cnt[k] += 1
    lat_mse /= np.maximum(cnt, 1)
    dec_mae /= np.maximum(cnt, 1)
    return lat_mse, dec_mae


@torch.no_grad()
def value_report(agent, buffer, discount, eval_horizon, device="cuda"):
    """V_hat(encode(s)) vs MC return-to-go v2g(s); and V_hat(rolled z) vs v2g."""
    m = agent.model
    v_enc, v_roll, v_true_enc, v_true_roll = [], [], [], []
    for ep in buffer.episodes:
        s = ep["state"].to(device)
        a = ep["action"].to(device)
        v2g = compute_v2g(ep["reward"], discount).to(device)  # [T+1]
        T = a.shape[0]
        # V_hat(encode(s_t)) for all states
        z_enc = m.encode(s)                       # [T+1, z]
        vpred = m.predict_value(z_enc)            # [T+1]
        v_enc.append(vpred.cpu().numpy())
        v_true_enc.append(v2g.cpu().numpy())
        # V_hat(rolled z): roll eval_horizon steps from each start, score endpoint
        for start in range(0, T):
            hmax = min(eval_horizon, T - start)
            if hmax <= 0:
                continue
            z = m.encode(s[start].unsqueeze(0))
            for k in range(hmax):
                z = m.next(z, a[start + k].unsqueeze(0))
            v_roll.append(float(m.predict_value(z)))
            v_true_roll.append(float(v2g[start + hmax]))
    v_enc = np.concatenate(v_enc); v_true_enc = np.concatenate(v_true_enc)
    v_roll = np.array(v_roll); v_true_roll = np.array(v_true_roll)
    out = {
        "enc_spearman": _spearman(v_enc, v_true_enc),
        "enc_pearson": float(np.corrcoef(v_enc, v_true_enc)[0, 1]),
        "enc_mse": float(((v_enc - v_true_enc) ** 2).mean()),
        "enc_pred_range": (float(v_enc.min()), float(v_enc.max())),
        "enc_true_range": (float(v_true_enc.min()), float(v_true_enc.max())),
        "roll_spearman": _spearman(v_roll, v_true_roll),
        "roll_mse": float(((v_roll - v_true_roll) ** 2).mean()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_cfg", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target_demos", required=True)
    ap.add_argument("--num_demos", type=int, default=20)
    ap.add_argument("--eval_horizon", type=int, default=8)
    ap.add_argument("--discount", type=float, default=0.99)
    args = ap.parse_args()

    agent, cfg = build_agent(args.stage1_cfg)
    buffer = SequenceReplayBuffer(max_episodes=10000, device="cpu")
    load_demos_to_sequence_buffer(buffer, args.target_demos, num_data=args.num_demos,
                                  reward_scale=cfg.env_reward_scale)
    print(f"loaded {buffer.size()} target demos, {buffer.num_transitions()} transitions\n")

    for ckpt in args.ckpts:
        agent.load_state_dict(torch.load(ckpt))
        agent = agent.to("cuda")
        agent.eval()
        print("=" * 78)
        print(f"CKPT: {ckpt}")
        lat, dec = dynamics_report(agent, buffer, args.discount, args.eval_horizon)
        print("  DYNAMICS rollout error (target demos, demo actions):")
        print("    step:        " + "  ".join(f"{k+1:>7d}" for k in range(len(lat))))
        print("    latent_mse:  " + "  ".join(f"{x:7.4f}" for x in lat))
        print("    decode_mae:  " + "  ".join(f"{x:7.4f}" for x in dec))
        v = value_report(agent, buffer, args.discount, args.eval_horizon)
        print("  VALUE vs MC return-to-go (target demos):")
        print(f"    V(encode s):  spearman={v['enc_spearman']:.3f}  pearson={v['enc_pearson']:.3f}  "
              f"mse={v['enc_mse']:.4f}")
        print(f"      pred range={v['enc_pred_range'][0]:.3f}..{v['enc_pred_range'][1]:.3f}  "
              f"true range={v['enc_true_range'][0]:.3f}..{v['enc_true_range'][1]:.3f}")
        print(f"    V(rolled z, H={args.eval_horizon}): spearman={v['roll_spearman']:.3f}  "
              f"mse={v['roll_mse']:.4f}")
        print()


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()
