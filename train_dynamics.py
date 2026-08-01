"""Offline SOURCE dynamics-only trainer (no env / no MPPI / no target).

The first step of the "dynamics-is-the-only-learned-component" plan (CLAUDE.md
§17 raw-state arm): train JUST the raw-state world model's dynamics head on a
pre-collected SOURCE dataset, and report HELD-OUT multi-step rollout prediction
error — the real measure of a dynamics model, not an MPPI score. Value / reward /
policy are deliberately NOT trained here (they will come from the source agent /
a heuristic at plan time).

The model is `rl/world_model.py::WorldModel(encoder_type="identity")`: the
"latent" z is the z-scored raw state (dim=obs_dim), `next` is a residual
next-(normalized-)state predictor `z' = z + MLP([z,a])`, `decode` is the analytic
denorm inverse. z-score stats are fit on the TRAIN split and stored in the ckpt
buffers, so the saved model self-normalizes when reloaded.

Data: any `load_source_replay_to_sequence_buffer` format (obs/state precomputed
+ optional expert_flags), e.g. release/data/mimicgen/threading_d0_source_diverse_v2.

Run:
  python train_dynamics.py \
      --data_path release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
      --num_data 10000 --num_steps 100000 \
      --save_dir exps/mimicgen/threading_dyn_diverse_v2_seed1
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rl.world_model import WorldModel, WorldModelConfig
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str,
                   default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--num_data", type=int, default=10000,
                   help="number of source trajectories to load (-1 = all; the full "
                        "diverse_v2 has 100k = 4.1GB, a subset is plenty for a 44->37 MLP)")
    p.add_argument("--save_dir", type=str, default="exps/mimicgen/threading_dyn_seed1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    # model
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layer", type=int, default=2)
    p.add_argument("--layer_norm", type=int, default=1)
    p.add_argument("--horizon", type=int, default=5, help="H: multi-step rollout length")
    p.add_argument("--rho", type=float, default=0.5, help="per-step discount on the rollout loss")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--augment_velocity", type=int, default=0,
                   help="1 = append finite-difference velocity (s_t - s_{t-1})/dt to each "
                        "state, doubling obs_dim (37->74). The dynamics then predicts the FULL "
                        "augmented Delta and rolls it autoregressively (fd_predict variant). "
                        "needle = first-37 dims [9:12] regardless.")
    p.add_argument("--ctrl_dt", type=float, default=0.05, help="control step (20 Hz) for the finite diff")
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=20.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=100000)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--val_batch", type=int, default=2048)
    p.add_argument("--log_every", type=int, default=2000)
    return p.parse_args()


def split_train_val(full: SequenceReplayBuffer, val_ratio: float, seed: int):
    """Episode-level train/val split (no leakage across the segment sampler)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(full.size())
    rng.shuffle(idx)
    n_val = int(round(full.size() * val_ratio))
    val_ids, train_ids = set(idx[:n_val].tolist()), idx[n_val:]
    train = SequenceReplayBuffer(max_episodes=full.size() + 1)
    val = SequenceReplayBuffer(max_episodes=full.size() + 1)
    for i, ep in enumerate(full.episodes):
        dst = val if i in val_ids else train
        dst.add_episode(ep["state"], ep["action"], ep["reward"], success=ep.get("success", False))
    return train, val


def consistency_loss(model: WorldModel, states, actions, rho):
    """Multi-step latent (= normalized-state) rollout MSE, rho**t weighted."""
    H = states.shape[1] - 1
    z = model.encode(states[:, 0])
    loss, wsum = 0.0, 0.0
    for t in range(H):
        w = rho ** t
        wsum += w
        with torch.no_grad():
            z_next = model.encode(states[:, t + 1])
        z = model.next(z, actions[:, t])
        loss = loss + w * F.mse_loss(z, z_next)
    return loss / wsum


@torch.no_grad()
def val_metrics(model: WorldModel, states, actions):
    """Compounding rollout error. Returns normalized MSE + raw-unit MAE at the
    1st and H-th step (so you can see how fast multi-step error grows)."""
    H = states.shape[1] - 1
    z = model.encode(states[:, 0])
    norm_mse, raw_mae, needle_mm = [], [], []
    for t in range(H):
        z = model.next(z, actions[:, t])  # rolled (uses its OWN prediction)
        z_true = model.encode(states[:, t + 1])
        norm_mse.append(F.mse_loss(z, z_true).item())
        raw_mae.append((model.decode(z) - states[:, t + 1]).abs().mean().item())
        # needle = first-37 dims [9:12]; L2 position error in mm (the task-relevant number)
        nd = torch.norm(model.decode(z)[:, 9:12] - states[:, t + 1, 9:12], dim=1) * 1000.0
        needle_mm.append(nd.mean().item())
    return norm_mse, raw_mae, needle_mm


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---- data (offline, no env) ----
    full = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(full, args.data_path, num_data=args.num_data)
    if args.augment_velocity:
        # append finite-difference velocity (s_t - s_{t-1})/dt; first frame vel=0.
        # keep torch dtype (the buffer stores torch tensors + calls .detach()).
        for ep in full.episodes:
            s = ep["state"] if torch.is_tensor(ep["state"]) else torch.as_tensor(ep["state"])
            s = s.float()
            vel = torch.zeros_like(s)
            vel[1:] = (s[1:] - s[:-1]) / args.ctrl_dt
            ep["state"] = torch.cat([s, vel], dim=1)
        print(f"[augment] appended finite-diff velocity: obs_dim "
              f"{full.episodes[0]['state'].shape[1] // 2} -> {full.episodes[0]['state'].shape[1]}")
    assert full.can_sample(args.horizon), f"no episode of length >= horizon={args.horizon}"
    train, val = split_train_val(full, args.val_ratio, args.seed)
    del full
    print(f"[data] train {train.size()} ep / {train.num_transitions()} tr | "
          f"val {val.size()} ep / {val.num_transitions()} tr")

    # ---- model (raw-state world model; only the dynamics head is trained) ----
    obs_dim = train.episodes[0]["state"].shape[1]
    action_dim = train.episodes[0]["action"].shape[1]
    wm_cfg = WorldModelConfig(
        encoder_type="identity", mlp_hidden_dim=args.hidden_dim, num_layer=args.num_layer,
        layer_norm=args.layer_norm, horizon=args.horizon, rho=args.rho,
        activation=args.activation, grad_clip=args.grad_clip,
    )
    model = WorldModel(obs_dim, action_dim, wm_cfg).to(dev)
    # z-score normalization fit on the TRAIN split (no val leakage), stored in buffers
    mean, std = train.state_stats()
    model.set_normalization(mean, std)
    print(f"[norm] fit on train states: |mean| avg={mean.abs().mean():.3f}, std avg={std.mean():.3f}")
    print(f"[model] raw-state dynamics: in={obs_dim + action_dim} -> out={obs_dim}, "
          f"hidden={args.hidden_dim}x{args.num_layer}, "
          f"params={sum(p.numel() for p in model.dynamics.parameters())/1e3:.1f}K")

    opt = torch.optim.Adam(model.dynamics.parameters(), lr=args.lr)

    # save cfg for reload
    json.dump({**vars(args), "obs_dim": obs_dim, "action_dim": action_dim},
              open(os.path.join(args.save_dir, "dyn_cfg.json"), "w"), indent=2)

    pbar = tqdm(total=args.num_steps, desc="train dynamics", dynamic_ncols=True, mininterval=1.0)
    best_h = float("inf")
    for step in range(1, args.num_steps + 1):
        states, actions, _ = train.sample(args.batch_size, args.horizon, dev)
        loss = consistency_loss(model, states, actions, args.rho)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), args.grad_clip)
        opt.step()
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.log_every == 0 or step == args.num_steps:
            vs, va, _ = val.sample(args.val_batch, args.horizon, dev)
            nmse, rmae, ndmm = val_metrics(model, vs, va)
            print(f"\n[step {step}] train/consistency={loss.item():.5f} grad={float(gnorm):.2f}"
                  f" | val 1-step: norm_mse={nmse[0]:.5f} needle_mm={ndmm[0]:.3f}"
                  f" | val {args.horizon}-step: norm_mse={nmse[-1]:.5f} needle_mm={ndmm[-1]:.3f}")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "latest.pt"))
            if ndmm[-1] < best_h:   # select by needle H-step MAE (comparable across obs_dim)
                best_h = ndmm[-1]
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
    pbar.close()
    print(f"[done] best val {args.horizon}-step needle_mm={best_h:.3f}; "
          f"saved -> {args.save_dir}/(latest|best).pt")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
