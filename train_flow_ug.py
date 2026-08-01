"""Flow BC policy for the SimDist policy-replacement experiment (CLAUDE.md §24 follow-up).

Goal: replace the SimDist world model's deterministic MLP policy head with a
stronger FLOW BC policy, to test whether a better base policy (± Universal-
Guidance steering by the frozen world model + V) beats the MLP policy-only
baseline (~42% on TARGET after 20-demo fine-tune).

This script only TRAINS the flow policy (offline, no env):
  Phase 1 — SOURCE pretrain: BC action-chunks from the expert (`expert_flags==1`)
            subset of the diverse_v2 dataset — the SAME data the SimDist MLP
            policy was distilled on (`bc_expert_only=1`). Only chunks fully
            inside a contiguous expert run are used (no noised actions leak in;
            padded at the run boundary with the last expert action).
  Phase 2 — TARGET finetune: BC chunks from the 20 TARGET demos (all steps are
            clean success demos).

Normalization (§15, state z-score + action→[-1,1]) is fit on the SOURCE expert
data and FROZEN through finetune + eval (20 target demos are too few to re-fit).

Saves `<save_dir>/flow.pt` (StateFlowPolicy state_dict incl. EMA + norm buffers)
and `<save_dir>/flow_cfg.json` (obs/action dims + the flow config) so
eval_flow_ug.py can rebuild the policy without a dataset.

Run:
  python train_flow_ug.py \
      --source_data release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
      --target_data release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5 \
      --save_dir exps/mimicgen/threading_flow_ug_seed1
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig


class _Batch:
    """Minimal stand-in for the BC dataset Batch that StateFlowPolicy.loss reads."""

    def __init__(self, state, action_chunk):
        self.obs = {"state": state}
        self.action = {"action_chunk": action_chunk}


# --------------------------------------------------------------------------- data
def _expert_runs(flag: np.ndarray):
    """Yield [lo, hi) index ranges of maximal contiguous expert_flag==1 runs."""
    T = flag.shape[0]
    t = 0
    while t < T:
        if flag[t] > 0.5:
            lo = t
            while t < T and flag[t] > 0.5:
                t += 1
            yield lo, t
        else:
            t += 1


def extract_source_expert_chunks(path: str, num_data: int, Tp: int, clamp: bool = True):
    """Extract (state, action_chunk[Tp]) pairs from the expert subset of the
    diverse_v2 source dataset. Only chunks fully inside a contiguous expert run
    are kept; a chunk shorter than Tp (near the run boundary) is padded with the
    last expert action in the run. state[t] is the obs BEFORE action[t]."""
    f = h5py.File(path, "r")
    g = f["data"]
    ids = sorted((k for k in g.keys() if k.startswith("demo_")),
                 key=lambda k: int(k.split("_")[1]))
    if num_data > 0:
        ids = ids[:num_data]
    states, chunks = [], []
    for k in tqdm(ids, desc=f"extract expert chunks (Tp={Tp})", dynamic_ncols=True,
                  mininterval=1.0):
        ep = g[k]
        if "expert_flags" not in ep:
            continue
        flag = np.array(ep["expert_flags"]).astype(np.float32)
        state = np.array(ep["obs"]["state"]).astype(np.float32)   # [T, obs] = s_0..s_{T-1}
        action = np.array(ep["actions"]).astype(np.float32)       # [T, act]
        if clamp:
            action = np.clip(action, -1.0, 1.0)
        T = action.shape[0]
        for lo, hi in _expert_runs(flag):
            for t in range(lo, hi):
                end = min(t + Tp, hi)
                ck = action[t:end]                                 # [<=Tp, act]
                if ck.shape[0] < Tp:                               # pad at run boundary
                    pad = np.repeat(ck[-1:], Tp - ck.shape[0], axis=0)
                    ck = np.concatenate([ck, pad], axis=0)
                states.append(state[t])
                chunks.append(ck)
    f.close()
    states = torch.from_numpy(np.stack(states))          # [N, obs]
    chunks = torch.from_numpy(np.stack(chunks))          # [N, Tp, act]
    print(f"[source] {states.shape[0]} expert chunks, obs={states.shape[1]}, "
          f"act={chunks.shape[2]}")
    return states, chunks


def extract_target_demo_chunks(path: str, num_data: int, Tp: int, clamp: bool = True):
    """Extract (state, action_chunk[Tp]) from the 20 TARGET demos. Each demo is a
    clean success trajectory, truncated at the first success step; chunks are
    padded at the (truncated) episode end with the last action."""
    from env.robosuite_wrapper import STATE_KEYS
    f = h5py.File(path, "r")
    task = json.load(open(os.path.join(os.path.dirname(path), "env_cfg.json")))["env_name"]
    state_keys = STATE_KEYS[task]
    g = f["data"]
    ids = sorted((k for k in g.keys() if k.startswith("demo_")),
                 key=lambda k: int(k.split("_")[1]))
    if num_data > 0:
        ids = ids[:num_data]
    states, chunks = [], []
    for k in ids:
        ep = g[k]
        action = np.array(ep["actions"]).astype(np.float32)
        reward = np.array(ep["rewards"]).astype(np.float32)
        if not (reward == 1).any():
            continue
        state = np.concatenate([np.array(ep["obs"][sk]) for sk in state_keys],
                               axis=1).astype(np.float32)          # [T, obs]
        if clamp:
            action = np.clip(action, -1.0, 1.0)
        T = int(np.argmax(reward == 1)) + 1                        # truncate at 1st success
        for t in range(T):
            end = min(t + Tp, T)
            ck = action[t:end]
            if ck.shape[0] < Tp:
                pad = np.repeat(ck[-1:], Tp - ck.shape[0], axis=0)
                ck = np.concatenate([ck, pad], axis=0)
            states.append(state[t])
            chunks.append(ck)
    f.close()
    states = torch.from_numpy(np.stack(states))
    chunks = torch.from_numpy(np.stack(chunks))
    print(f"[target] {states.shape[0]} demo chunks from {len(ids)} demos")
    return states, chunks


# ---------------------------------------------------------------------- training
def train_phase(policy, states, chunks, steps, batch_size, lr, device, desc):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    N = states.shape[0]
    policy.train()
    pbar = tqdm(range(steps), desc=desc, dynamic_ncols=True, mininterval=1.0)
    last = 0.0
    for step in pbar:
        idx = torch.randint(0, N, (batch_size,))
        s = states[idx].to(device)
        a = chunks[idx].to(device)
        loss = policy.loss(_Batch(s, a))
        opt.zero_grad()
        loss.backward()
        opt.step()
        policy.update_ema()
        last = loss.item()
        if (step + 1) % max(steps // 20, 1) == 0:
            pbar.set_postfix(loss=f"{last:.5f}")
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_data", type=str,
                   default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--target_data", type=str,
                   default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--save_dir", type=str, default="exps/mimicgen/threading_flow_ug_seed1")
    p.add_argument("--source_num_data", type=int, default=20000,
                   help="# diverse_v2 demos to draw expert chunks from (-1 = all 100k)")
    p.add_argument("--target_num_data", type=int, default=20)
    # Tp must be UNet1D-compatible (power-of-2 multiple for the 3-level down/up
    # sampling); H=5 is not, so use the standard flow/diffusion default Tp=16,
    # Ta=8. UG's rollout depth is DECOUPLED from Tp — the agent rolls only the
    # first world-model H (=5) actions through the (frozen) dynamics for the
    # value guidance, where the dynamics was actually trained.
    p.add_argument("--pred_horizon", type=int, default=16, help="Tp (UNet-safe)")
    p.add_argument("--action_horizon", type=int, default=8, help="Ta (<= Tp)")
    p.add_argument("--backbone", type=str, default="unet1d")
    p.add_argument("--source_steps", type=int, default=100000)
    p.add_argument("--target_steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- data ----
    src_s, src_a = extract_source_expert_chunks(
        args.source_data, args.source_num_data, args.pred_horizon)
    tgt_s, tgt_a = extract_target_demo_chunks(
        args.target_data, args.target_num_data, args.pred_horizon)

    obs_dim = src_s.shape[1]
    action_dim = src_a.shape[2]

    # ---- flow policy ----
    fcfg = StateFlowPolicyConfig(
        backbone=args.backbone,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
    )
    policy = StateFlowPolicy((obs_dim,), action_dim, fcfg).to(device)

    # normalization fit on SOURCE expert data, frozen through FT + eval
    state_mean = src_s.mean(0)
    state_std = src_s.std(0)
    a_flat = src_a.reshape(-1, action_dim)
    action_min = a_flat.min(0).values
    action_max = a_flat.max(0).values
    policy.set_normalization(state_mean, state_std, action_min, action_max)
    print(f"[norm] state_mean|std fit on {src_s.shape[0]} states; "
          f"action range [{action_min.min():.2f},{action_max.max():.2f}]")

    # ---- phase 1: source pretrain ----
    l1 = train_phase(policy, src_s, src_a, args.source_steps, args.batch_size,
                     args.lr, device, "phase1 source pretrain")
    torch.save(policy.state_dict(), os.path.join(args.save_dir, "flow_source.pt"))
    print(f"[phase1] done, loss={l1:.5f}; saved flow_source.pt")

    # ---- phase 2: target finetune ----
    l2 = train_phase(policy, tgt_s, tgt_a, args.target_steps, args.batch_size,
                     args.ft_lr, device, "phase2 target finetune")
    torch.save(policy.state_dict(), os.path.join(args.save_dir, "flow.pt"))

    json.dump({
        "obs_dim": obs_dim, "action_dim": action_dim,
        "backbone": args.backbone, "pred_horizon": args.pred_horizon,
        "action_horizon": args.action_horizon,
        "num_inference_steps": args.num_inference_steps,
        "source_num_data": args.source_num_data, "target_num_data": args.target_num_data,
        "source_steps": args.source_steps, "target_steps": args.target_steps,
        "final_source_loss": l1, "final_target_loss": l2,
    }, open(os.path.join(args.save_dir, "flow_cfg.json"), "w"), indent=2)
    print(f"[phase2] done, loss={l2:.5f}; saved flow.pt + flow_cfg.json")


if __name__ == "__main__":
    main()
