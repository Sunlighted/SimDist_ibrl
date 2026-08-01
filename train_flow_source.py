"""Train a flow BC policy on SOURCE data (diverse_v2), with periodic SOURCE eval.

Two data regimes (`--data_mode`) for the 2026-07-06 Q-UG transfer experiment:
  * "all"    -> flow policy 1: BC on ALL steps of diverse_v2 (expert + noised/
                exploratory actions), broad coverage.
  * "expert" -> flow policy 2: BC on the EXPERT subset only (expert_flags==1
                contiguous runs), clean expert.

No target fine-tuning: the flow is PURE source BC. Transfer to target is done
later at inference by Q-UG steering (eval_flow_qug.py) with a target expert Q.

Every `--eval_every` steps, evaluate flow-only on the SOURCE env (mode=none),
`--num_eval_game` (300) games, and save latest.pt (+ best.pt by source success).
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")

from evaluate import run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig
from rl.flow_qug_agent import FlowQUGAgent, FlowQUGConfig
from train_flow_ug import extract_source_expert_chunks, _Batch


def extract_source_all_chunks(path: str, num_data: int, Tp: int, clamp: bool = True):
    """Extract (state, action_chunk[Tp]) from EVERY step of the source dataset —
    no expert filtering (flow policy 1). Chunks are padded at episode end with the
    last action. state[t] is the obs BEFORE action[t]."""
    f = h5py.File(path, "r")
    g = f["data"]
    ids = sorted((k for k in g.keys() if k.startswith("demo_")),
                 key=lambda k: int(k.split("_")[1]))
    if num_data > 0:
        ids = ids[:num_data]
    states, chunks = [], []
    for k in tqdm(ids, desc=f"extract ALL chunks (Tp={Tp})", dynamic_ncols=True,
                  mininterval=1.0):
        ep = g[k]
        state = np.array(ep["obs"]["state"]).astype(np.float32)   # [T, obs]
        action = np.array(ep["actions"]).astype(np.float32)       # [T, act]
        if clamp:
            action = np.clip(action, -1.0, 1.0)
        T = action.shape[0]
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
    print(f"[source/all] {states.shape[0]} chunks from {len(ids)} demos, "
          f"obs={states.shape[1]}, act={chunks.shape[2]}")
    return states, chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default="release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5")
    p.add_argument("--data_mode", type=str, default="all", choices=["all", "expert"])
    p.add_argument("--num_data", type=int, default=30000,
                   help="# source demos to draw from (-1 = all; RAM/time heavy)")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--pred_horizon", type=int, default=16, help="Tp (UNet-safe)")
    p.add_argument("--action_horizon", type=int, default=8, help="Ta (train); eval overrides to 1")
    p.add_argument("--backbone", type=str, default="unet1d")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=10000)
    p.add_argument("--num_eval_game", type=int, default=300)
    p.add_argument("--num_proc", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_wb", type=int, default=1)
    p.add_argument("--wb_project", type=str, default="mimicgen")
    p.add_argument("--wb_entity", type=str, default="cao1233")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    wb_run = None
    if args.use_wb:
        try:
            import wandb
            wb_run = wandb.init(project=args.wb_project, entity=args.wb_entity,
                                name=os.path.basename(args.save_dir.rstrip("/")),
                                config=vars(args))
            print(f"[wandb] logging to {args.wb_entity}/{args.wb_project} run={wb_run.name}")
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing with log+json only")
            wb_run = None

    def wb_log(d, step):
        if wb_run is not None:
            wb_run.log(d, step=step)

    # ---- data ----
    if args.data_mode == "all":
        s, a = extract_source_all_chunks(args.data, args.num_data, args.pred_horizon)
    else:
        s, a = extract_source_expert_chunks(args.data, args.num_data, args.pred_horizon)
    obs_dim, action_dim = s.shape[1], a.shape[2]
    N = s.shape[0]
    # keep pools on CPU (can be several GB); move per-batch to GPU
    s, a = s.cpu(), a.cpu()

    # ---- flow policy + normalization (fit on this training data) ----
    fcfg = StateFlowPolicyConfig(
        backbone=args.backbone, pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon, num_inference_steps=args.num_inference_steps)
    flow = StateFlowPolicy((obs_dim,), action_dim, fcfg).to(device)
    state_mean, state_std = s.mean(0), s.std(0)
    a_flat = a.reshape(-1, action_dim)
    flow.set_normalization(state_mean, state_std, a_flat.min(0).values, a_flat.max(0).values)
    print(f"[norm] fit on {N} states; action range "
          f"[{a_flat.min():.2f},{a_flat.max():.2f}]")

    # ---- SOURCE eval env (flow-only) ----
    env_params = dict(
        env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        strict_success=StrictSuccessConfig(enabled=bool(args.strict_success),
                                           dwell_steps=3, max_tripod_disp=0.04),
        domain_shift=DomainShiftConfig(mode="none"),
        hand_init_perturb_joint_sigma=(args.hand_init_sigma if args.hand_init_sigma > 0 else None),
    )
    agent = FlowQUGAgent(flow, None, FlowQUGConfig(use_ug=0), device=device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr)

    def eval_source(seed):
        flow.eval()
        applied = flow.apply_ema()
        rng = np.random.get_state()
        try:
            scores = run_eval_mp(env_params=env_params, agent=agent,
                                 num_proc=args.num_proc, num_game=args.num_eval_game,
                                 seed=seed, verbose=False)
        finally:
            np.random.set_state(rng)
            if applied:
                flow.restore_from_ema()
        flow.train()
        return float(np.mean(scores))

    def save_cfg(extra):
        json.dump({"obs_dim": obs_dim, "action_dim": action_dim,
                   "backbone": args.backbone, "pred_horizon": args.pred_horizon,
                   "action_horizon": args.action_horizon,
                   "num_inference_steps": args.num_inference_steps,
                   "data_mode": args.data_mode, "num_data": args.num_data,
                   "num_chunks": N, **extra},
                  open(os.path.join(args.save_dir, "flow_cfg.json"), "w"), indent=2)

    curve = []
    best_sr, best_step, last_loss = -1.0, -1, float("nan")
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            sr = eval_source(seed=1000 + step)
            curve.append({"step": step, "source_success": sr, "train_loss": last_loss})
            torch.save(flow.state_dict(), os.path.join(args.save_dir, "latest.pt"))
            tag = "  latest"
            if sr > best_sr:
                best_sr, best_step = sr, step
                torch.save(flow.state_dict(), os.path.join(args.save_dir, "best.pt"))
                tag += " + best"
            save_cfg({"best_step": best_step, "best_source_success": best_sr, "curve": curve})
            wb_log({"source/success": sr, "train/loss": last_loss,
                    "source/best": best_sr}, step=step)
            print(f"[flow-{args.data_mode} @ {step:6d}] source_success={sr:.3f}  "
                  f"train_loss={last_loss:.4f}{tag}", flush=True)
            if step == args.steps:
                break
        flow.train()
        idx = torch.randint(0, N, (args.batch_size,))
        loss = flow.loss(_Batch(s[idx].to(device), a[idx].to(device)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        flow.update_ema()
        last_loss = loss.item()

    print(f"\n[done] flow-{args.data_mode}: best source_success={best_sr:.3f} @ step {best_step}")
    print("curve:", " ".join(f"{c['step']}:{c['source_success']:.2f}" for c in curve))
    if wb_run is not None:
        wb_run.finish()
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
