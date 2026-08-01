"""TARGET fine-tune of the source-pretrained flow policy WITH eval-based
checkpoint selection (overfit guard) — the honest version of train_flow_ug.py's
phase 2.

Loads `flow_source.pt` (source-pretrained flow, before any target FT) and
fine-tunes on the 20 TARGET demos, but instead of a blind fixed step budget it
evaluates TARGET strict-success every `eval_every` steps (arm-A, flow-only) and
saves the BEST-by-success checkpoint. The success curve rising-then-falling is
the overfit signal; step 0 (= source-only, no target FT) is the left anchor.

This directly mirrors what the SimDist adapt pipeline does for the MLP policy
(periodic target eval, best-by-target), so the flow (A) number is selected the
same honest way as the 42% baseline — no train/eval mismatch, no memorized-demo
overfit slipping through.

Run:
  python finetune_flow_ug.py \
      --source_flow exps/mimicgen/threading_flow_ug_seed1/flow_source.pt \
      --save_dir exps/mimicgen/threading_flow_ug_sel_seed1 \
      --steps 8000 --eval_every 1000 --num_eval_game 100
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from evaluate import run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig
from rl.flow_ug_agent import FlowUGAgent, FlowUGConfig
from train_flow_ug import extract_target_demo_chunks, _Batch

TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=[0.0, 0.0, 0.0],
    ring_size_scale=0.85, ring_friction_scale=2.0, mask_hole_offset_in_obs=False,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_flow", type=str,
                   default="exps/mimicgen/threading_flow_ug_seed1/flow_source.pt")
    p.add_argument("--save_dir", type=str,
                   default="exps/mimicgen/threading_flow_ug_sel_seed1")
    p.add_argument("--target_data", type=str,
                   default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--target_num_data", type=int, default=20)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--num_eval_game", type=int, default=100)
    p.add_argument("--num_proc", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- rebuild the flow from the SOURCE dir's flow_cfg.json + flow_source.pt ----
    src_dir = os.path.dirname(args.source_flow)
    fj = json.load(open(os.path.join(src_dir, "flow_cfg.json")))
    fcfg = StateFlowPolicyConfig(
        backbone=fj["backbone"], pred_horizon=fj["pred_horizon"],
        action_horizon=fj["action_horizon"], num_inference_steps=fj["num_inference_steps"])
    flow = StateFlowPolicy((fj["obs_dim"],), fj["action_dim"], fcfg).to(device)
    flow.load_state_dict(torch.load(args.source_flow, map_location=device))
    print(f"[load] source-pretrained flow: {args.source_flow}")

    # ---- target demo chunks (all 20) ----
    tgt_s, tgt_a = extract_target_demo_chunks(args.target_data, args.target_num_data,
                                              fj["pred_horizon"])
    tgt_s, tgt_a = tgt_s.to(device), tgt_a.to(device)
    N = tgt_s.shape[0]

    # ---- target env for eval ----
    env_params = dict(
        env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        strict_success=StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04),
        domain_shift=DomainShiftConfig(**TARGET_SHIFT),
        hand_init_perturb_joint_sigma=args.hand_init_sigma,
    )
    agent = FlowUGAgent(flow, None, FlowUGConfig(use_ug=0), device=device)

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr)

    def eval_target(seed):
        flow.eval()
        applied = flow.apply_ema()
        random_state = np.random.get_state()
        try:
            scores = run_eval_mp(env_params=env_params, agent=agent,
                                 num_proc=args.num_proc, num_game=args.num_eval_game,
                                 seed=seed, verbose=False)
        finally:
            np.random.set_state(random_state)
            if applied:
                flow.restore_from_ema()
        flow.train()
        return float(np.mean(scores))

    curve = []
    best_sr, best_step, last_loss = -1.0, -1, float("nan")
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            sr = eval_target(seed=1000 + step)
            curve.append({"step": step, "success": sr, "train_loss": last_loss})
            tag = ""
            if sr > best_sr:
                best_sr, best_step = sr, step
                torch.save(flow.state_dict(), os.path.join(args.save_dir, "flow.pt"))
                tag = "  <- best (saved)"
            print(f"[ft @ {step:5d}] target_success={sr:.3f}  "
                  f"train_loss={last_loss:.4f}{tag}", flush=True)
            if step == args.steps:
                break
        # one train step
        flow.train()
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        loss = flow.loss(_Batch(tgt_s[idx], tgt_a[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        flow.update_ema()
        last_loss = loss.item()

    # save cfg + curve; flow.pt is the best-by-success checkpoint
    json.dump({**fj, "best_step": best_step, "best_success": best_sr,
               "ft_steps": args.steps, "eval_every": args.eval_every,
               "num_eval_game": args.num_eval_game, "hand_init_sigma": args.hand_init_sigma},
              open(os.path.join(args.save_dir, "flow_cfg.json"), "w"), indent=2)
    json.dump(curve, open(os.path.join(args.save_dir, "ft_curve.json"), "w"), indent=2)
    print(f"\n[done] best target_success={best_sr:.3f} @ step {best_step}  "
          f"(saved flow.pt); reference MLP policy-only ≈ 0.42")
    print("curve:", " ".join(f"{c['step']}:{c['success']:.2f}" for c in curve))
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()
