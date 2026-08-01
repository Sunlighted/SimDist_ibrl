"""Fine-tune a SOURCE flow BC on N TARGET demos, then measure how well a TARGET-Q
Universal-Guidance steers it — as a function of N (2026-07-06 study).

Question: how many target demos does the source flow need before ∇Q (target
expert critic) guidance actually works? Sweep N in {20,50,100,200,500}.

Per run: load the n500 SOURCE flow (train_bc ckpt), fine-tune on the first N
demos of threading_d0_target_expert_n1000 (§12 dynamics-only target), with
periodic TARGET flow-only eval + best-by-success checkpoint (overfit guard).
After FT, eval the BEST checkpoint on TARGET: flow-only AND Q-UG@coeff (300
games each). Logs to wandb.
"""

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from evaluate import run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.flow_qug_agent import FlowQUGAgent, FlowQUGConfig
from train_flow_ug import extract_target_demo_chunks, _Batch
from eval_flow_ug import TARGET_SHIFT
from eval_flow_qug import load_q_agent


def build_target_env_params(strict, sigma):
    return dict(
        env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        strict_success=StrictSuccessConfig(enabled=bool(strict), dwell_steps=3,
                                           max_tripod_disp=0.04),
        domain_shift=DomainShiftConfig(**TARGET_SHIFT),
        hand_init_perturb_joint_sigma=(sigma if sigma > 0 else None),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_flow", type=str,
                   default="exps/mimicgen/threading_flow_bc_n500_seed1/latest.pt",
                   help="SOURCE flow to fine-tune (train_bc ckpt)")
    p.add_argument("--target_data", type=str,
                   default="release/data/mimicgen/threading_d0_target_expert_n1000/dataset.hdf5")
    p.add_argument("--target_num_data", type=int, required=True, help="N target demos")
    p.add_argument("--q_ckpt", type=str,
                   default="exps/mimicgen/threading_ibrl_target_snapshots_rlprior_seed1/latest.pt")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--steering_coeff", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--eval_every", type=int, default=3000)
    p.add_argument("--num_eval_game", type=int, default=100, help="flow-only eval during FT")
    p.add_argument("--final_num_game", type=int, default=300, help="flow-only + Q-UG at end")
    p.add_argument("--num_proc", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
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
        except Exception as e:
            print(f"[wandb] init failed ({e})"); wb_run = None

    def wb_log(d, step):
        if wb_run is not None:
            wb_run.log(d, step=step)

    # ---- load SOURCE flow (train_bc ckpt) ----
    import train_bc
    flow, _, _ = train_bc.load_model(args.source_flow, device, verbose=False)
    flow = flow.to(device)
    # EMA shadow/backup are plain dicts (not moved by .to()); force to device so
    # update_ema (param.data on cuda) doesn't hit a cpu/cuda mismatch.
    flow._ema_state = {k: v.to(device) for k, v in flow._ema_state.items()}
    flow._ema_backup = {k: v.to(device) for k, v in flow._ema_backup.items()}
    flow.train()
    Tp = flow.Tp
    print(f"[load] source flow {args.source_flow} | Tp={Tp} Ta={flow.Ta}")

    # ---- N target demo chunks ----
    tgt_s, tgt_a = extract_target_demo_chunks(args.target_data, args.target_num_data, Tp)
    tgt_s, tgt_a = tgt_s.to(device), tgt_a.to(device)
    N = tgt_s.shape[0]
    print(f"[data] {args.target_num_data} target demos -> {N} chunks")

    # ---- target env + agents ----
    env_params = build_target_env_params(args.strict_success, args.hand_init_sigma)
    env = PixelRobosuite(**env_params)  # type: ignore
    flowonly = FlowQUGAgent(flow, None, FlowQUGConfig(use_ug=0), device=device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr)

    def eval_flowonly(seed):
        flow.eval(); applied = flow.apply_ema()
        rng = np.random.get_state()
        try:
            sc = run_eval_mp(env_params=env_params, agent=flowonly, num_proc=args.num_proc,
                             num_game=args.num_eval_game, seed=seed, verbose=False)
        finally:
            np.random.set_state(rng)
            if applied: flow.restore_from_ema()
        flow.train(); return float(np.mean(sc))

    # ---- fine-tune with periodic TARGET flow-only eval + best ----
    best_sr, best_step, last_loss = -1.0, -1, float("nan")
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            sr = eval_flowonly(seed=1000 + step)
            torch.save(flow.state_dict(), os.path.join(args.save_dir, "latest.pt"))
            tag = "latest"
            if sr > best_sr:
                best_sr, best_step = sr, step
                torch.save(flow.state_dict(), os.path.join(args.save_dir, "best.pt"))
                tag += "+best"
            wb_log({"ft/flowonly_target": sr, "ft/loss": last_loss, "ft/best": best_sr}, step)
            print(f"[N={args.target_num_data} ft@{step:5d}] flowonly_target={sr:.3f} "
                  f"loss={last_loss:.4f} {tag}", flush=True)
            if step == args.steps:
                break
        flow.train()
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        loss = flow.loss(_Batch(tgt_s[idx], tgt_a[idx]))
        opt.zero_grad(); loss.backward(); opt.step(); flow.update_ema()
        last_loss = loss.item()

    # ---- final eval on BEST: flow-only + Q-UG ----
    flow.load_state_dict(torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device))
    flow.eval(); flow.apply_ema()
    q_agent = load_q_agent(args.q_ckpt, env, device)

    def final_eval(agent, seed):
        rng = np.random.get_state()
        try:
            sc = run_eval_mp(env_params=env_params, agent=agent, num_proc=args.num_proc,
                             num_game=args.final_num_game, seed=seed, verbose=False)
        finally:
            np.random.set_state(rng)
        return float(np.mean(sc)), float(np.std([float(s) for s in sc]) / max(len(sc) ** 0.5, 1))

    fo, fo_se = final_eval(FlowQUGAgent(flow, None, FlowQUGConfig(use_ug=0), device=device), seed=7)
    qug, qug_se = final_eval(
        FlowQUGAgent(flow, q_agent, FlowQUGConfig(use_ug=1, steering_coeff=args.steering_coeff),
                     device=device), seed=7)

    result = {"N": args.target_num_data, "best_step": best_step,
              "flowonly_best_ft": best_sr, "final_flowonly": fo, "final_flowonly_se": fo_se,
              "final_qug": qug, "final_qug_se": qug_se, "steering_coeff": args.steering_coeff}
    json.dump(result, open(os.path.join(args.save_dir, "result.json"), "w"), indent=2)
    wb_log({"final/flowonly": fo, "final/qug": qug}, args.steps)
    print(f"\n[RESULT N={args.target_num_data}] flow-only={fo:.3f}±{fo_se:.3f}  "
          f"Q-UG@{args.steering_coeff}={qug:.3f}±{qug_se:.3f}  (best_ft_step={best_step})", flush=True)
    if wb_run is not None:
        wb_run.finish()
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
