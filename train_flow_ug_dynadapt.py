"""Adapt a SimDist world-model's DYNAMICS on the TARGET, driven by flow BC + UG
(coeff 0.3) as the policy — INSTEAD of MPPI (CLAUDE.md §24 / project_flow_ug_simdist).

The §24 adaptation (train_simdist_adapt.py) rolls the MPPI planner in the TARGET
env to gather transitions and adapt the dynamics head. Here we replace MPPI, for
BOTH the data-collection policy AND eval, with the frozen flow BC policy + forward
Universal-Guidance steering (coeff 0.3), where the guidance world model IS the
dynamics being adapted (same WorldModel object). So as the dynamics migrates
source->target, the UG guidance it provides improves.

Start point: the SOURCE stage-1 world model
(threading_simdist_tdmpc2_stage1_seed1/latest.pt) — the exact ckpt that
threading_simdist_adapt_seed1 was MPPI-adapted from, so the endpoint of this run
is directly comparable to that MPPI-adapted WM (flow+UG@0.3 there = ~76-81%).

Expected curve: UG @ SOURCE dynamics is HARMFUL (memory: coeff 0.3 -> 8%), so
round 0 starts LOW and should RISE as the dynamics adapts toward target, ideally
approaching the MPPI-adapted endpoint. Every `adapt_interval` (1000) TARGET
transitions: train ONLY `model.dynamics` for `dyn_update_steps` steps (encoder/
reward/value/policy frozen), then EVAL flow+UG on TARGET, logged vs #transitions.

Run:
  python train_flow_ug_dynadapt.py \
      --stage1_ckpt exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
      --flow exps/mimicgen/threading_flow_ug_seed1/flow.pt \
      --save_dir exps/mimicgen/threading_flow_ug_dynadapt_seed1
"""

import argparse
import json
import os
import sys

import numpy as np
import pyrallis
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval_mp
from rl.sequence_replay import SequenceReplayBuffer
from rl.flow_ug_agent import FlowUGAgent, FlowUGConfig
from train_simdist import MainConfig, Workspace
from eval_flow_ug import load_flow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str,
                   default="exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt",
                   help="SOURCE stage-1 world model to adapt (its cfg.yaml rebuilds agent+env)")
    p.add_argument("--flow", type=str,
                   default="exps/mimicgen/threading_flow_ug_seed1/flow.pt",
                   help="frozen flow BC policy (target-FT'd). Pass the v2 self-distilled "
                        "flow to drive with the stronger base.")
    p.add_argument("--save_dir", type=str,
                   default="exps/mimicgen/threading_flow_ug_dynadapt_seed1")
    p.add_argument("--steering_coeff", type=float, default=0.3)
    p.add_argument("--ug_rollout_H", type=int, default=0, help="0 = world-model horizon")
    # adaptation loop
    p.add_argument("--adapt_interval", type=int, default=1000,
                   help="collect this many TARGET transitions per round")
    p.add_argument("--max_transitions", type=int, default=20000)
    p.add_argument("--dyn_update_steps", type=int, default=1000)
    p.add_argument("--dyn_lr", type=float, default=3e-4)
    p.add_argument("--adapt_batch_size", type=int, default=256)
    # eval
    p.add_argument("--num_eval_episode", type=int, default=200,
                   help="games per round for the curve (200 keeps 20 rounds tractable; "
                        "re-eval the best ckpt at 500 afterwards for the headline)")
    p.add_argument("--num_eval_proc", type=int, default=10)
    p.add_argument("--hand_init_sigma", type=float, default=0.1,
                   help="TARGET eval/collect hand-init joint sigma (match the 79-81% evals)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_wb", type=int, default=1)
    p.add_argument("--wb_project", type=str, default="mimicgen")
    p.add_argument("--wb_entity", type=str, default="cao1233")
    args = p.parse_args()

    common_utils.set_all_seeds(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # --- optional wandb ---
    wb_run = None
    if args.use_wb:
        try:
            import wandb
            wb_run = wandb.init(project=args.wb_project, entity=args.wb_entity,
                                name=os.path.basename(args.save_dir.rstrip("/")),
                                config=vars(args))
            print(f"[wandb] logging to {args.wb_entity}/{args.wb_project} "
                  f"run={wb_run.name}")
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing with log+json only")
            wb_run = None

    def wb_log(d, step):
        if wb_run is not None:
            wb_run.log(d, step=step)

    # --- rebuild the stage-1 SimDist agent + TARGET env from its cfg.yaml ---
    stage1_dir = os.path.dirname(args.stage1_ckpt)
    base_cfg: MainConfig = pyrallis.load(  # type: ignore
        MainConfig, open(os.path.join(stage1_dir, "cfg.yaml"), "r"))
    base_cfg.simdist.use_planner = 0            # MPPI unused; flow+UG drives everything
    base_cfg.eval_hand_init_perturb_joint_sigma = args.hand_init_sigma  # match evals
    ws = Workspace(base_cfg, from_main=False)
    sim_agent = ws.agent
    sim_agent.load_state_dict(torch.load(args.stage1_ckpt))
    sim_agent = sim_agent.to(device)
    model = sim_agent.model                     # WorldModel; only .dynamics will adapt

    eval_env = ws.eval_env                       # TARGET (eval_domain_shift)
    eval_env_params = ws.eval_env_params
    H = base_cfg.simdist.world_model.horizon
    print(f"[setup] source WM={os.path.basename(stage1_dir)} horizon={H} | "
          f"TARGET eval, hand_init_sigma={args.hand_init_sigma}")

    # --- frozen flow + UG agent sharing the SAME (adapting) world model ---
    flow = load_flow(args.flow, device)
    flow.Ta = 1                                  # re-query every step (the 79-81% regime)
    for prm in flow.parameters():
        prm.requires_grad_(False)
    ug_cfg = FlowUGConfig(use_ug=1, steering_coeff=args.steering_coeff,
                          ug_rollout_H=args.ug_rollout_H, use_reward=1, discount=0.99)
    ug_agent = FlowUGAgent(flow, model, ug_cfg, device=device)
    # flow-only reference agent (constant across adaptation; dynamics-independent)
    flowonly_agent = FlowUGAgent(flow, None, FlowUGConfig(use_ug=0), device=device)
    print(f"[setup] policy = flow({os.path.basename(os.path.dirname(args.flow))}) "
          f"+ UG(coeff={args.steering_coeff}) Ta=1")

    adapt_replay = SequenceReplayBuffer(max_episodes=100000, device="cpu")
    _obs = {"v": None}  # persistent rollout episode state across rounds

    def eval_agent(agent, seed):
        rng = np.random.get_state()
        try:
            with utils.eval_mode(agent):
                scores = run_eval_mp(env_params=eval_env_params, agent=agent,
                                     num_proc=args.num_eval_proc,
                                     num_game=args.num_eval_episode, seed=seed, verbose=False)
        finally:
            np.random.set_state(rng)
        return float(np.mean(scores))

    def collect(n_transitions):
        if _obs["v"] is None:
            o, _ = eval_env.reset()
            _obs["v"] = o
            adapt_replay.new_episode(o["state"])
        collected = 0
        with utils.eval_mode(ug_agent):
            while collected < n_transitions:
                action = ug_agent.act(_obs["v"], eval_mode=True, cpu=True)  # UG, Ta=1
                o, reward, terminal, success, _ = eval_env.step(action)
                _obs["v"] = o
                adapt_replay.add(o["state"], action, reward, terminal, success)
                collected += 1
                if terminal:
                    o, _ = eval_env.reset()
                    _obs["v"] = o
                    adapt_replay.new_episode(o["state"])

    def train_dynamics():
        if not adapt_replay.can_sample(H):
            print(f"[warn] buffer has no episode >= horizon={H}; skip dyn update")
            return float("nan")
        sim_agent.train()
        last = 0.0
        for _ in range(args.dyn_update_steps):
            states, actions, _ = adapt_replay.sample(args.adapt_batch_size, H, "cuda")
            m = sim_agent.update_dynamics_only(states, actions, lr=args.dyn_lr)
            last = m["adapt/consistency"]
        return last

    # ---- flow-only reference (one-time; independent of the adapting dynamics) ----
    flow_only_ref = eval_agent(flowonly_agent, seed=777)
    print(f"[ref] flow-only (no UG, dynamics-independent) target_success = {flow_only_ref:.3f}")

    # ---- round 0: flow+UG @ SOURCE dynamics (expected LOW / harmful) ----
    curve = []
    best_sr, best_step = -1.0, -1
    sr = eval_agent(ug_agent, seed=0)
    curve.append({"transitions": 0, "success": sr, "consistency": float("nan")})
    best_sr, best_step = sr, 0
    torch.save(model.state_dict(), os.path.join(args.save_dir, "wm.pt"))
    wb_log({"score/score": sr, "score/flow_only_ref": flow_only_ref,
            "adapt/transitions": 0}, step=0)
    print(f"[dynadapt @ 0 transitions] flow+UG target_success = {sr:.3f}  (source dyn, baseline)",
          flush=True)

    n_rounds = args.max_transitions // args.adapt_interval
    total = 0
    for r in range(1, n_rounds + 1):
        collect(args.adapt_interval)
        total += args.adapt_interval
        cons = train_dynamics()
        sr = eval_agent(ug_agent, seed=r * args.num_eval_episode)
        tag = ""
        if sr > best_sr:
            best_sr, best_step = sr, total
            torch.save(model.state_dict(), os.path.join(args.save_dir, "wm.pt"))
            tag = "  <- best (saved wm.pt)"
        curve.append({"transitions": total, "success": sr, "consistency": cons})
        wb_log({"score/score": sr, "score/flow_only_ref": flow_only_ref,
                "adapt/consistency": cons, "adapt/transitions": total,
                "other/buffer_episodes": adapt_replay.size()}, step=total)
        print(f"[dynadapt @ {total:5d} transitions] flow+UG target_success = {sr:.3f}  "
              f"consistency={cons:.4f}  buf_ep={adapt_replay.size()}{tag}", flush=True)
        json.dump({"flow": args.flow, "stage1_ckpt": args.stage1_ckpt,
                   "steering_coeff": args.steering_coeff, "flow_only_ref": flow_only_ref,
                   "best_success": best_sr, "best_transitions": best_step,
                   "num_eval_episode": args.num_eval_episode, "curve": curve},
                  open(os.path.join(args.save_dir, "dynadapt_curve.json"), "w"), indent=2)

    print(f"\n[done] flow+UG dynamics adaptation: source-dyn {curve[0]['success']:.3f} -> "
          f"best {best_sr:.3f} @ {best_step} transitions")
    print(f"       flow-only ref={flow_only_ref:.3f}; MPPI-adapted-WM reference "
          f"(same flow, flow+UG@{args.steering_coeff}) — compare to your 500-game number")
    print("curve:", " ".join(f"{c['transitions']}:{c['success']:.2f}" for c in curve))
    if wb_run is not None:
        wb_run.finish()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
