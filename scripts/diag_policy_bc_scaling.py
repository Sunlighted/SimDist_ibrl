"""H3 test: does scaling the policy-head BC data lift the MPPI ceiling?

Takes a base SimDist ckpt (e.g. the adapted 0.816 run, best dynamics), and for each
N in --demo_counts: reload the base ckpt (reset policy/dynamics/value), BC ONLY the
policy head on N TARGET demos, then eval BOTH policy-only (use_planner=0) and MPPI
(use_planner=1) on the TARGET env. If MPPI success rises with the policy's
standalone success, the frozen mediocre policy is the binding constraint (H3);
if MPPI plateaus regardless of policy quality, the policy is NOT the bottleneck.

Matches the 0.816 run's eval (sigma + the user's MPPI recipe) for comparability.

Run:
  python scripts/diag_policy_bc_scaling.py \
      --stage1_cfg exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/cfg.yaml \
      --base_ckpt exps/mimicgen/threading_simdist_adapt_seed1/model0.pt \
      --demos release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5 \
      --demo_counts 20 100 500 --bc_steps 3000 --num_eval 200 --sigma 0.1 --original_mppi 1
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pyrallis

import common_utils
from evaluate import run_eval_mp
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from train_simdist import MainConfig, Workspace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_cfg", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--demo_counts", nargs="+", type=int, default=[20, 100, 500])
    ap.add_argument("--bc_steps", type=int, default=3000)
    ap.add_argument("--bc_lr", type=float, default=3e-4)
    ap.add_argument("--bc_batch", type=int, default=256)
    ap.add_argument("--num_eval", type=int, default=200)
    ap.add_argument("--num_proc", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--original_mppi", type=int, default=1)
    ap.add_argument("--policy_from_scratch", type=int, default=0,
                    help="random-reinit the policy head each round (train it from "
                         "scratch on the target demos) instead of finetuning the "
                         "base ckpt's policy. encoder/dynamics/value stay frozen base.")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="episode-level held-out fraction for policy-BC early "
                         "stopping (model selection). 0 disables (use the final "
                         "policy). Guards against overfitting the few target demos.")
    ap.add_argument("--val_every", type=int, default=50,
                    help="compute val loss every N BC steps; keep the policy with "
                         "the lowest val loss for eval.")
    ap.add_argument("--min_val_ep", type=int, default=3,
                    help="minimum held-out episodes when val_frac>0.")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    common_utils.set_all_seeds(args.seed)
    cfg: MainConfig = pyrallis.load(MainConfig, open(args.stage1_cfg, "r"))  # type: ignore
    cfg.simdist.use_planner = 1
    cfg.eval_hand_init_perturb_joint_sigma = args.sigma
    if args.original_mppi:
        from rl.simdist_planner import apply_original_recipe
        apply_original_recipe(cfg.simdist.planner)
    ws = Workspace(cfg, from_main=False)
    agent = ws.agent
    H = cfg.simdist.world_model.horizon
    # snapshot the freshly-constructed (random) policy-head weights so that, in
    # --policy_from_scratch mode, every round starts the policy from the SAME clean
    # random init (isolates "how much target demo does the policy head need" from
    # any base-ckpt policy warm start).
    init_policy_sd = {k: v.detach().clone() for k, v in agent.model.policy.state_dict().items()}
    base_sd = torch.load(args.base_ckpt)

    def evaluate(use_planner, seed):
        agent.cfg.use_planner = use_planner
        rs = np.random.get_state()
        scores = run_eval_mp(env_params=ws.eval_env_params, agent=agent,
                             num_proc=args.num_proc, num_game=args.num_eval,
                             seed=seed, verbose=False)
        np.random.set_state(rs)
        return float(np.mean(scores))

    results = []
    for n in args.demo_counts:
        agent.load_state_dict(base_sd)   # reset policy/dynamics/value to base
        if args.policy_from_scratch:
            agent.model.policy.load_state_dict(init_policy_sd)  # wipe base policy -> random
        agent = agent.to("cuda")
        agent._bc_opt = None             # fresh BC optimizer
        buf = SequenceReplayBuffer(max_episodes=20000, device="cpu")
        load_demos_to_sequence_buffer(buf, args.demos, num_data=n,
                                      reward_scale=cfg.env_reward_scale)
        # Episode-level held-out split for early stopping: BC-finetuning the policy
        # on a handful of demos overfits fast, so we track a val loss and keep the
        # policy weights at the val-loss minimum instead of the (overfit) final step.
        n_val = 0
        val_buf = None
        if args.val_frac > 0 and buf.size() > args.min_val_ep:
            n_val = max(args.min_val_ep, int(round(args.val_frac * buf.size())))
            n_val = min(n_val, buf.size() - 1)  # keep >=1 train episode
            val_buf = SequenceReplayBuffer(max_episodes=20000, device="cpu")
            val_buf.episodes = buf.episodes[-n_val:]   # deterministic split
            buf.episodes = buf.episodes[:-n_val]

        def val_loss():
            agent.eval()
            rs = np.random.get_state()
            np.random.seed(0)  # fixed val minibatch across steps for comparability
            with torch.no_grad():
                vs, va, _ = val_buf.sample(1024, H, "cuda")
                pred = agent.model.policy_action(vs[:, :H].reshape(-1, vs.shape[-1]))
                vl = F.mse_loss(pred, va.reshape(-1, va.shape[-1])).item()
            np.random.set_state(rs)
            agent.train()
            return vl

        agent.train()
        last = 0.0
        best_val, best_step, best_sd = float("inf"), -1, None
        for step in range(args.bc_steps):
            s, a, _ = buf.sample(args.bc_batch, H, "cuda")
            last = agent.update_bc_policy(s, a, lr=args.bc_lr)["bc/policy_loss"]
            if val_buf is not None and (step % args.val_every == 0 or step == args.bc_steps - 1):
                vl = val_loss()
                if vl < best_val:
                    best_val, best_step = vl, step
                    best_sd = {k: v.detach().clone()
                               for k, v in agent.model.policy.state_dict().items()}
        if best_sd is not None:  # restore the early-stopped (best-val) policy
            agent.model.policy.load_state_dict(best_sd)
        agent.eval()
        pol = evaluate(use_planner=0, seed=10_000)
        mppi = evaluate(use_planner=1, seed=10_000)
        results.append((n, buf.size(), last, best_val, best_step, pol, mppi))
        print(f"[bc-scale] demos={n:4d} (train {buf.size()} / val {n_val})  "
              f"train_loss={last:.5f}  best_val={best_val:.5f}@{best_step}  "
              f"policy_only={pol:.3f}  MPPI={mppi:.3f}")
        sys.stdout.flush()

    mode = "policy FROM SCRATCH" if args.policy_from_scratch else "policy FINETUNED from base"
    print(common_utils.wrap_ruler(f"H3: policy-BC-data scaling [{mode}]"))
    print(f"{'demos':>6} | {'train_loss':>10} | {'best_val':>9} | {'@step':>6} | "
          f"{'policy_only':>11} | {'MPPI':>6} | {'MPPI-policy':>11}")
    for n, sz, bl, bv, bstep, pol, mppi in results:
        print(f"{n:>6} | {bl:>10.5f} | {bv:>9.5f} | {bstep:>6} | "
              f"{pol:>11.3f} | {mppi:>6.3f} | {mppi - pol:>+11.3f}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.allow_tf32 = True   # type: ignore
    torch.backends.cudnn.benchmark = True    # type: ignore
    main()
