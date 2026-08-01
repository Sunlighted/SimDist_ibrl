"""Diagnostic: during rollout with the ORIGINAL SimDist MPPI recipe, what
fraction of the planner's chosen action comes from the policy-seeded "base"
candidates vs the pure-random candidates?

MPPI executes mean[0] = a softmax-weighted average of the topk elites. So we
report two views, aggregated over all rollout planning steps, on the FINAL
iteration (the one that produces the executed action) AND the FIRST iteration:
  * base count %  = fraction of the K elites whose candidate is a base candidate
  * base weight % = softmax-weight mass on base elites (= the actual % of the
                    executed action attributable to the base proposal)

Run on SOURCE (mode=none) and TARGET (§12 dynamics-only) with the stage-1 ckpt.

Usage:
  python scripts/diag_mppi_origin.py \
      --weight exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/model0.pt \
      --num_game 20
"""
import argparse
import os

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import train_simdist
from env.domain_shift import DomainShiftConfig
from env.robosuite_wrapper import PixelRobosuite
from rl.simdist_planner import apply_original_recipe


def rollout_domain(agent, env_params, num_game, seed, label):
    """Single-process rollouts; returns a list of (per-episode step-records,
    success_bool) so we can split base/random stats by episode outcome."""
    env = PixelRobosuite(**env_params)
    pl = agent.planner
    pl.record_origin = True

    episodes = []
    for g in range(num_game):
        obs, _ = env.reset()
        pl._origin_records.clear()            # records for THIS episode only
        steps, success, done = 0, 0.0, False
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, _, terminal, succ, _ = env.step(action)
            steps += 1
            success = max(success, float(succ))
            done = bool(terminal)
        episodes.append((list(pl._origin_records), success > 0.5))
        print(f"  [{label}] game {g+1}/{num_game}  len={steps}  success={success:.0f}")

    pl.record_origin = False
    return episodes


def _stats(step_records):
    """step_records: list (per step) of list (per iter) of (count, weight)."""
    if not step_records:
        return None
    first = np.array([r[0] for r in step_records])   # [N,2]
    last = np.array([r[-1] for r in step_records])    # [N,2]
    return len(step_records), first.mean(0), last.mean(0)


def summarize(episodes, label):
    succ = [(e, s) for e, s in episodes if s]
    fail = [(e, s) for e, s in episodes if not s]
    print(f"\n========== {label} ==========")
    print(f"episodes: {len(episodes)}  (success {len(succ)}, fail {len(fail)})")
    print(f"{'group':14s}{'steps':>8s}{'iter1 base%':>13s}{'final base%':>13s}   (rand = 100 - base; by softmax weight)")
    for name, group in [("ALL", episodes), ("SUCCESS only", succ), ("FAIL only", fail)]:
        recs = [r for eps, _ in group for r in eps]
        st = _stats(recs)
        if st is None:
            print(f"{name:14s}{0:>8d}{'--':>13s}{'--':>13s}")
            continue
        n, first, last = st
        # column = softmax-weight view (count view is ~identical; printed compactly)
        print(f"{name:14s}{n:>8d}{first[1]*100:>12.1f}%{last[1]*100:>12.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str,
                   default="exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/model0.pt")
    p.add_argument("--num_game", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    agent, eval_env, target_params = train_simdist.load_model(args.weight, "cuda")
    # FORCE planner + ORIGINAL recipe (matches --original_mppi 1)
    apply_original_recipe(agent.cfg.planner)
    agent.cfg.use_planner = 1
    from rl.simdist_planner import MppiPlanner
    agent.planner = MppiPlanner(agent.model, agent.cfg.planner, agent.action_dim, "cuda")
    p_ = agent.cfg.planner
    nb = p_.num_base_override
    print(f"ORIGINAL recipe: base={nb}+rand={p_.num_samples-nb}, iters={p_.iterations}, "
          f"init_std={p_.init_std}, temp={p_.temperature}, elites={p_.num_elites}")

    # target params come straight from the ckpt cfg (§12 dynamics-only target);
    # source = same env with domain_shift disabled.
    source_params = dict(target_params)
    source_params["domain_shift"] = DomainShiftConfig(mode="none")

    common_utils_seed(args.seed)
    src_eps = rollout_domain(agent, source_params, args.num_game, args.seed, "SOURCE")
    common_utils_seed(args.seed)
    tgt_eps = rollout_domain(agent, target_params, args.num_game, args.seed, "TARGET")

    summarize(src_eps, "SOURCE")
    summarize(tgt_eps, "TARGET")
    import sys
    sys.stdout.flush()
    os._exit(0)


def common_utils_seed(seed):
    import common_utils
    common_utils.set_all_seeds(seed)


if __name__ == "__main__":
    main()
