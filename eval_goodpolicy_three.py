"""Eval threading_simdist_adapt_goodpolicy_seed1 (strong 500-demo BC base) three
ways at sigma=0.1, 500 games, TARGET + strict:
  (1) policy-head only          (the ~0.78 bar)
  (2) MPPI NARROW  (trust-proposal, stage-1 cfg defaults)
  (3) MPPI WIDE    (ORIGINAL SimDist recipe — what training used)
Uses the ADAPTED latest.pt (20k transitions, most dynamics-adapted).
AdaptWorkspace redirects stdout to its log, so RESULT lines land in
<save_dir>/train.log.
"""
import os
import numpy as np
import pyrallis
import torch

from train_simdist_adapt import AdaptConfig, AdaptWorkspace
from evaluate import run_eval_mp
from rl.simdist_planner import apply_original_recipe

RUN_DIR = "exps/mimicgen/threading_simdist_adapt_goodpolicy_seed1"
CKPT = os.path.join(RUN_DIR, "latest.pt")
NUM_GAME = 500
NUM_PROC = 10


def run(agent, env_params, use_planner, seed):
    prev = agent.cfg.use_planner
    agent.cfg.use_planner = use_planner
    rng = np.random.get_state()
    try:
        scores = run_eval_mp(env_params=env_params, agent=agent, num_proc=NUM_PROC,
                             num_game=NUM_GAME, seed=seed, verbose=False)
    finally:
        np.random.set_state(rng)
        agent.cfg.use_planner = prev
    return float(np.mean(scores))


def main():
    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(RUN_DIR, "cfg.yaml"), "r"))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1   # run's native
    cfg.num_eval_episode = NUM_GAME
    cfg.mp_eval = 1
    cfg.use_wb = 0
    # build with NARROW planner (don't mutate to original at construction);
    # we switch to wide in-process afterwards.
    cfg.original_mppi = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/6489b744-685c-4008-8662-9f1ebbbd9c66/scratchpad/goodpolicy_eval"

    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(CKPT))
    ws.agent = ws.agent.to("cuda")
    agent = ws.agent
    p = agent.planner.cfg
    print(f"[eval] latest.pt (20k transitions) | sigma=0.1 | {NUM_GAME} games | TARGET strict")
    print(f"[planner NARROW] num_samples={p.num_samples} mixture={p.mixture_coef} "
          f"init_std={p.init_std} iters={p.iterations} temp={p.temperature} horizon={p.horizon}")

    s_pol = run(agent, ws.eval_env_params, use_planner=0, seed=0)
    print(f"[RESULT] policy-only          | score = {s_pol:.4f}")

    s_narrow = run(agent, ws.eval_env_params, use_planner=1, seed=0)
    print(f"[RESULT] MPPI narrow          | score = {s_narrow:.4f}")

    apply_original_recipe(agent.planner.cfg)
    s_wide = run(agent, ws.eval_env_params, use_planner=1, seed=0)
    print(f"[RESULT] MPPI wide (original) | score = {s_wide:.4f}")

    print("\n===== SUMMARY (goodpolicy, sigma=0.1, 500 games, TARGET strict) =====")
    print(f"  policy-only          : {s_pol:.4f}")
    print(f"  MPPI narrow          : {s_narrow:.4f}")
    print(f"  MPPI wide (original) : {s_wide:.4f}")


if __name__ == "__main__":
    main()
