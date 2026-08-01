"""Eval threading_simdist_target_distill_sig01_seed1 three ways at sigma=0.1:
  (1) policy-head only (use_planner=0)
  (2) MPPI NARROW  (ibrl-tuned trust-the-proposal: mixture 0.8 / init_std 0.5)
  (3) MPPI WIDE    (ORIGINAL SimDist recipe: base=100+rand=250 / init_std 1.0)
All on TARGET (cfg eval_domain_shift §12) + strict_success, 500 games, sigma 0.1.
"""
import os
import numpy as np
import pyrallis
import torch

from train_simdist import MainConfig, Workspace
from evaluate import run_eval_mp
from rl.simdist_planner import apply_original_recipe

RUN_DIR = "exps/mimicgen/threading_simdist_target_distill_sig01_seed1"
CKPT = os.path.join(RUN_DIR, "model0.pt")
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
    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(RUN_DIR, "cfg.yaml"), "r"))  # type: ignore
    cfg.num_eval_episode = NUM_GAME
    # sigma stays at the run's native 0.1 (cfg.eval_hand_init_perturb_joint_sigma)
    ws = Workspace(cfg, from_main=False)
    agent = ws.agent
    agent.load_state_dict(torch.load(CKPT))
    agent = agent.to("cuda")
    env_params = ws.eval_env_params
    print(f"[eval] sigma = {cfg.eval_hand_init_perturb_joint_sigma}, "
          f"{NUM_GAME} games, TARGET + strict_success")

    # (1) policy-only
    s_pol = run(agent, env_params, use_planner=0, seed=0)
    print(f"[RESULT] policy-only          | score = {s_pol:.4f}")

    # (2) MPPI narrow (cfg defaults: mixture 0.8 / init_std 0.5)
    s_narrow = run(agent, env_params, use_planner=1, seed=0)
    print(f"[RESULT] MPPI narrow          | score = {s_narrow:.4f}")

    # (3) MPPI wide (ORIGINAL recipe) — mutate the planner cfg IN PLACE
    apply_original_recipe(agent.planner.cfg)
    s_wide = run(agent, env_params, use_planner=1, seed=0)
    print(f"[RESULT] MPPI wide (original) | score = {s_wide:.4f}")

    print("\n===== SUMMARY (sigma=0.1, 500 games, TARGET strict) =====")
    print(f"  policy-only          : {s_pol:.4f}")
    print(f"  MPPI narrow          : {s_narrow:.4f}")
    print(f"  MPPI wide (original) : {s_wide:.4f}")


if __name__ == "__main__":
    main()
