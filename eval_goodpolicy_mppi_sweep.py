"""Can a better-tuned MPPI beat a STRONG base policy (goodpolicy, ~0.73)?

Sweep planner variants on the SAME adapted ckpt (latest.pt, 20k transitions),
sigma=0.1, 300 games each, TARGET strict. Re-measures policy-only at 300 games
as the matched bar. All variants push toward "trust the proposal" (candidates
hug the policy rollout; value only re-ranks) — the only regime where MPPI could
plausibly help a good policy. Also a short-horizon / pure-value decisive test.

AdaptWorkspace redirects stdout to <save_dir>/train.log.
"""
import os
import numpy as np
import pyrallis
import torch

from train_simdist_adapt import AdaptConfig, AdaptWorkspace
from evaluate import run_eval_mp

RUN_DIR = "exps/mimicgen/threading_simdist_adapt_goodpolicy_seed1"
CKPT = os.path.join(RUN_DIR, "latest.pt")
NUM_GAME = 300
NUM_PROC = 10
SEED = 0


def set_horizon(planner, H):
    """horizon changes need discounts/final_discount recomputed (built in __init__)."""
    planner.cfg.horizon = H
    dev = planner.discounts.device
    planner.discounts = (planner.cfg.discount ** torch.arange(H, device=dev)).view(H, 1)
    planner.final_discount = planner.cfg.discount ** H


def apply_cfg(planner, base, overrides):
    """reset planner.cfg to `base` dict, then apply overrides; handle horizon."""
    for k, v in base.items():
        setattr(planner.cfg, k, v)
    H = overrides.pop("horizon", base["horizon"])
    for k, v in overrides.items():
        setattr(planner.cfg, k, v)
    set_horizon(planner, H)


def run(agent, env_params, use_planner):
    prev = agent.cfg.use_planner
    agent.cfg.use_planner = use_planner
    rng = np.random.get_state()
    try:
        scores = run_eval_mp(env_params=env_params, agent=agent, num_proc=NUM_PROC,
                             num_game=NUM_GAME, seed=SEED, verbose=False)
    finally:
        np.random.set_state(rng)
        agent.cfg.use_planner = prev
    return float(np.mean(scores))


def main():
    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(RUN_DIR, "cfg.yaml"), "r"))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.num_eval_episode = NUM_GAME
    cfg.mp_eval = 1
    cfg.use_wb = 0
    cfg.original_mppi = 0  # build with the NARROW (stage-1) planner
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/6489b744-685c-4008-8662-9f1ebbbd9c66/scratchpad/goodpolicy_sweep"

    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(CKPT))
    ws.agent = ws.agent.to("cuda")
    agent = ws.agent
    ep = ws.eval_env_params
    p = agent.planner.cfg

    # snapshot the narrow base cfg (the fields we mutate)
    base = dict(horizon=p.horizon, num_samples=p.num_samples, num_elites=p.num_elites,
                iterations=p.iterations, mixture_coef=p.mixture_coef,
                num_base_override=p.num_base_override, temperature=p.temperature,
                momentum=p.momentum, init_std=p.init_std, min_std=p.min_std,
                base_policy_std=p.base_policy_std, use_reward=p.use_reward,
                seed_mean_from_policy=p.seed_mean_from_policy)
    print(f"[sweep] goodpolicy latest.pt | sigma=0.1 | {NUM_GAME} games/variant | TARGET strict")
    print(f"[base narrow] {base}")

    results = []
    # 0) policy-only bar
    results.append(("policy-only", run(agent, ep, use_planner=0)))
    print(f"[RESULT] policy-only : {results[-1][1]:.4f}")

    variants = {
        # narrow default (stage-1) — reconfirm at 300g
        "narrow-default": {},
        # ultra-trust: ALL candidates hug the proposal, tiny spread, seed CEM mean from policy
        "ultra-trust": dict(num_base_override=512, mixture_coef=1.0, init_std=0.1,
                            base_policy_std=0.05, seed_mean_from_policy=1, temperature=2.0),
        # ultra-trust + GREEDY value pick (sharp softmax, few elites)
        "trust-greedyV": dict(num_base_override=512, mixture_coef=1.0, init_std=0.1,
                              base_policy_std=0.05, seed_mean_from_policy=1,
                              temperature=8.0, num_elites=16),
        # short horizon H=1, pure value ranking of policy-seeded first actions
        "H1-pureV": dict(horizon=1, num_base_override=512, mixture_coef=1.0,
                         init_std=0.1, base_policy_std=0.05, seed_mean_from_policy=1,
                         use_reward=0, temperature=8.0, num_elites=16),
        # H=2 trust (dynamics still within the H=3 distilled range)
        "H2-trust": dict(horizon=2, num_base_override=512, mixture_coef=1.0,
                         init_std=0.1, base_policy_std=0.05, seed_mean_from_policy=1,
                         temperature=4.0, num_elites=32),
    }
    for name, ov in variants.items():
        apply_cfg(agent.planner, base, dict(ov))
        s = run(agent, ep, use_planner=1)
        results.append((name, s))
        print(f"[RESULT] {name:14s}: {s:.4f}")

    print("\n===== SWEEP SUMMARY (goodpolicy, sigma=0.1, 300 games) =====")
    bar = results[0][1]
    for name, s in results:
        tag = "" if name == "policy-only" else f"  (vs policy {s - bar:+.4f})"
        print(f"  {name:14s}: {s:.4f}{tag}")


if __name__ == "__main__":
    main()
