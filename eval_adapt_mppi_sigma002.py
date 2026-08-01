"""Standalone MPPI eval of an adapted SimDist ckpt at a chosen hand-init sigma.

Reuses AdaptWorkspace's exact env/agent/planner setup (forces use_planner=1 +
ORIGINAL SimDist MPPI recipe), overrides the eval sigma, loads the ADAPTED
ckpt, and runs num_eval_episode MPPI games on TARGET.
"""
import os
import pyrallis
import torch

from train_simdist_adapt import AdaptConfig, AdaptWorkspace


def main():
    run_dir = "exps/mimicgen/threading_simdist_adapt_h5_seed1"
    ckpt = os.path.join(run_dir, "latest.pt")

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(run_dir, "cfg.yaml"), "r"))  # type: ignore
    # eval-only overrides
    cfg.eval_hand_init_perturb_joint_sigma = 0.02
    cfg.num_eval_episode = 500
    cfg.mp_eval = 1
    cfg.use_wb = 0
    # NARROW search: ibrl-tuned "trust-the-proposal" planner (mixture 0.8 /
    # init_std 0.15), i.e. keep the stage-1 planner defaults instead of the
    # broad ORIGINAL SimDist recipe.
    cfg.original_mppi = 0
    # don't clobber the real run's dir / cfg.yaml
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/6489b744-685c-4008-8662-9f1ebbbd9c66/scratchpad/adapt_eval_sigma002_narrow"

    ws = AdaptWorkspace(cfg)
    print(f"loading ADAPTED ckpt over stage-1: {ckpt}")
    ws.agent.load_state_dict(torch.load(ckpt))
    ws.agent = ws.agent.to("cuda")

    score = ws.eval(seed=0, use_planner=1)
    print(f"\n[RESULT] MPPI eval | sigma=0.02 | {cfg.num_eval_episode} games "
          f"| score = {score:.4f}")


if __name__ == "__main__":
    main()
