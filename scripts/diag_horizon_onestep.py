"""Does the distillation horizon cost one-step accuracy?

Dyna generates rollout_length=1 transitions, but every SimDist stage-1 here was
distilled with a multi-step latent-consistency loss (H=5 for the arm we use).
MBPO's model is 1-step BY CONSTRUCTION; ours is trained to be stable under five
applications, which is a different objective and may trade away one-step accuracy.

Rather than distil an H=1 model to find out (hours), compare the H=3/5/7/10
checkpoints that already exist, all distilled from the same `diverse_v2` source.

Protocol
--------
ONE set of TARGET transitions, collected on-policy (rule 5: never judge a model on
a distribution it does not produce), is scored by every model, so the states and
actions are identical across models and only the model varies.

Latent spaces are NOT comparable across checkpoints (different encoders, different
scales), so the headline is scale-free:

    rel = ||next(z,a) - encode(s')||  /  ||encode(s') - encode(s)||

i.e. one-step error measured against the one-step motion it has to explain.
rel >= 1 means the model is no better than predicting "nothing moved".
Reported as sup and Var_s[mean] per CLAUDE.md rule 5, not MSE/R^2.
"""
import argparse
import numpy as np
import pyrallis
import torch

import train_simdist
from common_utils import ibrl_utils as utils

MODELS = [
    ("h3",  "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"),
    ("h5",  "exps/mimicgen/threading_simdist_diversev2_h5_seed1"),
    ("h7",  "exps/mimicgen/threading_simdist_diversev2_h7_seed1"),
    ("h10", "exps/mimicgen/threading_simdist_diversev2_h10_seed1"),
    ("h5_dynens", "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1"),
]


def load(d):
    cfg = pyrallis.load(train_simdist.MainConfig, open(f"{d}/cfg.yaml"))
    ws = train_simdist.Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(f"{d}/latest.pt"))
    return ws.agent.to("cuda"), ws.eval_env_params


def collect(agent, env_params, n, seed, noise):
    """Target on-policy (s, a, s'). One collector for all models."""
    from env.robosuite_wrapper import PixelRobosuite
    ep = dict(env_params)
    ep["hand_init_perturb_joint_sigma"] = 0.1
    ep["device"] = "cuda"
    env = PixelRobosuite(**ep)
    agent.cfg.use_planner = 0                      # the policy head, as Dyna collects
    S, A, S2 = [], [], []
    np.random.seed(seed)
    obs, _ = env.reset()
    with utils.eval_mode(agent):
        while len(S) < n:
            s = obs["state"].clone()
            a = agent.act(obs, eval_mode=False, stddev=noise, cpu=True)
            obs, _, term, _, _ = env.step(a)
            S.append(s); A.append(a.float()); S2.append(obs["state"].clone())
            if term:
                obs, _ = env.reset()
    return (torch.stack(S).cuda().float(), torch.stack(A).cuda().float(),
            torch.stack(S2).cuda().float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.1, help="collect_std, as Dyna now uses")
    args = ap.parse_args()

    ref, ep = load(MODELS[1][1])                   # h5 collects; every model is scored on it
    S, A, S2 = collect(ref, ep, args.n, args.seed, args.noise)
    print(f"[data] {len(S)} TARGET on-policy transitions (collector = h5 policy, "
          f"stddev={args.noise})\n")
    del ref
    torch.cuda.empty_cache()

    print(f"{'model':11s} {'rel_mean':>9s} {'rel_sup':>8s} {'Var_s[rel]':>11s} "
          f"{'||err||':>9s} {'||step||':>9s}")
    print("-" * 62)
    for tag, d in MODELS:
        agent, _ = load(d)
        m = agent.model
        with torch.no_grad():
            z, z2t = m.encode(S), m.encode(S2)
            z2p = m.next(z, A)
            err = (z2p - z2t).norm(dim=-1)
            step = (z2t - z).norm(dim=-1).clamp_min(1e-8)
            rel = err / step
        print(f"{tag:11s} {rel.mean():9.4f} {rel.max():8.3f} {rel.var():11.4f} "
              f"{err.mean():9.4f} {step.mean():9.4f}")
        del agent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
