"""Ablations on a trained dynamics-ENSEMBLE SimDist model (CLAUDE.md §31).

Two questions, ONE frozen model, everything else held fixed (encoder / value /
policy / planner recipe / env / seed) so each run changes exactly one variable:

 1. ATTRIBUTION (--heads): is the +0.135 over plain SimDist due to PLAN-TIME
    ensembling (rolling the 5-head MEAN => lower model error) or to a side effect
    of training with an ensemble (a differently-shaped encoder)? `--heads 1`
    truncates the ensemble to head 0 at eval time, so the SAME weights are used
    with a single-head rollout. If heads=1 still scores ~0.96, the gain lives in
    the encoder/value/policy, NOT in the ensemble average.

 2. PESSIMISM (--disagree_coef): the actual §31 test — subtract
    coef * sum_t gamma^t * (cross-head std of the predicted next latent) from each
    candidate's planning return. coef=0 is the no-pessimism control.

`--mode probe` reports, on REAL target states along a planner rollout, the
per-candidate spread of the plain return vs the discounted disagreement sum, so
the coef sweep can be centred where the penalty is comparable to the value signal
(instead of blind-guessing decades).

Usage:
  python scripts/eval_dynens_ablate.py --mode probe  --ckpt <adapt ckpt>
  python scripts/eval_dynens_ablate.py --mode eval --heads 5 --disagree_coef 0 --num_game 100
"""
import argparse
import os
import sys

import numpy as np
import pyrallis
import torch
import torch.nn as nn

sys.path.insert(0, "/home/Chenyang/ibrl")

from evaluate import run_eval, run_eval_mp          # noqa: E402
from train_simdist import MainConfig, Workspace     # noqa: E402
from rl.simdist_planner import apply_original_recipe  # noqa: E402

DEFAULT_CKPT = ("exps/mimicgen/threading_simdist_dynens_adapt_long_h5_seed1/"
                "ablate_frozen.pt")


def log(*a):
    """stdout may be hijacked by common_utils.Logger elsewhere; stderr is safe."""
    print(*a, file=sys.stderr, flush=True)


def build(ckpt: str, stage1_dir: str, sigma: float, original_mppi: int):
    cfg: MainConfig = pyrallis.load(  # type: ignore
        MainConfig, open(os.path.join(stage1_dir, "cfg.yaml"), "r"))
    cfg.simdist.use_planner = 1
    if original_mppi:
        apply_original_recipe(cfg.simdist.planner)
    if sigma is not None:
        cfg.eval_hand_init_perturb_joint_sigma = sigma
    ws = Workspace(cfg, from_main=False)   # builds envs + agent only (no replay)
    agent = ws.agent
    agent.load_state_dict(torch.load(ckpt))
    agent = agent.to("cuda")
    agent.eval()
    return cfg, ws, agent


def set_heads(agent, n: int) -> int:
    """Truncate the dynamics ensemble to `n` heads AT EVAL TIME (head 0 first).
    The weights are untouched; only how many members `next()`/disagreement use."""
    m = agent.model
    total = m.num_dynamics_heads()
    assert 1 <= n <= total, f"--heads must be in [1,{total}]"
    m.dynamics_extra = nn.ModuleList(list(m.dynamics_extra)[: n - 1])
    if len(m.dynamics_prior) > 0:      # keep priors aligned with the kept heads
        m.dynamics_prior = nn.ModuleList(list(m.dynamics_prior)[:n])
    assert m.num_dynamics_heads() == n
    return total


@torch.no_grad()
def probe(agent, ws, n_states: int = 12):
    """On REAL target states: per-candidate std of the plain planning return vs of
    the discounted disagreement sum. coef ~ std(return)/std(disagree) makes the
    penalty comparable in magnitude to the value signal it competes with."""
    planner = agent.planner
    env = ws.eval_env
    H, disc = planner.cfg.horizon, planner.discounts
    r_sd, d_sd, d_mu = [], [], []
    obs, _ = env.reset()
    for i in range(n_states):
        state = obs["state"].to("cuda").float().view(1, -1)
        z0 = agent.model.encode(state)
        # the planner's own candidate distribution (base proposal + noise)
        nb = planner._num_base()
        base = planner._policy_rollout(z0, nb)
        if planner.cfg.base_policy_std > 0:
            base = base + torch.randn_like(base) * planner.cfg.base_policy_std
        nr = planner.cfg.num_samples - nb
        mean = torch.zeros(H, agent.action_dim, device="cuda")
        rand = (mean.unsqueeze(1)
                + planner.cfg.init_std * torch.randn(H, nr, agent.action_dim, device="cuda")
                ).clamp(-1, 1)
        acts = torch.cat([rand, base.clamp(-1, 1)], dim=1)      # [H, N, act]
        # plain return (coef=0) and the discounted disagreement sum, same rollout
        prev = planner.cfg.dynamics_disagree_coef
        planner.cfg.dynamics_disagree_coef = 0.0
        ret = planner._estimate_returns(z0, acts)
        planner.cfg.dynamics_disagree_coef = prev
        z = z0.expand(acts.shape[1], -1).clone()
        dl = []
        for t in range(H):
            dl.append(agent.model.dynamics_disagreement(z, acts[t]))
            z = agent.model.next(z, acts[t])
        dsum = (torch.stack(dl, 0) * disc).sum(0)               # [N]
        r_sd.append(ret.std().item())
        d_sd.append(dsum.std().item())
        d_mu.append(dsum.mean().item())
        # advance the env with the planner's action so states stay on-policy
        a = planner.plan(obs["state"].to("cuda").float(), eval_mode=True)
        obs, _, terminal, _, _ = env.step(a.cpu())
        if terminal:
            obs, _ = env.reset()
    r_sd, d_sd, d_mu = map(np.array, (r_sd, d_sd, d_mu))
    log(f"[probe] over {n_states} on-policy target states, {planner.cfg.num_samples} candidates each")
    log(f"  std_a(return)          = {r_sd.mean():.5f}  (the signal argmax ranks by)")
    log(f"  std_a(disagree_sum)    = {d_sd.mean():.5f}  (the penalty's spread ACROSS candidates)")
    log(f"  mean(disagree_sum)     = {d_mu.mean():.5f}  (constant part => argmax-immune)")
    ratio = r_sd.mean() / max(d_sd.mean(), 1e-9)
    log(f"  => coef making the penalty comparable to the value signal ~ {ratio:.1f}")
    log(f"  => suggested sweep: {ratio/10:.2f}, {ratio/3:.2f}, {ratio:.1f}, {ratio*3:.1f}")
    # the constant/spread split decides whether pessimism can do ANYTHING (§27 eps_bar)
    log(f"  spread/mean of disagreement = {d_sd.mean()/max(d_mu.mean(),1e-9):.3f} "
        f"(if <<1, the penalty is ~a constant shift => argmax-invariant => NO-OP)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--stage1_dir",
                    default="exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1")
    ap.add_argument("--mode", default="eval", choices=["eval", "probe"])
    ap.add_argument("--heads", type=int, default=5)
    ap.add_argument("--disagree_coef", type=float, default=0.0)
    ap.add_argument("--num_game", type=int, default=100)
    ap.add_argument("--num_proc", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--original_mppi", type=int, default=1)
    ap.add_argument("--policy_only", type=int, default=0)
    ap.add_argument("--mp", type=int, default=1)
    # dump per-episode scores so arms sharing a seed (=> the SAME initial states)
    # can be compared PAIRED, which is far more powerful than comparing two
    # independent means when the expected effect is small (<0.03).
    ap.add_argument("--save_scores", default="")
    args = ap.parse_args()

    cfg, ws, agent = build(args.ckpt, args.stage1_dir, args.sigma, args.original_mppi)
    total = set_heads(agent, args.heads)
    agent.planner.cfg.dynamics_disagree_coef = args.disagree_coef
    if args.policy_only:
        agent.cfg.use_planner = 0
    log(f"[cfg] ckpt={args.ckpt}")
    log(f"[cfg] heads={args.heads}/{total}  disagree_coef={args.disagree_coef}  "
        f"sigma={args.sigma}  original_mppi={args.original_mppi}  "
        f"planner={'OFF (policy only)' if args.policy_only else 'MPPI'}")

    if args.mode == "probe":
        probe(agent, ws)
        os._exit(0)

    if args.mp:
        scores = run_eval_mp(env_params=ws.eval_env_params, agent=agent,
                             num_proc=args.num_proc, num_game=args.num_game,
                             seed=args.seed, verbose=False)
    else:
        scores = run_eval(env_params=ws.eval_env_params, agent=agent,
                          num_game=args.num_game, seed=args.seed,
                          record_dir=None, verbose=False)
    s = float(np.mean(scores))
    se = float(np.std(scores) / np.sqrt(len(scores)))
    log(f"\n>>> RESULT heads={args.heads} coef={args.disagree_coef} "
        f"n={args.num_game}: score={s:.3f} +/- {se:.3f}")
    if args.save_scores:
        np.savez(args.save_scores, scores=np.asarray(scores, dtype=np.float32),
                 heads=args.heads, coef=args.disagree_coef, seed=args.seed,
                 num_game=args.num_game)
        log(f"[saved] per-episode scores -> {args.save_scores}")
    os._exit(0)


if __name__ == "__main__":
    main()
