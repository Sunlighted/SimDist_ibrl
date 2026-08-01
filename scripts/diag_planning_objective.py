"""Is the PLANNING OBJECTIVE the bottleneck? Measure J (what MPPI maximizes)
against J* (what actually happens), on the candidate sets MPPI really searches.

At probe steps along a real TARGET MPPI rollout we freeze the candidate set of
the LAST CEM iteration together with the planner's score
    J(a) = sum_t gamma^t r_hat(z_t) + gamma^H V_hat(z_H)          [imagined]
and then, in a SEPARATE target sim reset to the same sim_state, actually execute
each candidate's H actions and continue with the policy head to the end of the
episode, giving
    J*(a) = gamma^k  if the episode succeeds k steps later, else 0    [real]
Both envs are the deployment TARGET env (strict success), so J* is the true
discounted return of that candidate under the policy that follows it.

Reported per phase (free-space / contact):
  spearman(J, J*)            -- does the objective rank candidates at all?
  exec percentile            -- where the action MPPI actually executes lands in
                                the candidates' REAL return distribution
                                (50% = no better than picking at random)
  topJ vs all vs bestJ*      -- optimizer's curse check

Verdict:
  spearman ~ 0  => the objective carries no ranking signal => IT is the bottleneck
  spearman high but exec percentile low => model-error exploitation (optimizer's curse)
  spearman high AND exec percentile high => the planner picks fine; look elsewhere
"""
import os
import sys
import argparse
import json

import numpy as np
import torch
import pyrallis
from scipy.stats import spearmanr

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402

ERR = sys.stderr  # AdaptWorkspace hijacks stdout into <save_dir>/train.log

NEEDLE = slice(9, 12)
RING = slice(23, 26)
CONTACT_THRESH = 0.15  # needle-ring dist below this = contact/insert phase


def log(*a):
    print(*a, file=ERR, flush=True)


@torch.no_grad()
def plan_with_candidates(planner, state):
    """Replica of MppiPlanner.plan that ALSO returns the last iteration's
    candidate action sequences and their planner scores J."""
    cfg = planner.cfg
    H = cfg.horizon
    z0 = planner.model.encode(state.view(1, -1))

    num_base = planner._num_base()
    base_actions = planner._policy_rollout(z0, num_base) if num_base > 0 else None
    if base_actions is not None and cfg.base_policy_std > 0:
        base_actions = base_actions + torch.randn_like(base_actions) * cfg.base_policy_std

    mean = torch.zeros(H, planner.action_dim, device=planner.device)
    std = torch.full((H, planner.action_dim), cfg.init_std, device=planner.device)
    num_rand = cfg.num_samples - num_base

    actions = returns = None
    for _ in range(cfg.iterations):
        noise = torch.randn(H, num_rand, planner.action_dim, device=planner.device)
        rand_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
        actions = (torch.cat([rand_actions, base_actions.clamp(-1, 1)], dim=1)
                   if base_actions is not None else rand_actions)
        returns = planner._estimate_returns(z0, actions)          # [N]
        elite_val, elite_idx = torch.topk(returns, cfg.num_elites)
        elite_acts = actions[:, elite_idx]
        score = torch.exp(cfg.temperature * (elite_val - elite_val.max()))
        score = score / (score.sum() + 1e-9)
        new_mean = (score.view(1, -1, 1) * elite_acts).sum(1)
        new_std = torch.sqrt((score.view(1, -1, 1) * (elite_acts - new_mean.unsqueeze(1)) ** 2).sum(1))
        mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
        std = new_std.clamp(min=cfg.min_std)

    # J of the EXECUTED sequence (the CEM mean) under the same objective
    j_exec = planner._estimate_returns(z0, mean.unsqueeze(1)).item()
    return actions, returns, mean, j_exec


def restore(env, sim_state):
    """Hard-reset a PixelRobosuite's sim to sim_state (fresh episode bookkeeping)."""
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    env.time_step = 0
    if getattr(env, "_success_validator", None) is not None:
        env._success_validator.reset(raw, env._bindings)


@torch.no_grad()
def real_return(env, sim_state, action_seq, agent, max_steps, discount):
    """Execute action_seq (H actions) from sim_state in the REAL target sim, then
    continue with the policy head. Return gamma^k on success (k = step index of
    the success), else 0."""
    restore(env, sim_state)
    state = None
    k = 0
    for a in action_seq:                       # the candidate's H actions
        obs_d, reward, terminal, _, _ = env.step(a.cpu())
        k += 1
        if reward > 0:
            return discount ** k
        if terminal or k >= max_steps:
            return 0.0
        state = obs_d["state"]

    while k < max_steps:                       # then follow the policy head
        a = agent.model.policy_action(state.view(1, -1)).view(-1).clamp(-1, 1)
        obs_d, reward, terminal, _, _ = env.step(a.cpu())
        k += 1
        if reward > 0:
            return discount ** k
        if terminal:
            return 0.0
        state = obs_d["state"]
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--probe_every", type=int, default=25)
    ap.add_argument("--contact_probe_every", type=int, default=8)
    ap.add_argument("--max_probes_per_ep", type=int, default=4)
    ap.add_argument("--num_cand", type=int, default=32, help="candidates evaluated per probe")
    ap.add_argument("--max_rollout", type=int, default=250, help="cap on real steps per candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/planning_objective.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0
    cfg.mp_eval = 0
    cfg.save_dir = os.path.dirname(args.out) + "/po_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt)))
    agent = ws.agent.to("cuda")
    agent.cfg.use_planner = 1
    planner = agent.planner
    discount = planner.cfg.discount

    ep_params = dict(ws.eval_env_params)
    main_env = PixelRobosuite(**ep_params)   # rollout env (MPPI, untouched by probes)
    probe_env = PixelRobosuite(**ep_params)  # counterfactual env
    # CRITICAL: the domain shift is applied inside reset() (robosuite's hard_reset
    # rebuilds the sim, so the mutations must be re-applied every reset). A freshly
    # constructed env is still NOMINAL/SOURCE physics, and `restore()` only writes
    # qpos/qvel -- it never touches the model. Without this reset the counterfactual
    # rollouts silently run in the WRONG domain.
    probe_env.reset()
    log(f"[setup] ckpt={args.run_dir}/{args.ckpt}  planner: H={planner.cfg.horizon} "
        f"N={planner.cfg.num_samples} iters={planner.cfg.iterations} elites={planner.cfg.num_elites}")
    log(f"[setup] episodes={args.episodes} probe_every={args.probe_every} "
        f"cand/probe={args.num_cand} max_rollout={args.max_rollout}")

    recs = []
    for ep in range(args.episodes):
        obs, _ = main_env.reset()
        n_probe = 0
        last_probe = -999
        for t in range(cfg_episode_len(ws)):
            state = obs["state"]
            nrd_now = float(np.linalg.norm(
                state[NEEDLE].cpu().numpy() - state[RING].cpu().numpy()))
            gap = args.contact_probe_every if nrd_now < CONTACT_THRESH else args.probe_every
            do_probe = (t - last_probe >= gap) and (n_probe < args.max_probes_per_ep)

            if do_probe:
                cands, J, mean_seq, j_exec = plan_with_candidates(planner, state)
                sim_state = main_env.env.sim.get_state().flatten().copy()
                Jn = J.cpu().numpy()

                # pick num_cand candidates spanning the J range: the top 8 by J
                # (what the planner actually leans on) + a stratified sample of
                # the rest (so the correlation is measured over real spread)
                order = np.argsort(-Jn)
                top = order[:8]
                rest = order[8:]
                stride = max(len(rest) // (args.num_cand - 8), 1)
                sel = np.concatenate([top, rest[::stride][:args.num_cand - 8]])

                budget = min(args.max_rollout, cfg_episode_len(ws) - t)
                jstar = []
                for i in sel:
                    seq = cands[:, i]  # [H, act]
                    jstar.append(real_return(probe_env, sim_state, seq, agent, budget, discount))
                jstar = np.array(jstar)
                j_exec_star = real_return(probe_env, sim_state, mean_seq, agent, budget, discount)

                nrd = float(np.linalg.norm(
                    state[NEEDLE].cpu().numpy() - state[RING].cpu().numpy()))
                sp = spearmanr(Jn[sel], jstar).correlation if jstar.std() > 0 else np.nan
                pct = float((jstar < j_exec_star).mean() + 0.5 * (jstar == j_exec_star).mean())
                recs.append(dict(
                    ep=ep, t=t, needle_ring=nrd, contact=bool(nrd < CONTACT_THRESH),
                    spearman=None if np.isnan(sp) else float(sp),
                    J=Jn[sel].tolist(), Jstar=jstar.tolist(),
                    j_exec=float(j_exec), j_exec_star=float(j_exec_star),
                    exec_pct=pct,
                    jstar_top4=float(jstar[np.argsort(-Jn[sel])[:4]].mean()),
                    jstar_mean=float(jstar.mean()), jstar_best=float(jstar.max()),
                    jstar_nonzero=int((jstar > 0).sum()), n_cand=len(sel),
                ))
                n_probe += 1
                last_probe = t
                log(f"  ep{ep} t{t:3d} {'CONTACT' if nrd < CONTACT_THRESH else 'free   '} "
                    f"nrd={nrd:.3f} sp={sp:+.3f} exec_pct={pct:.2f} "
                    f"J*[exec]={j_exec_star:.3f} top4={recs[-1]['jstar_top4']:.3f} "
                    f"mean={jstar.mean():.3f} best={jstar.max():.3f} nz={int((jstar>0).sum())}/{len(sel)}")

                # continue the real episode with what the planner actually chose
                restore(main_env, sim_state)
                action = mean_seq[0].cpu()
            else:
                action = agent.act(obs, eval_mode=True).cpu()

            obs, reward, terminal, _, _ = main_env.step(action)
            if terminal:
                break
        log(f"[ep {ep}] done ({len(recs)} probes so far)")
        with open(args.out, "w") as f:
            json.dump(recs, f)

    summarize(recs)


def cfg_episode_len(ws):
    return int(ws.eval_env_params.get("episode_length", 400))


def summarize(recs):
    log("\n===== PLANNING-OBJECTIVE DIAGNOSIS =====")
    for name, sub in [("ALL", recs),
                      ("free-space", [r for r in recs if not r["contact"]]),
                      ("CONTACT", [r for r in recs if r["contact"]])]:
        if not sub:
            continue
        sp = [r["spearman"] for r in sub if r["spearman"] is not None]
        log(f"\n--- {name}  (n_probe={len(sub)}, {len(sp)} with non-degenerate J*)")
        if sp:
            log(f"  spearman(J, J*)   : mean {np.mean(sp):+.3f}  median {np.median(sp):+.3f} "
                f" [>0 in {100*np.mean(np.array(sp) > 0):.0f}% of probes]")
        log(f"  exec percentile   : {100*np.mean([r['exec_pct'] for r in sub]):.1f}%  (50% = random pick)")
        log(f"  J* of top-4 by J  : {np.mean([r['jstar_top4'] for r in sub]):.3f}")
        log(f"  J* of all cands   : {np.mean([r['jstar_mean'] for r in sub]):.3f}")
        log(f"  J* of BEST cand   : {np.mean([r['jstar_best'] for r in sub]):.3f}   <- headroom")
        log(f"  J* of EXECUTED    : {np.mean([r['j_exec_star'] for r in sub]):.3f}")


if __name__ == "__main__":
    main()
