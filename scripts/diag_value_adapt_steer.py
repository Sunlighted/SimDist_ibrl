"""Fourth probe -- measure the quantity that is actually causal.

Probes 1-3 refuted every per-candidate story:
  1 ordering   : retrained ranks imagined terminals BETTER vs the expert V
                 (+0.176 vs -0.010) yet scores 0.25 vs 0.80.
  2 softmax    : temperature*std = 0.003-0.006 => elite participation 64.00/64 =>
                 the softmax reweighting is INERT for both heads.
  3 elite compo: stock picks base/proposal candidates only 3.3% of the time (chance
                 28.6%) and its plan lands FARTHER from the policy action (0.474 vs
                 0.357) -- the opposite of "stock works because it trusts the BC prior".

Since the softmax is inert, the executed action is the plain MEAN of the 64 elites'
first actions: a heavily averaged "soft gradient step" in the direction the value
head prefers. Fine per-candidate ranking is washed out by that average; what
survives is the AVERAGE DIRECTION. And because MPPI re-plans every step, closed-loop
performance is the integral of those directions -- so the causal quantity is:

    does this value head's committed action move the REAL system in a better direction?

Measured directly: at every decision point, form each head's committed first action
from the SAME candidate set, execute each in the TRUE target sim from the SAME
sim_state (1 step), and score the outcome with the independent target expert
V_expert(s') (98% strict IBRL) and with the needle->ring distance. The pure policy
proposal action is included as a third arm (§27: policy-only ~0.34-0.46 here).

Run:
  python scripts/diag_value_adapt_steer.py --episodes 8 --stride 6
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist import MainConfig, Workspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def restore(env, sim_state):
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    env.time_step = 0
    if getattr(env, "_success_validator", None) is not None:
        env._success_validator.reset(raw, env._bindings)


def build_agent(base_cfg_dir, ckpt, sigma, dev):
    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(base_cfg_dir, "cfg.yaml")))  # type: ignore
    cfg.simdist.use_planner = 1
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(cfg.simdist.planner)
    if sigma is not None:
        cfg.eval_hand_init_perturb_joint_sigma = sigma
    cfg.use_wb = 0
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(ckpt))
    agent = ws.agent.to(dev)
    agent.eval()
    return agent, ws.eval_env_params, cfg


@torch.no_grad()
def cand_returns(model, s0, actions, discount, final_discount, vmodel):
    N = actions.shape[1]
    z = model.encode(s0.view(1, -1)).expand(N, -1).contiguous()
    rsum = torch.zeros(N, device=z.device)
    g = 1.0
    for t in range(actions.shape[0]):
        a = actions[t]
        rsum = rsum + g * model.predict_reward(z, a).view(-1)
        g = g * discount
        z = model.next(z, a)
    return rsum + vmodel.predict_value(z, target=False).view(-1) * final_discount


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg_dir", default="exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1")
    ap.add_argument("--stock_ckpt", default="exps/mimicgen/threading_simdist_adapt_seed1/latest.pt")
    ap.add_argument("--retrained_ckpt",
                    default="exps/mimicgen/threading_simdist_ablate_val_ws_600k_seed1/latest.pt")
    ap.add_argument("--ref_ckpt",
                    default="exps/mimicgen/threading_ibrl_rlprior_prior_sig01_seed1/model0.pt")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=-1.0)
    ap.add_argument("--arm", default="stock", choices=["stock", "retrained"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigma = None if args.sigma < 0 else args.sigma
    dev = "cuda"
    agentS, ep_params, cfgS = build_agent(args.base_cfg_dir, args.stock_ckpt, sigma, dev)
    agentR, _, _ = build_agent(args.base_cfg_dir, args.retrained_ckpt, sigma, dev)
    model = agentS.model
    pl = agentS.planner
    K, N = pl.cfg.num_elites, pl.cfg.num_samples
    disc = cfgS.discount
    fdisc = pl.final_discount if hasattr(pl, "final_discount") else disc ** pl.cfg.horizon
    heads = {"stock": agentS.model, "retrained": agentR.model}

    main_env = PixelRobosuite(**ep_params)
    probe_env = PixelRobosuite(**ep_params)
    probe_env.reset()   # CRITICAL (§27): applies the domain shift

    import train_rl
    ref_cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.ref_ckpt), "cfg.yaml")))
    ref = train_rl.load_frozen_agent(args.ref_ckpt, main_env, ref_cfg, dev)

    @torch.no_grad()
    def vref(state):
        return float(ref.value({"state": state.view(1, -1).to(dev)}).view(-1)[0])

    def needle_ring(state):
        s = state.view(-1).cpu().numpy()
        return float(np.linalg.norm(s[9:12] - s[23:26]))

    log(f"[setup] elites={K}/{N} | walking the {args.arm} arm | 1-step steering quality "
        f"in the TRUE target sim")

    arms = ("stock", "retrained", "policy")
    acc = {a: {"dv": [], "dd": []} for a in arms}
    n = 0
    arm_agent = agentS if args.arm == "stock" else agentR
    planner = arm_agent.planner
    cap = {}
    orig = planner._estimate_returns

    def wrapped(z0, actions, _o=orig, _c=cap):
        r = _o(z0, actions)
        _c["actions"] = actions
        return r
    planner._estimate_returns = wrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for e in range(args.episodes):
        obs, _ = main_env.reset()
        for t in range(args.max_steps):
            cap.clear()
            with torch.no_grad():
                plan = planner.plan(obs["state"], eval_mode=True, return_full=True)
            if t % args.stride == 0 and "actions" in cap:
                cand = cap["actions"]
                sim_state = main_env.env.sim.get_state().flatten().copy()
                v0, d0 = vref(obs["state"]), needle_ring(obs["state"])
                acts = {}
                for h, m in heads.items():
                    with torch.no_grad():
                        ret = cand_returns(model, obs["state"], cand, disc, fdisc, m)
                        idx = torch.topk(ret, K).indices
                        acts[h] = cand[0, idx].mean(0)
                with torch.no_grad():
                    acts["policy"] = planner._proposal(obs["state"].view(1, -1)).view(-1)
                for a_name, a in acts.items():
                    restore(probe_env, sim_state)
                    o2, _, _, _, _ = probe_env.step(a.detach().cpu())
                    acc[a_name]["dv"].append(vref(o2["state"]) - v0)
                    acc[a_name]["dd"].append(needle_ring(o2["state"]) - d0)
                n += 1
            obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
            if terminal:
                break
        log(f"  ep{e}: points={n}")
    planner._estimate_returns = orig

    def ms(x):
        x = np.asarray(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    log("\n" + "=" * 78)
    log("ONE-STEP STEERING QUALITY in the TRUE target sim (same state, same candidate set)")
    log("=" * 78)
    log(f"{'committed action':<20}{'dV_expert':>18}{'d(needle-ring dist)':>24}")
    for a in arms:
        dv = ms(acc[a]["dv"]); dd = ms(acc[a]["dd"])
        log(f"{a:<20}{dv[0]:>+13.5f} ±{dv[1]:.5f}{dd[0]:>+19.5f} ±{dd[1]:.5f}")
    sv = np.array(acc["stock"]["dv"]); rv = np.array(acc["retrained"]["dv"])
    d = sv - rv
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    log(f"\n  paired dV(stock) - dV(retrained) = {m:+.5f} ± {se:.5f}   (z={m/max(se,1e-12):+.2f})")
    log(f"  stock steers better at {100*np.mean(sv>rv):.1f}% of the {n} decision points")
    log("  (+dV = moves toward higher expert value; -d(dist) = needle approaches the ring)")
    os._exit(0)


if __name__ == "__main__":
    main()
