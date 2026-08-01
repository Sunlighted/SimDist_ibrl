"""Third probe. The first two REFUTED the obvious stories:
  1) ordering  (diag_value_adapt_ranking.py): the retrained head ranks imagined
     latents BETTER vs the target expert V (+0.176 vs -0.010) yet scores 0.25 vs 0.80.
  2) softmax scale (diag_value_adapt_scale.py): temperature*std = 0.003-0.006 for
     BOTH heads => elite participation ratio 64.00/64 => the softmax reweighting
     is INERT; the committed plan is the plain average of the 64 elites.

=> the value's ONLY channel is WHICH 64 of the 350 candidates topk selects
   (V carries 98% of the return spread; the reward term is 4x smaller).

The candidate pool is structured (rl/simdist_planner.py:253-261):
    actions = cat([ rand_actions (num_rand, from the CEM gaussian) ,
                    base_actions (num_base, NOISED POLICY PROPOSAL) ])
so base candidates occupy the LAST num_base indices. §27 established the planner
CANNOT plan from scratch (num_base=0 => 0.000 success in both domains): it only
refines a proposal's neighbourhood. So the operative question is whether the two
value heads DISAGREE ON HOW MUCH TO TRUST THE PROPOSAL:

  stock      elites mostly base  => committed plan stays near the BC policy (safe)
  retrained  elites mostly rand  => plan drifts into random-action space, where the
                                    learned dynamics is unreliable => collapse

Measures on the SAME on-policy candidate sets (no real sim => fast):
  * base fraction among each head's top-64 elites (chance level = num_base/num_samples)
  * ||mean(elites) - policy proposal|| : how far the committed plan lands from the
    pure policy action (action-space drift), per head
  * rank of the base block: mean percentile of base candidates in each head's return

Run:
  python scripts/diag_value_adapt_elites.py --episodes 6 --stride 6
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
def terms(model, s0, actions, discount, final_discount, vmodel):
    """reward sum + gamma^H V (with vmodel's value head) for [H,N,act] candidates."""
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
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=-1.0)
    ap.add_argument("--arm", default="stock", choices=["stock", "retrained"],
                    help="whose on-policy rollout to walk")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigma = None if args.sigma < 0 else args.sigma
    dev = "cuda"
    agentS, ep_params, cfgS = build_agent(args.base_cfg_dir, args.stock_ckpt, sigma, dev)
    agentR, _, _ = build_agent(args.base_cfg_dir, args.retrained_ckpt, sigma, dev)
    model = agentS.model
    pl = agentS.planner
    H, K, N = pl.cfg.horizon, pl.cfg.num_elites, pl.cfg.num_samples
    num_base = pl._num_base()
    num_rand = N - num_base
    disc = cfgS.discount
    fdisc = pl.final_discount if hasattr(pl, "final_discount") else disc ** H
    heads = {"stock": agentS.model, "retrained": agentR.model}
    chance = num_base / N

    main_env = PixelRobosuite(**ep_params)
    log(f"[setup] H={H} num_samples={N} (rand={num_rand} + base={num_base}) elites={K} "
        f"| chance base-fraction = {chance:.3f} | walking the {args.arm} arm")

    acc = {h: {k: [] for k in ("basefrac", "drift", "basepct")} for h in heads}
    arm = agentS if args.arm == "stock" else agentR
    planner = arm.planner
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
                cand = cap["actions"]                     # [H,N,act], base = last num_base
                with torch.no_grad():
                    a_pi = planner._proposal(obs["state"].view(1, -1)).view(-1)  # [act]
                for h, m in heads.items():
                    with torch.no_grad():
                        ret = terms(model, obs["state"], cand, disc, fdisc, m)
                    idx = torch.topk(ret, K).indices
                    is_base = (idx >= num_rand).float()
                    elite_mean0 = cand[0, idx].mean(0)              # committed first action
                    order = torch.argsort(torch.argsort(ret)).float() / (N - 1)  # percentile
                    acc[h]["basefrac"].append(float(is_base.mean()))
                    acc[h]["drift"].append(float((elite_mean0 - a_pi).norm()))
                    acc[h]["basepct"].append(float(order[num_rand:].mean()))
            obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
            if terminal:
                break
        log(f"  ep{e}: points={len(acc['stock']['basefrac'])}")
    planner._estimate_returns = orig

    def ms(x):
        x = np.asarray(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    log("\n" + "=" * 80)
    log("WHICH candidates does each value head's topk select?  "
        f"(chance base-fraction {chance:.3f})")
    log("=" * 80)
    log(f"{'value head':<12}{'base frac in top-64':>22}{'base mean percentile':>23}"
        f"{'||plan - policy action||':>26}")
    for h in heads:
        bf = ms(acc[h]["basefrac"]); bp = ms(acc[h]["basepct"]); dr = ms(acc[h]["drift"])
        log(f"{h:<12}{bf[0]:>17.3f} ±{bf[1]:.3f}{bp[0]:>18.3f} ±{bp[1]:.3f}"
            f"{dr[0]:>21.4f} ±{dr[1]:.4f}")
    log(f"\n  n = {len(acc['stock']['basefrac'])} decision points ({args.arm} on-policy)")
    log("  base frac >> chance  => that head TRUSTS the policy proposal (stays in its"
        " neighbourhood)")
    log("  base frac ~  chance  => the head is indifferent; the plan is dominated by"
        " RANDOM candidates,")
    log("                          i.e. it drifts where the learned dynamics is"
        " unconstrained (§27: no-proposal MPPI = 0.000)")
    os._exit(0)


if __name__ == "__main__":
    main()
