"""Follow-up to scripts/diag_value_adapt_ranking.py (which REFUTED the ordering
hypothesis: the retrained value head ranks imagined latents BETTER than stock,
+0.176 vs -0.010, yet scores 0.25 vs 0.80).

Remaining channel that rank-correlation is structurally blind to: the CEM
softmax CONSUMES the value's absolute cross-candidate SPREAD.

  rl/simdist_planner.py:268
      score    = exp(temperature * (elite_val - elite_val.max()))
      new_mean = (score * elite_acts).sum(1)

`topk` is rank-only (scale-free), but the softmax that FORMS the committed plan
is not: it sees `temperature * spread`. Two concrete failure modes invisible to
Spearman:
  * spread COLLAPSES  -> temperature*spread ~ 0 -> softmax ~ uniform over the 64
    elites -> the plan is the AVERAGE of 64 candidates (mush), and the sparse
    reward term (dead noise, §17) can take over the ranking entirely.
  * spread EXPLODES   -> near-argmax on a noisy signal (winner's curse).

Measures, on the SAME on-policy candidate sets (no real sim needed => fast):
  std_i(gamma^H * V_h(z_H))     per head h        <- what the softmax sees
  std_i(sum_t gamma^t r_hat_t)                    <- the competing reward term
  effective softmax entropy / participation ratio over the elites, per head
  the value term's share of the total return spread

Run:
  python scripts/diag_value_adapt_scale.py --episodes 6 --stride 6
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
def rollout_terms(model, s0, actions, discount, final_discount):
    """actions [H,N,act] -> (reward_sum [N], terminal latents z_H [N,z]) with THIS
    model's dynamics + reward head (identical across the two ckpts)."""
    N = actions.shape[1]
    z = model.encode(s0.view(1, -1)).expand(N, -1).contiguous()
    rsum = torch.zeros(N, device=z.device)
    g = 1.0
    for t in range(actions.shape[0]):
        a = actions[t]
        rsum = rsum + g * model.predict_reward(z, a).view(-1)
        g = g * discount
        z = model.next(z, a)
    return rsum, z


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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigma = None if args.sigma < 0 else args.sigma
    dev = "cuda"
    agentS, ep_params, cfgS = build_agent(args.base_cfg_dir, args.stock_ckpt, sigma, dev)
    agentR, _, _ = build_agent(args.base_cfg_dir, args.retrained_ckpt, sigma, dev)
    model = agentS.model
    pl = agentS.planner
    H, temp, K = pl.cfg.horizon, pl.cfg.temperature, pl.cfg.num_elites
    disc = agentS.discount if hasattr(agentS, "discount") else cfgS.discount
    fdisc = pl.final_discount if hasattr(pl, "final_discount") else disc ** H
    heads = {"stock": agentS.model, "retrained": agentR.model}

    main_env = PixelRobosuite(**ep_params)
    log(f"[setup] H={H} temperature={temp} num_elites={K} final_discount={fdisc:.4f}")

    acc = {h: {k: [] for k in ("vstd", "vmean", "vrange", "part", "share")} for h in heads}
    acc["rstd"] = []
    for arm_name, arm in (("stock", agentS),):     # stock's on-policy distribution
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
                    cand = cap["actions"]                       # [H,N,act] final CEM iter
                    with torch.no_grad():
                        rsum, zH = rollout_terms(model, obs["state"], cand, disc, fdisc)
                    acc["rstd"].append(float(rsum.std()))
                    for h, m in heads.items():
                        with torch.no_grad():
                            v = m.predict_value(zH, target=False).view(-1) * fdisc
                        ret = rsum + v
                        # elite softmax participation ratio: 1 => one candidate
                        # dominates (argmax), K => uniform average (mush)
                        ev = torch.topk(ret, K).values
                        w = torch.exp(temp * (ev - ev.max()))
                        w = w / (w.sum() + 1e-9)
                        acc[h]["vstd"].append(float(v.std()))
                        acc[h]["vmean"].append(float(v.mean()))
                        acc[h]["vrange"].append(float(v.max() - v.min()))
                        acc[h]["part"].append(float(1.0 / (w ** 2).sum()))
                        acc[h]["share"].append(float(v.std() / max(float(ret.std()), 1e-9)))
                obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
                if terminal:
                    break
            log(f"  ep{e}: points={len(acc['rstd'])}")
        planner._estimate_returns = orig

    def ms(x):
        x = np.asarray(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    log("\n" + "=" * 78)
    log("cross-candidate spread of the terms the CEM softmax consumes "
        f"(temperature={temp}, elites={K})")
    log("=" * 78)
    rm, rs = ms(acc["rstd"])
    log(f"{'':<12}{'std_i(gamma^H V)':>18}{'range_i(V)':>12}{'mean_i(V)':>12}"
        f"{'temp*std':>10}{'V share of ret':>16}{'elite particip.':>17}")
    for h in heads:
        vs = ms(acc[h]["vstd"]); vr = ms(acc[h]["vrange"]); vm = ms(acc[h]["vmean"])
        pa = ms(acc[h]["part"]); sh = ms(acc[h]["share"])
        log(f"{h:<12}{vs[0]:>18.5f}{vr[0]:>12.5f}{vm[0]:>12.4f}"
            f"{temp*vs[0]:>10.5f}{sh[0]:>15.2f}{pa[0]:>16.2f}  (±{pa[1]:.2f})")
    log(f"{'reward term':<12}{rm:>18.5f}   (std_i of sum_t gamma^t r_hat; §17 calls it dead noise)")
    log(f"\n  ratio std_i(V): retrained / stock = "
        f"{ms(acc['retrained']['vstd'])[0]/max(ms(acc['stock']['vstd'])[0],1e-12):.3f}")
    log(f"  n = {len(acc['rstd'])} decision points (stock on-policy)")
    log("\n  participation ratio: 1 = softmax picks ONE elite (argmax), "
        f"{K} = uniform average of all {K} elites (mush).")
    os._exit(0)


if __name__ == "__main__":
    main()
