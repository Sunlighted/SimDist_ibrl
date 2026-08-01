"""Sixth probe -- exploit the STRUCTURAL constraint the earlier probes ignored.

Established facts (probes 2-3):
  * the CEM softmax is INERT (temperature*value_spread = 0.003-0.006 => elite
    participation ratio 64.00/64), so the committed action is the plain MEAN of the
    top-64 candidates' first actions;
  * the reward head is shared by both ckpts and carries only ~2% of the return
    spread (V carries 98%).
=> the ONLY channel by which the two value heads can differ in behaviour is WHICH
   64 of the 350 candidates topk selects. Any real mechanism must be a property of
   the SELECTED SET -- not of ranking accuracy (probes 1,4,5 measured accuracy and
   found retrained equal-or-better, yet it scores 0.25 vs 0.80).

Untested degree of freedom: the selected set's GEOMETRY IN ACTION SPACE. Averaging
64 first-actions cancels whatever disagrees. If retrained's high-value set is less
COHERENT in action space, the mean is damped in MAGNITUDE while keeping direction:

  H: ||mean_{i in top64} a_i^(0)||  is systematically SMALLER for retrained
     (and/or the elite set's action dispersion is larger)

which predicts exactly the observed signature: same one-step direction (dV z=+0.09),
30-40% slower needle->ring closing, 345-vs-186-step episodes, timeouts, and every
static value metric looking fine.

Reported per head, on the SAME on-policy candidate sets:
  ||centroid||         magnitude of the committed first action (the damping)
  dispersion           mean ||a_i - centroid|| over the elites (incoherence)
  coherence            ||centroid|| / mean||a_i||  in [0,1]; 1 = all elites agree
  cos to stock's       direction agreement between the two heads' committed actions
  per-dim |centroid|   which action dims get damped (last dim = gripper)

Run:
  python scripts/diag_elite_action_geometry.py --episodes 10 --stride 4 --arm stock
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


def build(base_cfg_dir, ckpt, sigma, dev):
    cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(base_cfg_dir, "cfg.yaml")))  # type: ignore
    cfg.simdist.use_planner = 1
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(cfg.simdist.planner)
    if sigma is not None:
        cfg.eval_hand_init_perturb_joint_sigma = sigma
    cfg.use_wb = 0
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(ckpt))
    return ws.agent.to(dev).eval(), ws.eval_env_params, cfg


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
    v = vmodel.predict_value(z, target=False).view(-1) * final_discount
    return rsum + v, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg_dir", default="exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1")
    ap.add_argument("--stock_ckpt", default="exps/mimicgen/threading_simdist_adapt_seed1/latest.pt")
    ap.add_argument("--retrained_ckpt",
                    default="exps/mimicgen/threading_simdist_ablate_val_ws_600k_seed1/latest.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=-1.0)
    ap.add_argument("--arm", default="stock", choices=["stock", "retrained"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigma = None if args.sigma < 0 else args.sigma
    dev = "cuda"
    aS, ep_params, cfg = build(args.base_cfg_dir, args.stock_ckpt, sigma, dev)
    aR, _, _ = build(args.base_cfg_dir, args.retrained_ckpt, sigma, dev)
    model, pl = aS.model, aS.planner
    K, N = pl.cfg.num_elites, pl.cfg.num_samples
    disc = cfg.discount
    fdisc = pl.final_discount if hasattr(pl, "final_discount") else disc ** pl.cfg.horizon
    heads = {"stock": aS.model, "retrained": aR.model}
    main_env = PixelRobosuite(**ep_params)
    log(f"[setup] elites={K}/{N} | arm={args.arm} | measuring elite-set action geometry")

    acc = {h: {k: [] for k in ("cnorm", "disp", "coh", "amean", "cdim")} for h in heads}
    cos_sr = []
    arm = aS if args.arm == "stock" else aR
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
    n = 0
    for e in range(args.episodes):
        obs, _ = main_env.reset()
        for t in range(args.max_steps):
            cap.clear()
            with torch.no_grad():
                plan = planner.plan(obs["state"], eval_mode=True, return_full=True)
            if t % args.stride == 0 and "actions" in cap:
                cand = cap["actions"]                       # [H,N,act]
                cents = {}
                for h, m in heads.items():
                    with torch.no_grad():
                        ret, _ = cand_returns(model, obs["state"], cand, disc, fdisc, m)
                    idx = torch.topk(ret, K).indices
                    el = cand[0, idx]                        # [K, act] elite FIRST actions
                    c = el.mean(0)
                    cents[h] = c
                    an = el.norm(dim=-1).mean()
                    acc[h]["cnorm"].append(float(c.norm()))
                    acc[h]["disp"].append(float((el - c).norm(dim=-1).mean()))
                    acc[h]["amean"].append(float(an))
                    acc[h]["coh"].append(float(c.norm() / max(float(an), 1e-9)))
                    acc[h]["cdim"].append(c.abs().cpu().numpy())
                cs = torch.nn.functional.cosine_similarity(
                    cents["stock"].view(1, -1), cents["retrained"].view(1, -1)).item()
                cos_sr.append(cs)
                n += 1
            obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
            if terminal:
                break
        log(f"  ep{e}: points={n}")
    planner._estimate_returns = orig

    def ms(x):
        x = np.asarray(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    log("\n" + "=" * 88)
    log(f"ELITE-SET ACTION GEOMETRY  (committed action = mean of top-{K} first actions)")
    log("=" * 88)
    log(f"{'head':<12}{'||centroid||':>16}{'elite dispersion':>19}{'mean||a_i||':>14}"
        f"{'coherence':>12}")
    for h in heads:
        cn = ms(acc[h]["cnorm"]); dp = ms(acc[h]["disp"])
        am = ms(acc[h]["amean"]); co = ms(acc[h]["coh"])
        log(f"{h:<12}{cn[0]:>11.4f} ±{cn[1]:.4f}{dp[0]:>14.4f} ±{dp[1]:.4f}"
            f"{am[0]:>14.4f}{co[0]:>9.3f} ±{co[1]:.3f}")
    cs_, cr_ = np.array(acc["stock"]["cnorm"]), np.array(acc["retrained"]["cnorm"])
    d = cr_ - cs_
    log(f"\n  paired ||centroid||: retrained - stock = {d.mean():+.4f} ± "
        f"{d.std(ddof=1)/np.sqrt(len(d)):.4f}   (z={d.mean()/max(d.std(ddof=1)/np.sqrt(len(d)),1e-12):+.2f}"
        f", ratio {cr_.mean()/max(cs_.mean(),1e-12):.3f}x)")
    coh_s, coh_r = np.array(acc["stock"]["coh"]), np.array(acc["retrained"]["coh"])
    dc = coh_r - coh_s
    log(f"  paired coherence   : retrained - stock = {dc.mean():+.3f} ± "
        f"{dc.std(ddof=1)/np.sqrt(len(dc)):.3f}   (z={dc.mean()/max(dc.std(ddof=1)/np.sqrt(len(dc)),1e-12):+.2f})")
    log(f"  cos(committed_stock, committed_retrained) = {np.mean(cos_sr):+.3f} ± "
        f"{np.std(cos_sr,ddof=1)/np.sqrt(len(cos_sr)):.3f}  (direction agreement)")
    log("\n  per-dim |centroid| (dims 0-2 = eef pos, 3-5 = eef rot, 6 = gripper):")
    for h in heads:
        cd = np.stack(acc[h]["cdim"]).mean(0)
        log(f"    {h:<11}" + "  ".join(f"{v:.4f}" for v in cd))
    log(f"\n  n = {n} decision points ({args.arm} on-policy)")
    log("  H predicts: retrained ||centroid|| SMALLER (averaging cancellation) with"
        " cos ~ +1 (same direction, damped magnitude).")
    os._exit(0)


if __name__ == "__main__":
    main()
