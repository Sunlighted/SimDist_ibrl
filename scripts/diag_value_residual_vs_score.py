"""Final probe: does ONE quantity explain every arm's score?

The recipe 2x2 (scripts/value_head_recipe_ablate.py) produced 5 re-fit value heads
whose closed-loop scores span 0.28-0.56 while stock is 0.745-0.80:

    A  real   x expertq     0.280
    C  real   x vsrc        0.430
    B  rolled x expertq     0.515
    D  rolled x vsrc        0.560
    E0 rolled x stock_self  0.555      <- fits stock's OWN function, still loses ~0.2
    (historical retrained arm ~0.37)

Every arm fit its own target well (held-out mse 2.6e-4..1.7e-3), so accuracy w.r.t.
its target is not the variable. Candidate single explanatory variable: how far the
head deviates from the WORKING (co-adapted) function AT THE POINTS THE PLANNER
SCORES, and specifically the part of that deviation argmax is exposed to.

For every arm, on ONE FIXED distribution (stock's own on-policy candidate sets, so
all arms are compared at identical latents):

    delta_i   = V_arm(z_H^i) - V_stock(z_H^i)        over the candidate set at a state
    |delta|            absolute deviation (mean)
    std_i(delta~)      WITHIN-SET deviation, i.e. after removing the per-state mean
                       -> the only part that can change topk (a per-state constant
                          cannot: §27's eps_bar vs eps_tilde)
    Spearman_i(V_arm, V_stock)   ranking agreement with the working function
    top64 overlap      |top64(arm) ∩ top64(stock)| / 64  -- the ONLY channel by which
                       the value can act (probes 2-3: softmax inert, so the committed
                       action is the mean of the selected 64)

If score is monotone in top64-overlap / std_i(delta~), the root cause is a single
scalar: deviation-at-the-query-points, and the "recipe" axes matter only through it.

Run:
  python scripts/diag_value_residual_vs_score.py --episodes 8 --stride 4
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
ARMS = [
    # LONG runs (200k steps) -- scores are the mean over their 200-game eval points
    ("stock", "exps/mimicgen/threading_simdist_adapt_seed1/latest.pt", 0.775),
    ("E0 long (self)", "exps/mimicgen/threading_valrecipe_E0_long_seed1/latest.pt", 0.759),
    ("C long real/vsrc", "exps/mimicgen/threading_valrecipe_C_long_seed1/latest.pt", 0.466),
    ("B long rolled/expQ", "exps/mimicgen/threading_valrecipe_B_long_seed1/latest.pt", 0.389),
    ("D long rolled/vsrc", "exps/mimicgen/threading_valrecipe_D_long_seed1/latest.pt", 0.403),
    ("historical 600k", "exps/mimicgen/threading_simdist_ablate_val_ws_600k_seed1/latest.pt", 0.370),
]


def log(*a):
    print(*a, file=ERR, flush=True)


def build(ckpt, dev="cuda"):
    cfg: MainConfig = pyrallis.load(MainConfig, open(  # type: ignore
        "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/cfg.yaml"))
    cfg.simdist.use_planner = 1
    cfg.use_wb = 0
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(cfg.simdist.planner)
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(ckpt))
    return ws.agent.to(dev).eval(), ws.eval_env_params, cfg


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(len(x))
    return r


def spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    avail = [(n, p, s) for n, p, s in ARMS if os.path.exists(p)]
    for n, p, s in ARMS:
        if not os.path.exists(p):
            log(f"  !! missing ckpt for {n}: {p}")
    base, ep_params, cfg = build(avail[0][1], dev)
    model, pl = base.model, base.planner
    K = pl.cfg.num_elites
    fdisc = pl.final_discount if hasattr(pl, "final_discount") else cfg.discount ** pl.cfg.horizon
    # every arm shares encoder/dynamics/reward -> one model rolls, many value heads read
    vheads = {}
    for n, p, s in avail:
        ag, _, _ = build(p, dev)
        vheads[n] = (ag.model, s)
    log(f"[setup] {len(vheads)} arms | elites={K} | rolling stock's on-policy candidate sets")

    main_env = PixelRobosuite(**ep_params)
    acc = {n: {k: [] for k in ("adev", "wdev", "sp", "ov")} for n in vheads}
    planner = base.planner
    cap = {}
    orig = planner._estimate_returns

    def wrapped(z0, actions, _o=orig, _c=cap):
        r = _o(z0, actions)
        _c["actions"] = actions
        return r
    planner._estimate_returns = wrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_pts = 0
    for e in range(args.episodes):
        obs, _ = main_env.reset()
        for t in range(args.max_steps):
            cap.clear()
            with torch.no_grad():
                plan = planner.plan(obs["state"], eval_mode=True, return_full=True)
            if t % args.stride == 0 and "actions" in cap:
                cand = cap["actions"]
                N = cand.shape[1]
                with torch.no_grad():
                    z = model.encode(obs["state"].view(1, -1)).expand(N, -1).contiguous()
                    rsum = torch.zeros(N, device=dev)
                    g = 1.0
                    for tt in range(cand.shape[0]):
                        rsum = rsum + g * model.predict_reward(z, cand[tt]).view(-1)
                        g = g * cfg.discount
                        z = model.next(z, cand[tt])
                    vals = {n: (m.predict_value(z, target=False).view(-1) * fdisc)
                            for n, (m, _) in vheads.items()}
                vs = vals["stock"].cpu().numpy()
                acc["stock"].setdefault("sig", []).append(float(np.std(vs)))
                top_s = set(torch.topk(rsum + vals["stock"], K).indices.cpu().numpy().tolist())
                for n, v in vals.items():
                    vn = v.cpu().numpy()
                    d = vn - vs
                    acc[n]["adev"].append(float(np.abs(d).mean()))
                    acc[n]["wdev"].append(float((d - d.mean()).std()))
                    acc[n]["sp"].append(spearman(vn, vs))
                    top_n = set(torch.topk(rsum + v, K).indices.cpu().numpy().tolist())
                    acc[n]["ov"].append(len(top_n & top_s) / K)
                n_pts += 1
            obs, reward, terminal, _, _ = main_env.step(plan[0].detach().cpu())
            if terminal:
                break
        log(f"  ep{e}: points={n_pts}")
    planner._estimate_returns = orig

    def m(x):
        return float(np.nanmean(x))

    log("\n" + "=" * 100)
    log("DEVIATION FROM THE WORKING FUNCTION, measured at stock's on-policy candidate sets")
    log("=" * 100)
    log(f"{'arm':<24}{'score':>8}{'|delta|':>11}{'std_i(delta~)':>15}"
        f"{'Spearman vs stock':>19}{'top64 overlap':>15}")
    rows = []
    for n, (mm, sc) in vheads.items():
        a = acc[n]
        rows.append((n, sc, m(a["adev"]), m(a["wdev"]), m(a["sp"]), m(a["ov"])))
        log(f"{n:<24}{sc:>8.3f}{m(a['adev']):>11.5f}{m(a['wdev']):>15.5f}"
            f"{m(a['sp']):>19.3f}{m(a['ov']):>15.3f}")
    log(f"\n  n = {n_pts} decision points (stock on-policy)")
    sig = float(np.nanmean(acc["stock"]["sig"])) if "sig" in acc["stock"] else float("nan")
    log(f"  SIGNAL: stock cross-candidate spread std_i(gamma^H V) = {sig:.5f}")
    log(f"\n{'arm':<24}{'score':>8}{'residual/signal':>18}")
    for n, sc, ad, wd, sp, ov in rows:
        log(f"{n:<24}{sc:>8.3f}{(wd/max(sig,1e-12)):>18.2f}")

    sc = np.array([r[1] for r in rows])
    for j, name in ((2, "|delta|"), (3, "std_i(delta~)"), (4, "Spearman vs stock"),
                    (5, "top64 overlap")):
        x = np.array([r[j] for r in rows])
        log(f"  Spearman(score, {name:<20}) = {spearman(sc, x):+.3f}   "
            f"Pearson = {np.corrcoef(sc, x)[0,1]:+.3f}   (n={len(rows)} arms)")
    log("\n  a strong monotone relation => the root cause is ONE scalar: deviation from the")
    log("  co-adapted function at the query points; the recipe axes act only through it.")
    os._exit(0)


if __name__ == "__main__":
    main()
