"""Is the 300-episodes-per-dataset training pool too small? (i.e. did the recipe
arms overfit, making their reported fit error optimistic?)

NOTE on a measurement flaw this checks: value_head_recipe_ablate.py logs
`heldout_mse`, but that quantity is sampled from the SAME SegmentPool used for
training (in-sample). With 174,974 segments and 200k steps x batch 256 (~293
epochs) overfitting is possible and would be invisible in that log.

This script recomputes, for each converged long-run arm, the fit error against
ITS OWN target on:
    (a) the exact TRAIN pool  (same rng seed / episodes_per / dataset order), and
    (b) a FRESH pool of episodes never used for training (different rng draw),
plus (c) a much LARGER fresh pool, to see whether more data would even help.

train ~ fresh  => 300 episodes was already enough; data quantity is not the limit
fresh >> train => overfitting; rerun the arms with more episodes

Run:
  python scripts/diag_valrecipe_generalization.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist import MainConfig, Workspace  # noqa: E402

ERR = sys.stderr
spec = importlib.util.spec_from_file_location(
    "vhra", "/home/Chenyang/ibrl/scripts/value_head_recipe_ablate.py")
vhra = importlib.util.module_from_spec(spec)
sys.modules["vhra"] = vhra
spec.loader.exec_module(vhra)

ARMS = [   # name -> (ckpt, latent, teacher)
    ("E0 long", "exps/mimicgen/threading_valrecipe_E0_long_seed1/latest.pt", "rolled", "stock_self"),
    ("C long", "exps/mimicgen/threading_valrecipe_C_long_seed1/latest.pt", "real", "vsrc"),
    ("D long", "exps/mimicgen/threading_valrecipe_D_long_seed1/latest.pt", "rolled", "vsrc"),
    ("B long", "exps/mimicgen/threading_valrecipe_B_long_seed1/latest.pt", "rolled", "expertq"),
]


def log(*a):
    print(*a, file=ERR, flush=True)


def main():
    dev = "cuda"
    stock_ckpt = "exps/mimicgen/threading_simdist_adapt_seed1/latest.pt"
    stock, ep_params, cfg = vhra.build(stock_ckpt, None, dev)
    H = stock.planner.cfg.horizon
    wm = cfg.simdist.world_model
    paths = [vhra.DATA["v3"], vhra.DATA["v4"]]

    vsrc = None
    log("[setup] building V_src teacher (frozen source IBRL critic)")
    import train_rl
    from env.robosuite_wrapper import PixelRobosuite
    tmp_env = PixelRobosuite(**ep_params)
    src_ckpt = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
    src_cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(src_ckpt), "cfg.yaml")))
    src = train_rl.load_frozen_agent(src_ckpt, tmp_env, src_cfg, dev)

    @torch.no_grad()
    def vsrc(states):
        return src.value({"state": states}).view(-1)

    # (a) exact TRAIN pool: same seed(1) / episodes_per(300) / order as the long runs
    pools = {}
    log("[pool] TRAIN  (seed=1, 300 eps/dataset -- identical to the long runs)")
    pools["train"] = vhra.SegmentPool(paths, 300, H, np.random.default_rng(1), dev, True)
    log("[pool] FRESH  (seed=777, 300 eps/dataset -- never trained on)")
    pools["fresh300"] = vhra.SegmentPool(paths, 300, H, np.random.default_rng(777), dev, True)
    log("[pool] FRESH-L(seed=999, 2000 eps/dataset)")
    pools["fresh2000"] = vhra.SegmentPool(paths, 2000, H, np.random.default_rng(999), dev, True)
    for k, p in pools.items():
        log(f"    {k:<10} {p.n} segments")

    @torch.no_grad()
    def fit_error(agent_model, pool, latent, teacher, n=20000, bs=4096):
        errs = []
        g = torch.Generator(device=dev).manual_seed(0)
        for _ in range(max(1, n // bs)):
            s0, a, sH, vH = pool.sample(bs, g)
            z0 = stock.model.encode(s0)
            if latent == "rolled":
                z = z0
                for t in range(H):
                    z = stock.model.next(z, a[:, t])
                zq = z
            else:
                zq = stock.model.encode(sH)
            if teacher == "stock_self":
                tgt = stock.model.predict_value(zq, target=False).view(-1)
            elif teacher == "expertq":
                tgt = vH
            else:
                tgt = vsrc(sH)
            pred = agent_model.predict_value(zq, target=False).view(-1)
            errs.append(float(F.mse_loss(pred, tgt)))
        return float(np.mean(errs))

    log("\n" + "=" * 88)
    log("FIT ERROR vs OWN TARGET: train pool vs UNSEEN episodes  (is 300 eps/dataset enough?)")
    log("=" * 88)
    log(f"{'arm':<10}{'train pool':>14}{'fresh 300ep':>14}{'fresh 2000ep':>15}"
        f"{'fresh/train':>13}{'RMS(fresh)':>13}")
    for name, ck, latent, teacher in ARMS:
        if not os.path.exists(ck):
            log(f"  !! missing {ck}")
            continue
        ag, _, _ = vhra.build(ck, None, dev)
        e_tr = fit_error(ag.model, pools["train"], latent, teacher)
        e_f3 = fit_error(ag.model, pools["fresh300"], latent, teacher)
        e_f2 = fit_error(ag.model, pools["fresh2000"], latent, teacher)
        log(f"{name:<10}{e_tr:>14.2e}{e_f3:>14.2e}{e_f2:>15.2e}"
            f"{e_f3/max(e_tr,1e-12):>13.2f}{np.sqrt(e_f2):>13.4f}")
    log("\n  fresh/train ~ 1.0  => no overfitting; 300 eps/dataset was enough,")
    log("                        the arms' floors are FUNCTION floors, not data floors.")
    log("  fresh/train >> 1   => overfitting; rerun with more episodes.")
    log("  (signal for reference: stock cross-candidate spread = 0.0068)")
    os._exit(0)


if __name__ == "__main__":
    main()
