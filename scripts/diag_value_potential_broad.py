"""Fifth probe -- re-measure the "local raggedness" claim WHERE THE PLANNER LOOKS,
on the LARGE target datasets (fixing all three flaws of the 20-demo probe).

The 20-demo potential probe (which produced "retrained's V reverses at 24% of steps
vs stock's 10.5%") was flawed three ways, and that finding must not be trusted:
  (a) only 20 clean 100%-success expert demos (narrow; nothing like the states the
      stalling retrained arm actually visits),
  (b) evaluated at the REAL encoder latent encode(s), but the planner scores the
      IMAGINED latent f^H(z_0, a),
  (c) it measured the derivative ALONG TIME (V(s_{t+1}) - V(s_t)) whereas the planner
      compares DIFFERENT ACTIONS FROM THE SAME STATE (cross-action derivative).
And probe 1 (on-policy, imagined latents, cross-candidate) said the OPPOSITE.

This probe fixes (a),(b),(c) using the big target sets, no sim needed, because a
dataset gives both the executed action and the real s_{t+H}:

(A) CROSS-STATE ordering at IMAGINED latents, huge n:
      predicted progress  p_h = V_h(f^H(z_t, a_exec)) - V_h(encode(s_t))
      true progress       y   = values[t+H] - values[t]     (target-expert critic Q)
    -> Spearman(p_h, y). CAVEAT: `values` IS the retrained head's training target,
    so this metric structurally favours it. Reported for completeness.

(B) CROSS-ACTION spurious spread at IMAGINED latents (REFERENCE-FREE -- the actual
    test of the raggedness claim):
      perturb a_exec with the planner's own noise scale (base_policy_std) into M
      candidates, roll H steps, and measure std_a V_h(f^H) -- variation the planner
      is forced to rank on. Compare it to the real progress signal:
        potential SNR_h = Var_state(y_h_real) / Var_a(V_h(f^H))
      where y_h_real = V_h(encode(s_{t+H})) - V_h(encode(s_t)) is that head's own
      reading of the true H-step progress. A head whose cross-action spread swamps
      the true progress increment is "ragged" AT THE POINTS THE PLANNER SCORES.

Also reported: the time-direction backward-step fraction (the 20-demo metric) on
these big sets, split by success/failure and by phase, so the earlier number can be
checked at scale; and the imagination bias |V(f^H) - V(encode(s_{t+H}))|.

Datasets: v2 is the retrained head's TRAINING set; v3/v4 are HELD OUT; expert_n1000
is clean expert. Phase split by needle->ring distance (state[9:12] vs state[23:26]).

Run:
  python scripts/diag_value_potential_broad.py --datasets expert_n1000,v3,v4,v2 --episodes 150
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist import MainConfig, Workspace  # noqa: E402

ERR = sys.stderr
PATHS = {
    "dynamics20": "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5",
    "expert_n1000": "release/data/mimicgen/threading_d0_target_expert_n1000/dataset.hdf5",
    "expert_n500": "release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5",
    "v2": "release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5",
    "v3": "release/data/mimicgen/threading_d0_target_diverse_v3/dataset.hdf5",
    "v4": "release/data/mimicgen/threading_d0_target_diverse_v4/dataset.hdf5",
}


def log(*a):
    print(*a, file=ERR, flush=True)


def build(ckpt, dev="cuda"):
    cfg: MainConfig = pyrallis.load(MainConfig, open(  # type: ignore
        "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/cfg.yaml"))
    cfg.simdist.use_planner = 1
    cfg.use_wb = 0
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(cfg.simdist.planner)   # deployed recipe (base_policy_std 0.1, min_std 0.05)
    ws = Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(ckpt))
    return ws.agent.to(dev).eval(), cfg


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(len(x))
    return r


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(_rank(a[m]), _rank(b[m]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock_ckpt", default="exps/mimicgen/threading_simdist_adapt_seed1/latest.pt")
    ap.add_argument("--retrained_ckpt",
                    default="exps/mimicgen/threading_simdist_ablate_val_ws_600k_seed1/latest.pt")
    ap.add_argument("--datasets", default="expert_n1000,v3,v4,v2")
    ap.add_argument("--episodes", type=int, default=150, help="episodes sampled per dataset")
    ap.add_argument("--per_ep", type=int, default=8, help="time indices sampled per episode")
    ap.add_argument("--cloud", type=int, default=32, help="perturbed candidates per state")
    ap.add_argument("--contact_thresh", type=float, default=0.13,
                help="needle-ring BODY distance; expert success sits at ~0.094 (see §17)")
    ap.add_argument("--perturb_std", type=float, default=-1.0,
                help="cross-action perturbation; <0 => planner min_std (the CEM converged spread)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    aS, cfg = build(args.stock_ckpt)
    aR, _ = build(args.retrained_ckpt)
    model = aS.model                    # encoder/dynamics identical in both ckpts
    H = aS.planner.cfg.horizon
    astd = aS.planner.cfg.min_std if args.perturb_std < 0 else args.perturb_std
    heads = {"stock": aS.model, "retrained": aR.model}
    log(f"[setup] H={H} | perturb std = {astd} (planner min_std={aS.planner.cfg.min_std}, base_policy_std={aS.planner.cfg.base_policy_std}) | cloud={args.cloud}")

    @torch.no_grad()
    def V(m, z):
        return m.predict_value(z, target=False).view(-1)

    @torch.no_grad()
    def roll(z0, acts):
        """z0 [B,z] , acts [B,H,act] -> imagined terminal latent [B,z]"""
        z = z0
        for t in range(acts.shape[1]):
            z = model.next(z, acts[:, t])
        return z

    for dname in args.datasets.split(","):
        path = PATHS[dname]
        if not os.path.exists(path):
            log(f"  !! missing {dname} ({path})")
            continue
        rows = {h: {k: [] for k in ("p", "yreal", "spread", "bias", "back")} for h in heads}
        meta = {"y": [], "succ": [], "contact": []}
        with h5py.File(path, "r") as f:
            d = f["data"]
            keys = list(d.keys())
            sel = rng.choice(len(keys), size=min(args.episodes, len(keys)), replace=False)
            for ei in sel:
                ep = d[keys[ei]]
                s = np.asarray(ep["obs"]["state"][()], dtype=np.float32)
                a = np.asarray(ep["actions"][()], dtype=np.float32)
                T = min(s.shape[0], a.shape[0] + 1)
                if T < H + 3:
                    continue
                has_v = "values" in ep
                vals = np.asarray(ep["values"][()], dtype=np.float32).reshape(-1) if has_v else None
                succ = float(np.max(np.asarray(ep["rewards"][()])) > 0.5) if "rewards" in ep else np.nan
                ts = rng.choice(T - H - 1, size=min(args.per_ep, T - H - 1), replace=False)
                st = torch.from_numpy(s).to(dev)
                with torch.no_grad():
                    z_all = model.encode(st)
                # time-direction backward-step fraction (the 20-demo metric), whole ep
                for h, m in heads.items():
                    with torch.no_grad():
                        v_line = V(m, z_all).cpu().numpy()
                    rows[h]["back"].append(float(np.mean(np.diff(v_line) < 0)))
                for t in ts:
                    t = int(t)
                    a_exec = torch.from_numpy(a[t:t + H]).to(dev)             # [H,act]
                    cloud = a_exec.unsqueeze(0).repeat(args.cloud, 1, 1)
                    cloud[1:] = (cloud[1:] + torch.randn_like(cloud[1:]) * astd).clamp(-1, 1)
                    z0 = z_all[t].unsqueeze(0).repeat(args.cloud, 1)
                    with torch.no_grad():
                        zH = roll(z0, cloud)                                  # [M,z]
                    nd = float(np.linalg.norm(s[t, 9:12] - s[t, 23:26]))
                    meta["contact"].append(nd < args.contact_thresh)
                    meta["succ"].append(succ)
                    meta["y"].append(float(vals[t + H] - vals[t]) if has_v else np.nan)
                    for h, m in heads.items():
                        with torch.no_grad():
                            vH = V(m, zH)                                     # [M]
                            v0 = float(V(m, z_all[t].unsqueeze(0))[0])
                            vreal = float(V(m, z_all[t + H].unsqueeze(0))[0])
                        rows[h]["p"].append(float(vH[0]) - v0)          # exec-action progress
                        rows[h]["yreal"].append(vreal - v0)             # true H-step progress
                        rows[h]["spread"].append(float(vH.std()))       # cross-action spread
                        rows[h]["bias"].append(float(vH[0]) - vreal)    # imagination bias

        n = len(meta["y"])
        ct = np.array(meta["contact"])
        sc = np.array(meta["succ"])
        y = np.array(meta["y"])
        log("\n" + "=" * 96)
        tag = "TRAIN SET of the retrained head" if dname == "v2" else "held out"
        log(f"dataset {dname}  ({tag})   n={n} states, {len(rows['stock']['back'])} episodes"
            f" | contact-phase {100*np.mean(ct):.0f}% | success {np.nanmean(sc):.2f}")
        log("=" * 96)
        log(f"{'head':<11}{'(A) Sp(pred,true prog)':>23}{'(B) std_a V(f^H)':>18}"
            f"{'|true prog|':>13}{'SNR=Var(prog)/Var(spread)':>27}{'|imag bias|':>13}")
        for h in heads:
            r = {k: np.array(v) for k, v in rows[h].items()}
            spA = spearman(r["p"], y)  # nan when the set has no `values`
            sprd = float(np.nanmean(r["spread"]))
            prog = float(np.nanstd(r["yreal"]))
            snr = (prog ** 2) / max(sprd ** 2, 1e-16)
            log(f"{h:<11}{spA:>+18.3f}     {sprd:>17.5f}{prog:>13.5f}{snr:>22.2f}"
                f"{float(np.nanmean(np.abs(r['bias']))):>18.5f}")
        log(f"\n{'head':<11}{'back-step frac (time dir, the 20-demo metric)':<48}"
            f"{'contact':>10}{'free':>10}{'success ep':>12}{'fail ep':>10}")
        for h in heads:
            b = np.array(rows[h]["back"])
            sp_c = np.nanmean(np.array(rows[h]["spread"])[ct]) if ct.any() else np.nan
            sp_f = np.nanmean(np.array(rows[h]["spread"])[~ct]) if (~ct).any() else np.nan
            log(f"{h:<11}{np.nanmean(b):>18.3f} ±{np.nanstd(b, ddof=1)/np.sqrt(len(b)):.3f}"
                f"{'':>27}{sp_c:>10.5f}{sp_f:>10.5f}"
                f"{'(spread by phase)':>22}")
        # paired contrast of the reference-free quantity
        ss = np.array(rows["stock"]["spread"]); rs = np.array(rows["retrained"]["spread"])
        dd = rs - ss
        log(f"\n  paired std_a: retrained - stock = {dd.mean():+.5f} ± "
            f"{dd.std(ddof=1)/np.sqrt(len(dd)):.5f}  (ratio {rs.mean()/max(ss.mean(),1e-12):.2f}x)")
        bs = np.array(rows["stock"]["back"]); br = np.array(rows["retrained"]["back"])
        db = br - bs
        log(f"  paired back-frac: retrained - stock = {db.mean():+.3f} ± "
            f"{db.std(ddof=1)/np.sqrt(len(db)):.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
