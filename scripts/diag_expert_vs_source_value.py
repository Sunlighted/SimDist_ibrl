"""WHERE does the wide-trained expert's value disagree with the frozen source value?

Setup (CLAUDE.md §33). In the wide1.5 domain (§12 target physics + needle init box x1.5),
ORACLE dynamics + the FROZEN source value scores 0.933, and bucketing by needle-init
eccentricity puts the whole loss in the outer quartile (0.967/0.967/0.967 -> 0.833). Since
oracle dynamics has zero model error, that loss is the value's. `threading_ibrl_target_wide15_seed1`
is an IBRL agent trained DIRECTLY on this domain (train score 1.00), so its
V(s) = min-ensemble Q(s, pi_hybrid(s)) is a much better value here -- a reference for what
the source value SHOULD look like.

This script asks where the two disagree, which does two jobs at once:
  (a) localizes the value error without any hand-designed label, and
  (b) VALIDATES THE GATE: a support-based gate is only useful if novelty predicts the
      disagreement. `Spearman(novelty, |V_expert - V_source|)` is therefore the honest
      version of the check that scripts/diag_value_gate_calib.py got wrong (it used
      failure-vs-success episodes, on the wrong physics, with a policy that succeeded
      once in 50 tries).

Both values are read at the point the ORACLE arm actually queries (`encode(real s_H)` --
see eval_oracle_dyn_mppi.py::oracle_returns), and the expert value is taken with its IBRL
prior attached (`use_ibrl=True`), i.e. the policy it really executes -- load_frozen_agent
alone would silently drop the prior (§17/§27, a bug that has bitten this project 3x).

The environment is inherited from the run's own cfg (`Workspace.eval_env_params`) plus
`--needle_scale`, NOT hand-written: writing `DomainShiftConfig(mode="fixed")` by hand is
exactly how the first attempt ended up on nominal physics instead of the §12 target.

Run:
  python scripts/diag_expert_vs_source_value.py --episodes 40
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

ERR = sys.stderr
SRC = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
BOX_C, BOX_HW = np.array([-0.125, 0.2]), np.array([0.075, 0.05])
NEEDLE, RING = slice(9, 12), slice(9 + 14, 9 + 17)


def log(*a):
    print(*a, file=ERR, flush=True)


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


@torch.no_grad()
def knn_dist(query, ref, k=5, chunk=2048):
    out = []
    for i in range(0, query.shape[0], chunk):
        d = torch.cdist(query[i:i + chunk], ref)
        out.append(d.topk(k, dim=1, largest=False).values.mean(1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt_run", default="exps/mimicgen/threading_simdist_dynens_adapt_wide1.5_seed1")
    ap.add_argument("--ckpt", default="model0.pt")
    ap.add_argument("--expert", default="exps/mimicgen/threading_ibrl_target_wide15_seed1/model0.pt")
    ap.add_argument("--needle_scale", type=float, default=1.5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--collect", default="mppi", choices=["mppi", "expert"],
                    help="mppi = the DEPLOYED learned-MPPI on-policy distribution (what the "
                         "value is actually queried on); expert = the wide-trained expert")
    ap.add_argument("--ref_episodes", type=int, default=3000)
    ap.add_argument("--cal_episodes", type=int, default=600)
    ap.add_argument("--ref_stride", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stride", type=int, default=4, help="record every k-th control step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    from train_simdist import MainConfig, Workspace
    from train_simdist_adapt import AdaptConfig
    from env.domain_shift import InitDistConfig
    from env.robosuite_wrapper import PixelRobosuite

    # ---- the system under test: env inherited from its own cfg (never hand-written) ----
    acfg: AdaptConfig = pyrallis.load(AdaptConfig, open(  # type: ignore
        os.path.join(args.adapt_run, "cfg.yaml")))
    scfg: MainConfig = pyrallis.load(MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(acfg.stage1_ckpt), "cfg.yaml")))
    scfg.simdist.use_planner = 1
    scfg.use_wb = 0
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(scfg.simdist.planner)
    if acfg.eval_hand_init_perturb_joint_sigma is not None:
        scfg.eval_hand_init_perturb_joint_sigma = acfg.eval_hand_init_perturb_joint_sigma
    ws = Workspace(scfg, from_main=False)
    ws.agent.load_state_dict(torch.load(os.path.join(args.adapt_run, args.ckpt)))
    agent = ws.agent.to(dev).eval()
    wm = agent.model

    ep = dict(ws.eval_env_params)          # <- §12 target physics, straight from the cfg
    nsc = args.needle_scale
    ep["init_distribution"] = InitDistConfig(
        enabled=True,
        needle_x=(BOX_C[0] - BOX_HW[0] * nsc, BOX_C[0] + BOX_HW[0] * nsc),
        needle_y=(BOX_C[1] - BOX_HW[1] * nsc, BOX_C[1] + BOX_HW[1] * nsc),
        needle_z_rot=(-2.0 * np.pi / 3.0, -np.pi / 3.0))
    env = PixelRobosuite(**ep)
    log(f"[env] from {args.adapt_run}/cfg.yaml: shift={ep['domain_shift'].mode} "
        f"friction={ep['domain_shift'].friction_scale} kp={ep['domain_shift'].gripper_kp_scale} "
        f"ring={ep['domain_shift'].ring_size_scale}/{ep['domain_shift'].ring_friction_scale} "
        f"| needle box x{nsc}")

    # ---- the expert value (IBRL hybrid, prior attached) ----
    ecfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.expert), "cfg.yaml")))
    expert = train_rl.load_frozen_agent(args.expert, env, ecfg, dev)
    edesc = train_rl.attach_ibrl_prior(expert, args.expert, env, dev)
    expert.cfg.act_method = "ibrl" if edesc else "rl"
    log(f"[expert] {args.expert} | prior: {edesc or 'none'} | act={expert.cfg.act_method}")

    @torch.no_grad()
    def V_src(states):
        return wm.predict_value(wm.encode(states.to(dev)), target=False).view(-1).cpu().numpy()

    @torch.no_grad()
    def V_exp(states):
        return expert.value({"state": states.to(dev)}, use_ibrl=bool(edesc)).view(-1).cpu().numpy()

    # ---- reference set for novelty (the value's OWN training data) ----
    rng = np.random.default_rng(args.seed)
    with h5py.File(SRC, "r") as f:
        d = f["data"]
        keys = list(d.keys())
        pick = rng.choice(len(keys), size=args.ref_episodes + args.cal_episodes, replace=False)

        def grab(idxs):
            return np.concatenate([np.asarray(d[keys[i]]["obs"]["state"][()],
                                              dtype=np.float32)[:: args.ref_stride] for i in idxs])
        ref = torch.from_numpy(grab(pick[: args.ref_episodes])).to(dev)
        cal = torch.from_numpy(grab(pick[args.ref_episodes:])).to(dev)
    mu, sd = ref.mean(0, keepdim=True), ref.std(0, keepdim=True).clamp_min(1e-6)
    refw, calw = (ref - mu) / sd, (cal - mu) / sd          # whitened: else large-scale dims dominate
    u_in = knn_dist(calw, refw, args.k).cpu().numpy()
    T99 = float(np.percentile(u_in, 99))
    log(f"[novelty] ref {tuple(ref.shape)} cal {tuple(cal.shape)} | in-dist kNN p50 "
        f"{np.percentile(u_in,50):.3f} p99 {T99:.3f} p99.5 {np.percentile(u_in,99.5):.3f}")

    # ---- collect on-policy states ----
    S, ECC, TT, OK = [], [], [], []
    for e in range(args.episodes):
        np.random.seed(args.seed + e)
        torch.manual_seed(args.seed + e)
        obs, _ = env.reset()
        n0 = obs["state"][9:12].cpu().numpy()[:2]
        ecc = float(np.max(np.abs((n0 - BOX_C) / (BOX_HW * nsc))))
        traj, ok = [], 0.0
        for t in range(400):
            with torch.no_grad():
                a = (agent.act(obs, eval_mode=True) if args.collect == "mppi"
                     else expert.act(obs, eval_mode=True))
            obs, reward, terminal, _, _ = env.step(a)
            if t % args.stride == 0:
                traj.append((t, obs["state"].cpu().numpy().copy()))
            if reward > 0:
                ok = 1.0
            if terminal:
                break
        for t, s in traj:
            S.append(s); ECC.append(ecc); TT.append(t); OK.append(ok)
        if (e + 1) % 10 == 0:
            log(f"    ep{e+1}: success so far {np.mean(OK[-1:] + [o for o in OK]):.2f} "
                f"({len(S)} states)")
    S = np.stack(S); ECC = np.array(ECC); TT = np.array(TT); OK = np.array(OK)
    St = torch.from_numpy(S).float()
    log(f"[data] {len(S)} states from {args.episodes} eps | collect={args.collect}")

    vs, ve = V_src(St), V_exp(St)
    nov = knn_dist((St.to(dev) - mu) / sd, refw, args.k).cpu().numpy()
    diff = ve - vs
    ad = np.abs(diff)

    def blk(t):
        log("\n" + "=" * 82 + f"\n{t}\n" + "=" * 82)

    blk("scales (the two values live on different scales -- compare SHAPE, not level)")
    log(f"  V_source : mean {vs.mean():+.4f} std {vs.std():.4f}  [min {vs.min():+.3f} max {vs.max():+.3f}]")
    log(f"  V_expert : mean {ve.mean():+.4f} std {ve.std():.4f}  [min {ve.min():+.3f} max {ve.max():+.3f}]")
    log(f"  Spearman(V_source, V_expert) over all states = {spearman(vs, ve):+.3f}"
        "   <- global ordering agreement")

    blk("★ GATE VALIDATION: does novelty predict where they disagree?")
    # z-score both first so |diff| is not dominated by the scale mismatch
    zs = (vs - vs.mean()) / max(vs.std(), 1e-9)
    ze = (ve - ve.mean()) / max(ve.std(), 1e-9)
    adz = np.abs(ze - zs)
    log(f"  Spearman(novelty, |dV| raw)        = {spearman(nov, ad):+.3f}")
    log(f"  Spearman(novelty, |dV| z-scored)   = {spearman(nov, adz):+.3f}   <- the gate criterion")
    log(f"  frac of states above novelty p99   = {np.mean(nov > T99):.3f}")
    hi = nov > T99
    if hi.sum() and (~hi).sum():
        log(f"  |dV|z  above-p99 {adz[hi].mean():.3f}  vs  below {adz[~hi].mean():.3f}  "
            f"(ratio {adz[hi].mean()/max(adz[~hi].mean(),1e-9):.2f}x)")

    blk("localization: by eccentricity / by episode phase / by success")
    log(f"{'bucket':>18}{'n':>7}{'novelty':>10}{'|dV|z':>9}{'V_src':>9}{'V_exp':>9}")
    for lo, hi_ in [(0.0, 0.48), (0.48, 0.74), (0.74, 0.89), (0.89, 1.01)]:
        m = (ECC >= lo) & (ECC < hi_)
        if m.sum():
            log(f"{f'ecc {lo:.2f}-{hi_:.2f}':>18}{int(m.sum()):>7}{nov[m].mean():>10.3f}"
                f"{adz[m].mean():>9.3f}{vs[m].mean():>9.3f}{ve[m].mean():>9.3f}")
    for lo, hi_ in [(0, 40), (40, 120), (120, 400)]:
        m = (TT >= lo) & (TT < hi_)
        if m.sum():
            log(f"{f'step {lo}-{hi_}':>18}{int(m.sum()):>7}{nov[m].mean():>10.3f}"
                f"{adz[m].mean():>9.3f}{vs[m].mean():>9.3f}{ve[m].mean():>9.3f}")
    for v, nm in ((1.0, "success eps"), (0.0, "failed eps")):
        m = OK == v
        if m.sum():
            log(f"{nm:>18}{int(m.sum()):>7}{nov[m].mean():>10.3f}"
                f"{adz[m].mean():>9.3f}{vs[m].mean():>9.3f}{ve[m].mean():>9.3f}")

    blk("VERDICT")
    sp = spearman(nov, adz)
    log(f"  Spearman(novelty, |dV|z) = {sp:+.3f}")
    if sp > 0.4:
        log("  => novelty PREDICTS where the source value is wrong: a support-based gate is")
        log("     the right instrument. Proceed to the gated residual.")
    elif sp > 0.15:
        log("  => weak. The gate would fire only partly where it matters; consider a")
        log("     subspace novelty (needle pose dims) or a different support measure.")
    else:
        log("  => novelty does NOT predict the disagreement. A support-based gate is the")
        log("     WRONG instrument here -- rethink before implementing anything.")
    os._exit(0)


if __name__ == "__main__":
    main()
