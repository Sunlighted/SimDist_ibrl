"""Step 1a of the gated value-residual design: is a support-based gate VIABLE at all?

Context (CLAUDE.md §33). With ORACLE dynamics (model error = 0, so the value's ranking is
the only thing that can still be wrong) the frozen source value scores 0.993 on the §12
target and 0.967 on the jam domain, but only 0.933 on wide1.5 (needle init box x1.5) --
and bucketing by needle-init eccentricity r shows the oracle loss is concentrated in the
OUTER quartile (r 0.89-1.00: 0.833 vs 0.967 for r<0.89). So the value is locally blind in
a region it was never trained on, and the fix must be LOCAL: keep V_frozen bit-exact where
it works, add a gated residual only where it does not.

The gate cannot be "which episode started outside the D0 box" -- that is an EPISODE-level
label, while the value is a function of the STATE (a needle that started off-box but is now
grasped and lifted may be in perfectly familiar territory). So the gate must be a
state-level support measure. Here it is a k-NN distance to the value's ACTUAL training data
(`source_diverse_v2`, what dynens stage-1 distilled on; §24 adapt ran with adapt_value=0 so
the value never moved afterwards) -- no training, no fitting, an exact support probe. RND is
just the memory-compressed version and is only needed for deployment.

Three questions, all gating the design:

  (1) SEPARATION: does novelty actually light up where the failures are? Measured as
      AUC(novelty; states from failing wide episodes vs states from succeeding ones) and by
      eccentricity bucket. If novelty cannot separate them the gate is broken -- stop.
  (2) RETURN-TO-SUPPORT: for OOD states, is novelty(s_{t+H}) < novelty(s_t), and how often
      does s_{t+H} fall back inside the threshold? This sets lambda for the backup target:
      high => V_frozen is a trustworthy bootstrap a few steps out (small lambda is fine);
      low => V_frozen is untrustworthy everywhere in the annulus, so lambda must go to 1
      and the real terminal reward becomes the only anchor.
  (3) LATENT vs RAW: the value consumes encode(s), so latent distance is the relevant one;
      raw 37-d distance is free of the encoder's geometric distortions (§30: the latent
      drops mm-scale geometry). Disagreement between them is itself informative.

NOT measured here (needs real oracle-MPPI candidate sets, so it is merged into the data
collection step): std_i(g) WITHIN a candidate set. If the gate itself varies across the 350
candidates of one decision point, the gate becomes the within-set fluctuation eps_tilde that
§33 identified as the one thing that kills argmax.

Run:
  python scripts/diag_value_gate_calib.py --ref_episodes 2000 --wide_episodes 40
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

from env.domain_shift import DomainShiftConfig, InitDistConfig  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.strict_success import StrictSuccessConfig  # noqa: E402

ERR = sys.stderr
SRC = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
# D0 nominal needle box (same numbers eval_oracle_dyn_mppi.py uses)
BOX_C, BOX_HW = np.array([-0.125, 0.2]), np.array([0.075, 0.05])


def log(*a):
    print(*a, file=ERR, flush=True)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


@torch.no_grad()
def knn_dist(query, ref, k=5, chunk=4096):
    """mean distance to the k nearest reference points. query [M,D], ref [N,D] on GPU."""
    out = []
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        d = torch.cdist(q, ref)                       # [c, N]
        out.append(d.topk(k, dim=1, largest=False).values.mean(1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt_run", default="exps/mimicgen/threading_simdist_dynens_adapt_wide1.5_seed1")
    ap.add_argument("--ckpt", default="model0.pt")
    ap.add_argument("--agent", default="exps/mimicgen/threading_ibrl_source_strict_seed1/model0.pt")
    ap.add_argument("--ref_episodes", type=int, default=2000, help="source_diverse_v2 episodes for the reference set")
    ap.add_argument("--cal_episodes", type=int, default=400, help="HELD-OUT source episodes for the threshold")
    ap.add_argument("--ref_stride", type=int, default=8, help="subsample steps within an episode")
    ap.add_argument("--wide_episodes", type=int, default=40)
    ap.add_argument("--needle_scale", type=float, default=1.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=0, help="0 => use the world model's horizon")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--explore_std", type=float, default=0.0,
                    help="action noise while collecting wide-domain states (widens coverage; "
                         "these rollouts are only for the gate probe, not for training)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    from train_simdist import MainConfig, Workspace
    from train_simdist_adapt import AdaptConfig

    acfg: AdaptConfig = pyrallis.load(AdaptConfig, open(  # type: ignore
        os.path.join(args.adapt_run, "cfg.yaml")))
    scfg: MainConfig = pyrallis.load(MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(acfg.stage1_ckpt), "cfg.yaml")))
    scfg.use_wb = 0
    ws = Workspace(scfg, from_main=False)
    ws.agent.load_state_dict(torch.load(os.path.join(args.adapt_run, args.ckpt)))
    wm = ws.agent.to(dev).eval().model
    H = args.horizon or scfg.simdist.world_model.horizon

    @torch.no_grad()
    def enc(states):
        return wm.encode(states.to(dev))

    # ---------- reference + calibration sets from the value's OWN training data ----------
    rng = np.random.default_rng(args.seed)
    with h5py.File(SRC, "r") as f:
        d = f["data"]
        keys = list(d.keys())
        pick = rng.choice(len(keys), size=args.ref_episodes + args.cal_episodes, replace=False)
        ref_k, cal_k = pick[: args.ref_episodes], pick[args.ref_episodes:]

        def grab(idxs):
            S = []
            for i in idxs:
                s = np.asarray(d[keys[i]]["obs"]["state"][()], dtype=np.float32)
                S.append(s[:: args.ref_stride])
            return np.concatenate(S)
        ref_raw = torch.from_numpy(grab(ref_k)).to(dev)
        cal_raw = torch.from_numpy(grab(cal_k)).to(dev)
    ref_lat, cal_lat = enc(ref_raw), enc(cal_raw)

    W = {}   # per-space whitening stats, reused for every later batch
    for nm, R in (("raw", ref_raw), ("latent", ref_lat)):
        W[nm] = (R.mean(0, keepdim=True), R.std(0, keepdim=True).clamp_min(1e-6))

    def wh(nm, x):
        mu, sd = W[nm]
        return (x - mu) / sd
    ref_raw, cal_raw = wh("raw", ref_raw), wh("raw", cal_raw)
    ref_lat, cal_lat = wh("latent", ref_lat), wh("latent", cal_lat)
    log(f"[ref] source_diverse_v2: reference {tuple(ref_raw.shape)}, "
        f"held-out calibration {tuple(cal_raw.shape)} | latent dim {ref_lat.shape[1]} | H={H}")

    spaces = {"raw": (ref_raw, cal_raw), "latent": (ref_lat, cal_lat)}
    thr = {}
    for name, (r, c) in spaces.items():
        u_in = knn_dist(c, r, args.k).cpu().numpy()
        thr[name] = dict(p50=float(np.percentile(u_in, 50)),
                         p90=float(np.percentile(u_in, 90)),
                         p99=float(np.percentile(u_in, 99)),
                         p995=float(np.percentile(u_in, 99.5)))
        # heavy tails in the reference data itself make p99.5 a poor threshold, so print
        # the whole quantile curve and let the separation numbers pick the operating point
        log(f"  [{name}] in-dist kNN(k={args.k}) whitened dist: p50 {thr[name]['p50']:.3f} "
            f"p90 {thr[name]['p90']:.3f} p99 {thr[name]['p99']:.3f} p99.5 {thr[name]['p995']:.3f}"
            f"  (tail ratio p99.5/p50 = {thr[name]['p995']/max(thr[name]['p50'],1e-9):.1f}x)")

    # ---------- wide-domain rollouts (cheap: policy + noise, NOT oracle MPPI) ----------
    nsc = args.needle_scale
    ds = DomainShiftConfig(mode="fixed")
    init = InitDistConfig(
        enabled=True,
        needle_x=(BOX_C[0] - BOX_HW[0] * nsc, BOX_C[0] + BOX_HW[0] * nsc),
        needle_y=(BOX_C[1] - BOX_HW[1] * nsc, BOX_C[1] + BOX_HW[1] * nsc),
        needle_z_rot=(-2.0 * np.pi / 3.0, -np.pi / 3.0))
    env = PixelRobosuite(
        env_name="Threading_D0", robots="Panda", episode_length=400, use_state=True,
        rl_cameras=[], camera_names=[], device="cuda", domain_shift=ds,
        init_distribution=init, strict_success=StrictSuccessConfig(enabled=True),
        hand_init_perturb_joint_sigma=args.sigma)
    cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.agent), "cfg.yaml")))
    agent = train_rl.load_frozen_agent(args.agent, env, cfg, dev)
    desc = train_rl.attach_ibrl_prior(agent, args.agent, env, dev)   # §17/§27: prior!
    agent.cfg.act_method = "ibrl" if desc else "rl"
    log(f"[wide] needle box x{nsc} | policy prior: {desc or 'none'} | "
        f"{args.wide_episodes} eps, explore_std {args.explore_std}")

    trajs = []
    for e in range(args.wide_episodes):
        np.random.seed(args.seed + e)
        torch.manual_seed(args.seed + e)
        obs, _ = env.reset()
        n0 = obs["state"][9:12].cpu().numpy()[:2]
        uv = (n0 - BOX_C) / (BOX_HW * nsc)
        S, ok = [], 0.0
        for t in range(400):
            with torch.no_grad():
                a = agent.act(obs, eval_mode=True)
            if args.explore_std > 0:
                a = (a + torch.randn_like(a) * args.explore_std).clamp(-1, 1)
            obs, reward, terminal, _, _ = env.step(a)
            S.append(obs["state"].cpu().numpy().copy())
            if reward > 0:
                ok = 1.0
            if terminal:
                break
        trajs.append(dict(S=np.stack(S), ok=ok, r=float(np.max(np.abs(uv)))))
        if (e + 1) % 10 == 0:
            log(f"    ep{e+1}: success so far {np.mean([x['ok'] for x in trajs]):.2f}")

    succ = np.array([t["ok"] for t in trajs])
    ecc = np.array([t["r"] for t in trajs])
    log(f"[wide] success {succ.mean():.2f} | mean |states| {np.mean([len(t['S']) for t in trajs]):.0f}")

    def blk(t):
        log("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)

    for name in ("raw", "latent"):
        r, _ = spaces[name]
        blk(f"[{name}] (1) SEPARATION + (2) RETURN-TO-SUPPORT   thr(p99.5)={thr[name]['p995']:.4f}")
        # novelty over all states of every trajectory
        nov, tag_fail, tag_ecc, pair = [], [], [], []
        for t in trajs:
            S = torch.from_numpy(t["S"]).to(dev)
            X = wh(name, S if name == "raw" else enc(S))
            u = knn_dist(X, r, args.k).cpu().numpy()
            nov.append(u)
            tag_fail.append(np.full(len(u), 1.0 - t["ok"]))
            tag_ecc.append(np.full(len(u), t["r"]))
            if len(u) > H:
                pair.append(np.stack([u[:-H], u[H:]]))         # (u_t, u_{t+H})
        nov = np.concatenate(nov); tag_fail = np.concatenate(tag_fail)
        tag_ecc = np.concatenate(tag_ecc); pair = np.concatenate(pair, axis=1)
        T = thr[name]["p995"]
        for q in ("p90", "p99", "p995"):
            Tq = thr[name][q]
            log(f"  thr={q} ({Tq:.3f}): frac>thr overall {np.mean(nov > Tq):.3f}" + "".join(
                f" | ecc {lo:.2f}-{hi:.2f}: {np.mean(nov[(tag_ecc>=lo)&(tag_ecc<hi)] > Tq):.3f}"
                for lo, hi in [(0.0, 0.48), (0.48, 0.74), (0.74, 0.89), (0.89, 1.01)]
                if ((tag_ecc >= lo) & (tag_ecc < hi)).sum()))
        log(f"  novelty: mean {nov.mean():.4f} | frac above thr {np.mean(nov > T):.3f}")
        log(f"  (1) AUC(novelty; states of FAILED eps > states of SUCCESS eps) = "
            f"{auc(nov[tag_fail > 0], nov[tag_fail == 0]):+.3f}")
        for lo, hi in [(0.0, 0.48), (0.48, 0.74), (0.74, 0.89), (0.89, 1.01)]:
            m = (tag_ecc >= lo) & (tag_ecc < hi)
            if m.sum():
                log(f"      eccentricity {lo:.2f}-{hi:.2f}: novelty {nov[m].mean():.4f}, "
                    f"frac>thr {np.mean(nov[m] > T):.3f}  (n={int(m.sum())} states)")
        ood = pair[0] > T
        if ood.sum():
            log(f"  (2) among OOD states (n={int(ood.sum())}):")
            log(f"      frac with novelty DECREASING over H steps = {np.mean(pair[1][ood] < pair[0][ood]):.3f}")
            log(f"      frac that fall BACK INSIDE the threshold  = {np.mean(pair[1][ood] <= T):.3f}"
                f"   <- high => V_frozen is a usable bootstrap (small lambda ok)")
        else:
            log("  (2) no OOD states above threshold -- the gate would never open (bad sign)")

    blk("VERDICT (gate viability)")
    log("  need: (1) AUC clearly > 0.5 and novelty rising with eccentricity;")
    log("        (2) a decent return-to-support fraction, else push lambda -> 1 (MC anchor).")
    log("  if (1) fails the gate is broken -- fix the support measure before anything else.")
    os._exit(0)


if __name__ == "__main__":
    main()
