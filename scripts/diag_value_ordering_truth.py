"""Detector-free test: is V_source's ORDERING wrong in the window where the damage happens?

Why this exists. In wide1.5 the oracle arm loses ~0.095 entirely in the annulus (core 1.000
n=121 vs annulus 0.905 n=179, against a 0.987 ceiling). I attributed that to OOD value
blindness, but the 10-head value ensemble showed annulus/core disagreement ~1.0x over
on-policy trajectories (only 1.58x at the reset frame), and I concluded "the OOD lasts one
frame". That conclusion silently assumed the ensemble is a valid OOD detector -- it is only
a weak one (absolute disagreement 0.001-0.007, §29 documented head collapse; its far-OOD
control needs ~0.4 sigma of full-state noise to reach the 1.58x seen at reset). So
"ann/core ~ 1.0" cannot distinguish "no OOD" from "OOD below the detector's sensitivity".

This measures the thing itself, with no detector in the loop. The takeover experiment
(scripts/diag_when_damage_happens.py) showed p_recover falls from 0.94 at t=0 to 0.60 at
t=200 in failing annulus episodes, i.e. the damage accrues over t~40-200. In that window we
ask what MPPI actually needs:

    at state s_t, take K candidate action sequences, roll them H steps in the REAL sim,
    and for each terminal measure the TRUE value = p_recover (success rate when the 0.987
    wide-trained expert takes over from there). Then compare the true ordering with
    V_source's ordering.

  Spearman(V_source, p_recover) across candidates ~ 0  => the value's ordering IS broken in
      the damage window, and the ensemble missed it => the detector is insensitive, not the
      OOD absent. Then a better gate (kNN / subspace / genuinely diverse heads) is the fix.
  Spearman clearly > 0                              => the value ranks fine where the damage
      happens => the deficit is not the value's, and the gated-residual plan should stop.

The expert's own value is reported alongside as a reference ordering (it is a different
function, trained in this domain, though with narrow on-policy coverage).

Run:
  python scripts/diag_value_ordering_truth.py --episodes 40 --cands 6 --restarts 2
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr
BOX_C, BOX_HW = np.array([-0.125, 0.2]), np.array([0.075, 0.05])


def log(*a):
    print(*a, file=ERR, flush=True)


def _rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(len(x))
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut_run", default="exps/mimicgen/threading_simdist_dynens_adapt_wide1.5_seed1")
    ap.add_argument("--sut_ckpt", default="model0.pt")
    ap.add_argument("--expert", default="exps/mimicgen/threading_ibrl_target_wide15_seed1/model0.pt")
    ap.add_argument("--needle_scale", type=float, default=1.5)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--probe_at", default="110,170,230",
                    help="steps inside the damage window (p_recover falls 0.81->0.60 over 40-200)")
    ap.add_argument("--cands", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--cand_std", type=float, default=0.5,
                    help="action noise defining the candidate cloud around the executed plan")
    ap.add_argument("--max_ep", type=int, default=10, help="cap the cost")
    ap.add_argument("--annulus_only", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    from train_simdist import MainConfig, Workspace
    from train_simdist_adapt import AdaptConfig
    from env.domain_shift import InitDistConfig
    from env.robosuite_wrapper import PixelRobosuite

    a = pyrallis.load(AdaptConfig, open(os.path.join(args.sut_run, "cfg.yaml")))  # type: ignore
    s = pyrallis.load(MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(a.stage1_ckpt), "cfg.yaml")))
    s.use_wb = 0
    s.simdist.use_planner = 1
    from rl.simdist_planner import apply_original_recipe
    apply_original_recipe(s.simdist.planner)
    if a.eval_hand_init_perturb_joint_sigma is not None:
        s.eval_hand_init_perturb_joint_sigma = a.eval_hand_init_perturb_joint_sigma
    ws = Workspace(s, from_main=False)
    ws.agent.load_state_dict(torch.load(os.path.join(args.sut_run, args.sut_ckpt)))
    sut = ws.agent.to(dev).eval()
    wm = sut.model
    H = s.simdist.world_model.horizon

    nsc = args.needle_scale
    ep = dict(ws.eval_env_params)
    ep["init_distribution"] = InitDistConfig(
        enabled=True,
        needle_x=(BOX_C[0] - BOX_HW[0] * nsc, BOX_C[0] + BOX_HW[0] * nsc),
        needle_y=(BOX_C[1] - BOX_HW[1] * nsc, BOX_C[1] + BOX_HW[1] * nsc),
        needle_z_rot=(-2 * np.pi / 3, -np.pi / 3),
        tripod_x=(0.0, 0.0), tripod_y=(-0.15, -0.15), tripod_z_rot=(np.pi / 2, np.pi / 2))
    env = PixelRobosuite(**ep)

    ecfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.expert), "cfg.yaml")))
    expert = train_rl.load_frozen_agent(args.expert, env, ecfg, dev)
    edesc = train_rl.attach_ibrl_prior(expert, args.expert, env, dev)
    expert.cfg.act_method = "ibrl" if edesc else "rl"
    log(f"[setup] H={H} | needle box x{nsc} | expert prior={edesc or 'none'} | "
        f"probe at {args.probe_at}, {args.cands} candidates, {args.restarts} restarts")

    @torch.no_grad()
    def V_src(states):
        return wm.predict_value(wm.encode(states.to(dev)), target=False).view(-1).cpu().numpy()

    @torch.no_grad()
    def V_exp(states):
        return expert.value({"state": states.to(dev)}, use_ibrl=bool(edesc)).view(-1).cpu().numpy()

    GAMMA = s.discount if hasattr(s, "discount") else 0.99

    @torch.no_grad()
    def expert_rollout_success(sim_state, budget):
        """GRADED true value: gamma^(steps the expert needs), 0 if it never succeeds."""
        restore(env, sim_state)
        obs, _ = env._extract_images(env.env._get_observations())
        for k in range(budget):
            aa = expert.act(obs, eval_mode=True)
            obs, reward, terminal, _, _ = env.step(aa)
            if reward > 0:
                return float(GAMMA ** (k + 1))
            if terminal:
                break
        return 0.0

    probes = [int(v) for v in args.probe_at.split(",")]
    rows = []          # (t, sp_src, sp_exp, spread_src, true_spread)
    n_used = 0
    for e in range(args.episodes):
        if n_used >= args.max_ep:
            break
        np.random.seed(args.seed + e)
        torch.manual_seed(args.seed + e)
        obs, _ = env.reset()
        n0 = obs["state"][9:12].cpu().numpy()[:2]
        ecc = float(np.max(np.abs((n0 - BOX_C) / (BOX_HW * nsc))))
        if args.annulus_only and ecc <= 1.0 / nsc:
            continue
        traj = []
        ok = 0.0
        for t in range(400):
            with torch.no_grad():
                act = sut.act(obs, eval_mode=True)
            ss = env.env.sim.get_state().flatten().copy()
            traj.append((t, ss, act.clone()))
            obs, reward, terminal, _, _ = env.step(act)
            if reward > 0:
                ok = 1.0
            if terminal:
                break
        if ok > 0:                      # the damage window only exists in failures
            continue
        n_used += 1
        log(f"  ep{e} (ecc {ecc:.2f}, FAILED, len {len(traj)}): probing")
        for t0 in probes:
            if t0 >= len(traj):
                continue
            _, ss0, a0 = traj[t0]
            # candidate cloud around the executed action, rolled H steps in the REAL sim
            terms, sims = [], []
            for c in range(args.cands):
                restore(env, ss0)
                o, _ = env._extract_images(env.env._get_observations())
                for k in range(H):
                    aa = a0 if c == 0 else (a0 + torch.randn_like(a0) * args.cand_std).clamp(-1, 1)
                    o, _, term, _, _ = env.step(aa)
                    if term:
                        break
                terms.append(o["state"].cpu().numpy().copy())
                sims.append(env.env.sim.get_state().flatten().copy())
            T = torch.from_numpy(np.stack(terms)).float()
            vs, ve = V_src(T), V_exp(T)
            # TRUE value of each terminal: expert takeover success rate
            true = np.array([np.mean([expert_rollout_success(sm, 400 - t0 - H)
                                      for _ in range(args.restarts)]) for sm in sims])
            rows.append((t0, spearman(vs, true), spearman(ve, true),
                         float(np.std(vs)), float(np.std(true)), float(np.mean(true))))
            log(f"     t={t0:3d}  Sp(V_src,true)={rows[-1][1]:+.3f}  "
                f"Sp(V_exp,true)={rows[-1][2]:+.3f}  true={true.round(2).tolist()}")

    def blk(t):
        log("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    blk("ordering quality inside the damage window (true value = expert takeover success)")
    R = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    if len(R) == 0:
        log("  no probes collected"); os._exit(0)
    log(f"{'t':>6}{'n':>5}{'Sp(V_src,true)':>17}{'Sp(V_exp,true)':>17}"
        f"{'std(V_src)':>12}{'std(true)':>11}{'mean(true)':>12}")
    for t0 in probes:
        m = R[:, 0] == t0
        if m.sum():
            sp1 = np.nanmean(R[m, 1]); sp2 = np.nanmean(R[m, 2])
            log(f"{t0:>6}{int(m.sum()):>5}{sp1:>17.3f}{sp2:>17.3f}"
                f"{np.nanmean(R[m,3]):>12.4f}{np.nanmean(R[m,4]):>11.3f}{np.nanmean(R[m,5]):>12.3f}")
    sp_all = np.nanmean(R[:, 1])
    se = np.nanstd(R[:, 1], ddof=1) / max(np.sqrt(np.sum(~np.isnan(R[:, 1]))), 1)
    log(f"\n  ALL: Spearman(V_src, true) = {sp_all:+.3f} ± {se:.3f}  (n={len(R)} probe points)")
    log(f"       Spearman(V_exp, true) = {np.nanmean(R[:,2]):+.3f}")

    blk("VERDICT")
    if sp_all < 0.2:
        log(f"  V_source's ordering IS broken here ({sp_all:+.3f}): the value is wrong exactly")
        log("  where the damage accrues, and the ensemble did not see it => the DETECTOR is")
        log("  insensitive, not the OOD absent. A better gate is worth building.")
    elif sp_all > 0.5:
        log(f"  V_source ranks candidates well here ({sp_all:+.3f}) => the deficit is NOT the")
        log("  value's ordering. Stop the gated-residual line; look at dynamics / search.")
    else:
        log(f"  intermediate ({sp_all:+.3f}) -- compare against Sp(V_exp,true) and the spread")
        log("  columns before concluding; std(true) ~ 0 would mean the candidates are all")
        log("  equivalent and the probe carries no signal.")
    os._exit(0)


if __name__ == "__main__":
    main()
