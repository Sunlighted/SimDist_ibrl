"""THE decisive measurement for "when is value adapt actually necessary".

Established so far:
  * a tipped-over tripod is genuinely IRREVERSIBLE (scripts/diag_tripod_trap_causal.py:
    p_recover = 0.00 from tipped states vs CONTROL 0.29-0.53 from matched non-tipped
    states), i.e. it is a real trap and not a symptom of an already-doomed episode;
  * with the correctly-loaded source expert (IBRL prior attached -- load_frozen_agent
    silently drops it, see train_rl.attach_ibrl_prior) the tripod NEVER tips in source
    (0/20 episodes, max tilt 2.8 deg) but tips in the SS12 target;
  * the tripod knobs cannot produce a "pure reachability" domain: any change that makes
    an 18 g free body easier to disturb also makes it slide, and strict_success gates on
    max_tripod_disp=0.04 (scan: displacement 0.010 -> 0.672 m).

But the decisive question does NOT need a clean domain -- it only needs PAIRED states:

    does the frozen source value score a TIPPED state below a matched UPRIGHT one?

  AUC ~ 0.5  => the value is structurally BLIND to an irreversible state. Then value
               adapt IS necessary (and the right form is a GATED residual that only
               pushes the trap region down -- not the global re-fit that SS33 killed).
  AUC >> 0.5 => the frozen value already knows tipping is bad; adapting it stays
               unnecessary and SS33's conclusion extends to reachability shifts.

Pairing controls for progress: for every tipped episode we take the state right after
the tilt crosses the threshold, and pair it with a state from a NON-tipped episode at
the SAME step index (so both are "equally far along" in time). Values are read from the
SimDist world model's value head -- the function the planner actually scores with --
both at encode(s) and, optionally, at the rolled latent the planner really queries.

Run:
  python scripts/diag_value_trap_blindness.py --episodes 40
"""
import argparse
import os
import sys

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from env.domain_shift import DomainShiftConfig  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.strict_success import StrictSuccessConfig  # noqa: E402

ERR = sys.stderr
TRIPOD_QUAT = slice(9 + 17, 9 + 21)
TRIPOD_POS = slice(9 + 14, 9 + 17)
NEEDLE_POS = slice(9 + 0, 9 + 3)     # object-state[0:3]


def needle_ring_dist(s):
    return float(np.linalg.norm(s[NEEDLE_POS] - s[TRIPOD_POS]))


def log(*a):
    print(*a, file=ERR, flush=True)


def quat_tilt_deg(q_now, q_ref):
    def zaxis(q):
        w, x, y, z = q
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    a, b = zaxis(q_now), zaxis(q_ref)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def auc(pos, neg):
    """P(score(pos) > score(neg)) -- rank-based, ties count 1/2."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt_run", default="exps/mimicgen/threading_simdist_dynens_adapt_long_h5_seed1",
                    help="SimDist run whose FROZEN value head is the one under test")
    ap.add_argument("--ckpt", default="model0.pt")
    ap.add_argument("--agent", default="exps/mimicgen/threading_ibrl_source_strict_seed1/model0.pt",
                    help="policy used to GENERATE the states (its prior is re-attached)")
    ap.add_argument("--trap_mass", type=float, default=1.0)
    ap.add_argument("--trap_com", type=float, default=-4.0)
    ap.add_argument("--use_target_shift", type=int, default=0,
                    help="1 = SS12 target + trap knobs; 0 = nominal physics + trap knobs")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--tilt_deg", type=float, default=30.0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    from train_simdist import MainConfig, Workspace
    from train_simdist_adapt import AdaptConfig

    # ---- the frozen value head under test (the deployed SimDist system) ----
    acfg: AdaptConfig = pyrallis.load(AdaptConfig, open(  # type: ignore
        os.path.join(args.adapt_run, "cfg.yaml")))
    scfg: MainConfig = pyrallis.load(MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(acfg.stage1_ckpt), "cfg.yaml")))
    scfg.use_wb = 0
    ws = Workspace(scfg, from_main=False)
    ws.agent.load_state_dict(torch.load(os.path.join(args.adapt_run, args.ckpt)))
    wm = ws.agent.to(dev).eval().model

    @torch.no_grad()
    def V(states):                       # [B,37] -> [B]
        return wm.predict_value(wm.encode(states.to(dev)), target=False).view(-1).cpu().numpy()

    # ---- env with the trap knobs ----
    if args.use_target_shift:
        ds = DomainShiftConfig(
            mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
            solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
            mask_hole_offset_in_obs=False, ring_size_scale=0.85, ring_friction_scale=2.0,
            tripod_mass_scale=args.trap_mass, tripod_com_height_scale=args.trap_com)
    else:
        ds = DomainShiftConfig(mode="fixed", tripod_mass_scale=args.trap_mass,
                               tripod_com_height_scale=args.trap_com)
    env = PixelRobosuite(
        env_name="Threading_D0", robots="Panda", episode_length=400, use_state=True,
        rl_cameras=[], camera_names=[], device="cuda", domain_shift=ds,
        strict_success=StrictSuccessConfig(enabled=True),
        hand_init_perturb_joint_sigma=args.sigma)

    cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.agent), "cfg.yaml")))
    agent = train_rl.load_frozen_agent(args.agent, env, cfg, dev)
    desc = train_rl.attach_ibrl_prior(agent, args.agent, env, dev)   # SS17/SS27: prior!
    agent.cfg.act_method = "ibrl" if desc else "rl"
    log(f"[setup] value head <- {args.adapt_run}/{args.ckpt}")
    log(f"        states    <- {args.agent} (prior: {desc or 'none'}, {agent.cfg.act_method})")
    log(f"        trap: mass={args.trap_mass} com_h={args.trap_com} "
        f"target_shift={args.use_target_shift} | tilt>{args.tilt_deg} | {args.episodes} eps")

    tipped, upright = [], []      # (step, state, tilt)
    n_tip_ep = 0
    for e in range(args.episodes):
        np.random.seed(args.seed + e)
        torch.manual_seed(args.seed + e)
        obs, _ = env.reset()
        q_ref = obs["state"][TRIPOD_QUAT].cpu().numpy().copy()
        traj, fired = [], False
        for t in range(400):
            with torch.no_grad():
                a = agent.act(obs, eval_mode=True)
            obs, reward, terminal, _, _ = env.step(a)
            s = obs["state"].cpu().numpy().copy()
            tilt = quat_tilt_deg(s[TRIPOD_QUAT], q_ref)
            traj.append((t, s, tilt))
            if tilt > args.tilt_deg and not fired:
                fired = True
                tipped.append((t, s, tilt))      # first frame past the threshold
            if terminal:
                break
        if fired:
            n_tip_ep += 1
        else:
            upright.extend(traj[::10])           # subsample the whole clean trajectory
    log(f"\n[data] {n_tip_ep}/{args.episodes} episodes tipped | "
        f"{len(tipped)} tipped states, {len(upright)} upright states")
    if not tipped:
        log("  no tipped states -> raise the trap knobs or the episode count")
        os._exit(0)

    # progress-matched pairing: for each tipped state at step t, use upright states
    # within +-15 steps of t
    v_tip = V(torch.from_numpy(np.stack([s for _, s, _ in tipped])).float())
    up_steps = np.array([t for t, _, _ in upright])
    up_states = np.stack([s for _, s, _ in upright])
    v_up_all = V(torch.from_numpy(up_states).float())

    up_d = np.array([needle_ring_dist(s) for _, s, _ in upright])
    tip_d = np.array([needle_ring_dist(s) for _, s, _ in tipped])

    matched_pos, matched_neg = [], []          # time-matched
    dmatched_pos, dmatched_neg = [], []        # needle-ring-DISTANCE matched (the real control)
    for i, (t, _, _) in enumerate(tipped):
        m = np.abs(up_steps - t) <= 15
        if m.sum():
            matched_pos.append(v_tip[i])
            matched_neg.append(float(v_up_all[m].mean()))
        dm = np.abs(up_d - tip_d[i]) <= 0.01   # within 1 cm of the same proximity
        if dm.sum():
            dmatched_pos.append(v_tip[i])
            dmatched_neg.append(float(v_up_all[dm].mean()))

    def blk(t):
        log("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

    blk("does the FROZEN value score a TIPPED (irreversible) state below an UPRIGHT one?")
    log(f"  V(tipped)        mean {v_tip.mean():+.4f}  std {v_tip.std():.4f}  n={len(v_tip)}")
    log(f"  V(upright, all)  mean {v_up_all.mean():+.4f}  std {v_up_all.std():.4f}  n={len(v_up_all)}")
    log(f"  AUC(upright > tipped), unmatched        = {auc(v_up_all, v_tip):.3f}")
    if matched_pos:
        mp, mn = np.array(matched_pos), np.array(matched_neg)
        d = mn - mp
        log(f"  progress-MATCHED (|dt|<=15 steps), n={len(mp)}")
        log(f"    V(upright)-V(tipped) = {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f}"
            f"   (z={d.mean()/max(d.std(ddof=1)/np.sqrt(len(d)),1e-12):+.2f})")
        log(f"    AUC(matched upright > tipped)      = {auc(mn, mp):.3f}")
    if dmatched_pos:
        mp, mn = np.array(dmatched_pos), np.array(dmatched_neg)
        d = mn - mp
        log(f"  ** needle-ring-DISTANCE matched (|dd|<=1cm), n={len(mp)}  <- controls for "
            f"'tipping happens when the needle is closest'")
        log(f"    V(upright)-V(tipped) = {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f}"
            f"   (z={d.mean()/max(d.std(ddof=1)/np.sqrt(len(d)),1e-12):+.2f})")
        log(f"    AUC(matched upright > tipped)      = {auc(mn, mp):.3f}")
    log(f"  needle-ring dist: tipped {tip_d.mean():.4f} vs upright {up_d.mean():.4f}")
    log(f"\n  planner signal for scale: cross-candidate value spread ~0.0068 (H=3) / "
        f"0.0111 (H=5)")

    blk("VERDICT")
    a_m = auc(np.array(dmatched_neg), np.array(dmatched_pos)) if dmatched_pos else (
        auc(np.array(matched_neg), np.array(matched_pos)) if matched_pos else np.nan)
    if np.isnan(a_m):
        log("  not enough matched pairs")
    elif a_m < 0.65:
        log(f"  AUC={a_m:.3f} ~ chance => the frozen value is BLIND to an IRREVERSIBLE state.")
        log("  This is the regime where value adapt IS necessary -- and the right form is a")
        log("  GATED residual that only pushes the trap region down, NOT the global re-fit")
        log("  that SS33 measured at -0.26.")
    else:
        log(f"  AUC={a_m:.3f} >> chance => the frozen value ALREADY scores the trap lower.")
        log("  Even under a reachability change it is not blind, so SS33's 'do not adapt")
        log("  value' extends here too; the gap must be closed elsewhere (policy/dynamics).")
    os._exit(0)


if __name__ == "__main__":
    main()
