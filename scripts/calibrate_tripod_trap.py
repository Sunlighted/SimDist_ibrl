"""Calibrate the tripod TIP-OVER trap (CLAUDE.md §33 follow-up).

Every Threading shift so far only makes the task uniformly HARDER, which leaves
the state ORDERING intact -- that is exactly why a frozen source value head keeps
working and only dynamics needs adapting (§33: adapting value costs -0.26 and buys
nothing). The one regime where value adapt should be NECESSARY is a shift that
changes REACHABILITY: states that are fine in source but from which success is
impossible in target. A tipped-over tripod is that, and it is OBSERVABLE
(tripod_quat lives at obs["object-state"][17:21] == state[26:30]) -- an
unobservable trap could not be fixed by ANY value function, so observability is a
hard requirement of the design.

A usable trap must satisfy FOUR conditions; this script measures the first two
(the cheap ones) so a bad recipe is thrown out before any training is run:

  (1) irreversible in TARGET     -> tip-over rate > 0 and those episodes fail
  (2) ~absent in SOURCE          -> tip rate ~ 0 with the SAME policy
  (3) observable                 -> by construction (tripod_quat in obs)
  (4) the planner can steer into/away from it  -> tested later, in the planner

Rolls the frozen expert (default: the source IBRL agent) under a grid of
(tripod_mass_scale, tripod_com_height_scale) and reports, per cell:
    tip rate, success rate, and the joint table success|tipped vs success|upright.

The tip test is the ANGLE of the tripod's z-axis from vertical, computed from
tripod_quat; `--tilt_deg` sets the threshold. Note the nominal tripod quat is
(0.7071, 0, 0, 0.7071), i.e. already rotated 90 deg about x, so "upright" is
measured RELATIVE TO THE RESET POSE, not to the world z-axis.

Run:
  python scripts/calibrate_tripod_trap.py --episodes 30 \
      --grid "1.0:1.0,0.3:1.0,0.3:0.0,0.3:-1.0,0.15:-1.0,0.15:-2.0"
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
# state layout (37-d): object-state starts at 9; tripod_quat is object-state[17:21]
TRIPOD_QUAT = slice(9 + 17, 9 + 21)
TRIPOD_POS = slice(9 + 14, 9 + 17)   # object-state[14:17]


def log(*a):
    print(*a, file=ERR, flush=True)


def quat_tilt_deg(q_now, q_ref):
    """Angle between the two frames' local z axes, in degrees. q = (w,x,y,z)."""
    def zaxis(q):
        w, x, y, z = q
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    a, b = zaxis(q_now), zaxis(q_ref)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def make_env(target: bool, mass_s, com_s, sigma, strict):
    """target=False -> the §12 dynamics-only TARGET shift is NOT applied (source)."""
    if target:
        ds = DomainShiftConfig(
            mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
            solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
            mask_hole_offset_in_obs=False, ring_size_scale=0.85,
            ring_friction_scale=2.0,
            tripod_mass_scale=mass_s, tripod_com_height_scale=com_s)
    else:
        # SOURCE physics, but the SAME trap knobs, so condition (2) is tested
        # against the trap itself and not against "source is just easier".
        ds = DomainShiftConfig(mode="fixed", tripod_mass_scale=mass_s,
                               tripod_com_height_scale=com_s)
    return PixelRobosuite(
        env_name="Threading_D0", robots="Panda", episode_length=400,
        use_state=True, rl_cameras=[], camera_names=[], device="cuda",
        domain_shift=ds, strict_success=StrictSuccessConfig(enabled=bool(strict)),
        hand_init_perturb_joint_sigma=sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="exps/mimicgen/threading_ibrl_source_strict_seed1/model0.pt",
                    help="use the STRICT-trained source agent: the non-strict one was never "
                         "penalised for shoving the tripod (max_tripod_disp), so its tip rate "
                         "reflects its own sloppiness rather than the environment")
    ap.add_argument("--grid", default="1.0:1.0,0.3:1.0,0.3:0.0,0.3:-1.0,0.15:-1.0,0.15:-2.0",
                    help="comma list of mass_scale:com_height_scale")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tilt_deg", type=float, default=30.0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--strict", type=int, default=1)
    ap.add_argument("--domains", default="source,target",
                    help="'source' = NOMINAL physics + the trap knobs (the clean TRAP-ONLY "
                         "domain: reachability changes, contact difficulty does not); "
                         "'target' = the SS12 dynamics-only target + the trap knobs")
    ap.add_argument("--spontaneous", type=int, default=1,
                    help="also roll ZERO actions to check the tripod does not tip by itself "
                         "(a self-tipping env is broken, not a trap)")
    ap.add_argument("--bare_head", type=int, default=0,
                    help="1 = do NOT re-attach the IBRL prior (bare RL actor); for ablation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    cells = [tuple(float(x) for x in c.split(":")) for c in args.grid.split(",")]
    log(f"[setup] {len(cells)} cells x {args.episodes} eps x 2 domains | "
        f"tilt>{args.tilt_deg} deg = tipped | sigma={args.sigma} | strict={args.strict}")
    log(f"        agent = {args.agent}")

    # one env just to build the frozen agent (shapes only)
    probe = make_env(True, 1.0, 1.0, args.sigma, args.strict)
    cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.agent), "cfg.yaml")))
    agent = train_rl.load_frozen_agent(args.agent, probe, cfg, dev)
    # CRITICAL (§17/§27): load_frozen_agent pins act_method="rl", which DROPS the
    # IBRL prior the agent was trained with. This ckpt trained as act_method=ibrl
    # (max-Q over {RL head, flow-BC prior}) and scored 0.90 in-training; the bare
    # RL head alone only gets ~0.44. Re-attach the prior and restore the hybrid.
    if not args.bare_head:
        desc = train_rl.attach_ibrl_prior(agent, args.agent, probe, dev)
        agent.cfg.act_method = "ibrl" if desc else "rl"
        log(f"        prior  = {desc or '(none declared -> bare RL head)'} "
            f"| act_method={agent.cfg.act_method}")

    log(f"\n{'mass':>6}{'com_h':>7}{'domain':>9}{'tip%':>8}{'succ%':>8}"
        f"{'succ|tip':>10}{'succ|up':>9}{'maxtilt':>9}{'maxdisp':>9}{'disp>4cm':>10}")
    rows = []
    for mass_s, com_s in cells:
        if args.spontaneous:
            for dom in args.domains.split(","):
                env = make_env(dom == "target", mass_s, com_s, args.sigma, args.strict)
                mx = 0.0
                for e in range(3):
                    np.random.seed(1000 + e); torch.manual_seed(1000 + e)
                    obs, _ = env.reset()
                    q_ref = obs["state"][TRIPOD_QUAT].cpu().numpy().copy()
                    for _ in range(150):
                        obs, _, term, _, _ = env.step(torch.zeros(env.action_dim))
                        mx = max(mx, quat_tilt_deg(obs["state"][TRIPOD_QUAT].cpu().numpy(), q_ref))
                        if term:
                            break
                flag = "  <-- SELF-TIPPING, env is broken" if mx > args.tilt_deg else ""
                log(f"  [spontaneous] mass={mass_s} com_h={com_s} {dom}: "
                    f"max tilt under ZERO actions = {mx:.1f} deg{flag}")
    for mass_s, com_s in cells:
        for target in [d == "target" for d in args.domains.split(",")]:
            env = make_env(target, mass_s, com_s, args.sigma, args.strict)
            tips, succs, maxtilts, disps = [], [], [], []
            for e in range(args.episodes):
                np.random.seed(args.seed + e)
                torch.manual_seed(args.seed + e)
                obs, _ = env.reset()
                q_ref = obs["state"][TRIPOD_QUAT].cpu().numpy().copy()
                p_ref = obs["state"][TRIPOD_POS].cpu().numpy().copy()
                tipped, succ, mx, mdisp = 0.0, 0.0, 0.0, 0.0
                for t in range(400):
                    with torch.no_grad():
                        a = agent.act(obs, eval_mode=True)
                    obs, reward, terminal, success, _ = env.step(a)
                    tilt = quat_tilt_deg(obs["state"][TRIPOD_QUAT].cpu().numpy(), q_ref)
                    mx = max(mx, tilt)
                    mdisp = max(mdisp, float(np.linalg.norm(
                        obs["state"][TRIPOD_POS].cpu().numpy() - p_ref)))
                    if tilt > args.tilt_deg:
                        tipped = 1.0
                    if reward > 0:
                        succ = 1.0
                    if terminal:
                        break
                tips.append(tipped); succs.append(succ); maxtilts.append(mx)
                disps.append(mdisp)
            tips, succs = np.array(tips), np.array(succs)
            s_tip = succs[tips > 0].mean() if (tips > 0).any() else np.nan
            s_up = succs[tips == 0].mean() if (tips == 0).any() else np.nan
            dom = "target" if target else "source"
            log(f"{mass_s:>6.2f}{com_s:>7.2f}{dom:>9}{100*tips.mean():>8.1f}"
                f"{100*succs.mean():>8.1f}{100*s_tip:>10.1f}{100*s_up:>9.1f}"
                f"{np.mean(maxtilts):>9.1f}{np.mean(disps):>9.3f}"
                f"{100*np.mean(np.array(disps) > 0.04):>10.0f}")
            rows.append((mass_s, com_s, dom, tips.mean(), succs.mean(), s_tip, s_up))
    log("\n  a USABLE trap cell needs: target tip% clearly > 0, source tip% ~ 0,")
    log("  and succ|tip ~ 0 (tipping really is unrecoverable).")
    log("  Then difficulty-match a NO-trap shift to the same overall succ% before")
    log("  comparing 'dyn-only adapt' vs 'dyn + trap-aware value'.")
    os._exit(0)


if __name__ == "__main__":
    main()
