"""VALUE-ORDER INVERSION test on the JAM trap (the user's insight, and the design that
fixes what the tripod trap got wrong).

Why the tripod trap failed as a test of value blindness: tipping the tripod CHANGES THE
TASK GEOMETRY (tripod_pos/quat), and the frozen value already tracks that geometry, so it
scored tipped states 0.35 below matched upright ones (AUC 0.92) -- it did not need to have
"seen" tipping.

The jam trap is structurally different: a half-inserted needle has the SAME geometry in
both domains (probe: min needle-ring distance 0.122 nominal vs 0.124 jam, deep-reach rate
86.7% in both) but a completely different VALUE -- in source you just push through, in the
jam domain you are stuck and must retract, realign and retry. So the value cannot tell them
apart from any observable geometric quantity: the geometry really is identical, only the
value of that geometry changed. That is value-ORDER INVERSION, and unlike irreversibility
it does not require the state to be unrecoverable, only expensive relative to the horizon.

Two things are measured together, because one of them is a hard FEASIBILITY GATE:

  (A) FEASIBILITY: is the jam domain still solvable? From a pre-insertion state (needle
      approaching, not yet deep), restart K times with action noise. If p_success is ~0 for
      EVERY such state, the domain is unsolvable and no value function could help -- the
      experiment would be meaningless. The source expert scoring 0% is NOT sufficient
      evidence of unsolvability (it was never optimised for this domain).

  (B) INVERSION: at DEEP states (needle at the hole) with matched geometry, compare the
      frozen value V(s) against the ground-truth p_success(s) measured by restarting in the
      same domain. Blindness looks like:
          source: V high, p_success high      (calibrated)
          jam:    V high, p_success ~ 0       (inverted -> the value is structurally fooled)

Run:
  python scripts/diag_jam_value_inversion.py --episodes 25 --restarts 8
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
NEEDLE, RING = slice(9, 12), slice(9 + 14, 9 + 17)


def log(*a):
    print(*a, file=ERR, flush=True)


def d_nr(s):
    return float(np.linalg.norm(s[NEEDLE] - s[RING]))


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


@torch.no_grad()
def restart_success(env, agent, sim_state, steps, stddev):
    restore(env, sim_state)
    obs, _ = env._extract_images(env.env._get_observations())
    for _ in range(steps):
        a = agent.act(obs, eval_mode=True)
        if stddev > 0:
            a = (a + torch.randn_like(a) * stddev).clamp(-1, 1)
        obs, reward, terminal, _, _ = env.step(a)
        if reward > 0:
            return 1.0
        if terminal:
            break
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt_run", default="exps/mimicgen/threading_simdist_dynens_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="model0.pt")
    ap.add_argument("--agent", default="exps/mimicgen/threading_ibrl_source_strict_seed1/model0.pt")
    ap.add_argument("--ring_friction", type=float, default=5.0)
    ap.add_argument("--ring_size", type=float, default=0.85)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--restart_stddev", type=float, default=0.1,
                    help="action noise on restarts: FEASIBILITY needs varied attempts, "
                         "otherwise every replay repeats the same failure")
    ap.add_argument("--deep", type=float, default=0.14)
    ap.add_argument("--approach", type=float, default=0.20,
                    help="pre-insertion band (needle approaching): feasibility snapshots")
    ap.add_argument("--max_snap", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.1)
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

    @torch.no_grad()
    def V(states):
        return wm.predict_value(wm.encode(states.to(dev)), target=False).view(-1).cpu().numpy()

    def make(jam):
        ds = (DomainShiftConfig(mode="fixed", ring_friction_scale=args.ring_friction,
                                ring_size_scale=args.ring_size, mask_hole_offset_in_obs=False)
              if jam else DomainShiftConfig(mode="fixed"))
        return PixelRobosuite(
            env_name="Threading_D0", robots="Panda", episode_length=400, use_state=True,
            rl_cameras=[], camera_names=[], device="cuda", domain_shift=ds,
            strict_success=StrictSuccessConfig(enabled=True),
            hand_init_perturb_joint_sigma=args.sigma)

    probe = make(False)
    cfg = pyrallis.load(train_rl.MainConfig, open(  # type: ignore
        os.path.join(os.path.dirname(args.agent), "cfg.yaml")))
    agent = train_rl.load_frozen_agent(args.agent, probe, cfg, dev)
    desc = train_rl.attach_ibrl_prior(agent, args.agent, probe, dev)
    agent.cfg.act_method = "ibrl" if desc else "rl"
    log(f"[setup] value <- {args.adapt_run}/{args.ckpt} | policy prior: {desc or 'none'}")
    log(f"        jam = ring_friction {args.ring_friction} x ring_size {args.ring_size} | "
        f"{args.episodes} eps, {args.restarts} restarts (noise {args.restart_stddev})")

    out = {}
    for jam in (False, True):
        env = make(jam)
        deep_snaps, appr_snaps, succ = [], [], []
        for e in range(args.episodes):
            np.random.seed(args.seed + e)
            torch.manual_seed(args.seed + e)
            obs, _ = env.reset()
            got_deep = got_appr = False
            ok = 0.0
            for t in range(400):
                with torch.no_grad():
                    a = agent.act(obs, eval_mode=True)
                obs, reward, terminal, _, _ = env.step(a)
                s = obs["state"].cpu().numpy().copy()
                d = d_nr(s)
                ss = env.env.sim.get_state().flatten().copy()
                if not got_appr and args.deep <= d < args.approach:
                    got_appr = True
                    appr_snaps.append((ss, s, t, d))
                if not got_deep and d < args.deep:
                    got_deep = True
                    deep_snaps.append((ss, s, t, d))
                if reward > 0:
                    ok = 1.0
                if terminal:
                    break
            succ.append(ok)

        def measure(snaps, tag):
            snaps = snaps[: args.max_snap]
            if not snaps:
                return None
            ps = [np.mean([restart_success(env, agent, ss, 400 - t, args.restart_stddev)
                           for _ in range(args.restarts)]) for ss, _, t, _ in snaps]
            vs = V(torch.from_numpy(np.stack([s for _, s, _, _ in snaps])).float())
            ds = [d for _, _, _, d in snaps]
            return dict(tag=tag, n=len(snaps), p=float(np.mean(ps)),
                        v=float(np.mean(vs)), d=float(np.mean(ds)),
                        p_any=float(np.max(ps)))

        dom = "jam" if jam else "source"
        out[dom] = dict(succ=float(np.mean(succ)),
                        deep=measure(deep_snaps, "deep"),
                        appr=measure(appr_snaps, "approach"))
        log(f"  [{dom}] episode success {np.mean(succ):.2f} | "
            f"deep snaps {len(deep_snaps)}, approach snaps {len(appr_snaps)}")

    def blk(t):
        log("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    blk("(A) FEASIBILITY GATE -- is the jam domain still solvable?")
    log(f"{'domain':>8}{'episode succ':>14}{'p_success @approach':>21}{'best snapshot':>15}")
    for dom in ("source", "jam"):
        a = out[dom]["appr"]
        if a:
            log(f"{dom:>8}{out[dom]['succ']:>14.2f}{a['p']:>21.2f}{a['p_any']:>15.2f}")
    log("  jam p_success @approach ~ 0 AND best snapshot 0 => UNSOLVABLE, lower the friction.")
    log("  > 0 => solvable; the expert just is not aligned enough, which is the point.")

    blk("(B) VALUE-ORDER INVERSION at DEEP states (geometry matched by construction)")
    log(f"{'domain':>8}{'needle-ring d':>15}{'V_frozen':>11}{'true p_success':>16}")
    for dom in ("source", "jam"):
        d = out[dom]["deep"]
        if d:
            log(f"{dom:>8}{d['d']:>15.3f}{d['v']:>11.4f}{d['p']:>16.2f}")
    s, j = out["source"]["deep"], out["jam"]["deep"]
    if s and j:
        log(f"\n  dV  = {j['v'] - s['v']:+.4f}   (planner discriminates candidates at ~0.0111)")
        log(f"  dP  = {j['p'] - s['p']:+.2f}")
        blk("VERDICT")
        if j["p"] < 0.25 and abs(j["v"] - s["v"]) < 0.05:
            log("  INVERSION CONFIRMED: same geometry, same value, but the jam state is")
            log("  actually doomed. The frozen value is structurally blind here -- unlike the")
            log("  tripod trap, no observable geometric cue separates the two.")
        elif j["v"] < s["v"] - 0.05:
            log("  the frozen value ALREADY scores the jam state lower -- not blind after all.")
            log("  (check what observable it is using: needle orientation? velocity?)")
        else:
            log("  inconclusive -- see the numbers above.")
    os._exit(0)


if __name__ == "__main__":
    main()
