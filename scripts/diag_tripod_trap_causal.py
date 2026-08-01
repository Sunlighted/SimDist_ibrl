"""Is a tipped tripod actually a TRAP, or just a symptom of an already-failing
episode?  (follow-up to scripts/calibrate_tripod_trap.py)

The calibration measured a CORRELATION: episodes where the tripod tilts past a
threshold almost never succeed. That is not enough -- three explanations fit:
  (1) real trap        : tipping causes the failure (irreversible)
  (2) reverse causation: the episode was already failing, the flailing arm then
                         happened to nudge the tripod
  (3) loose threshold  : 30 deg is just wobble, not a tip
and the source numbers (tip 20-33%, max tilt only 22-40 deg) look a lot like (2).

This measures IRREVERSIBILITY directly. Roll the frozen expert; at the FIRST
frame whose tilt crosses each threshold, snapshot sim_state and RESTART the
expert from it N times (same domain) -- if it never succeeds, that state really
is unrecoverable. The control is the matched-time question: from the SAME step
index in episodes that did NOT tip, restarting succeeds at rate p_ctrl. So

    trap  <=>  p_recover(tipped) ~ 0  AND  p_ctrl >> 0

If both are ~0, the state was doomed for reasons unrelated to the tripod
(explanation 2) and the tilt is a symptom, not a trap.

Reported per threshold and per domain: tip rate, p_recover from the tip state,
p_ctrl from the matched non-tipped state, and the tilt at the moment of crossing.

Run:
  python scripts/diag_tripod_trap_causal.py --episodes 25 --restarts 8
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


def log(*a):
    print(*a, file=ERR, flush=True)


def quat_tilt_deg(q_now, q_ref):
    def zaxis(q):
        w, x, y, z = q
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    a, b = zaxis(q_now), zaxis(q_ref)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def make_env(target, mass_s, com_s, sigma):
    if target:
        ds = DomainShiftConfig(
            mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
            solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
            mask_hole_offset_in_obs=False, ring_size_scale=0.85, ring_friction_scale=2.0,
            tripod_mass_scale=mass_s, tripod_com_height_scale=com_s)
    else:
        ds = DomainShiftConfig(mode="fixed", tripod_mass_scale=mass_s,
                               tripod_com_height_scale=com_s)
    return PixelRobosuite(
        env_name="Threading_D0", robots="Panda", episode_length=400, use_state=True,
        rl_cameras=[], camera_names=[], device="cuda", domain_shift=ds,
        strict_success=StrictSuccessConfig(enabled=True),
        hand_init_perturb_joint_sigma=sigma, record_sim_state=True)


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
def rollout_from(env, agent, sim_state, steps, stddev):
    """restart from sim_state, return 1.0 if the episode succeeds."""
    restore(env, sim_state)
    # rebuild the obs dict from the restored sim WITHOUT stepping (a no-op step
    # would advance the physics and change the very state we are testing)
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
    ap.add_argument("--agent", default="exps/mimicgen/threading_ibrl_source_strict_seed1/model0.pt",
                    help="use the STRICT-trained source agent: the non-strict one was never "
                         "penalised for shoving the tripod (max_tripod_disp), so its tip rate "
                         "reflects its own sloppiness rather than the environment")
    ap.add_argument("--cells", default="1.0:1.0,0.15:-2.0")
    ap.add_argument("--domains", default="source,target")
    ap.add_argument("--thresholds", default="30,45,60")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--restarts", type=int, default=8, help="restarts per snapshot")
    ap.add_argument("--restart_stddev", type=float, default=0.05,
                    help="action noise on restarts so the N replays are not identical")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--bare_head", type=int, default=0,
                    help="1 = do NOT re-attach the IBRL prior (bare RL actor); for ablation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    import train_rl
    ths = [float(x) for x in args.thresholds.split(",")]
    cells = [tuple(float(x) for x in c.split(":")) for c in args.cells.split(",")]
    doms = args.domains.split(",")

    probe = make_env(True, 1.0, 1.0, args.sigma)
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
    log(f"[setup] cells={cells} domains={doms} thresholds={ths} | {args.episodes} eps, "
        f"{args.restarts} restarts/snapshot (stddev {args.restart_stddev})")

    for mass_s, com_s in cells:
        for dom in doms:
            env = make_env(dom == "target", mass_s, com_s, args.sigma)
            # collect: per episode, the first crossing step + sim_state for each th,
            # plus a matched-time snapshot from non-tipping episodes
            snaps = {t: [] for t in ths}          # (sim_state, step)
            ctrl = []                             # (sim_state, step) from non-tipped eps
            outcome = []
            for e in range(args.episodes):
                np.random.seed(args.seed + e)
                torch.manual_seed(args.seed + e)
                obs, _ = env.reset()
                q_ref = obs["state"][TRIPOD_QUAT].cpu().numpy().copy()
                fired = {t: False for t in ths}
                traj = []                         # (step, sim_state) every 5 steps
                succ = 0.0
                for t in range(400):
                    with torch.no_grad():
                        a = agent.act(obs, eval_mode=True)
                    obs, reward, terminal, _, _ = env.step(a)
                    tilt = quat_tilt_deg(obs["state"][TRIPOD_QUAT].cpu().numpy(), q_ref)
                    ss = env.env.sim.get_state().flatten().copy()
                    if t % 5 == 0:
                        traj.append((t, ss))
                    for th in ths:
                        if not fired[th] and tilt > th:
                            fired[th] = True
                            snaps[th].append((ss, t, tilt))
                    if reward > 0:
                        succ = 1.0
                    if terminal:
                        break
                outcome.append(succ)
                if not fired[ths[0]] and traj:     # never tipped -> control snapshot
                    k = min(len(traj) - 1, max(1, len(traj) // 2))
                    ctrl.append((traj[k][1], traj[k][0]))

            # measure recoverability
            log(f"\n=== cell mass={mass_s} com_h={com_s} | domain={dom} | "
                f"episode success {np.mean(outcome):.2f} (n={len(outcome)})")
            log(f"{'thresh':>8}{'tip rate':>10}{'p_recover':>12}{'tilt@cross':>12}{'n':>5}")
            for th in ths:
                sn = snaps[th]
                if not sn:
                    log(f"{th:>8.0f}{0.0:>10.2f}{'-':>12}{'-':>12}{0:>5}")
                    continue
                rec = []
                for ss, t0, tilt in sn[: 12]:      # cap the cost
                    r = [rollout_from(env, agent, ss, 400 - t0, args.restart_stddev)
                         for _ in range(args.restarts)]
                    rec.append(np.mean(r))
                log(f"{th:>8.0f}{len(sn)/len(outcome):>10.2f}{np.mean(rec):>12.2f}"
                    f"{np.mean([s[2] for s in sn]):>12.1f}{len(sn):>5}")
            if ctrl:
                crec = []
                for ss, t0 in ctrl[: 12]:
                    r = [rollout_from(env, agent, ss, 400 - t0, args.restart_stddev)
                         for _ in range(args.restarts)]
                    crec.append(np.mean(r))
                log(f"{'CONTROL':>8}{'(no tip)':>10}{np.mean(crec):>12.2f}"
                    f"{'-':>12}{len(ctrl):>5}   <- matched-time restart from NON-tipped states")

    log("\n  TRAP  <=> p_recover(tipped) ~ 0  AND  CONTROL p_recover >> 0.")
    log("  both ~ 0 => the tilt is a SYMPTOM of an already-doomed episode, not a trap.")
    os._exit(0)


if __name__ == "__main__":
    main()
