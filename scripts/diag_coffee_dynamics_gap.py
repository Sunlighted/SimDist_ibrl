"""Measure the Coffee_D0 SOURCE->TARGET *dynamics* gap directly.

WHY THIS EXISTS
---------------
Sweeping a shift knob and reading the expert's SUCCESS RATE conflates two
different things: "how much did P(s'|s,a) move" and "was the expert robust
enough to still finish". A knob can change the physics a lot and cost zero
success (Coffee: mass_scale 3.0, pod friction 0.3, holder friction 3.0 all
scored 100%). If the goal is a genuine dynamics gap, success rate is the WRONG
selection metric.

This script measures the gap the honest way: take real (sim_state, action)
pairs from the demos, replay each ONE step in a SOURCE env and in a TARGET env,
and report how far the next states diverge -- in physically meaningful units
(pod mm, lid degrees, eef mm, gripper mm) rather than a 66-dim norm, so
quaternion sign flips can't contaminate it and the numbers stay interpretable.

Both arms are cold-started identically (qacc_warmstart=0 + controller
reset_goal), so the ONLY difference between them is the model parameters.

TRAPS THIS SCRIPT AVOIDS (both cost real time before)
  * The target env MUST be reset() before use -- domain_shift is applied in
    PixelRobosuite.reset(), and set_state_from_flattened only writes qpos/qvel.
    An unreset probe env silently runs SOURCE physics (see CLAUDE.md 27).
  * sim_state does NOT contain the OSC controller's internal goal, which is
    initialized at reset from whatever pose the arm was in. Replaying without
    resetting the goal makes the first step depend on episode history rather
    than on (s, a) (see CLAUDE.md 35). We call controller.reset_goal() after
    restoring the state so both arms start from an identical, well-defined
    controller state.

Usage
  python scripts/diag_coffee_dynamics_gap.py --num_demos 20 --stride 4
  python scripts/diag_coffee_dynamics_gap.py --arms A,B,maxgap
"""
import argparse
import dataclasses
import os
import sys

import numpy as np
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.domain_shift import DomainShiftConfig  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402


def log(*a):
    # AdaptWorkspace-style stdout hijacking bit us before; stderr is safe.
    print(*a, file=sys.stderr, flush=True)


# Candidate shifts. `source` is the control arm: it must come out at exactly 0.
ARMS = {
    "source":      dict(),
    # single knobs, at the values the success sweep called "inert"
    "mass3":       dict(mass_scale=3.0),
    "podfric0.3":  dict(friction_scale=0.3),
    "holderfric3": dict(pod_holder_friction_scale=3.0),
    "solref0.5":   dict(solref_scale=0.5),
    # single knobs the success sweep called "effective"
    "kp0.2":       dict(gripper_kp_scale=0.2),
    "lid100":      dict(lid_damping_scale=100.0),
    # the success-rate-designed targets
    "A":           dict(gripper_kp_scale=0.20, lid_damping_scale=100.0),
    "B":           dict(gripper_kp_scale=0.20, lid_damping_scale=75.0),
    # stack everything that genuinely moves the physics
    "maxgap":      dict(gripper_kp_scale=0.20, lid_damping_scale=100.0,
                        mass_scale=3.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0, solref_scale=0.5),
    "maxgapB":     dict(gripper_kp_scale=0.20, lid_damping_scale=75.0,
                        mass_scale=3.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0, solref_scale=0.5),
    "maxgap2":     dict(gripper_kp_scale=0.20, lid_damping_scale=75.0,
                        mass_scale=5.0, friction_scale=0.2,
                        pod_holder_friction_scale=5.0, solref_scale=0.3),
    "maxgap3":     dict(gripper_kp_scale=0.30, lid_damping_scale=50.0,
                        mass_scale=5.0, friction_scale=0.2,
                        pod_holder_friction_scale=5.0, solref_scale=0.3),
    # solref dropped on purpose: Coffee's pod is already at timeconst 0.001 s,
    # BELOW MuJoCo's 2*timestep = 0.004 s floor, so scaling it is at best
    # partly clamped and not a knob to lean on.
    "C50":         dict(gripper_kp_scale=0.30, lid_damping_scale=50.0,
                        mass_scale=3.0, friction_scale=0.5,
                        pod_holder_friction_scale=2.0),
    "C75":         dict(gripper_kp_scale=0.30, lid_damping_scale=75.0,
                        mass_scale=3.0, friction_scale=0.5,
                        pod_holder_friction_scale=2.0),
    "C75kp25":     dict(gripper_kp_scale=0.25, lid_damping_scale=75.0,
                        mass_scale=3.0, friction_scale=0.5,
                        pod_holder_friction_scale=2.0),
    # --- lid backed off; difficulty moved onto kp / mass / friction ---
    "E25":         dict(gripper_kp_scale=0.25, lid_damping_scale=25.0,
                        mass_scale=3.0, friction_scale=0.5,
                        pod_holder_friction_scale=2.0),
    "E10":         dict(gripper_kp_scale=0.25, lid_damping_scale=10.0,
                        mass_scale=3.0, friction_scale=0.5,
                        pod_holder_friction_scale=2.0),
    "E10hard":     dict(gripper_kp_scale=0.20, lid_damping_scale=10.0,
                        mass_scale=5.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0),
    "E1nolid":     dict(gripper_kp_scale=0.20, lid_damping_scale=1.0,
                        mass_scale=5.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0),
    "F2":          dict(gripper_kp_scale=0.20, lid_damping_scale=2.0,
                        mass_scale=5.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0),
    "F5":          dict(gripper_kp_scale=0.20, lid_damping_scale=5.0,
                        mass_scale=5.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0),
    # even harder physics, lid essentially untouched
    "F5x":         dict(gripper_kp_scale=0.15, lid_damping_scale=5.0,
                        mass_scale=8.0, friction_scale=0.2,
                        pod_holder_friction_scale=5.0),
    "G5x":         dict(gripper_kp_scale=0.30, lid_damping_scale=5.0,
                        mass_scale=8.0, friction_scale=0.2,
                        pod_holder_friction_scale=5.0),
    "G10":         dict(gripper_kp_scale=0.30, lid_damping_scale=10.0,
                        mass_scale=5.0, friction_scale=0.3,
                        pod_holder_friction_scale=3.0),
}


def make_env(**shift_kwargs):
    ds = (DomainShiftConfig(mode="fixed", **shift_kwargs) if shift_kwargs
          else DomainShiftConfig(mode="none"))
    env = PixelRobosuite(
        "Coffee_D0", "Panda", episode_length=400, use_state=True,
        device="cpu", rl_image_size=96, camera_names=["agentview"],
        domain_shift=ds,
    )
    env.reset()  # REQUIRED: this is where the shift is applied.
    return env


def probe(env, sim_state, action):
    """One cold-start step from `sim_state` under `action`; return the physical
    quantities we compare across arms."""
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    # Pin the controller goal to the restored pose so the step depends on
    # (s, a) alone and not on this env's episode history.
    ctrl = raw.robots[0].controller
    if hasattr(ctrl, "update"):
        ctrl.update(force=True)
    if hasattr(ctrl, "reset_goal"):
        ctrl.reset_goal()
    raw.step(np.asarray(action, dtype=np.float64))

    d, m = raw.sim.data, raw.sim.model
    hinge = m.jnt_qposadr[m.joint_name2id("coffee_machine_lid_main_joint0")]
    return dict(
        pod=np.array(d.body_xpos[raw.obj_body_id["coffee_pod"]], copy=True),
        lid=float(d.qpos[hinge]),
        eef=np.array(d.site_xpos[m.site_name2id(
            raw.robots[0].controller.eef_name)], copy=True),
        grip=np.array(d.qpos[raw.robots[0]._ref_gripper_joint_pos_indexes],
                      copy=True),
    )


def phase_of(sim_state, env):
    """Cheap 3-way phase label so the gap can be read per task stage."""
    raw = env.env
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.forward()
    d, m = raw.sim.data, raw.sim.model
    pod = np.array(d.body_xpos[raw.obj_body_id["coffee_pod"]])
    hold = np.array(d.body_xpos[raw.obj_body_id["coffee_pod_holder"]])
    hinge = m.jnt_qposadr[m.joint_name2id("coffee_machine_lid_main_joint0")]
    lid = float(d.qpos[hinge])
    if np.linalg.norm(pod[:2] - hold[:2]) > 0.05:
        return "reach/transport"
    if lid > 1.0:
        return "insert"
    return "close-lid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default="release/data/mimicgen/core/coffee_d0/"
                            "processed_data96.hdf5")
    ap.add_argument("--num_demos", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4,
                    help="subsample every k-th transition per demo")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    args = ap.parse_args()

    # ---- gather (sim_state, action) pairs -------------------------------
    pairs = []
    with h5py.File(args.data, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys[:args.num_demos]:
            g = f["data"][k]
            S, A = g["states"][:], g["actions"][:]
            for t in range(0, len(A), args.stride):
                pairs.append((S[t], A[t]))
    log(f"[data] {len(pairs)} (sim_state, action) pairs "
        f"from {args.num_demos} demos (stride {args.stride})")

    src = make_env()
    phases = np.array([phase_of(s, src) for s, _ in pairs])
    for ph in ("reach/transport", "insert", "close-lid"):
        log(f"       phase {ph:16s}: {int((phases == ph).sum()):5d}")

    base = [probe(src, s, a) for s, a in pairs]

    hdr = (f"{'arm':<13}{'pod mm':>22}{'lid deg':>20}"
           f"{'eef mm':>18}{'grip mm':>12}")
    log("\n" + hdr)
    log(f"{'':13}{'mean / p90 / max':>22}{'mean / p90 / max':>20}"
        f"{'mean / p90':>18}{'mean':>12}")
    log("-" * len(hdr))

    results = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm not in ARMS:
            log(f"  (skip unknown arm {arm})")
            continue
        env = make_env(**ARMS[arm])
        cur = [probe(env, s, a) for s, a in pairs]
        pod = np.array([np.linalg.norm(c["pod"] - b["pod"])
                        for c, b in zip(cur, base)]) * 1000
        lid = np.degrees(np.abs(np.array([c["lid"] - b["lid"]
                                          for c, b in zip(cur, base)])))
        eef = np.array([np.linalg.norm(c["eef"] - b["eef"])
                        for c, b in zip(cur, base)]) * 1000
        grip = np.array([np.linalg.norm(c["grip"] - b["grip"])
                         for c, b in zip(cur, base)]) * 1000
        results[arm] = dict(pod=pod, lid=lid, eef=eef, grip=grip)
        log(f"{arm:<13}"
            f"{pod.mean():7.3f} /{np.percentile(pod, 90):7.3f} /{pod.max():6.2f}"
            f"{lid.mean():7.3f} /{np.percentile(lid, 90):6.3f} /{lid.max():5.2f}"
            f"{eef.mean():8.3f} /{np.percentile(eef, 90):7.3f}"
            f"{grip.mean():10.3f}")
        del env

    # ---- per-phase breakdown (pod gap only; it is the task-relevant one) --
    log("\nper-phase pod-position gap (mean mm):")
    log(f"{'arm':<13}" + "".join(f"{p:>18}" for p in
                                 ("reach/transport", "insert", "close-lid")))
    for arm, r in results.items():
        row = "".join(
            f"{r['pod'][phases == p].mean():18.3f}" if (phases == p).any()
            else f"{'-':>18}"
            for p in ("reach/transport", "insert", "close-lid"))
        log(f"{arm:<13}" + row)

    log("\nNOTE the `source` row is the harness control and MUST be all zeros; "
        "any nonzero value there means the replay is not deterministic and "
        "every other number in this table is contaminated.")


if __name__ == "__main__":
    main()
