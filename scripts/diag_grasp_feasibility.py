"""Friction-cone feasibility of holding the Threading needle under a shift.

No rollout — just reads the model so we can compute whether ANY policy could
hold the needle (max friction force vs needle weight) at the shifted dynamics.
"""
import os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import resolve_threading_bindings

G = 9.81

env = PixelRobosuite(env_name="Threading_D0", robots="Panda", episode_length=200,
                     use_state=True, rl_cameras=[], camera_names=["agentview"])
env.reset()
raw = env.env
m = raw.sim.model
b = resolve_threading_bindings(raw)

needle_bid = b["object_body_ids"][0]
needle_mass = float(m.body_mass[needle_bid])

# handle geoms (what the gripper holds): needle-body geoms whose sliding friction
# is the nominal 1.0 (the tip is already 0.3). Report the range.
handle_fric = []
for g in b["object_geom_ids"]:
    handle_fric.append(float(m.geom_friction[g, 0]))
handle_fric = np.array(handle_fric)

# gripper position-servo: force = gainprm[0]*(ctrl-qpos) - kv*qvel, clamped to
# forcerange. Max steady force ~ min(forcerange, kp * max_finger_travel).
aids = b["gripper_act_ids"]
kp = np.array([m.actuator_gainprm[a, 0] for a in aids])
frng = np.array([m.actuator_forcerange[a] for a in aids])
ctrlrng = np.array([m.actuator_ctrlrange[a] for a in aids])
# finger joint travel (each gripper actuator drives a prismatic finger joint)
trnid = np.array([m.actuator_trnid[a, 0] for a in aids])
jrange = np.array([m.jnt_range[m.jnt_qposadr.tolist().index(m.jnt_qposadr[j]) if False else j]
                   for j in trnid]) if False else np.array([m.jnt_range[j] for j in trnid])

print("=" * 64)
print("NOMINAL model values")
print("=" * 64)
print(f"needle mass            : {needle_mass*1000:.2f} g")
print(f"object-body geom μ      : min {handle_fric.min():.3f}  max {handle_fric.max():.3f}")
print(f"gripper actuator kp     : {kp}")
print(f"gripper forcerange      : {frng.tolist()}")
print(f"gripper ctrlrange       : {ctrlrng.tolist()}")
print(f"finger joint range (m)  : {jrange.tolist()}")

def feasibility(mass_scale, fric_scale, kp_scale, load_factor, label):
    mass = needle_mass * mass_scale
    weight = mass * G
    mu = handle_fric.max() * fric_scale          # best (grippiest) handle contact
    kp_s = kp * kp_scale
    travel = float(np.max(np.abs(jrange[:, 1] - jrange[:, 0])))  # max finger stroke
    # Max normal force a position servo can hold = min(forcerange_max, kp*travel).
    f_force = float(np.max(np.abs(frng)))
    N_kp = float(np.max(kp_s)) * travel
    N_max = min(f_force, N_kp) if f_force > 0 else N_kp
    # Two fingers grip => friction holding force ~ 2*mu*N (one pad each side).
    hold = 2.0 * mu * N_max
    need = weight * load_factor
    print(f"\n--- {label} (mass×{mass_scale}, fric×{fric_scale}, kp×{kp_scale}, "
          f"load×{load_factor}) ---")
    print(f"  needle weight         : {weight:.3f} N   (mass {mass*1000:.1f} g)")
    print(f"  handle μ (best)        : {mu:.3f}")
    print(f"  max grip normal N      : {N_max:.2f} N "
          f"(forcerange {f_force:.1f} N vs kp·travel {N_kp:.1f} N)")
    print(f"  max friction hold 2μN : {hold:.3f} N")
    print(f"  required (weight×load): {need:.3f} N")
    margin = hold / need if need > 0 else float("inf")
    verdict = "FEASIBLE" if margin >= 1.0 else "INFEASIBLE"
    print(f"  hold/need ratio        : {margin:.2f}x  ->  {verdict}")
    return margin

print("\n" + "=" * 64)
print("FRICTION-CONE FEASIBILITY (best-case: grippiest handle geom, max grip)")
print("load×1 = static hang; load×2-3 ~= accel + insertion contact during task")
print("=" * 64)
# the user's target
feasibility(10.0, 0.05, 0.15, 1.0, "TARGET static")
feasibility(10.0, 0.05, 0.15, 2.5, "TARGET w/ task load")
# moderated suggestion
feasibility(5.0, 0.25, 0.4, 1.0, "moderated static")
feasibility(5.0, 0.25, 0.4, 2.5, "moderated w/ task load")
# nominal sanity check
feasibility(1.0, 1.0, 1.0, 2.5, "nominal w/ task load (sanity)")
