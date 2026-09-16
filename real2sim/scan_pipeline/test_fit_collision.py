"""Validate fit_collision against SYNTHETIC clouds with known ground truth,
then run it on real assets. Synthetic first: if the fitter cannot recover a box
it generated itself, its verdict on a scan is meaningless.
"""
import sys
import numpy as np
sys.path.insert(0, "/scratch/yifanr/gsbench")
from fit_collision import fit_collision, to_mjcf

rng = np.random.default_rng(0)
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


def surf_box(h, n=20000):
    X = rng.uniform(-1, 1, (n, 3)) * h
    f = rng.integers(0, 3, n)
    for a in range(3):
        m = f == a
        X[m, a] = np.sign(rng.uniform(-1, 1, m.sum())) * h[a]
    return X


def surf_cyl(r, hl, n=20000):
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-hl, hl, n)
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


def surf_sph(r, n=20000):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * r


print("=== synthetic: type recovery ===")
h = np.array([0.04, 0.02, 0.06])
f, _, _ = fit_collision(surf_box(h))
check("box -> box", f["type"] == "box", f"got {f['type']}")
check("box dims", np.allclose(np.sort(f["size"]), np.sort(h), atol=1e-3),
      f"{np.round(np.sort(f['size']),4)} vs {np.sort(h)}")

f, _, _ = fit_collision(surf_cyl(0.03, 0.05))
check("cylinder -> cylinder", f["type"] == "cylinder", f"got {f['type']}")
if f["type"] == "cylinder":
    check("cyl dims", abs(f["size"][0]-0.03) < 2e-3 and abs(f["size"][1]-0.05) < 2e-3,
          f"r={f['size'][0]:.4f} hl={f['size'][1]:.4f}")

f, _, _ = fit_collision(surf_sph(0.035))
check("sphere -> sphere or box", f["type"] in ("sphere", "box"), f"got {f['type']}")

print("=== synthetic: a flat plate must NOT be called a cylinder ===")
plate = surf_box(np.array([0.05, 0.05, 0.002]))
f, cands, _ = fit_collision(plate)
cy = [c for c in cands if c["type"] == "cylinder"][0]
bx = [c for c in cands if c["type"] == "box"][0]
check("plate -> box", f["type"] == "box", f"got {f['type']}")
check("box fill beats cylinder", bx["fill"] > cy["fill"],
      f"box={bx['fill']:.3f} cyl={cy['fill']:.3f}")

print("=== synthetic: an L-shape must fall back to hull ===")
a = rng.uniform(0, 1, (2500, 3)) * np.array([0.08, 0.02, 0.02])
b = rng.uniform(0, 1, (2500, 3)) * np.array([0.02, 0.08, 0.02])
f, _, _ = fit_collision(np.vstack([a, b]), tol=0.003)
check("L-shape -> hull", f["type"] == "hull", f"got {f['type']}")

print("=== robustness: 2% floaters must not blow up the fit ===")
X = surf_box(h)
X = np.vstack([X, rng.uniform(-0.5, 0.5, (int(0.02*len(X)), 3))])
f, _, _ = fit_collision(X)
check("floaters tolerated (p95 not max)", f["type"] in ("box", "hull"), f"got {f['type']}")

print("=== MJCF emission ===")
f, _, _ = fit_collision(surf_cyl(0.03, 0.05))
xml = to_mjcf(f, "peg")
check("mjcf has type+size+quat", all(k in xml for k in ("type=", "size=", "quat=")), xml[:90])

print("=== real asset: robosuite Robotiq inner finger (visual mesh) ===")
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco, robosuite
from robosuite.controllers import load_controller_config
cc = load_controller_config(default_controller="OSC_POSE")
env = robosuite.make(env_name="Lift", robots="Panda", gripper_types="Robotiq85Gripper",
                     controller_configs=cc, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, horizon=5)
m = env.sim.model
for body in ["gripper0_left_inner_finger", "robot0_link5"]:
    bid = m.body_name2id(body)
    pts = []
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == bid and m.geom_group[i] == 1 and m.geom_type[i] == 7 \
           and m.geom_dataid[i] >= 0:
            did = m.geom_dataid[i]
            a_, n_ = m.mesh_vertadr[did], m.mesh_vertnum[did]
            pts.append(np.array(m.mesh_vert[a_:a_+n_]).reshape(-1, 3))
    if pts:
        V = np.concatenate(pts, 0)
        print(f"  {body} ({len(V)} verts):")
        f, _, _ = fit_collision(V, verbose=True)
        print(f"    -> {f['type']}   {to_mjcf(f, body)[:110]}")
env.close()

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
