"""The sim side of the scripted grasp grid (env/grasp_script.py): same script, same execution layer, per-trial needle pose.

  --dry           no robot data: the needle at the nominal pose + a simulated placement error (--place_sd); checks
                  the script (clearances, no table hit) and gives the sim's success map before the visit.
  --real RUN_DIR  a `policy_server --scripted grasp --save_dir RUN_DIR` run: per trial, the needle pose is fitted to the
                  hover wrist frame (hover_trialNN_wrist.jpg + _state.json, close range), the sim arm and reference start
                  at the real trial's first q, and the trial is re-run; prints sim lifted vs the operator's y / n / s.
Contact model knobs for the sweep that this data is for: --grip_solref, --grip_friction, --needle_mass_scale.

    python scripts/grasp_twin.py --dry --out /tmp/grasp_dry
    python scripts/grasp_twin.py --real runs/grasp_1020_1400 --grip_solref 0.02 1.0
"""
import argparse
import dataclasses
import glob
import io
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402


def build_env(args):
    import env.threading_variants as tv
    from env.robosuite_wrapper import PixelRobosuite
    from env.sim2real import Sim2RealConfig, load_sim2real
    cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
    off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
    cell = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **off)
    cams = ["robot0_eye_in_hand"] if args.real else []
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 10 ** 6, image_size=64, rl_image_size=64,
                         camera_names=cams, rl_cameras=cams, use_state=True, state_only_no_render=not cams,
                         device="cpu", hand_init_perturb_joint_sigma=0.0, sim2real=cell)
    return env, cell


def contact_knobs(m, args):
    for g in range(m.ngeom):
        n = m.geom_id2name(g) or ""
        if n.startswith("gripper0") and "collision" in n:
            if args.grip_solref is not None:
                m.geom_solref[g] = args.grip_solref
            if args.grip_friction is not None:
                m.geom_friction[g][0] = args.grip_friction
    if args.needle_mass_scale != 1.0:
        for b in range(m.nbody):
            if (m.body_id2name(b) or "").startswith("needle"):
                m.body_mass[b] *= args.needle_mass_scale
                m.body_inertia[b] *= args.needle_mass_scale


def place(m, d, base, obj, x, y, yaw):
    a = m.get_joint_qpos_addr(f"{obj}_obj_joint0")[0]
    v = m.get_joint_qvel_addr(f"{obj}_obj_joint0")[0]
    d.qpos[a:a + 2] = [base[0] + x, base[1] + y]
    h = np.radians(yaw) / 2
    d.qpos[a + 3:a + 7] = [np.cos(h), 0, 0, np.sin(h)]
    d.qvel[v:v + 6] = 0.0


def run_trial(env, cell, args, trial, needle, q_start=None, park_tripod=True, grid_point=None):
    """One grasp trial in the sim with the needle at `needle` (base frame). `grid_point` = the trial's
    (height, along, across); default default_grid()[trial] (a --grasp_trials run renumbers its trials from 0). -> dict."""
    from env.grasp_script import GraspGeometry, GraspScript, default_grid
    np.random.seed(0)
    env.reset()
    e = env.env
    m, d = e.sim.model, e.sim.data
    contact_knobs(m, args)
    rob = e.robots[0]
    L = e.exec_layer
    jidx = np.asarray(rob._ref_joint_pos_indexes)
    base = d.body_xpos[m.body_name2id("robot0_base")].copy()
    home = np.asarray(cell.init_qpos, np.float64)
    geo = GraspGeometry.from_sim(e.sim, rob, home)
    nominal = tuple(args.nominal)
    gp = tuple(grid_point) if grid_point is not None else default_grid()[trial]
    sc = GraspScript(geo=geo, needle=nominal, grid=[gp], hz=float(e.control_freq), wait_forever=False)
    place(m, d, base, "needle", *needle)
    if park_tripod:
        a = m.get_joint_qpos_addr("tripod_obj_joint0")[0]
        d.qpos[a:a + 3] = [0.0, 3.0, -5.0]
    q0 = home if q_start is None else np.asarray(q_start, np.float64)
    d.qpos[jidx] = q0
    d.qvel[np.asarray(rob._ref_joint_vel_indexes)] = 0.0
    e.sim.forward()
    L.reset_goal()
    nb = m.body_name2id("needle_obj_root")
    z0 = float(d.body_xpos[nb][2])
    tid = m.geom_name2id("table_collision")
    gripper_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0")]
    table_hits, zmax, kn, phases = 0, z0, {}, []
    while not sc.done:
        T = L.ref_site_pose()
        act, info = sc.act(T[:3, 3], T[:3, :3])
        if sc.done:
            break
        e.timestep = 0
        e.step(act)
        for c in d.contact[:d.ncon]:
            if (c.geom1 == tid and c.geom2 in gripper_geoms) or (c.geom2 == tid and c.geom1 in gripper_geoms):
                table_hits += 1
                break
        zmax = max(zmax, float(d.body_xpos[nb][2]))
        ph = info.get("phase")
        kn[ph] = float(d.qpos[m.get_joint_qpos_addr("gripper0_finger_joint")])
        if ph == "hold_up":
            lifted_now = float(d.body_xpos[nb][2]) - z0
    return dict(trial=trial, grid=list(gp), needle=[round(float(v), 5) for v in needle],
                lifted_mm=round(1e3 * lifted_now, 1), max_rise_mm=round(1e3 * (zmax - z0), 1),
                driver_at_hold=round(kn.get("hold_up", np.nan), 3), table_contact_steps=table_hits,
                success=bool(lifted_now > 0.07))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--real", default="")
    ap.add_argument("--nominal", type=float, nargs=3, default=None, help="x y yaw (base); default the cell layout centre")
    ap.add_argument("--place_sd", type=float, nargs=2, default=(0.005, 5.0), help="dry: placement error sd (m, deg)")
    ap.add_argument("--reps", type=int, default=1, help="dry: placements per grid point")
    ap.add_argument("--grip_solref", type=float, nargs=2, default=None)
    ap.add_argument("--grip_friction", type=float, default=None)
    ap.add_argument("--needle_mass_scale", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    assert args.dry or args.real, "--dry or --real"
    env, cell = build_env(args)
    if args.nominal is None:
        args.nominal = [float(np.mean(r)) for r in (cell.layout_needle_x, cell.layout_needle_y, cell.layout_needle_z_rot_deg)]
    from env.grasp_script import default_grid
    rows = []
    if args.dry:
        rng = np.random.default_rng(0)
        for trial in range(len(default_grid())):
            for r in range(args.reps):
                nd = np.array(args.nominal) + np.r_[rng.normal(0, args.place_sd[0], 2), rng.normal(0, args.place_sd[1])]
                rows.append(run_trial(env, cell, args, trial, nd))
                print(json.dumps(rows[-1]), flush=True)
    else:
        rows = real_mode(env, cell, args)
    n = len(rows)
    print(f"[grasp twin] sim lifted {sum(r['success'] for r in rows)}/{n}, any table contact in "
          f"{sum(r['table_contact_steps'] > 0 for r in rows)}/{n}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json.dump(dict(args=vars(args), rows=rows), open(os.path.join(args.out, "grasp_twin.json"), "w"), indent=1)


def real_mode(env, cell, args):
    """Per trial: fit the needle to the hover wrist frame, re-run the trial from the real start q."""
    import cv2
    from PIL import Image
    from scipy.optimize import minimize
    from real2sim.deploy.preprocess import Rectifier, camera_intrinsics
    steps = [json.loads(ln) for ln in open(os.path.join(args.real, "steps.jsonl")) if ln.strip()]
    ops = {r["trial"]: r["operator"] for r in steps if "operator" in r}
    first_q = {}
    for r in steps:
        sc = r.get("script") or {}
        if "trial" in sc and sc["trial"] not in first_q:
            first_q[sc["trial"]] = r["q"]
    wcam = [c for c in cell.cameras if c.name == "robot0_eye_in_hand"][0]
    rows = []
    for st in sorted(glob.glob(os.path.join(args.real, "hover_trial*_state.json"))):
        S = json.load(open(st))
        t = S["trial"]
        raw = np.array(Image.open(st.replace("_state.json", "_wrist.jpg")).convert("RGB"))
        j = json.load(open(wcam.real_intrinsics))
        K, D = np.array(j["K"]), np.array(j["D"])
        rect = Rectifier(K, D, raw.shape[1], raw.shape[0], 448, wcam.resolved_fovy())
        real = rect(raw)
        needle = fit_needle(env, cell, S["q"], S["fingers"][0], real, np.array(args.nominal))
        row = run_trial(env, cell, args, t, needle, q_start=first_q.get(t), grid_point=S.get("grid"))
        row["operator"] = ops.get(t)
        rows.append(row)
        print(json.dumps(row), flush=True)
    agree = [r for r in rows if r["operator"] in ("y", "n", "s")]
    ok = sum((r["operator"] == "y") == r["success"] for r in agree)
    print(f"[grasp twin] sim vs real agree on {ok}/{len(agree)} trials "
          f"(real lifted {sum(r['operator'] == 'y' for r in agree)}, sim {sum(r['success'] for r in agree)})")
    return rows


def fit_needle(env, cell, q, knuckle, real448, init):
    """(x, y, yaw) of the needle (base frame) maximising the wrist-view IoU at the hover pose."""
    import cv2
    from scipy.optimize import minimize
    import env.threading_variants as tv
    np.random.seed(0)
    env.reset()
    e = env.env
    m, d = e.sim.model, e.sim.data
    rob = e.robots[0]
    base = d.body_xpos[m.body_name2id("robot0_base")].copy()
    d.qpos[np.asarray(rob._ref_joint_pos_indexes)] = q
    for a_, v in zip([m.get_joint_qpos_addr(n) for n in tv.ROBOTIQ_DRIVER_JOINTS], tv.robotiq_real_to_sim(knuckle)):
        d.qpos[a_] = v
    a = m.get_joint_qpos_addr("tripod_obj_joint0")[0]
    d.qpos[a:a + 3] = [0.0, 3.0, -5.0]
    cls = np.zeros(m.ngeom + 1, np.int8)
    for g in range(m.ngeom):
        n = m.geom_id2name(g) or ""
        b = m.body_id2name(int(m.geom_bodyid[g])) or ""
        cls[g] = 2 if n.startswith("needle") else 4 if b.startswith("gripper0") else 0
    r, g_, b_ = (real448[..., i].astype(int) for i in range(3))
    red = ((r - g_) > 12) & (r >= b_)

    def loss(p):
        place(m, d, base, "needle", *p)
        e.sim.forward()
        s = e.sim.render(camera_name="robot0_eye_in_hand", width=448, height=448, segmentation=True)[::-1][..., 1]
        s = cls[np.where(s >= 0, s, m.ngeom)]
        ns = s == 2
        ok = ~cv2.dilate((s == 4).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        rr = red & cv2.dilate(ns.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool) & ok
        return 1 - (ns & ok & rr).sum() / max(((ns & ok) | rr).sum(), 1)

    best = None
    for dyaw in (0, -8, 8):
        x0 = init + [0, 0, dyaw]
        res = minimize(loss, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=300,
                       initial_simplex=[x0, x0 + [0.006, 0, 0], x0 + [0, 0.006, 0], x0 + [0, 0, 4.0]]))
        if best is None or res.fun < best.fun:
            best = res
    return best.x


if __name__ == "__main__":
    main()
