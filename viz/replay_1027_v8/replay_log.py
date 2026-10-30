"""Replay converted real Robotiq demos in the sim cell (calib_20261015/SIM_TODO.md §4).

Input: the dataset.hdf5 written by scripts/convert_real_teleop.py on the Robotiq cell (it stores
`obs/real_joint_pos`, the measured q, next to the contract keys).

Per demo, in the training cell (Threading_RealCellRobotiq, the cell's control_freq, every DR off):
  * objects at a FIXED layout (default: the 10-15 layout-shot fit, README §5 -- the demos were
    recorded right after it with the objects in that spot);
  * the arm starts at the demo's measured q[0], gripper open;
  * --mode track (default): closed-loop OSC servo onto the REAL measured grip-site trajectory,
    a = gain * (pose_real[t+1] - pose_sim) / scale, gripper = the demo's sign. Tests whether the
    sim's geometry / contacts let the same TCP path grasp and thread. Reports the tracking error,
    so a failure can be told apart from "the servo did not follow".
  * --mode open: the stored OSC actions open loop (tests the action convention + execution layer;
    without the real arm's lag / cap in the sim this is expected to drift).
Outputs per demo: success, first success step, grasp (needle lifted), tracking error, final needle
pose; optional --video dir with agentview + wrist frames side by side with the real frames.

  python scripts/replay_real_demos_sim.py \
      --dataset /user_data/Chenyang/real_stage2/threading_robotiq_1015/dataset.hdf5 \
      --cell real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml --video <dir>
"""
import argparse
import dataclasses
import io
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/Chenyang/ibrl")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

import env.threading_variants as tv  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import Sim2RealConfig, load_sim2real  # noqa: E402

POS_SCALE, ROT_SCALE = 0.05, 0.5
# 10-15 layout shot, scene-camera render fit (calib_20261015/README.md §5): base frame, deg
LAYOUT_1015 = dict(needle=(0.5561, -0.1108, -110.2), tripod=(0.5204, 0.1832, -36.0))


def place(sim, layout):
    """Objects at base-frame (x, y, yaw) AFTER the reset, z kept: the convention of
    calib_20261014/tools/fit_layout_wrist.py place(), which produced the per-demo fits.
    (Patching the placement bounds does not work: the wrapper's apply_cell_layout wraps
    them on the instance with the cell's full ranges -- that silently randomised the
    first version of this script.)"""
    m, d = sim.model, sim.data
    base = d.body_xpos[m.body_name2id("robot0_base")]
    for obj, (x, y, yaw) in layout.items():
        a = m.get_joint_qpos_addr(f"{obj}_obj_joint0")[0]
        v = m.get_joint_qvel_addr(f"{obj}_obj_joint0")[0]
        d.qpos[a:a + 3] = [base[0] + x, base[1] + y, d.qpos[a + 2]]
        h = np.radians(yaw) / 2
        d.qpos[a + 3:a + 7] = [np.cos(h), 0, 0, np.sin(h)]
        d.qvel[v:v + 6] = 0.0
    sim.forward()


def axisangle(R):
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle
    return np.asarray(quat2axisangle(mat2quat(R)))


def redmask_rgb(rgb):
    """calib_20261014/tools/fit_layout_wrist.py redmask (R - max(G, B) > 22), on RGB."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (r - np.maximum(g, b)) > 22


def jpeg(buf):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(np.asarray(buf).tobytes())).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cell", default=tv.REAL_CELL_ROBOTIQ_YAML)
    ap.add_argument("--mode", default="track", choices=("track", "open"))
    ap.add_argument("--gain", type=float, default=2.5,
                    help="track: servo gain on the per-step pose error (sim OSC moves ~0.4 of a "
                         "commanded delta per 10 Hz step)")
    ap.add_argument("--extra_steps", type=int, default=20, help="hold the last target this long")
    ap.add_argument("--exec_layer", default="auto", choices=("auto", "on", "off"),
                    help="the cell's execution layer (env/exec_layer.py). auto: ON for open (the deploy-consistent test: "
                         "ref_inverse actions through the same chain as on the robot), OFF for track (a servo onto the "
                         "measured path, which the layer's lag would only blur)")
    ap.add_argument("--demos", default="", help="comma list of demo indices (default all)")
    ap.add_argument("--video", default="", help="dir for per-demo side-by-side mp4")
    ap.add_argument("--video_every1", action="store_true", help="video: every step, not every 2nd")
    ap.add_argument("--grip", default="action", choices=("action", "knuckle"),
                    help="gripper sign: the stored action (cmd below the middle of its range) or "
                         "'knuckle': close from the step the MEASURED real knuckle starts to close "
                         "(> --grip_onset rad), shifted earlier by --grip_lead steps")
    ap.add_argument("--grip_onset", type=float, default=0.05)
    ap.add_argument("--needle_mass_scale", type=float, default=1.0,
                    help="DIAGNOSTIC: scale the needle's mass + inertia (sim 7.6 g)")
    ap.add_argument("--grip_solref", type=float, nargs=2, default=None,
                    help="DIAGNOSTIC: solref of every Robotiq collision geom (robosuite: fingers/tips 0.01 0.25 = stiff "
                         "and underdamped, pads/inner knuckles 0.02 1)")
    ap.add_argument("--grasp_condim", type=int, default=0,
                    help="DIAGNOSTIC: condim for the Robotiq pads/fingertips and the needle handle (sim 3 = "
                         "no torsional friction); 0 = unchanged")
    ap.add_argument("--perturb", default="",
                    help="'dx_mm,dy_mm,dyaw_deg' added to the needle's layout (grasp robustness sweeps)")
    ap.add_argument("--grip_lead", type=int, default=0)
    ap.add_argument("--grasp_log", action="store_true",
                    help="print, per step around the close: sim driver joint vs the mapped real knuckle, "
                         "needle displacement, and which gripper geoms touch which needle geoms")
    ap.add_argument("--layout_json", default="",
                    help="per source episode {needle: [x, y, yaw], tripod: [...]} (base frame); "
                         "default = the 10-15 layout shot for every demo")
    ap.add_argument("--inhand_dr_prob", type=float, default=0.0)
    ap.add_argument("--gripper_speed", type=float, default=0.006)
    ap.add_argument("--gripper_cmd_delay", type=int, nargs=2, default=[0, 0])
    args = ap.parse_args()

    cell = load_sim2real(args.cell)
    off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
    cell = dataclasses.replace(
        cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **off)
    cell = dataclasses.replace(cell, inhand_dr_prob=args.inhand_dr_prob, gripper_speed=args.gripper_speed,
                               gripper_cmd_delay=tuple(args.gripper_cmd_delay))
    print('[cell overrides] needle_com_y', cell.needle_com_y, 'inhand_dr_prob', cell.inhand_dr_prob, 'gripper_speed', cell.gripper_speed, 'delay', cell.gripper_cmd_delay, 'dr_gripper_speed', cell.dr_gripper_speed)
    per_demo = {}
    if args.layout_json:
        import json
        per_demo = {k: v for k, v in json.load(open(args.layout_json)).items() if not k.startswith("_")}
    cams = ["agentview", "robot0_eye_in_hand"]
    el = (args.mode == "open") if args.exec_layer == "auto" else (args.exec_layer == "on")
    from robosuite.controllers import load_controller_config  # noqa: F401
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 10 ** 6, image_size=224, rl_image_size=224,
                         camera_names=cams, rl_cameras=cams, use_state=True, device="cpu",
                         hand_init_perturb_joint_sigma=0.0, sim2real=cell, env_kwargs={"exec_layer": el})
    print(f"[replay] {env.env.control_freq:g} Hz | mode {args.mode} | exec layer {'ON' if el else 'off'} | layout "
          f"{args.layout_json or LAYOUT_1015}")

    f = h5py.File(args.dataset, "r")
    names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    if args.demos:
        names = [f"demo_{int(i)}" for i in args.demos.split(",")]
    rows = []
    for name in names:
        d = f["data"][name]
        q = np.asarray(d["obs/real_joint_pos"], np.float64)
        acts = np.asarray(d["actions"], np.float64)
        T = len(q)
        if args.grip == "knuckle":
            kr = np.asarray(d["obs/real_gripper"], np.float64)
            closed = kr > args.grip_onset
            # sign for step t = state the real gripper is in at t+1, led by grip_lead steps
            idx = np.clip(np.arange(T) + 1 + args.grip_lead, 0, T - 1)
            acts = acts.copy()
            acts[:, 6] = np.where(closed[idx], 1.0, -1.0)
        src = d.attrs.get("source_episode", name)
        layout = {k: tuple(v) for k, v in per_demo.get(src, LAYOUT_1015).items() if k in ("needle", "tripod")}
        if args.perturb:
            px, py, pyaw = (float(v) for v in args.perturb.split(","))
            x, y, yaw = layout["needle"]
            layout["needle"] = (x + 1e-3 * px, y + 1e-3 * py, yaw + pyaw)
        np.random.seed(0)
        env.reset()
        # robosuite's hard reset rebuilds the sim: take every handle AFTER it
        e = env.env
        rob = e.robots[0]
        sim = e.sim
        m = sim.model
        jidx = np.array(rob._ref_joint_pos_indexes)
        jvel = np.array(rob._ref_joint_vel_indexes)
        sid = rob.eef_site_id
        if args.needle_mass_scale != 1.0:
            for b in range(m.nbody):
                if (m.body_id2name(b) or "").startswith("needle"):
                    m.body_mass[b] *= args.needle_mass_scale
                    m.body_inertia[b] *= args.needle_mass_scale
        if args.grip_solref is not None:
            for g in range(m.ngeom):
                n = m.geom_id2name(g) or ""
                if n.startswith("gripper0") and "collision" in n:
                    m.geom_solref[g] = args.grip_solref
        if args.grasp_condim:
            for g in range(m.ngeom):
                n = m.geom_id2name(g) or ""
                if n == "needle_obj_handle" or (n.startswith("gripper0") and ("fingerpad" in n or "fingertip" in n)
                                                 and "collision" in n):
                    m.geom_condim[g] = args.grasp_condim

        def site():
            return sim.data.site_xpos[sid].copy(), sim.data.site_xmat[sid].reshape(3, 3).copy()

        def fk(qq):
            save = sim.data.qpos.copy(), sim.data.qvel.copy()
            sim.data.qpos[jidx] = qq
            sim.forward()
            out = site()
            sim.data.qpos[:], sim.data.qvel[:] = save
            sim.forward()
            return out
        place(sim, layout)
        sim.data.qpos[jidx] = q[0]
        sim.data.qvel[jvel] = 0.0
        sim.forward()
        # settle the objects on the table (the arm is held by the zero-action OSC)
        rob.controller.update(force=True)
        rob.controller.reset_goal()
        for _ in range(3):
            e.timestep = 0
            e.step(np.r_[np.zeros(6), -1.0])
        sim.data.qpos[jidx] = q[0]
        sim.data.qvel[jvel] = 0.0
        sim.forward()
        rob.controller.update(force=True)
        rob.controller.reset_goal()
        if getattr(e, "exec_layer", None) is not None:
            # the reference starts where ref_inverse started it: the GELLO command of frame 0 (the forced start
            # leaves the arm ~2.5 cm from it); the arm itself starts at the measured q[0]
            qc0 = np.asarray(d["obs/real_joint_cmd"][0], np.float64) if "obs/real_joint_cmd" in d else q[0]
            e.exec_layer.q_ref, e.exec_layer.v_ref = qc0.copy(), np.zeros(7)
            e.exec_layer.start = e.exec_layer.goal = e.exec_layer.last_target = np.array(q[0])
        nb = m.body_name2id("needle_obj_main") if "needle_obj_main" in [
            m.body_id2name(i) for i in range(m.nbody)] else \
            [i for i in range(m.nbody) if (m.body_id2name(i) or "").startswith("needle")][0]
        z0 = float(sim.data.body_xpos[nb][2])
        tb = m.body_name2id("tripod_obj_root")
        trip0 = sim.data.body_xpos[tb].copy()
        targets = [fk(q[t]) for t in range(T)]
        g_needle = m.geom_name2id("needle_obj_needle")
        g_ring = [m.geom_name2id(f"tripod_obj_ring_{i}") for i in range(e.tripod.num_ring_geoms)]
        radius = float(e.tripod.ring_size[1])
        dmin, dvec, inhand = np.inf, None, []
        gq = [m.get_joint_qpos_addr(n) for n in tv.ROBOTIQ_DRIVER_JOINTS]
        g_real = np.asarray(d["obs/real_gripper"], np.float64) if "obs/real_gripper" in d else None
        close_t = int(np.argmax(acts[:, 6] > 0))
        n0 = sim.data.body_xpos[nb].copy()
        # wrist view while carried, before the tripod enters it: needle mask sim vs real
        needle_geoms = np.array([i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith("needle")])
        wcmp, n_wcmp = [], 0
        err, zmax, first, frames = [], z0, -1, []
        LOG = dict(tcp_sim=[], tcp_real=[], needle=[], needle_tip=[], ring=[], tripod=[], grip_sim=[], ok=[])
        for t in range(T - 1 + args.extra_steps):
            k = min(t + 1, T - 1)
            p, R = site()
            if args.mode == "track":
                pt, Rt = targets[k]
                a = np.concatenate([args.gain * (pt - p) / POS_SCALE,
                                    args.gain * axisangle(Rt @ R.T) / ROT_SCALE,
                                    [acts[min(t, T - 1), 6]]])
            else:
                a = acts[t] if t < T else np.r_[np.zeros(6), acts[-1, 6]]
            e.timestep = 0
            e.step(np.clip(a, -1, 1))
            # contact health (the wrapper's probe; e.step bypasses the wrapper's own bookkeeping)
            for k_, v_ in env._contact_probe().items():
                env.contact_health[k_] = max(env.contact_health.get(k_, 0.0), v_)
            sim.forward()
            p, R = site()
            err.append(np.linalg.norm(p - targets[k][0]))
            zmax = max(zmax, float(sim.data.body_xpos[nb][2]))
            if args.grasp_log and close_t - 25 <= t <= close_t + 1 and (np.linalg.norm(sim.data.body_xpos[nb] - n0) > 5e-4 or t >= close_t - 1):
                pairs = set()
                for c in sim.data.contact[:sim.data.ncon]:
                    n1, n2 = m.geom_id2name(c.geom1) or "", m.geom_id2name(c.geom2) or ""
                    if ("needle" in n1 and "gripper" in n2) or ("needle" in n2 and "gripper" in n1):
                        gname = n1 if "gripper" in n1 else n2
                        nname = n2 if "gripper" in n1 else n1
                        pairs.add(f"{gname.replace('gripper0_', '')}~{nname.replace('needle_obj_', '')}")
                    if ("needle" in n1 and "table" in n2) or ("needle" in n2 and "table" in n1):
                        pairs.add("table")
                gs = sim.data.qpos[gq[0]]
                gr = tv.robotiq_real_to_sim(g_real[min(t + 1, T - 1)])[0] if g_real is not None else np.nan
                dn = 1e3 * (sim.data.body_xpos[nb] - n0)
                print(f"     t {t:3d} a_g {a[6]:+.0f} | driver sim {gs:.3f} real->sim {gr:.3f} | needle moved "
                      f"{np.round(dn, 1).tolist()} mm | {sorted(pairs)}", flush=True)
            ring = np.mean([sim.data.geom_xpos[g] for g in g_ring], 0)
            LOG['tcp_sim'].append(p.copy()); LOG['tcp_real'].append(targets[k][0].copy()); LOG['needle'].append(sim.data.body_xpos[nb].copy()); LOG['needle_tip'].append(sim.data.geom_xpos[g_needle].copy()); LOG['ring'].append(ring.copy()); LOG['tripod'].append(sim.data.body_xpos[tb].copy()); LOG['grip_sim'].append(float(sim.data.qpos[gq[0]])); LOG['ok'].append(bool(e._check_success())); LOG.setdefault('needle_quat', []).append(sim.data.body_xquat[nb].copy()); LOG.setdefault('kn_real', []).append(float(g_real[k]) if g_real is not None else np.nan)
            dn = sim.data.geom_xpos[g_needle] - ring
            if np.linalg.norm(dn) < dmin:
                dmin, dvec = float(np.linalg.norm(dn)), dn.copy()
            if float(sim.data.body_xpos[nb][2]) > z0 + 0.03:     # carried: needle pose in the grip-site frame
                inhand.append(R.T @ (sim.data.body_xpos[nb] - p))
                if n_wcmp < 8 and t + 1 < T and "obs/robot0_eye_in_hand_image" in d:
                    n_wcmp += 1
                    seg = np.flipud(sim.render(camera_name="robot0_eye_in_hand", width=224, height=224,
                                               segmentation=True))[..., 1]
                    ms = np.isin(seg, needle_geoms)
                    mr = redmask_rgb(jpeg(d["obs/robot0_eye_in_hand_image"][t + 1]))
                    if ms.sum() > 50 and mr.sum() > 50:
                        cs = np.argwhere(ms).mean(0); cr = np.argwhere(mr).mean(0)
                        wcmp.append(((ms & mr).sum() / (ms | mr).sum(), *(cs - cr)))
            if first < 0 and e._check_success():
                first = t
            if args.video and (t % 2 == 0 or args.video_every1):
                im = [np.flipud(sim.render(camera_name=c, width=224, height=224)) for c in cams]
                tt = min(t, T - 1)
                real = [jpeg(d[f"obs/{c}_image"][tt]) if f"obs/{c}_image" in d else np.zeros_like(im[0])
                        for c in cams]
                real = [r if r.shape == im[0].shape else np.zeros_like(im[0]) for r in real]
                frames.append(np.concatenate([np.concatenate(im, 1), np.concatenate(real, 1)], 0))
        err = np.array(err[:T - 1])
        os.makedirs(args.video, exist_ok=True); np.savez(os.path.join(args.video, f'{name}_log.npz'), base=sim.data.body_xpos[m.body_name2id('robot0_base')], **{k_: np.array(v_) for k_, v_ in LOG.items()})
        ok = bool(e._check_success())
        nz = sim.data.body_xpos[nb].copy()
        row = dict(demo=name, T=T, success_end=ok, first_success=first,
                   lifted_mm=1e3 * (zmax - z0), err_p50_mm=1e3 * np.median(err), err_max_mm=1e3 * err.max(),
                   needle_end=np.round(nz, 3).tolist(),
                   tripod_moved_mm=float(1e3 * np.linalg.norm(sim.data.body_xpos[tb] - trip0)), dmin_mm=1e3 * dmin, radius_mm=1e3 * radius,
                   dvec_mm=np.round(1e3 * dvec, 1).tolist(),
                   inhand_mm=(np.round(1e3 * np.median(inhand, 0), 1).tolist() if inhand else None),
                   inhand_sd_mm=(np.round(1e3 * np.std(inhand, 0), 1).tolist() if inhand else None),
                   wrist_iou=(float(np.median([w[0] for w in wcmp])) if wcmp else None),
                   wrist_dpx=(np.round(np.median([w[1:] for w in wcmp], 0), 1).tolist() if wcmp else None))
        h = dict(env.contact_health)                              # per-episode maxima, filled in the step loop
        row.update(table_force_max_N=h.get("table_force"), table_impact=bool(getattr(env, "table_impact", False)),
                   pen_needle_tripod_mm=1e3 * h.get("pen_needle_tripod", 0.0))
        rows.append(row)
        if h:
            print(f"  {name}: contact health: gripper-table force max {h.get('table_force', 0):.1f} N "
                  f"(impact {row['table_impact']}) | needle-ring pen {1e3 * h.get('pen_needle_tripod', 0):.2f} mm | "
                  f"needle-table {1e3 * h.get('pen_needle_table', 0):.2f} mm", flush=True)
        print(f"  {name}: T {T:3d} | success end {ok!s:5s} first {first:4d} | needle lifted "
              f"{row['lifted_mm']:5.1f} mm | TCP err p50 {row['err_p50_mm']:4.1f} max "
              f"{row['err_max_mm']:5.1f} mm | tripod moved {row['tripod_moved_mm']:.1f} mm | needle-ring min {row['dmin_mm']:5.1f} mm (radius "
              f"{row['radius_mm']:.1f}) at {row['dvec_mm']} | needle in grip frame {row['inhand_mm']} "
              f"sd {row['inhand_sd_mm']} | carried, wrist needle IoU sim/real {row['wrist_iou']} "
              f"centroid sim-real (row, col) px@224 {row['wrist_dpx']}", flush=True)
        if args.video and frames:
            import imageio
            os.makedirs(args.video, exist_ok=True)
            out = os.path.join(args.video, f"{name}_{args.mode}.mp4")
            imageio.mimsave(out, frames, fps=5)
    n = len(rows)
    print(f"[replay] {args.mode}: success {sum(r['success_end'] for r in rows)}/{n}, "
          f"ever {sum(r['first_success'] >= 0 for r in rows)}/{n}, "
          f"lifted >10 mm {sum(r['lifted_mm'] > 10 for r in rows)}/{n}")


if __name__ == "__main__":
    main()
