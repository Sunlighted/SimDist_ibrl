"""Convert a raw GELLO teleop recording into the stage-2 demo contract.

Input  : the recorder's per-episode hdf5 (`collect_demos.py`) -- JOINT space,
         native-resolution JPEG, one file per episode.
Output : one `dataset.hdf5` in the layout `train_simdist_adapt.py
         --target_demo_path` reads, per docs/real-teleop-stage2.md §5, plus an
         `env_cfg.json` sidecar.

Gate: scripts/validate_real_demos.py on the output. This script deliberately
does NOT re-implement those checks -- it produces, the validator judges.

★ THE ONE DESIGN DECISION WORTH READING
Forward kinematics is done BY THE SIM ITSELF: the measured joint angles are
written into the sim robot's qpos and the TCP/hand pose is read straight back
out of MuJoCo. It is not an external FK plus a hand-derived correction.

That matters because the contract's two nastiest traps -- prop position is the
TCP (`gripper0_grip_site`) while prop orientation is the HAND BODY
(`robot0_right_hand`, 90 deg off), and the 45.03 deg flange rotation of iron law
#27 -- are both *definitions inside the sim model*. Reading them out of the
model cannot get them wrong; re-deriving them can, and has. It also puts the
result in the sim WORLD frame for free (the cell places the base at
(-0.489,-0.034,0.912)), which is trap #1.

The assumption this buys it with: the real encoders and the sim model agree on
joint order and zero. Checked per episode -- frame 0 must land on the Franka
ready pose, which the recorder forces (`start_pose_forced`).

ACTIONS. The recording stores a joint-space command (`action/joint_positions` =
the GELLO target), not an EE command. Default `--action_source cmd_fk` FKs the
target as well and forms

    a[0:3] = (p_target - p_measured) / 0.05        clipped to [-1,1]
    a[3:6] = axisangle(R_target . R_measured^T) / 0.5
    a[6]   = +1 closing / -1 opening               (sign only, per the contract)

which is exactly OSC_POSE's semantics ("target = current measured TCP + delta"),
and is self-consistent for a P-type follower: target-minus-measured IS the lag,
and OSC's action is the same quantity. `--action_source delta_meas` differences
the measured poses instead; the contract says that is the WRONG thing to store
(it is the result, not the command), so it exists only for comparison -- the
summary prints both distributions either way.

Usage:
  python scripts/convert_real_teleop.py \
      --raw_dir /user_data/Chenyang/tmp_atm/threading2 \
      --out     /user_data/Chenyang/real_stage2/threading_atm/dataset.hdf5 \
      --expert  exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt \
      --cell    release/cfgs/sim2real/threading_real_cell_calibrated_wrist.yaml \
      --success release/cfgs/sim2real/threading_atm_success.csv
"""
import argparse
import dataclasses
import glob
import hashlib
import io
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pyrallis  # noqa: E402
from PIL import Image  # noqa: E402

import train_rl  # noqa: E402
from env.sim2real import Sim2RealConfig, load_sim2real  # noqa: E402
from real2sim.deploy.preprocess import Rectifier, camera_intrinsics  # noqa: E402
from scripts.generate_simdist_data import (  # noqa: E402
    _RES_OVERRIDE, _write_image_dataset, build_source_env, resolve_domain_shift,
)

# Which recorder camera feeds which sim camera. The crop recipe per camera is
# NOT hard-coded here: it is derived from the cell's fovy and the intrinsics the
# recorder stored in the episode, so a cell edit propagates and a camera swap is
# caught rather than silently mis-cropped.
CAM_MAP = {"scene": "agentview", "wrist": "robot0_eye_in_hand"}
OSC_POS_SCALE = 0.05      # m per unit action, robosuite OSC_POSE default
OSC_ROT_SCALE = 0.5       # rad per unit action
READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])
OBJECT_DIM = 28           # STATE_KEYS["Threading_D0"] -> obs/object placeholder


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def undistort_crop(img, K, D, fovy_deg, out=224):
    """Undistort + square crop about the PRINCIPAL POINT at the sim fovy + resize.

    Centring on the principal point rather than the image centre is not a
    detail: MuJoCo's camera is a symmetric pinhole with no principal-point
    offset, so an off-centre crop is a constant pixel-to-geometry shift that no
    amount of training removes (docs/real-teleop-stage2.md §3b).
    Delegates to the deploy Rectifier so offline demos and the live policy see
    the same pixels (it was a separate integer-crop copy until 2026-10-11)."""
    key = (img.shape[1], img.shape[0], tuple(np.ravel(K)), tuple(np.ravel(D)), fovy_deg, out)
    r = _RECT.get(key)
    if r is None:
        r = _RECT[key] = Rectifier(K, D, img.shape[1], img.shape[0], out, fovy_deg)
    return r(img), r.side, (1,) if r.padded else ()


_RECT = {}


def read_success_spec(path, names):
    """csv: episode,success[,success_frame]   (-1 or blank = last frame)."""
    spec = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.lower().startswith("episode,"):
                continue
            parts = [p.strip() for p in ln.split(",")]
            ep = parts[0]
            ok = parts[1] not in ("0", "false", "False", "no", "")
            fr = int(parts[2]) if len(parts) > 2 and parts[2] not in ("", "-1") else -1
            spec[ep] = (ok, fr)
    unknown = sorted(set(spec) - set(names))
    if unknown:
        raise SystemExit(f"--success names episodes that are not in --raw_dir: {unknown}")
    return spec


class SimFK:
    """The sim, used only as a forward-kinematics oracle in the WORLD frame."""

    def __init__(self, expert, cell_path):
        _RES_OVERRIDE["image_size"] = 84
        _RES_OVERRIDE["rl_image_size"] = 84
        cfg = pyrallis.load(  # type: ignore
            train_rl.MainConfig, open(os.path.join(os.path.dirname(expert), "cfg.yaml")))
        cell = load_sim2real(cell_path)
        # Franka-hand era (READY pose, finger width, world-frame prop). The 10-14 Robotiq
        # cell changes all three (DROID home, ONE knuckle angle via robotiq_real_to_sim,
        # prop_frame "base" via env.sim2real.eef_in_base) -- port before using it there,
        # do not let it write a silently wrong prop (docs/real-teleop-stage2.md §2b).
        if cell.prop_frame != "world" or "Robotiq" in str(cell.gripper_types):
            raise SystemExit(f"{cell_path}: prop_frame={cell.prop_frame!r} / gripper "
                             f"{cell.gripper_types!r} -- this converter is not ported to that "
                             f"cell yet (docs/real-teleop-stage2.md §2b)")
        # Geometry only. Base-pose DR would move the WORLD<-BASE transform that
        # every prop value is expressed in, so it must be off.
        for f in dataclasses.fields(Sim2RealConfig):
            if f.name.startswith("dr_"):
                setattr(cell, f.name, f.default)
        env, _ = build_source_env(
            cfg, 0.0, resolve_domain_shift(argparse.Namespace(domain="expert"), cfg),
            rl_cameras=("agentview",))
        env.sim2real = cell
        env._s2r_active = True
        env.reset()
        self.env = env
        self.sim = env.env.sim
        rb = env.env.robots[0]
        self.qi = np.asarray(rb._ref_joint_pos_indexes)
        self.gi = np.asarray(rb._ref_gripper_joint_pos_indexes)
        m = self.sim.model
        self.site = m.site_name2id("gripper0_grip_site")
        self.hand = m.body_name2id("robot0_right_hand")
        self.base = self.sim.data.body_xpos[m.body_name2id("robot0_base")].copy()
        self.cell = cell

    def fk(self, q, width=None):
        d = self.sim.data
        d.qpos[self.qi] = q
        if width is not None and len(self.gi) == 2:
            d.qpos[self.gi] = [width / 2.0, -width / 2.0]
        d.qvel[:] = 0.0
        self.sim.forward()
        return (d.site_xpos[self.site].copy(),               # TCP, world
                d.body_xmat[self.hand].reshape(3, 3).copy())  # HAND body, world

    def close(self):
        self.env.env.close()


def mat_to_quat_xyzw(R):
    from robosuite.utils.transform_utils import mat2quat
    return np.asarray(mat2quat(R), dtype=np.float64)      # robosuite returns xyzw


def mat_to_axisangle(R):
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle
    return np.asarray(quat2axisangle(mat2quat(R)), dtype=np.float64)


def convert_episode(path, fk, args, spec):
    name = os.path.basename(path)[:-3]
    h = h5py.File(path, "r")
    T = int(h.attrs["num_frames"])
    q_meas = np.asarray(h["observation/joint_positions"], dtype=np.float64)
    q_cmd = np.asarray(h["action/joint_positions"], dtype=np.float64)
    width = np.asarray(h["observation/gripper_width"], dtype=np.float64)
    g_cmd = np.asarray(h["action/gripper_width_percent"], dtype=np.float64)
    wall = np.asarray(h["time/wall"], dtype=np.float64)
    intr = json.loads(h.attrs["camera_intrinsics"])

    # --- sanity: does frame 0 sit on the ready pose the sim inits from? -------
    d0 = np.degrees(q_meas[0] - READY)
    if np.abs(d0).max() > args.ready_tol_deg:
        raise SystemExit(
            f"{name}: frame 0 is {np.abs(d0).max():.1f} deg from the Franka ready pose "
            f"(per joint {np.round(d0,2).tolist()}). Either the recorder did not force "
            f"the start pose, or the joint ORDER/zero disagrees with the sim model -- "
            f"in which case every prop value below would be wrong. Raise "
            f"--ready_tol_deg only if you know why.")

    # --- forward kinematics ---------------------------------------------------
    p_meas = np.zeros((T, 3)); R_meas = np.zeros((T, 3, 3))
    p_cmd = np.zeros((T, 3)); R_cmd = np.zeros((T, 3, 3))
    for t in range(T):
        p_meas[t], R_meas[t] = fk.fk(q_meas[t], width[t])
        p_cmd[t], R_cmd[t] = fk.fk(q_cmd[t], width[t])

    # --- gripper polarity, decided from the data and reported ------------------
    # The contract is SIGN ONLY (g > 0 closes). Which end of gripper_width_percent
    # means "closed" is a recorder convention, so it is measured, not assumed.
    fin = np.isfinite(g_cmd) & np.isfinite(width)
    corr = float(np.corrcoef(g_cmd[fin], width[fin])[0, 1]) if fin.sum() > 2 else float("nan")
    mid = 0.5 * (np.nanmax(g_cmd) + np.nanmin(g_cmd))
    closing = (g_cmd < mid) if (np.isnan(corr) or corr >= 0) else (g_cmd > mid)
    a_grip = np.where(closing, 1.0, -1.0)

    # --- actions ---------------------------------------------------------------
    dp_cmd = (p_cmd - p_meas) / OSC_POS_SCALE
    dr_cmd = np.array([mat_to_axisangle(R_cmd[t] @ R_meas[t].T) for t in range(T)]) / OSC_ROT_SCALE
    dp_dm = np.zeros_like(dp_cmd); dr_dm = np.zeros_like(dr_cmd)
    dp_dm[:-1] = (p_meas[1:] - p_meas[:-1]) / OSC_POS_SCALE
    dr_dm[:-1] = np.array([mat_to_axisangle(R_meas[t + 1] @ R_meas[t].T)
                           for t in range(T - 1)]) / OSC_ROT_SCALE
    src = dict(cmd_fk=(dp_cmd, dr_cmd), delta_meas=(dp_dm, dr_dm))[args.action_source]
    actions = np.clip(np.concatenate([src[0], src[1], a_grip[:, None]], axis=1), -1.0, 1.0)

    # --- prop -------------------------------------------------------------------
    eef_pos = p_meas.astype(np.float32)
    eef_quat = np.array([mat_to_quat_xyzw(R_meas[t]) for t in range(T)], dtype=np.float32)
    grip_qpos = np.stack([width / 2.0, -width / 2.0], axis=1).astype(np.float32)

    # --- trim -------------------------------------------------------------------
    lo = 0
    if args.trim_idle:
        mov = np.abs(np.diff(q_meas, axis=0)).max(axis=1)
        moving = np.where(mov > np.radians(args.trim_idle_deg))[0]
        if len(moving):
            lo = max(0, int(moving[0]) - args.trim_idle_keep)
    hi = T
    if args.max_steps and (hi - lo) > args.max_steps:
        lo = hi - args.max_steps          # keep the TAIL: the task ends there

    ok, frame = spec.get(name, (None, -1))
    if ok is None:
        if not args.assume_success_last:
            raise SystemExit(
                f"{name}: no entry in --success and --assume_success_last not set. "
                f"The stage-2 loader DROPS any demo without a reward==1 step, so an "
                f"unlabelled episode would silently vanish. Label it.")
        ok, frame = True, -1
    succ_idx = (T - 1) if frame < 0 else int(frame)
    rewards = np.zeros(T, dtype=np.float32)
    if ok:
        if not (lo <= succ_idx < hi):
            raise SystemExit(f"{name}: success frame {succ_idx} falls outside the kept "
                             f"window [{lo},{hi}) -- trimming would delete it.")
        rewards[succ_idx] = 1.0

    # --- images -------------------------------------------------------------------
    frames = {}
    crop_info = {}
    for raw_cam, sim_cam in CAM_MAP.items():
        k = intr[raw_cam]["K"]
        K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], dtype=np.float64)
        D = np.array(intr[raw_cam]["D"], dtype=np.float64)
        cc = [c for c in fk.cell.cameras if c.name == sim_cam]
        if not cc:
            raise SystemExit(f"cell {args.cell} has no camera {sim_cam!r}; the stage-2 "
                             f"contract needs both {sorted(CAM_MAP.values())}")
        fovy = float(cc[0].resolved_fovy())
        ds = h[f"observation/images/{raw_cam}"]
        # The cell's real_intrinsics (a re-calibration of this camera) beats the
        # factory K, D the recorder stored: the back-of-hand ZED-M's factory D is wrong.
        w0, h0 = Image.open(io.BytesIO(np.asarray(ds[lo]).tobytes())).size
        K, D = camera_intrinsics(cc[0], K, D, w0, h0)
        out = np.zeros((hi - lo, 3, 224, 224), dtype=np.uint8)
        for j, t in enumerate(range(lo, hi)):
            img = np.array(Image.open(io.BytesIO(np.asarray(ds[t]).tobytes())).convert("RGB"))
            sq, side, pad = undistort_crop(img, K, D, fovy)
            out[j] = sq.transpose(2, 0, 1)
            if j == 0:
                crop_info[sim_cam] = dict(fovy=fovy, side=side, padded=bool(any(pad)),
                                          src=list(img.shape), fy=float(K[1, 1]),
                                          cx=float(K[0, 2]), cy=float(K[1, 2]),
                                          intrinsics=cc[0].real_intrinsics or "recorder (factory)")
        frames[sim_cam] = out

    cap = {}
    for raw_cam, sim_cam in CAM_MAP.items():
        age = np.asarray(h[f"age/{raw_cam}"], dtype=np.float64)
        cap[sim_cam] = (wall - age)[lo:hi]
    h.close()

    return dict(
        name=name, T=hi - lo, T_raw=T, lo=lo, hi=hi, success=bool(ok),
        actions=actions[lo:hi].astype(np.float32), rewards=rewards[lo:hi],
        eef_pos=eef_pos[lo:hi], eef_quat=eef_quat[lo:hi], gripper=grip_qpos[lo:hi],
        timestamp=wall[lo:hi], capture=cap, frames=frames, crop=crop_info,
        grip_corr=corr, succ_idx=succ_idx - lo if ok else -1,
        diag=dict(a_cmd=np.abs(np.concatenate([dp_cmd, dr_cmd], 1)[lo:hi]).max(0),
                  a_dm=np.abs(np.concatenate([dp_dm, dr_dm], 1)[lo:hi]).max(0),
                  clipped=float((np.abs(np.concatenate([src[0], src[1]], 1)[lo:hi]) > 1).mean())),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="dir of the recorder's episode_*.h5")
    ap.add_argument("--out", required=True, help="path to write dataset.hdf5")
    ap.add_argument("--expert", default="exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt",
                    help="supplies the task + env the FK oracle is built from")
    ap.add_argument("--cell", default="release/cfgs/sim2real/threading_real_cell_calibrated_wrist.yaml",
                    help="THE source of both camera FOVs and the base pose. Must be the "
                         "same cell stage-1 was trained on, or the crops disagree.")
    ap.add_argument("--success", default="", help="csv: episode,success[,success_frame]")
    ap.add_argument("--assume_success_last", action="store_true",
                    help="DEBUG: treat every episode as a success ending on its last "
                         "frame. Never use this for a dataset you will train on.")
    ap.add_argument("--env_name", default="Threading_D0", help="written to env_cfg.json")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="0 = keep the full episode (default). Anything else keeps the "
                         "TAIL, which can cut the grasp out -- check before using it.")
    ap.add_argument("--trim_idle", type=int, default=0,
                    help="1 = drop the stationary lead-in before the first motion")
    ap.add_argument("--trim_idle_deg", type=float, default=0.05,
                    help="per-step max joint motion counted as 'moving' (deg)")
    ap.add_argument("--trim_idle_keep", type=int, default=10, help="steps kept before it")
    ap.add_argument("--action_source", default="cmd_fk", choices=("cmd_fk", "delta_meas"))
    ap.add_argument("--ready_tol_deg", type=float, default=5.0,
                    help="the start-pose check is there to catch a joint ORDER/zero "
                         "mismatch (which shows up as tens of degrees), not to reject "
                         "normal variation -- training adds N(0, 0.1 rad = 5.7 deg) per "
                         "joint, so a single joint at ~6 deg is in-distribution")
    ap.add_argument("--drop_longer_than", type=int, default=0,
                    help="skip whole episodes longer than this many steps (0 = keep "
                         "all). Distinct from --max_steps, which TRIMS an episode and "
                         "can cut the grasp out; this drops it instead.")
    args = ap.parse_args()

    raws = sorted(glob.glob(os.path.join(args.raw_dir, "*.h5")))
    good = []
    for p in raws:
        try:
            with h5py.File(p, "r"):
                good.append(p)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {os.path.basename(p)}: unreadable ({type(e).__name__}) -- "
                  f"truncated transfer?")
    if not good:
        raise SystemExit(f"no readable episodes in {args.raw_dir}")
    if args.drop_longer_than:
        keep = []
        for p in good:
            with h5py.File(p, "r") as h:
                T = int(h.attrs["num_frames"])
            if T <= args.drop_longer_than:
                keep.append(p)
            else:
                print(f"[drop] {os.path.basename(p)[:-3]}: {T} steps "
                      f"({T*0.05:.1f} s) > {args.drop_longer_than}")
        print(f"[filter] kept {len(keep)}/{len(good)} episodes at "
              f"<= {args.drop_longer_than} steps ({args.drop_longer_than*0.05:.1f} s)")
        good = keep
        if not good:
            raise SystemExit("--drop_longer_than removed every episode")
    names = [os.path.basename(p)[:-3] for p in good]
    spec = read_success_spec(args.success, names) if args.success else {}

    fk = SimFK(args.expert, args.cell)
    print(f"[fk] base in world = {np.round(fk.base, 4).tolist()}  "
          f"(prop is WORLD frame; base-frame storage is the classic 0.912 m error)")

    eps = [convert_episode(p, fk, args, spec) for p in good]
    fk.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        g = f.create_group("data")
        for i, e in enumerate(eps):
            d = g.create_group(f"demo_{i}")
            o = d.create_group("obs")
            o.create_dataset("robot0_eef_pos", data=e["eef_pos"])
            o.create_dataset("robot0_eef_quat", data=e["eef_quat"])
            o.create_dataset("robot0_gripper_qpos", data=e["gripper"])
            o.create_dataset("object", data=np.zeros((e["T"], OBJECT_DIM), np.float32))
            o.create_dataset("timestamp", data=e["timestamp"])
            for cam, arr in e["frames"].items():
                _write_image_dataset(o, f"{cam}_image", arr, args.jpeg_quality)
                o.create_dataset(f"{cam}_capture_ts", data=e["capture"][cam])
            d.create_dataset("actions", data=e["actions"])
            d.create_dataset("rewards", data=e["rewards"])
            d.attrs["success"] = bool(e["success"])
            d.attrs["num_samples"] = int(e["T"])
            d.attrs["source_episode"] = e["name"]
            d.attrs["source_window"] = [e["lo"], e["hi"]]
        g.attrs["num_demos"] = len(eps)
        g.attrs["total"] = int(sum(e["T"] for e in eps))

    side = dict(env_name=args.env_name, control_hz=20.0,
                action_source=args.action_source,
                osc_pos_scale=OSC_POS_SCALE, osc_rot_scale=OSC_ROT_SCALE,
                cell=args.cell, cell_sha256=sha256(args.cell),
                expert=args.expert, fk="sim model (gripper0_grip_site / robot0_right_hand)",
                base_in_world=np.round(fk.base, 6).tolist(),
                crops=eps[0]["crop"], jpeg_quality=args.jpeg_quality,
                raw_dir=args.raw_dir,
                raw_sha256={e["name"]: sha256(p) for e, p in zip(eps, good)})
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "env_cfg.json"), "w") as fh:
        json.dump(side, fh, indent=1)

    # ---- summary ---------------------------------------------------------------
    print(f"\nwrote {args.out}  ({len(eps)} demos, {sum(e['T'] for e in eps)} transitions)")
    for cam, c in eps[0]["crop"].items():
        print(f"  [crop] {cam}: fovy {c['fovy']:.3f} deg -> {c['side']} px of "
              f"{c['src'][0]} about ({c['cx']:.1f},{c['cy']:.1f}) -> 224"
              + ("   ⚠ needed padding" if c["padded"] else ""))
    print(f"  [gripper] corr(cmd_pct, measured_width) = {eps[0]['grip_corr']:+.3f} "
          f"-> {'low pct = closing' if eps[0]['grip_corr'] >= 0 else 'high pct = closing'}")
    print(f"\n{'episode':14s} {'T':>5s} {'kept':>11s} {'succ':>5s} {'@':>5s}  "
          f"{'max|a| cmd_fk (xyz|rpy)':<34s} {'max|a| delta_meas':<26s} clip%")
    for e in eps:
        c, m = e["diag"]["a_cmd"], e["diag"]["a_dm"]
        print(f"{e['name']:14s} {e['T_raw']:5d} {str([e['lo'], e['hi']]):>11s} "
              f"{str(e['success']):>5s} {e['succ_idx']:5d}  "
              f"{np.round(c[:3],2).tolist()}|{np.round(c[3:],2).tolist()}  "
              f"{np.round(m[:3],2).tolist()}|{np.round(m[3:],2).tolist()}  "
              f"{100*e['diag']['clipped']:.1f}")
    P = np.concatenate([e["eef_pos"] for e in eps])
    print(f"\n  eef_pos (world) median {np.round(np.median(P,0),3).tolist()}  "
          f"x[{P[:,0].min():.3f},{P[:,0].max():.3f}] y[{P[:,1].min():.3f},{P[:,1].max():.3f}] "
          f"z[{P[:,2].min():.3f},{P[:,2].max():.3f}]")
    print( "  sim stage-1 1%-99%:            x[-0.373,0.350] y[-0.543,0.790] z[0.925,1.473]")
    print(f"\nnext: python scripts/validate_real_demos.py --dataset {args.out}")


if __name__ == "__main__":
    main()
