"""Render a sim2real cell config and report whether the workspace is VISIBLE.

The occlusion report is the point. A camera pose can be perfectly calibrated and
still be useless: if the robot sits between the lens and the workpiece, the
policy simply never sees the thing it has to align. That failure is invisible in
a pose table and obvious in one render, so this script prints both.

Criterion: project each object into the image and compare its depth against the
robot base's. base_depth < object_depth means the arm is in front of it.

Usage:
  python scripts/preview_sim2real.py --config release/cfgs/sim2real/threading_real_cell.yaml
  python scripts/preview_sim2real.py --config ... --cam_pos 0.60 0.48 0.78   # try a move
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real

CAMS = ["agentview", "robot0_eye_in_hand"]
OBJECTS = ("needle_obj_root", "tripod_obj_root")


def hole_report(env, cam_name, res):
    """Angular resolution of the RING OPENING -- the 20 x 20 mm hole the needle
    has to be threaded through.

    Worth measuring separately from "is the object in frame", because the two
    answer different questions and only this one predicts whether a policy could
    ever align from this view. Measured here: 4.2 x 3.6 px from the real cell's
    camera. For scale, sim's own agentview gets 2.8 x 12.9 px at 74.7 deg
    incidence -- i.e. even the training-time third-person view never resolved the
    hole, which is why insertion is the wrist camera's job.

    `incidence` is the angle between the view ray and the hole's face normal
    (0 = looking straight down the hole, 90 = edge-on, hole invisible). The
    normal is recovered as the least-variance direction of the ring geoms, which
    needs no assumption about how the tripod was rotated.
    """
    m, d = env.env.sim.model, env.env.sim.data
    cid = m.camera_name2id(cam_name)
    cp = d.cam_xpos[cid].copy()
    R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    f = (res / 2) / np.tan(np.radians(float(m.cam_fovy[cid])) / 2)
    P = np.array([d.geom_xpos[g] for g in range(m.ngeom)
                  if (m.geom_id2name(g) or "").startswith("tripod_obj_ring")
                  and (m.geom_id2name(g) or "").endswith("_vis")])
    if len(P) < 4:
        return None
    pc = (P - cp) @ R
    if (pc[:, 2] <= 0).any():
        return None
    u = f * pc[:, 0] / pc[:, 2] + res / 2
    v = f * pc[:, 1] / pc[:, 2] + res / 2
    ctr = P.mean(0)
    dist = float(np.linalg.norm(ctr - cp))
    _, _, V = np.linalg.svd(P - ctr)
    inc = float(np.degrees(np.arccos(min(1.0, abs(V[2] @ ((ctr - cp) / dist))))))
    return u.max() - u.min(), v.max() - v.min(), dist, inc


def visible_fraction(env, cam_name, prefix):
    """Fraction of an object's geoms actually reachable by a ray from the camera.

    Replaces an earlier heuristic that compared the object's depth against
    `robot0_base`'s and called anything nearer than the base "occluded". That was
    a FALSE POSITIVE machine: the base is the pedestal, not the arm, so a camera
    mounted anywhere above the robot reports every workpiece as occluded while
    the render plainly shows it. Two rounds of camera-placement advice were built
    on that mistake.

    mj_ray returns the first geom hit. A hit on ANOTHER geom of the same object is
    self-occlusion, which does not stop you seeing the object, so it counts as
    visible; anything else (an arm link, a finger) does not.
    """
    import mujoco
    m, d = env.env.sim.model, env.env.sim.data
    mm = m._model if hasattr(m, "_model") else m
    dd = d._data if hasattr(d, "_data") else d
    cid = m.camera_name2id(cam_name)
    cp = d.cam_xpos[cid].copy()
    ids = [g for g in range(m.ngeom)
           if (m.geom_id2name(g) or "").startswith(prefix)
           and (m.geom_id2name(g) or "").endswith("_vis")]
    if not ids:
        ids = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(prefix)]
    if not ids:
        return None
    vis, blockers = 0, {}
    gid = np.zeros(1, dtype=np.int32)
    for g in ids:
        tgt = d.geom_xpos[g].copy()
        vec = tgt - cp
        n = np.linalg.norm(vec)
        if n < 1e-9:
            continue
        mujoco.mj_ray(mm, dd, cp, vec / n, None, 1, -1, gid)
        hit = m.geom_id2name(int(gid[0])) if gid[0] >= 0 else None
        if hit is not None and hit.startswith(prefix):
            vis += 1
        else:
            blockers[hit] = blockers.get(hit, 0) + 1
    worst = max(blockers, key=blockers.get) if blockers else None
    return vis / len(ids), worst


def occlusion_report(env, cam_name, res):
    m, d = env.env.sim.model, env.env.sim.data
    cid = m.camera_name2id(cam_name)
    cp = d.cam_xpos[cid].copy()
    # MuJoCo looks down -z with +y up; convert to the OpenCV optical frame so the
    # projection below is the textbook one.
    R_cv = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    f = (res / 2) / np.tan(np.radians(float(m.cam_fovy[cid])) / 2)
    rows = []
    for nm in OBJECTS:
        try:
            bid = m.body_name2id(nm)
        except Exception:
            continue
        pc = R_cv.T @ (d.body_xpos[bid] - cp)
        if pc[2] <= 0:
            rows.append((nm, None, None, None, "BEHIND"))
            continue
        u = f * pc[0] / pc[2] + res / 2
        v = f * pc[1] / pc[2] + res / 2
        inframe = 0 <= u < res and 0 <= v < res
        vf = visible_fraction(env, cam_name, nm.replace("_root", ""))
        frac, blocker = vf if vf else (1.0, None)
        if not inframe:
            tag = "OUT-OF-FRAME"
        elif frac < 0.5:
            tag = f"OCCLUDED {100*(1-frac):.0f}% by {blocker}"
        elif frac < 0.95:
            tag = f"partly occluded ({100*frac:.0f}% visible)"
        else:
            tag = "OK"
        rows.append((nm, u, v, pc[2], tag))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="figs/sim2real")
    ap.add_argument("--cam_pos", type=float, nargs=3, default=None,
                    help="override agentview position (base frame) for a what-if")
    ap.add_argument("--cam_look_at", type=float, nargs=3, default=None,
                    help="aim agentview at this point (base frame), replacing the "
                         "config's quat/look_dir. Moving a camera WITHOUT re-aiming "
                         "it just points the same ray somewhere else -- which is "
                         "exactly what happens on the real bench too.")
    ap.add_argument("--fovy", type=float, default=None,
                    help="override agentview vertical FOV. On a fixed-FOV camera "
                         "(a ZED) this corresponds to CROPPING the native image "
                         "about the optical axis and rescaling.")
    ap.add_argument("--no_dr", action="store_true", help="render the nominal only")
    args = ap.parse_args()

    cfg = load_sim2real(args.config)
    if args.cam_pos is not None:
        for c in cfg.cameras:
            if c.name == "agentview":
                c.pos = tuple(args.cam_pos)
                print(f"[override] agentview pos -> {c.pos}")
    if args.cam_look_at is not None:
        for c in cfg.cameras:
            if c.name == "agentview":
                c.quat = None
                c.look_dir = None
                c.look_at = tuple(args.cam_look_at)
                print(f"[override] agentview look_at -> {c.look_at}")
    if args.fovy is not None:
        for c in cfg.cameras:
            if c.name == "agentview":
                c.fovy = args.fovy
                c.fy = 0.0
                print(f"[override] agentview fovy -> {c.fovy}")
    if args.no_dr:
        # Reset every dr_* to its dataclass default, which is the INERT value.
        # (The old rule mapped any non-(1,1) tuple to (0,0), which turned the
        # MULTIPLICATIVE ranges -- light diffuse, rgb gain -- into x0: a black
        # frame.)
        import dataclasses
        from env.sim2real import Sim2RealConfig
        for f in dataclasses.fields(Sim2RealConfig):
            if f.name.startswith("dr_"):
                setattr(cfg, f.name, f.default)

    env = PixelRobosuite(
        "Threading_D0", "Panda", 400, image_size=args.res, rl_image_size=args.res,
        camera_names=CAMS, rl_cameras=CAMS, use_state=False, obs_stack=1,
        state_stack=1, prop_stack=1, flip_image=True, device="cpu",
        hand_init_perturb_joint_sigma=0.0, sim2real=cfg)

    os.makedirs(args.out, exist_ok=True)
    shots = {c: [] for c in CAMS}
    for i in range(args.num):
        np.random.seed(i)
        env.reset()
        obs, *_ = env.step(torch.zeros(env.action_dim))
        for c in CAMS:
            shots[c].append(obs[c].numpy().transpose(1, 2, 0).astype(np.uint8))
        if i == 0:
            m, d = env.env.sim.model, env.env.sim.data
            bid = m.body_name2id("robot0_base")
            bp = d.body_xpos[bid]; bR = d.body_xmat[bid].reshape(3, 3)
            sid = m.site_name2id("gripper0_grip_site")
            top = float(env.env.table_offset[2]) if cfg.table_above_base is not None else 0.8
            print(f"\ntable top world z   = {top:.4f}  "
                  f"({top - bp[2]:+.4f} m vs base origin)")
            print(f"eef above table     = {d.site_xpos[sid][2] - top:+.4f} m")
            for c in CAMS:
                rows = occlusion_report(env, c, args.res)
                print(f"\n[{c}]  fovy={float(m.cam_fovy[m.camera_name2id(c)]):.2f}")
                for nm, u, v, dep, tag in rows:
                    loc = "  (behind camera)" if u is None else \
                        f"  u={u:6.1f} v={v:6.1f} depth={dep:.3f}"
                    print(f"    {nm:18s}{loc}  -> {tag}")
                rv = visible_fraction(env, c, "tripod_obj_ring")
                if rv is not None:
                    print(f"    {'ring geoms visible':18s}  {100*rv[0]:5.1f} %"
                          + (f"   (blocked by {rv[1]})" if rv[1] else ""))
                hr = hole_report(env, c, args.res)
                if hr is not None:
                    w, h, dd, inc = hr
                    verdict = ("UNRESOLVABLE" if min(w, h) < 5 else
                               "marginal" if min(w, h) < 10 else "ok")
                    print(f"    {'RING OPENING':18s}  {w:5.1f} x {h:5.1f} px  "
                          f"at {dd:.3f} m, incidence {inc:4.1f} deg  -> {verdict}")

    S, P = args.res, 4
    cols = min(args.num, 3)
    rows_n = (args.num + cols - 1) // cols
    for c in CAMS:
        g = Image.new("RGB", (cols * (S + P) + P, rows_n * (S + P) + P), (30, 30, 30))
        for k, a in enumerate(shots[c]):
            g.paste(Image.fromarray(a), (P + (k % cols) * (S + P), P + (k // cols) * (S + P)))
        g.save(f"{args.out}/{c}.png")
    print(f"\nwrote {args.out}/agentview.png and {args.out}/robot0_eye_in_hand.png")


main()
