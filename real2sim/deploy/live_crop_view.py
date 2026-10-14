"""Live view of what the policy will see: both ZEDs -> the exact 224 crop of the sim camera.

Uses the deploy Rectifier (real2sim/deploy/preprocess.py) with the cell yaml's fovy and the same
K, D the policy server uses (scene: factory SN values; wrist: the cell's real_intrinsics), so
the window shows the policy input pixel for pixel (minus the splat colour map).

Top row: the two 224 crops (scaled up). Bottom row: the raw frames with the crop outline.
Red objects (needle, tripod) are outlined in the crops; a blob touching the crop edge is
flagged EDGE -- the needle must not be cut off.

--needle x,y,zrot draws a TARGET needle (sim NeedleObject geometry, base frame, z_rot in deg as in
the cell yaml) into the scene views in cyan, plus its placement range (--needle_range) on the table:
move the real needle onto the cyan outline.

Keys: f = toggle the scene fovy between the cell value and --alt_fovy (compare crops)
      SPACE = save raw frames + LIVE joints to --out (refused if joints are stale)
      q / ESC = quit

    cd ~/simdist/SimDist_ibrl
    # base env: its OpenCV has a GUI (Qt); the simdist env's OpenCV is headless
    ~/anaconda3/bin/python real2sim/deploy/live_crop_view.py \
        --out ~/simdist/calib/layout_1014
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from real2sim.deploy.preprocess import Rectifier  # noqa: E402

CELL = os.path.join(_REPO, "real2sim/zed_capture/calib_20261014/threading_real_cell_20261014.yaml")
FACTORY = os.path.join(_REPO, "release/cfgs/sim2real/zed_factory_intrinsics.json")
JOINTS = os.path.expanduser("~/simdist/gello_software/simdist_runner/joint_state_latest.json")
# Locked exposure / gain / white balance, same file and code as the camera publisher.
ZED_CONTROLS_DIR = os.path.expanduser(
    "~/simdist/gello_software/ros2/src/franka_gello_state_publisher/scripts")
CAMS = {"scene": ("ZED 2i", "agentview"), "wrist": ("ZED-M", "robot0_eye_in_hand")}
W, H, S = 1280, 720, 224


def find_device(model):
    for i in range(16):
        p = f"/sys/class/video4linux/video{i}/name"
        if os.path.exists(p) and open(p).read().startswith(model):
            return i
    raise SystemExit(f"{model} not found (USB 3? `cat /sys/bus/usb/devices/*/speed`)")


def load_cams(cell_path):
    cell = yaml.safe_load(open(cell_path))
    by_name = {c["name"]: c for c in cell["cameras"]}
    fac = json.load(open(FACTORY))
    out = {}
    for ros, (model, sim) in CAMS.items():
        c = by_name[sim]
        if c.get("real_intrinsics"):
            path = os.path.normpath(os.path.join(os.path.dirname(cell_path), c["real_intrinsics"]))
            j = json.load(open(path))
            K, D, src = np.array(j["K"]), np.array(j["D"]), path
        else:
            m = fac[ros]["modes"]["hd720"]
            K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1.0]])
            D, src = np.array(fac[ros]["distortion_k1k2p1p2k3"]), f"factory SN{fac[ros]['serial']}"
        out[ros] = dict(model=model, sim=sim, K=K, D=D, fovy=float(c["fovy"]), src=src)
    return out


def crop_outline(rect, K, D):
    """Crop border in RAW (distorted) pixel coordinates."""
    s = rect.side
    t = np.linspace(0, s - 1, 60)
    edge = np.r_[np.c_[t, 0 * t], np.c_[0 * t + s - 1, t], np.c_[t[::-1], 0 * t + s - 1],
                 np.c_[0 * t, t[::-1]]]
    rays = cv2.undistortPoints(edge.reshape(-1, 1, 2), rect.newK, None).reshape(-1, 2)
    p, _ = cv2.projectPoints(np.c_[rays, np.ones(len(rays))], np.zeros(3), np.zeros(3), K, D)
    return p.reshape(-1, 1, 2).astype(np.int32)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def needle_parts(x, y, zrot_deg, z_table):
    """Sim NeedleObject (mimicgen) in the base frame: shaft along body +y from the tip (-0.08)
    to 0.04, 20 mm cube handle 0.04..0.08; body origin 20 mm above the table."""
    c, s = np.cos(np.radians(zrot_deg)), np.sin(np.radians(zrot_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o = np.array([x, y, z_table + 0.02])
    handle = np.array([(hx, hy, hz) for hx in (-.02, .02) for hy in (.04, .08) for hz in (-.02, .02)])
    shaft = np.array([(0, -0.08, -0.015), (0, 0.04, -0.015)])
    return (Rz @ handle.T).T + o, (Rz @ shaft.T).T + o


def draw_target(img, project, needle, rng, z_table):
    """project: (N,3) base points -> (N,2) pixels of `img`."""
    x, y, zr = needle
    hx, hy = rng
    box = np.array([[x - hx, y - hy, z_table], [x + hx, y - hy, z_table],
                    [x + hx, y + hy, z_table], [x - hx, y + hy, z_table]])
    edge = np.concatenate([np.linspace(box[i], box[(i + 1) % 4], 20) for i in range(4)])
    cv2.polylines(img, [project(edge).astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 1)
    handle, shaft = needle_parts(x, y, zr, z_table)
    hull = cv2.convexHull(project(handle).astype(np.float32)).astype(np.int32)
    cv2.polylines(img, [hull], True, (255, 255, 0), 2)
    a, b = project(shaft).astype(np.int32)
    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 0), 2)


def red_blobs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 40))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    blobs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 15:
            continue
        edge = x <= 1 or y <= 1 or x + w >= img.shape[1] - 1 or y + h >= img.shape[0] - 1
        blobs.append(dict(mask=(lab == i), area=int(a), edge=bool(edge)))
    return sorted(blobs, key=lambda b: -b["area"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--alt_fovy", type=float, default=62.0, help="scene fovy shown with 'f'")
    ap.add_argument("--zoom", type=float, default=2.5, help="display scale of the 224 crops")
    ap.add_argument("--out", default="", help="SPACE saves shots here")
    ap.add_argument("--max_age", type=float, default=0.5)
    ap.add_argument("--auto_exposure", action="store_true",
                    help="leave the camera on auto exposure / WB (default: lock camera_controls.json)")
    ap.add_argument("--needle", default="", help="target needle 'x,y,zrot_deg' in the base frame")
    ap.add_argument("--needle_range", default="0.075,0.05", help="half range 'dx,dy' (m) to draw")
    args = ap.parse_args()
    cams = load_cams(args.cell)
    cell = yaml.safe_load(open(args.cell))
    z_table = 0.0207 + float(cell.get("table_top_dz", 0.0))     # mat + board (work surface)
    sc = next(c for c in cell["cameras"] if c["name"] == "agentview")
    R_bs, t_bs = quat_wxyz_to_R(sc["quat"]), np.array(sc["pos"], float)
    needle = [float(v) for v in args.needle.split(",")] if args.needle else None
    nrange = [float(v) for v in args.needle_range.split(",")]

    def scene_cam(P):
        return (R_bs.T @ (np.asarray(P, float) - t_bs).T).T
    for name, c in cams.items():
        c["dev"] = find_device(c["model"])
        cap = cv2.VideoCapture(c["dev"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {name}")
        c["cap"] = cap
        if not args.auto_exposure:   # every open resets the exposure
            sys.path.insert(0, ZED_CONTROLS_DIR)
            import zed_controls
            print(f"[{name}] controls locked:",
                  zed_controls.apply(c["dev"], fps=30, grab=lambda cap=cap: cap.read()))
        c["rects"] = {c["fovy"]: Rectifier(c["K"], c["D"], W, H, S, c["fovy"])}
        print(f"[{name}] -> {c['sim']}  fovy {c['fovy']}  crop {c['rects'][c['fovy']].side} px  K,D: {c['src']}")
    scene_fovy = cams["scene"]["fovy"]
    n_shot = len([d for d in os.listdir(args.out) if d.startswith("shot_")]) if args.out and \
        os.path.isdir(args.out) else 0
    msg, msg_t = "", 0.0
    Z = int(S * args.zoom)
    while True:
        raws, tiles, bottoms = {}, [], []
        for name, c in cams.items():
            ok, f = c["cap"].read()
            if not ok or f.shape[1] != 2 * W:
                raise SystemExit(f"[{name}] frame read failed (USB 2? wrong mode?)")
            raw = f[:, :W].copy()
            raws[name] = raw
            fovy = scene_fovy if name == "scene" else c["fovy"]
            if fovy not in c["rects"]:
                c["rects"][fovy] = Rectifier(c["K"], c["D"], W, H, S, fovy)
            rect = c["rects"][fovy]
            crop = rect(raw)
            blobs = red_blobs(crop)
            big = cv2.resize(crop, (Z, Z), interpolation=cv2.INTER_NEAREST)
            if needle and name == "scene":
                f = (Z / 2) / np.tan(np.radians(fovy) / 2)

                def proj_big(P, f=f):
                    p = scene_cam(P)
                    return np.c_[f * p[:, 0] / p[:, 2], f * p[:, 1] / p[:, 2]] + (Z - 1) / 2
                draw_target(big, proj_big, needle, nrange, z_table)
            for b in blobs:
                cs, _ = cv2.findContours(cv2.resize(b["mask"].astype(np.uint8), (Z, Z),
                                                    interpolation=cv2.INTER_NEAREST),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(big, cs, -1, (0, 0, 255) if b["edge"] else (0, 255, 0), 2)
            tag = "" if fovy == c["fovy"] else "  (ALT, not the cell)"
            cv2.putText(big, f"{name} -> {c['sim']}  fovy {fovy:g}{tag}", (8, 24), 0, 0.6,
                        (0, 255, 255), 2)
            txt = "red blobs @224: " + (", ".join(f"{b['area']}px{' EDGE' if b['edge'] else ''}"
                                                  for b in blobs[:3]) or "NONE")
            cv2.putText(big, txt, (8, Z - 12), 0, 0.6,
                        (0, 0, 255) if (not blobs or any(b["edge"] for b in blobs[:2])) else (0, 255, 0), 2)
            tiles.append(big)
            r = raw.copy()
            cv2.polylines(r, [crop_outline(rect, c["K"], c["D"])], True, (0, 255, 255), 3)
            if needle and name == "scene":
                def proj_raw(P, K=c["K"], D=c["D"]):
                    p, _ = cv2.projectPoints(scene_cam(P), np.zeros(3), np.zeros(3), K, D)
                    return p.reshape(-1, 2)
                draw_target(r, proj_raw, needle, nrange, z_table)
            bottoms.append(cv2.resize(r, (Z, int(Z * H / W))))
        frame = np.vstack([np.hstack(tiles), np.hstack(bottoms)])
        try:
            d = json.load(open(JOINTS))
            age = time.time() - d["t"]
        except Exception:
            d, age = None, float("inf")
        jtxt = f"joints {age:.2f} s old" if age < args.max_age else f"joints STALE ({age:.0f} s)"
        cv2.putText(frame, jtxt + ("" if not args.out else f"   SPACE=save shot_{n_shot:03d}")
                    + "   f=fovy  q=quit", (8, Z + 24), 0, 0.6,
                    (0, 255, 0) if age < args.max_age else (0, 0, 255), 2)
        if msg and time.time() - msg_t < 3:
            cv2.putText(frame, msg, (8, Z + 52), 0, 0.7, (255, 255, 0), 2)
        cv2.imshow("policy view (224 crops) -- live", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("f"):
            scene_fovy = args.alt_fovy if scene_fovy == cams["scene"]["fovy"] else cams["scene"]["fovy"]
        if k == ord(" ") and args.out:
            if age > args.max_age:
                msg = f"REFUSED: joints {age:.1f} s old (teleop + joint_tap up?)"
            else:
                sd = os.path.join(args.out, f"shot_{n_shot:03d}")
                os.makedirs(sd)
                for name, raw in raws.items():
                    cv2.imwrite(os.path.join(sd, f"{name}_left.png"), raw)
                cv2.imwrite(os.path.join(sd, "policy_view.png"), np.hstack(tiles))
                json.dump({"q": d["q"], "joint_t": d["t"], "capture_t": time.time(),
                           "joint_order": "fr3_joint1..7", "mode": "hd720", "eye": "left",
                           "rectified": False, "cell": os.path.relpath(args.cell, _REPO),
                           "scene_fovy_shown": scene_fovy},
                          open(os.path.join(sd, "state.json"), "w"), indent=1)
                msg = f"saved {sd}"
                print(msg, "q=", [round(x, 3) for x in d["q"]])
                n_shot += 1
            msg_t = time.time()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
