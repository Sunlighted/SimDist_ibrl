#!/usr/bin/env python
"""Real2sim scan recorder.

Move the arm by hand, press ENTER, and this captures both ZEDs plus the live
joint state as one sample. Run on the desktop (tisl-ws-22-5) in the `robot` env;
needs joint_server.py running on the NUC.

    conda activate robot
    python record_scan.py --out ~/scan_capture/run1

Keys at the prompt:
    ENTER / r   record a sample at the current pose
    d           drop the last sample
    s           show what has been recorded so far
    q           finish and write the manifest

Each sample writes, per camera: left/right rectified PNG, depth as float32 .npy
in metres, the confidence map, and a depth preview PNG. Camera intrinsics,
stereo baseline and distortion are written once to intrinsics.json — that plus
the recorded joints is what the extrinsics solve and the render-vs-real
comparison consume.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import pyzed.sl as sl

# The ZED-M fails to open with LOW USB BANDWIDTH if it is initialised before the
# 2i, even though the two sit on separate 5 Gbps root hubs -- the SDK contends
# over the other camera's HID interface during open. 2i first is the fix.
CAMERAS = [
    ("zed2i", 37597132),   # fixed scene camera
    ("zedm", 16478414),    # wrist camera (hand_camera_id in droid parameters.py)
]


class StaleStateError(RuntimeError):
    """The joint stream is frozen -- recording against it would be worthless."""


def fetch_state(url, timeout=5.0, require_live=True):
    """Read robot state, refusing frozen data.

    franka_panda_client dies whenever FCI drops, but polymetis keeps serving its
    last-seen state, so reads succeed forever with a stale joint vector. That
    cost a full 14-pose scan on 2026-09-16. The server now reports how long the
    joints have been bit-identical; anything past its threshold is not data.
    """
    with urllib.request.urlopen(url + "/state", timeout=timeout) as r:
        state = json.loads(r.read().decode())
    if require_live and not state.get("live", False):
        raise StaleStateError(
            "joint stream FROZEN for {}s (FCI down / franka_panda_client dead). "
            "Joints reported: [{}]".format(
                state.get("stale_s", "?"),
                ", ".join("{:.4f}".format(x) for x in state.get("joints", []))))
    return state


def open_cameras(fps, depth_mode, resolution):
    cams = []
    for name, serial in CAMERAS:
        cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_serial_number(serial)
        init.camera_resolution = resolution
        init.camera_fps = fps
        init.depth_mode = depth_mode
        init.coordinate_units = sl.UNIT.METER
        # Keep factory calibration fixed across the whole scan. Self-calibration
        # would silently re-solve the stereo extrinsics between samples, which
        # makes the intrinsics we write out no longer describe every frame.
        init.camera_disable_self_calib = True
        err = cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            for c, _, _ in cams:
                c.close()
            raise RuntimeError("{} (SN {}) failed to open: {}".format(name, serial, err))
        print("  opened {:6s} SN {}".format(name, serial))
        cams.append((cam, name, serial))
    return cams


def dump_intrinsics(cams, out_dir):
    data = {}
    for cam, name, serial in cams:
        info = cam.get_camera_information()
        conf = info.camera_configuration
        calib = conf.calibration_parameters
        def cam_params(p):
            return {
                "fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy,
                "distortion": [float(x) for x in p.disto],
                "h_fov": p.h_fov, "v_fov": p.v_fov,
            }
        data[name] = {
            "serial": serial,
            "model": str(info.camera_model),
            "firmware": conf.firmware_version,
            "resolution": [conf.resolution.width, conf.resolution.height],
            "fps": conf.fps,
            # Right-to-left transform: the stereo baseline, in metres.
            "baseline": float(calib.get_camera_baseline()),
            "left": cam_params(calib.left_cam),
            "right": cam_params(calib.right_cam),
        }
    path = os.path.join(out_dir, "intrinsics.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("wrote", path)
    return data


def depth_preview(depth, lo=0.3, hi=3.0):
    d = np.nan_to_num(depth, nan=hi, posinf=hi, neginf=hi)
    d = np.clip(d, lo, hi)
    norm = ((1.0 - (d - lo) / (hi - lo)) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def capture(cams, state_url, out_dir, index, settle, burst):
    """Capture one sample: both cameras plus the joint state around them.

    Joint state is read before and after the images so the manifest records how
    much the arm drifted during the capture -- with a hand-guided arm that drift
    is the dominant error in the extrinsics solve, so it is worth measuring
    rather than assuming.
    """
    name = "pose_{:03d}".format(index)
    sample_dir = os.path.join(out_dir, name)
    os.makedirs(sample_dir, exist_ok=True)

    state_before = fetch_state(state_url)
    time.sleep(settle)

    left_img, right_img, depth_mat, conf_mat = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()
    per_camera = {}
    for cam, cam_name, serial in cams:
        grabbed = 0
        for _ in range(burst):
            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                grabbed += 1
        if grabbed == 0:
            raise RuntimeError("{}: no frames grabbed".format(cam_name))

        cam.retrieve_image(left_img, sl.VIEW.LEFT)
        cam.retrieve_image(right_img, sl.VIEW.RIGHT)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        cam.retrieve_measure(conf_mat, sl.MEASURE.CONFIDENCE)

        left = left_img.get_data()[:, :, :3].copy()
        right = right_img.get_data()[:, :, :3].copy()
        depth = depth_mat.get_data().astype(np.float32)
        conf = conf_mat.get_data().astype(np.float32)

        cv2.imwrite(os.path.join(sample_dir, cam_name + "_left.png"), left)
        cv2.imwrite(os.path.join(sample_dir, cam_name + "_right.png"), right)
        np.save(os.path.join(sample_dir, cam_name + "_depth.npy"), depth)
        np.save(os.path.join(sample_dir, cam_name + "_confidence.npy"), conf)
        cv2.imwrite(os.path.join(sample_dir, cam_name + "_depth_preview.png"),
                    depth_preview(depth))

        valid = np.isfinite(depth)
        per_camera[cam_name] = {
            "serial": serial,
            "grabs": grabbed,
            "zed_timestamp_ns": cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds(),
            "depth_valid_fraction": float(valid.mean()),
            "depth_min_m": float(np.nanmin(depth[valid])) if valid.any() else None,
            "depth_max_m": float(np.nanmax(depth[valid])) if valid.any() else None,
            "mean_intensity": float(left.mean()),
        }

    state_after = fetch_state(state_url)
    drift = [abs(a - b) for a, b in zip(state_after["joints"], state_before["joints"])]

    record = {
        "name": name,
        "index": index,
        "wall_time": time.time(),
        "joints": state_after["joints"],
        "joints_before": state_before["joints"],
        "joint_drift_max_rad": max(drift),
        "ee_pos": state_after.get("ee_pos"),
        "ee_quat": state_after.get("ee_quat"),
        "cameras": per_camera,
    }
    with open(os.path.join(sample_dir, "state.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def fmt_joints(q):
    return "[" + ", ".join("{:7.4f}".format(x) for x in q) + "]"


def write_manifest(out_dir, samples, intrinsics):
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump({
            "created": time.time(),
            "n_samples": len(samples),
            "cameras": {n: {"serial": s} for n, s in CAMERAS},
            "intrinsics": intrinsics,
            "samples": samples,
        }, f, indent=2)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/scan_capture/run1"))
    ap.add_argument("--state-url", default="http://172.16.0.42:4243")
    ap.add_argument("--fps", type=int, default=15,
                    help="15 is what both cameras sustain together; 30 is flaky")
    ap.add_argument("--resolution", choices=["hd2k", "hd1080", "hd720", "vga"],
                    default="hd2k",
                    help="hd2k (2208x1242) is the max both cameras support; "
                         "verified to run on both simultaneously at 15 fps")
    ap.add_argument("--burst", type=int, default=10,
                    help="frames to grab per camera before keeping the last one")
    ap.add_argument("--settle", type=float, default=0.4,
                    help="seconds to wait after ENTER before grabbing")
    ap.add_argument("--depth", choices=["performance", "quality", "ultra", "neural"],
                    default="ultra",
                    help="neural is sharpest but builds a TensorRT engine on first use")
    args = ap.parse_args()

    depth_mode = {
        "performance": sl.DEPTH_MODE.PERFORMANCE,
        "quality": sl.DEPTH_MODE.QUALITY,
        "ultra": sl.DEPTH_MODE.ULTRA,
        "neural": sl.DEPTH_MODE.NEURAL,
    }[args.depth]
    resolution = {
        "hd2k": sl.RESOLUTION.HD2K,
        "hd1080": sl.RESOLUTION.HD1080,
        "hd720": sl.RESOLUTION.HD720,
        "vga": sl.RESOLUTION.VGA,
    }[args.resolution]

    os.makedirs(args.out, exist_ok=True)

    print("checking joint server at", args.state_url)
    try:
        state = fetch_state(args.state_url)
    except StaleStateError as e:
        print("ERROR: {}".format(e))
        print("Re-activate FCI in Desk, then restart launch_arm_ro.sh and")
        print("run_joint_server.sh on the NUC. Do not scan until this is live.")
        return 1
    except Exception as e:
        print("ERROR: cannot reach joint server: {}: {}".format(type(e).__name__, e))
        print("Start it on the NUC first (see run_joint_server.sh).")
        return 1
    print("  joints:", fmt_joints(state["joints"]))
    print("  stream live (last changed {}s ago)".format(state["stale_s"]))

    print("opening cameras ({}, {} depth)".format(args.resolution, args.depth))
    cams = open_cameras(args.fps, depth_mode, resolution)
    intrinsics = dump_intrinsics(cams, args.out)

    samples = []
    index = 0
    print()
    print("Ready. Move the arm by hand, then:")
    print("  ENTER/r = record    d = drop last    s = show    q = finish")
    print()
    try:
        while True:
            try:
                state = fetch_state(args.state_url)
                prompt = "[{:2d} saved] {} > ".format(len(samples), fmt_joints(state["joints"]))
            except StaleStateError:
                prompt = "[{:2d} saved] *** STREAM FROZEN - do not record *** > ".format(
                    len(samples))
            except Exception as e:
                prompt = "[{:2d} saved] <state unavailable: {}> > ".format(len(samples), e)
            try:
                key = input(prompt).strip().lower()
            except EOFError:
                break

            if key in ("", "r"):
                try:
                    rec = capture(cams, args.state_url, args.out, index, args.settle, args.burst)
                except Exception as e:
                    print("  capture failed: {}: {}".format(type(e).__name__, e))
                    continue
                samples.append(rec)
                index += 1
                print("  saved {}  joints {}".format(rec["name"], fmt_joints(rec["joints"])))
                print("       drift during capture: {:.5f} rad  |  depth valid {}".format(
                    rec["joint_drift_max_rad"],
                    "  ".join("{}={:.0%}".format(k, v["depth_valid_fraction"])
                              for k, v in rec["cameras"].items())))
                if rec["joint_drift_max_rad"] > 0.002:
                    print("       WARNING: arm moved during capture; consider re-recording")
            elif key == "d":
                if samples:
                    dropped = samples.pop()
                    print("  dropped {} (files left on disk)".format(dropped["name"]))
                else:
                    print("  nothing to drop")
            elif key == "s":
                for rec in samples:
                    print("  {}  {}".format(rec["name"], fmt_joints(rec["joints"])))
            elif key == "q":
                break
            else:
                print("  keys: ENTER/r record, d drop, s show, q finish")
    finally:
        for cam, _, _ in cams:
            cam.close()

    if samples:
        write_manifest(args.out, samples, intrinsics)
        print("\n{} samples in {}".format(len(samples), args.out))
    else:
        print("\nno samples recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
