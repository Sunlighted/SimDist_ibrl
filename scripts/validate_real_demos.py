"""Validate a REAL-robot stage-2 demo hdf5 against the data contract.

Contract: docs/real-teleop-stage2.md. Every check here guards a mistake that the
stage-2 loader would NOT catch -- it would happily train on it:

  * prop in the robot BASE frame instead of sim WORLD  (z off by 0.912 m)
  * eef_quat stored wxyz instead of xyzw, or the TCP's orientation instead of
    the hand body's
  * gripper_qpos as a width instead of [w/2, -w/2]
  * no reward==1 step (the loader silently DROPS the whole demo)
  * JPEG frames without the codec attrs (the loader reads garbage)
  * control loop not at 20 Hz

Exit code 0 = all checks pass; 1 = at least one FAIL. WARNs do not fail.

  python scripts/validate_real_demos.py --dataset <dir>/dataset.hdf5 [--sheet out.png]
"""
import argparse
import io
import sys

import h5py
import numpy as np

# ---- reference values measured in sim (nominal real cell, Franka ready pose) ----
READY_EEF_POS = np.array([-0.1821, -0.0340, 1.4058])        # world frame, TCP
READY_EEF_QUAT = np.array([1.0, 0.0, 0.0, 0.0])              # xyzw, hand body
BASE_IN_WORLD = np.array([-0.489, -0.034, 0.912])
# 1%-99% of the stage-1 pixel data, widened by a margin (metres)
POS_LO = np.array([-0.373, -0.543, 0.925]) - 0.10
POS_HI = np.array([0.350, 0.790, 1.473]) + 0.10
PROP_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
CAMS = ("agentview", "robot0_eye_in_hand")
FRAME_SHAPE = (3, 224, 224)
CONTROL_DT = 0.05

RESULTS = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(level, msg):
    RESULTS[level] += 1
    print(f"  [{level}] {msg}")


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def decode(ds, i):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(np.asarray(ds[i], dtype=np.uint8).tobytes())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sheet", default="contact_sheet.png")
    ap.add_argument("--sheet_demos", type=int, default=6)
    args = ap.parse_args()

    f = h5py.File(args.dataset, "r")
    assert "data" in f, "no top-level 'data' group"
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    print(f"{args.dataset}: {len(keys)} demos")

    pos_all, first_quats, dts, lens, first_succ = [], [], [], [], []
    n_nosucc = 0
    sheet_rows = []
    lags, reuse = {}, {}

    for k in keys:
        g = f["data"][k]
        tag = f"{k}:"
        # --- structure -------------------------------------------------------
        missing = [x for x in ("actions", "rewards") if x not in g]
        missing += [f"obs/{x}" for x in PROP_KEYS + ("object",) if x not in g.get("obs", {})]
        missing += [f"obs/{c}_image" for c in CAMS if f"{c}_image" not in g.get("obs", {})]
        if missing:
            report("FAIL", f"{tag} missing {missing}")
            continue
        T = g["actions"].shape[0]
        lens.append(T)
        shapes = {"actions": (T, 7), "rewards": (T,), "obs/robot0_eef_pos": (T, 3),
                  "obs/robot0_eef_quat": (T, 4), "obs/robot0_gripper_qpos": (T, 2),
                  "obs/object": (T, 28)}
        bad = {p: g[p].shape for p, s in shapes.items() if g[p].shape != s}
        for c in CAMS:
            if g[f"obs/{c}_image"].shape[0] != T:
                bad[f"obs/{c}_image"] = g[f"obs/{c}_image"].shape
        if bad:
            report("FAIL", f"{tag} shape/length mismatch (T={T}): {bad}")
            continue

        a = np.asarray(g["actions"], dtype=np.float64)
        r = np.asarray(g["rewards"], dtype=np.float64)
        pos = np.asarray(g["obs/robot0_eef_pos"], dtype=np.float64)
        quat = np.asarray(g["obs/robot0_eef_quat"], dtype=np.float64)
        grip = np.asarray(g["obs/robot0_gripper_qpos"], dtype=np.float64)
        obj = np.asarray(g["obs/object"], dtype=np.float64)
        pos_all.append(pos)
        first_quats.append(quat[0])

        # --- actions ---------------------------------------------------------
        if np.abs(a).max() > 1.0 + 1e-6:
            report("WARN", f"{tag} actions exceed [-1,1] (max |a|={np.abs(a).max():.3f}); "
                           "the loader clips them -- record the CLIPPED command")
        if T > 400:
            report("WARN", f"{tag} T={T} > 400 (episode_length); stage-1 never saw longer")

        # --- reward / success -----------------------------------------------
        if not (r == 1).any():
            n_nosucc += 1
        else:
            first_succ.append(int(np.argmax(r == 1)))
        if bool(g.attrs.get("success", (r == 1).any())) != bool((r == 1).any()):
            report("FAIL", f"{tag} attrs['success'] disagrees with rewards")

        # --- quaternion -----------------------------------------------------
        nq = np.linalg.norm(quat, axis=1)
        if np.abs(nq - 1).max() > 1e-3:
            report("FAIL", f"{tag} eef_quat not unit (norm in [{nq.min():.4f},{nq.max():.4f}])")

        # --- gripper ---------------------------------------------------------
        if grip[:, 0].min() < -0.002 or grip[:, 0].max() > 0.045 \
                or grip[:, 1].max() > 0.002 or grip[:, 1].min() < -0.045:
            report("FAIL", f"{tag} gripper_qpos out of range: left [{grip[:,0].min():.4f},"
                           f"{grip[:,0].max():.4f}] right [{grip[:,1].min():.4f},{grip[:,1].max():.4f}]"
                           " -- expected [w/2, -w/2] with w in [0, 0.08] m")
        # NO symmetry check: in sim the two fingers are independent joints and
        # contact pushes them apart (|l+r| up to ~0.03 m), so asymmetry is normal.
        # A real Panda hand reports ONE width => real data is always symmetric --
        # a small sim/real distribution gap, documented, not a data error.

        if np.abs(obj).max() > 0:
            report("WARN", f"{tag} obs/object is not all zeros -- fine only if it is a real "
                           "object pose in the SAME 28-dim layout as sim")

        # --- images ----------------------------------------------------------
        for c in CAMS:
            ds = g[f"obs/{c}_image"]
            if ds.attrs.get("codec", "raw") != "jpeg":
                if ds.dtype != np.uint8 or tuple(ds.shape[1:]) != FRAME_SHAPE:
                    report("FAIL", f"{tag} {c}: neither jpeg (codec attr) nor raw uint8 {FRAME_SHAPE}")
                continue
            if tuple(ds.attrs.get("frame_shape", ())) != FRAME_SHAPE:
                report("FAIL", f"{tag} {c}: frame_shape attr {tuple(ds.attrs.get('frame_shape', ()))} != {FRAME_SHAPE}")
                continue
            try:
                im = decode(ds, 0)
                if im.shape != (224, 224, 3):
                    report("FAIL", f"{tag} {c}: decoded frame is {im.shape}, want (224,224,3)")
            except Exception as e:  # noqa: BLE001
                report("FAIL", f"{tag} {c}: frame 0 not decodable ({type(e).__name__})")

        # --- timing ----------------------------------------------------------
        if "obs/timestamp" in g:
            ts = np.asarray(g["obs/timestamp"], dtype=np.float64)
            if len(ts) > 1:
                dts.append(np.median(np.diff(ts)))
            for c in CAMS:
                if f"obs/{c}_capture_ts" in g:
                    cts = np.asarray(g[f"obs/{c}_capture_ts"], dtype=np.float64)
                    lags.setdefault(c, []).append(np.median(ts - cts))
                    # a repeated capture stamp = the same frame reused on two ticks
                    reuse.setdefault(c, []).append(float(np.mean(np.diff(cts) == 0)))

        if len(sheet_rows) < args.sheet_demos:
            idx = [0, T // 2, T - 1]
            try:
                sheet_rows.append([decode(g[f"obs/{c}_image"], i) for c in CAMS for i in idx])
            except Exception:  # noqa: BLE001
                pass

    # ---- dataset-level checks ---------------------------------------------------
    print("\ndataset-level:")
    if pos_all:
        P = np.concatenate(pos_all)
        med = np.median(P, axis=0)
        if np.median(P[:, 2]) < 0.6:
            report("FAIL", f"eef_pos median z = {np.median(P[:,2]):.3f} m -- this is the robot "
                           f"BASE frame. Sim prop is WORLD frame: add {BASE_IN_WORLD.tolist()}")
        else:
            out = ((P < POS_LO) | (P > POS_HI)).any(axis=1).mean()
            (report("FAIL", f"{100*out:.1f}% of eef_pos outside the stage-1 range "
                            f"(median {np.round(med,3).tolist()})") if out > 0.05 else
             report("PASS", f"eef_pos in the stage-1 world-frame range "
                            f"({100*out:.1f}% outside; median {np.round(med,3).tolist()})"))
        Q = np.array(first_quats)
        ang = np.array([quat_angle_deg(q, READY_EEF_QUAT) for q in Q])
        ang_wxyz = np.array([quat_angle_deg(np.roll(q, -1), READY_EEF_QUAT) for q in Q])
        if np.median(ang) < 30:
            report("PASS", f"first-frame eef_quat within {np.median(ang):.1f} deg (median) of the "
                           "ready pose (xyzw, hand body)")
        elif np.median(ang_wxyz) < 30:
            report("FAIL", f"eef_quat looks like WXYZ (as stored: {np.median(ang):.0f} deg from ready; "
                           f"re-ordered as xyzw: {np.median(ang_wxyz):.0f} deg). Store XYZW.")
        else:
            report("FAIL", f"first-frame eef_quat is {np.median(ang):.0f} deg from the ready pose -- "
                           "wrong frame? It must be the HAND body (TCP is 90 deg off) incl. the 45 deg "
                           "flange->hand mapping (docs/real-teleop-stage2.md §2)")
    if n_nosucc:
        report("WARN", f"{n_nosucc}/{len(keys)} demos have NO reward==1 step -- the stage-2 loader "
                       "DROPS them entirely")
    if first_succ:
        print(f"  success step (first reward==1): median {int(np.median(first_succ))}, "
              f"range [{min(first_succ)}, {max(first_succ)}] (the loader truncates here)")
    if lens:
        print(f"  lengths: median {int(np.median(lens))}, range [{min(lens)}, {max(lens)}]")
    if dts:
        md = float(np.median(dts))
        (report("PASS", f"control period median {1000*md:.1f} ms (~{1/md:.1f} Hz)")
         if abs(md - CONTROL_DT) <= 0.15 * CONTROL_DT else
         report("FAIL", f"control period median {1000*md:.1f} ms (~{1/md:.1f} Hz); stage-1 is 20 Hz"))
    else:
        report("WARN", "no obs/timestamp -- cannot verify the 20 Hz control rate")
    for c in CAMS:
        if c not in lags:
            report("WARN", f"no obs/{c}_capture_ts -- cannot measure how stale the {c} frames are "
                           "(sim frames are 0 ms old)")
            continue
        lag = 1000 * float(np.median(lags[c])); rr = 100 * float(np.mean(reuse[c]))
        msg = f"{c}: frame is {lag:.0f} ms older than prop (median); {rr:.0f}% of ticks reuse the previous frame"
        if lag < 0:
            report("FAIL", msg + " -- NEGATIVE lag: capture_ts and timestamp are on different clocks")
        elif lag >= 50 or rr >= 10:
            report("WARN", msg + " -- >= one control step / frequent reuse: camera fps below 20 Hz "
                                 "(HD2K is 15 fps) or a slow pipeline; see docs §3b")
        else:
            report("PASS", msg)

    if sheet_rows:
        from PIL import Image
        H = 224
        sheet = Image.new("RGB", (len(sheet_rows[0]) * (H + 4), len(sheet_rows) * (H + 4)), (20, 20, 20))
        for i, row in enumerate(sheet_rows):
            for j, im in enumerate(row):
                sheet.paste(Image.fromarray(im), (j * (H + 4), i * (H + 4)))
        sheet.save(args.sheet)
        print(f"\ncontact sheet -> {args.sheet}  (per row: agentview first/mid/last | wrist "
              "first/mid/last). Eyeball: upright? RGB not BGR? crop centred?")

    print(f"\nPASS {RESULTS['PASS']}  WARN {RESULTS['WARN']}  FAIL {RESULTS['FAIL']}")
    sys.exit(1 if RESULTS["FAIL"] else 0)


if __name__ == "__main__":
    main()
