"""Gate for the render-replay contract (scripts/rerender_simdist_data.py).

Two halves, and BOTH are needed (iron law #26: a hook that exists but is never
called is the expensive failure mode):

  A. REPLAY IS FAITHFUL -- re-rendering an episode from its stored sim_state +
     model_state reproduces the collected frames. If this fails, "we can add the
     wrist camera later" is false and the collection is a one-shot commitment.
     ⚠️ The criterion is NOT byte-equality any more. Since soft shadows went in
     (2026-09-23) the renderer is not bit-deterministic: two re-renders of ONE
     episode differ by up to ~6/255 on ~0.01% of pixels. So the test is
     SELF-CALIBRATING -- re-render twice, and require source-vs-re-render to be
     no worse than re-render-vs-re-render. That still catches every real
     breakage by orders of magnitude (a wrong background photo measured 212, a
     wrong shadow set 118).
  B. THE RESTORE IS LOAD-BEARING -- re-rendering with --no_model_state (a fresh
     DR draw instead of the episode's own) produces DIFFERENT frames. Without
     this half, A could pass trivially on a dataset where DR never varied.

Run:  python scripts/test_rerender.py            # collects its own smoke set
      python scripts/test_rerender.py --dataset <existing>.hdf5
"""
import argparse
import io
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PY = "/home/Chenyang/.conda/envs/ibrl/bin/python"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "exps/mimicgen/threading_realcell_ibrl_seed1"   # --run overrides (e.g. the Square expert)
CAMS = ["agentview", "robot0_eye_in_hand"]
FAILED = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")
    if not ok:
        FAILED.append(name)


def frames(group, cam):
    """Decode one episode's camera dataset to (T, H, W, C) uint8."""
    d = group[f"obs/{cam}_image"]
    raw = np.asarray(d)
    if raw.dtype == np.uint8 and raw.ndim > 1:      # raw storage
        return raw
    return np.stack([np.asarray(Image.open(io.BytesIO(np.asarray(b).tobytes())))
                     for b in raw])


def jpeg_bytes(group, cam):
    d = np.asarray(group[f"obs/{cam}_image"])
    return [np.asarray(b).tobytes() for b in d]


def collect(out_dir, n=12):
    env = dict(os.environ, GPU=os.environ.get("GPU", "0"), GUARD="0", NUM_PROC="2",
               NUM_TRAJ=str(n), SEED="4242", RUN=RUN, OUT=out_dir)
    cmd = ["bash", "scripts/generate_simdist_data_threading.sh",
           "--rl_cameras", "+".join(CAMS), "--rl_image_size", "224",
           "--jpeg_quality", "90", "--store_sim_state", "1"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert os.path.exists(f"{out_dir}/dataset.hdf5"), r.stdout[-3000:] + r.stderr[-3000:]


def rerender(src, dst, no_model_state=0, limit=4):
    cmd = [PY, "scripts/rerender_simdist_data.py", "--dataset", src, "--out", dst,
           "--expert", f"{RUN}/model0.pt", "--cameras", "+".join(CAMS),
           "--rl_image_size", "224", "--jpeg_quality", "90",
           "--limit", str(limit), "--no_model_state", str(no_model_state)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert os.path.exists(dst), r.stdout[-3000:] + r.stderr[-3000:]


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--run", default=RUN,
                    help="IBRL run whose model0.pt (and hence task + sim2real cell) "
                         "the smoke set is collected and re-rendered with")
    args = ap.parse_args()
    RUN = args.run

    tmp = args.keep or tempfile.mkdtemp(prefix="rerender_gate_")
    src = args.dataset
    if not src:
        print("\ncollecting a 12-trajectory smoke set (2 cameras, sim_state on)")
        collect(f"{tmp}/collect")
        src = f"{tmp}/collect/dataset.hdf5"

    print("\nA. replay is faithful (against the renderer's own noise floor)")
    same, same2 = f"{tmp}/same.hdf5", f"{tmp}/same2.hdf5"
    rerender(src, same)
    rerender(src, same2)
    with h5py.File(src, "r") as a, h5py.File(same, "r") as b, h5py.File(same2, "r") as b2:
        keys = sorted(b["data"].keys(), key=lambda x: int(x.split("_")[1]))
        check("dataset carries sim_state + model_state",
              all(k in a["data"][keys[0]] for k in ("sim_state", "model_state")))
        check("model_state_layout survives to the file attrs",
              "model_state_layout" in a["data"].attrs)
        for cam in CAMS:
            shapes_ok = True
            sr_max = rr_max = 0          # source-vs-rerender, rerender-vs-rerender
            sr_mean = rr_mean = 0.0
            for k in keys:
                fa, fb, fc = (frames(a["data"][k], cam), frames(b["data"][k], cam),
                              frames(b2["data"][k], cam))
                if not (fa.shape == fb.shape == fc.shape):
                    shapes_ok = False
                    break
                dsr = np.abs(fa.astype(int) - fb.astype(int))
                drr = np.abs(fc.astype(int) - fb.astype(int))
                sr_max = max(sr_max, int(dsr.max())); rr_max = max(rr_max, int(drr.max()))
                sr_mean = max(sr_mean, float(dsr.mean())); rr_mean = max(rr_mean, float(drr.mean()))
            check(f"{cam}: frame shapes match", shapes_ok)
            # +2/255 of slack over the noise floor, and the mean must stay at
            # the floor's order of magnitude -- a real mismatch moves the MEAN,
            # a nondeterministic one only ever touches a few pixels.
            check(f"{cam}: re-render within the renderer's noise floor",
                  shapes_ok and sr_max <= rr_max + 2 and sr_mean <= max(rr_mean * 4, 1e-3),
                  f"source-vs-rerender max {sr_max} mean {sr_mean:.5f} | "
                  f"rerender-vs-rerender max {rr_max} mean {rr_mean:.5f}")
        # the labels must ride along untouched
        k = keys[0]
        check("actions/values/state copied through",
              np.array_equal(np.asarray(a["data"][k]["actions"]),
                             np.asarray(b["data"][k]["actions"]))
              and np.array_equal(np.asarray(a["data"][k]["obs/state"]),
                                 np.asarray(b["data"][k]["obs/state"])))

    print("\nB. the model_state restore is load-bearing")
    fresh = f"{tmp}/fresh.hdf5"
    rerender(src, fresh, no_model_state=1)
    with h5py.File(src, "r") as a, h5py.File(fresh, "r") as b:
        keys = sorted(b["data"].keys(), key=lambda x: int(x.split("_")[1]))
        diffs = []
        for cam in CAMS:
            for k in keys:
                fa, fb = frames(a["data"][k], cam), frames(b["data"][k], cam)
                diffs.append(np.abs(fa.astype(int) - fb.astype(int)).mean())
        check("skipping the restore CHANGES the frames (so A is not trivial)",
              max(diffs) > 1.0, f"max mean |diff| {max(diffs):.2f}")

    print("\n" + "=" * 60)
    if FAILED:
        print(f"FAILED ({len(FAILED)}):")
        for f in FAILED:
            print("   -", f)
        sys.exit(1)
    print("ALL PASS")


main()
