"""Offline end-to-end test of the policy runner on recorded GELLO episodes.

Starts policy_server's Server in-process on a free port, then plays the recorded
episodes through the REAL wire protocol from a client thread: raw JPEGs, joint
positions/velocities and finger positions exactly as the ROS topics carry them.
Everything the live runner does is exercised except moving the arm.

Reports per episode: latency, the policy's actions, how many steps the server's
safety checks would refuse, and -- as a sanity check of frames/prop/action
conventions -- the cosine between the policy's commanded translation and the
operator's actual EE motion during the APPROACH (before the first grasp), when
both should head for the peg.
"""
import argparse
import glob
import os
import socket
import sys
import threading
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
import h5py  # noqa: E402
import json  # noqa: E402
from real2sim.deploy import policy_server as ps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "exps/mimicgen/peg_pixel_stage1_splat_seed1/model0.pt"))
    ap.add_argument("--real_dir", default="/scratch/yifanr/simdist/gello_software/demos/peg")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no_color", action="store_true")
    ap.add_argument("--save_dir", default=os.path.join(_REPO, "viz/real_vs_sim/replay_test"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jpeg_reduce", type=int, default=2)
    a = ap.parse_args()

    sa = argparse.Namespace(ckpt=a.ckpt, host="127.0.0.1", port=0, device=a.device,
                            color_json=ps.DEFAULT_COLOR_JSON, no_color=a.no_color, z_min=0.995,
                            x_box=(-0.35, 0.35), y_box=(-0.35, 0.35), joint_margin=0.05,
                            save_dir=a.save_dir, save_every=25, jpeg_reduce=a.jpeg_reduce)
    srv = ps.Server(sa)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); srv.a.port = s.getsockname()[1]; s.close()
    threading.Thread(target=srv.serve, daemon=True).start()
    time.sleep(1.0)

    sock = socket.create_connection(("127.0.0.1", srv.a.port))
    open(os.path.join(a.save_dir, "steps.jsonl"), "w").close() if a.save_dir else None
    for path in sorted(glob.glob(f"{a.real_dir}/*.h5")):
        f = h5py.File(path, "r")
        intr = json.loads(f.attrs["camera_intrinsics"])
        info = {c: {"K": intr[c]["K"], "D": intr[c]["D"], "width": intr[c]["width"],
                    "height": intr[c]["height"]} for c in ("scene", "wrist")}
        ps.send_msg(sock, {"type": "hello", "camera_info": info}); h, _ = ps.recv_msg(sock)
        assert "error" not in h, h
        q = f["observation/joint_positions"][()]; dq = f["observation/joint_velocities"][()]
        fing = f["observation/gripper_finger_positions"][()]; grip = f["action/gripper_width_percent"][()]
        T = len(q); t_grasp = int(np.argmax(grip < 0.5)) if (grip < 0.5).any() else T
        acts, eefs, lat, refused, clipped = [], [], [], 0, 0
        for t in range(0, T, a.stride):
            t0 = time.perf_counter()
            ps.send_msg(sock, {"type": "step", "q": q[t].tolist(), "dq": dq[t].tolist(),
                               "fingers": fing[t].tolist(), "cams": ["scene", "wrist"]},
                        [bytes(f["observation/images/scene"][t]), bytes(f["observation/images/wrist"][t])])
            r, _ = ps.recv_msg(sock)
            assert "error" not in r, r
            lat.append((time.perf_counter() - t0) * 1e3)
            acts.append(r["action"]); eefs.append(r["eef"])
            refused += any("refused" in w for w in r["warn"])
            clipped += any("clipped" in w for w in r["warn"])
        acts, eefs = np.array(acts), np.array(eefs)
        # operator's EE motion over the next 0.5 s vs the policy's commanded translation
        k = max(1, 10 // a.stride)
        n_app = max(2, min(len(eefs) - k, t_grasp // a.stride))
        vel = eefs[k:k + n_app] - eefs[:n_app]
        cmd = acts[:n_app, :3]
        m = (np.linalg.norm(vel, axis=1) > 0.01) & (np.linalg.norm(cmd, axis=1) > 0.1)
        cos = (np.sum(vel[m] * cmd[m], 1) / (np.linalg.norm(vel[m], axis=1) * np.linalg.norm(cmd[m], axis=1)))
        name = os.path.basename(path)
        ms = [json.loads(l)["ms"] for l in open(os.path.join(a.save_dir, "steps.jsonl"))][-len(acts):]
        print(f"   server ms p50 pre {np.median([m['pre'] for m in ms]):.1f} policy "
              f"{np.median([m['policy'] for m in ms]):.1f} osc {np.median([m['osc'] for m in ms]):.1f}", flush=True)
        print(f"== {name}: {len(acts)} steps | latency ms p50 {np.median(lat):.1f} p95 {np.percentile(lat, 95):.1f} "
              f"| server refused {refused}, joint-clipped {clipped}", flush=True)
        print(f"   action mean {acts.mean(0).round(3).tolist()}  |a_xyz| p50 {np.median(np.linalg.norm(acts[:, :3], axis=1)):.2f} "
              f"| gripper close frac {(acts[:, 6] > 0).mean():.2f} (operator grasps at t={t_grasp})", flush=True)
        print(f"   approach (t<{t_grasp}): cos(policy dir, operator motion) mean {cos.mean() if len(cos) else float('nan'):.2f} "
              f"over {m.sum()} moving steps, frac>0 {np.mean(cos > 0) if len(cos) else float('nan'):.2f}", flush=True)
    sock.close()
    print(f"[write] policy-input images + steps.jsonl in {a.save_dir}")


if __name__ == "__main__":
    main()
