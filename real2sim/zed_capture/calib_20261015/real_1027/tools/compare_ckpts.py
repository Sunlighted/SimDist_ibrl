"""Same real input, two checkpoints: does a newer stage 1 still stall where the deployed one did (10-27 takeover run)?

Drives policy_server.Server A (the DEPLOYED ckpt) through the recorded run s1_policy_1027_2057 step by step with
exactly what the policy saw on the robot: the server's own saved obs PNGs (rectified + colour-mapped 224 px,
lossless), the recorded q / dq / fingers, and the recorded t_mono as the clock of the speed caps. A's reference arm,
exec_obs, box clamp and caps therefore evolve as on the robot; A's actions are checked against steps.jsonl (the
harness is only trusted if they match). At every step the other ckpts (B...) are queried on the SAME obs: same
images, prop of the same q, exec_obs of A's reference state and A's previous action. B never acts.

  python compare_ckpts.py OUT.json CKPT_A CKPT_B [CKPT_C ...]
"""
import json, os, sys, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, REPO); os.chdir(REPO)
import cv2  # noqa: E402
import h5py  # noqa: E402
import real2sim.deploy.policy_server as ps  # noqa: E402

RUN = "realdata/to_sim_1027/runs/s1_policy_1027_2057"
DEMO = "realdata/to_sim_1027/demos/threading_robotiq_1027/episode_0003.h5"   # camera_info only
EPISODES = (4, 5, 6, 7)                       # = demos 3-6 (policy segment, then the GELLO takeover)
COLOR = "real2sim/zed_capture/calib_20261014/color_correction_threading_1014.json"

CLOCK = [0.0]
ps.time.monotonic = lambda: CLOCK[0]         # _speed_caps measures the real time since the last step


def server(ckpt):
    a = ps.build_parser().parse_args(["--ckpt", ckpt, "--z_min", "0.960", "--color_json", COLOR])
    s = ps.Server(a)
    intr = json.loads(h5py.File(DEMO).attrs["camera_intrinsics"])
    s.hello({"hz": s.arm.control_freq, "camera_info": {c: {"K": np.array(v["K"]).ravel().tolist(), "D": list(v["D"]),
             "width": int(v["width"]), "height": int(v["height"])} for c, v in intr.items()}})
    return s


def main():
    out, ckpts = sys.argv[1], sys.argv[2:]
    A = server(ckpts[0])
    Bs = [server(c) for c in ckpts[1:]]
    for B in Bs:
        for cam in A.cammap:
            assert (A.fovy[A.cammap[cam]], A.rect[cam].side) == (B.fovy[B.cammap[cam]], B.rect[cam].side), \
                f"{cam}: the ckpts crop differently -- the saved PNGs are not B's input"
    cams = list(A.cammap)                      # the PNG row order (policy_server.step)
    steps = [json.loads(l) for l in open(os.path.join(RUN, "steps.jsonl"))]
    res = {"ckpts": ckpts, "episodes": {}}
    for ep in EPISODES:
        rows = [r for r in steps if r["episode"] == ep]
        for s in [A] + Bs:
            s._reset_ref(); s.last_action = np.zeros(7); s.last_cmd_pose = None; s.last_cmd_z = None
            s.last_step_t = None
        recs = []
        for r in rows:
            png = cv2.imread(os.path.join(RUN, f"obs_ep{ep:02d}_{r['step']:05d}.png"))[..., ::-1]
            S = png.shape[0]
            ims = {c: np.ascontiguousarray(png[:, i * S:(i + 1) * S]) for i, c in enumerate(cams)}
            q, dq, fingers = np.array(r["q"]), np.array(r["dq"]), np.array(r["fingers"])
            # B: same obs as A sees in this step (A's reference + previous action), queried BEFORE A steps it
            bacts = []
            for B in Bs:
                B.arm.q_ref = None if A.arm.q_ref is None else np.array(A.arm.q_ref)
                B.arm.v_ref = np.array(A.arm.v_ref)
                B.arm.set_state(q, dq, fingers)
                prop = B.arm.prop(fingers)
                if B.exec_obs:
                    prop = np.concatenate([prop, B.arm.exec_obs(q, A.last_action)])
                obs = {B.cammap[c]: B.torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) for c, im in ims.items()}
                obs["prop"] = B.torch.from_numpy(prop.astype(np.float32))
                with B.torch.no_grad():
                    bacts.append(np.clip(B.agent.act(obs, eval_mode=True).numpy().astype(np.float64), -1, 1))
            CLOCK[0] = r["t_mono"]
            A.obs_images = lambda blobs, names, ims=ims: ims
            ra = A.step({"q": r["q"], "dq": r["dq"], "fingers": r["fingers"], "cams": cams}, [b"", b""])
            recs.append(dict(step=r["step"], eef=r["eef_full"], fingers=r["fingers"][0], a_rec=r["action"],
                             a_A=np.asarray(ra["action"]).round(4).tolist(),
                             a_B=[b.round(4).tolist() for b in bacts],
                             qref_rec=r.get("q_ref_before"),
                             qref_A=None if A.arm.q_ref is None else np.asarray(A.arm.q_ref).round(6).tolist()))
        res["episodes"][ep] = recs
        d = np.array([np.abs(np.array(x["a_A"]) - np.array(x["a_rec"])).max() for x in recs])
        print(f"[ep {ep}] {len(recs)} steps | A vs recorded action: max|d| median {np.median(d):.4f} max {d.max():.4f}",
              flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
