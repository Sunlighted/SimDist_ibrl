"""Object-layout photos with LIVE joints: scene + wrist raw left eye + the FR3 q of that instant.

Press ENTER to take a shot (move the arm between shots for extra wrist views), q + ENTER to quit.
A shot is refused if the joint state is older than --max_age s (joint_tap.py not receiving), so
every saved frame has the q it was taken at -- the 10-13 layout photo did not (q 301 s stale).

    ~/anaconda3/envs/simdist/bin/python ~/simdist/calib/snap_layout.py --out ~/simdist/calib/layout_1014
"""
import argparse
import json
import os
import time

import cv2

JOINTS = os.path.expanduser("~/simdist/gello_software/simdist_runner/joint_state_latest.json")
MODEL = {"scene": "ZED 2i", "wrist": "ZED-M"}
W, H = 1280, 720


def find_device(model):
    for i in range(16):
        p = f"/sys/class/video4linux/video{i}/name"
        if os.path.exists(p) and open(p).read().startswith(model):
            return i  # first node of the camera = the video stream
    raise SystemExit(f"{model} not found")


def read_q():
    d = json.load(open(JOINTS))
    return d["q"], d["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_age", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    caps = {}
    for name, model in MODEL.items():
        cap = cv2.VideoCapture(find_device(model), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {name}")
        caps[name] = cap
    n = len([d for d in os.listdir(args.out) if d.startswith("shot_")])
    while True:
        if input(f"[shot {n}] ENTER = take, q = quit: ").strip().lower() == "q":
            break
        for cap in caps.values():  # drop buffered frames so the image is from now
            for _ in range(5):
                cap.grab()
        frames = {}
        for name, cap in caps.items():
            ok, f = cap.read()
            if not ok or f.shape[1] != 2 * W:
                print(f"  {name}: frame read failed")
                break
            frames[name] = f[:, :W]
        t_cap = time.time()
        q, t_q = read_q()
        if len(frames) < 2:
            continue
        if t_cap - t_q > args.max_age:
            print(f"  REFUSED: joints are {t_cap - t_q:.1f} s old -- is the teleop session + joint_tap up?")
            continue
        d = os.path.join(args.out, f"shot_{n:03d}")
        os.makedirs(d)
        for name, f in frames.items():
            cv2.imwrite(os.path.join(d, f"{name}_left.png"), f)
        json.dump({"q": q, "joint_t": t_q, "capture_t": t_cap, "joint_order": "fr3_joint1..7",
                   "mode": "hd720", "eye": "left", "rectified": False},
                  open(os.path.join(d, "state.json"), "w"), indent=1)
        print(f"  saved {d}  q={[round(x, 3) for x in q]}  joint age {t_cap - t_q:.3f} s")
        n += 1


if __name__ == "__main__":
    main()
