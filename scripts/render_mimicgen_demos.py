"""Render MimicGen-generated PegInsertion demos to mp4 + a contact-sheet PNG.

Replays the stored sim states (no re-simulation), so what you see is exactly
what was saved. Three views per frame:
  zedview  -- the real external ZED's pose (what the deployed policy sees)
  frontview -- robosuite's wide view of the whole scene
  holecam  -- close-up on the target pocket, injected into the model xml,
              to check insertions are real and not clip-through

    python scripts/render_mimicgen_demos.py \
        --dataset release/data/mimicgen/generated/peg_insertion/peg_try1/demo.hdf5 \
        --out viz/mimicgen_peg/peg_try1 --num 6
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: F401,E402
import h5py  # noqa: E402
import cv2  # noqa: E402
import imageio  # noqa: E402
import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402

import env.peg_insertion_mg  # noqa: F401,E402

CAMS = ["zedview", "frontview", "holecam"]
# holecam offset from the pocket centre (world frame; robot faces +x)
HOLECAM_OFFSET = np.array([0.22, -0.22, 0.14])


def with_holecam(xml, board_pos):
    pos = board_pos + HOLECAM_OFFSET
    cam = ('<camera name="holecam" mode="targetbody" target="board_obj_main" '
           f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" fovy="35"/>')
    return xml.replace("<worldbody>", "<worldbody>" + cam, 1)


def label(img, text, color=(255, 255, 255)):
    img = img.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def render_demo(env, g, size):
    states = g["states"][()]
    xml = g.attrs["model_file"]
    env.reset_to({"states": states[0], "model": xml})
    board = env.env.sim.data.body_xpos[env.env.obj_body_id["board"]].copy()
    env.reset_to({"states": states[0], "model": with_holecam(xml, board)})
    frames, succ = [], []
    for s in states:
        env.reset_to({"states": s})
        frames.append([env.render(mode="rgb_array", height=size, width=size, camera_name=c)
                       for c in CAMS])
        succ.append(bool(env.env._check_success()))
    return frames, succ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    f = h5py.File(args.dataset, "r")
    # robomimic refuses to build an env before the obs-modality registry exists
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["robot0_eef_pos"], "rgb": []})
    env_meta = json.loads(f["data"].attrs["env_args"])
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:args.num]

    writer = imageio.get_writer(args.out + ".mp4", fps=args.fps, macro_block_size=1)
    sheet_rows = []
    for k in keys:
        g = f[f"data/{k}"]
        frames, succ = render_demo(env, g, args.size)
        T = len(frames)
        grasp = g["datagen_info/subtask_term_signals/grasp"][()] if "datagen_info" in g else None
        t_grasp = int(np.argmax(grasp)) if grasp is not None and grasp.any() else None
        t_succ = succ.index(True) if any(succ) else None
        print(f"{k}: T={T} grasp@{t_grasp} first_success@{t_succ} final_success={succ[-1]}")
        for t, views in enumerate(frames):
            col = (80, 255, 80) if succ[t] else (255, 255, 255)
            row = [label(v, f"{k} t={t}/{T} {c}" + ("  SUCCESS" if succ[t] else ""), col)
                   for v, c in zip(views, CAMS)]
            writer.append_data(np.concatenate(row, axis=1))
        # contact sheet: start / grasp / halfway to success / final, holecam + zedview
        key_t = [0, t_grasp or T // 3, ((t_grasp or 0) + (t_succ or T - 1)) // 2, T - 1]
        tiles = []
        for t in key_t:
            tiles.append(label(frames[t][0], f"{k} t={t} zed"))
            tiles.append(label(frames[t][2], f"t={t} hole" + (" OK" if succ[t] else ""),
                               (80, 255, 80) if succ[t] else (255, 255, 255)))
        sheet_rows.append(np.concatenate(tiles, axis=1))
    writer.close()
    small = [cv2.resize(r, (r.shape[1] // 2, r.shape[0] // 2)) for r in sheet_rows]
    imageio.imwrite(args.out + "_sheet.png", np.concatenate(small, axis=0))
    print("wrote", args.out + ".mp4", args.out + "_sheet.png")


if __name__ == "__main__":
    main()
