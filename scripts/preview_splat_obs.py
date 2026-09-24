"""Preview the PHOTOREAL (3DGS) observations the peg pixel pipeline feeds the policy.

Builds PixelRobosuite exactly the way train_simdist.py does when splat_obs=1
(same SplatCompositor args, the two real cameras at 224 px), drives one
scripted-expert episode, and at a few key moments saves:
    policy obs (splat composite)  |  plain MuJoCo render of the SAME state
for both cameras. If the two columns look alike, the splat never happened.

    python scripts/preview_splat_obs.py --out viz/splat_preview
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import real2sim.splat_obs  # noqa: F401,E402  FIRST: fixes the gs_mujoco import order
from real2sim.splat_obs import DEFAULT_CALIB, DEFAULT_SEG_DIR, DEFAULT_SCENE_PLY, DEFAULT_GROUND  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402

from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.smoke_peg_insertion import scripted_expert_action  # noqa: E402

CAMS = ["zedview", "robot0_zedm_wrist"]


def to_hwc(x):
    x = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return np.ascontiguousarray(x.transpose(1, 2, 0) if x.shape[0] == 3 else x).astype(np.uint8)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="viz/splat_preview")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--size", type=int, default=224)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    env = PixelRobosuite(
        env_name="PegInsertion_D0", robots=["Panda"], episode_length=300,
        use_state=False, extra_state=True, device="cuda",
        image_size=a.size, rl_image_size=a.size, camera_names=CAMS, rl_cameras=CAMS,
        hand_init_perturb_joint_sigma=0.1, end_on_success=True,
        env_kwargs=dict(board_pos_noise=0.01, board_yaw_noise=5.0),
        splat_obs=dict(calib_path=DEFAULT_CALIB, seg_dir=DEFAULT_SEG_DIR,
                       scene_ply=DEFAULT_SCENE_PLY, ground_json=DEFAULT_GROUND))
    np.random.seed(a.seed)
    obs, _ = env.reset()
    raw = env.env
    print("obs keys:", sorted(obs.keys()), {c: tuple(obs[c].shape) for c in CAMS if c in obs})

    def grab(tag):
        row = []
        for c in CAMS:
            pol = to_hwc(obs[c])                                           # what the policy sees
            mj = raw.sim.render(camera_name=c, width=a.size, height=a.size)[::-1]  # plain MuJoCo, top-down
            diff = float(np.abs(pol.astype(float) - mj.astype(float)).mean())
            row += [label(pol, f"{tag} {c.replace('robot0_', '')} SPLAT"),
                    label(np.ascontiguousarray(mj), f"MuJoCo only  |diff|={diff:.0f}")]
        return np.concatenate(row, axis=1)

    rows, frames, ctl, t, done = [grab("t=0")], [], {}, 0, False
    marks = {"grasped": False, "above": False}
    while not done:
        obs, _, done, succ, _ = env.step(torch.from_numpy(scripted_expert_action(raw, ctl)))
        t += 1
        frames.append(np.concatenate([to_hwc(obs[c]) for c in CAMS], axis=1))
        mode = ctl.get("mode", "")
        if mode == "lift" and not marks["grasped"]:
            marks["grasped"] = True; rows.append(grab(f"t={t} grasp"))
        if mode == "insert" and not marks["above"]:
            marks["above"] = True; rows.append(grab(f"t={t} at hole"))
        if succ:
            rows.append(grab(f"t={t} inserted"))
            break
    imageio.imwrite(os.path.join(a.out, "splat_vs_mujoco.png"), np.concatenate(rows, axis=0))
    imageio.mimsave(os.path.join(a.out, "splat_episode.mp4"), frames, fps=20, macro_block_size=1)
    print(f"episode: {t} steps, success={bool(succ)}; wrote {a.out}/splat_vs_mujoco.png and splat_episode.mp4")


if __name__ == "__main__":
    main()
