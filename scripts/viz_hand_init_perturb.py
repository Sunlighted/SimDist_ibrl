"""Visualize the EE-space init-noise distribution under different
`hand_init_perturb` settings.

Renders a 3×N grid of agentview frames at t=0 for Threading_D0:
  - row 1: hand_init_perturb=None     (robosuite default, σ=0.02 joint qpos,
           ~22 mm EE std — matches MimicGen demo collection)
  - row 2: hand_init_perturb=(0.03, 0.03, 0.02)  (current source-handperturb YAML)
  - row 3: hand_init_perturb=(0.05, 0.05, 0.03)  (~1.5× demo σ)

Also prints per-row EE position stats (per-axis std, range) so you can
correlate the visual spread with concrete mm numbers.

Run:
    source set_env.sh
    python scripts/viz_hand_init_perturb.py --out /tmp/perturb_grid.png --n 8
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def render_row(perturb, label, n_samples, image_size, base_seed,
               joint_sigma=None):
    """Reset env n_samples times under `perturb` (or `joint_sigma`), collect
    agentview frames + EE pos. The two knobs are mutually exclusive — pass
    exactly one (or neither for robosuite default)."""
    from env.robosuite_wrapper import PixelRobosuite

    env = PixelRobosuite(
        env_name="Threading_D0",
        robots=["Panda"],
        episode_length=400,
        image_size=image_size,
        rl_image_size=image_size,
        camera_names=["agentview"],
        rl_cameras=["agentview"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        hand_init_perturb=perturb,
        hand_init_perturb_joint_sigma=joint_sigma,
    )

    frames = []
    ee_positions = []
    ee_quats = []
    for i in range(n_samples):
        np.random.seed(base_seed + i)
        obs, hires = env.reset()
        img = hires["agentview"].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        frames.append(img)
        s = obs["state"]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        ee_positions.append(s[:3])
        ee_quats.append(s[3:7])  # eef_quat (xyzw or wxyz — robosuite uses xyzw)

    del env  # free EGL ctx promptly to reduce __del__ noise
    ee = np.stack(ee_positions)
    quats = np.stack(ee_quats)
    # angle between each quat and the mean quat, in degrees
    mean_q = quats.mean(0); mean_q /= np.linalg.norm(mean_q)
    dots = np.clip(np.abs(quats @ mean_q), 0.0, 1.0)
    angles_deg = 2 * np.arccos(dots) * 180.0 / np.pi
    stats = {
        "label": label,
        "std_mm": (ee.std(0) * 1000).round(1).tolist(),
        "range_mm": ((ee.max(0) - ee.min(0)) * 1000).round(1).tolist(),
        "ori_std_deg": float(angles_deg.std().round(2)),
        "ori_max_deg": float(angles_deg.max().round(2)),
    }
    print(
        f"[{label:32s}] pos std (mm) = {stats['std_mm']}, "
        f"ori std (deg) = {stats['ori_std_deg']}, ori max (deg) = {stats['ori_max_deg']}"
    )
    return frames, stats


def make_grid(rows, labels, out_path, gap=4, label_h=22):
    """Stack frame-rows into one PNG with row labels on the left."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    h, w = rows[0][0].shape[:2]

    label_w = 220  # left strip for row labels
    grid_w = label_w + n_cols * w + (n_cols - 1) * gap
    grid_h = n_rows * h + (n_rows - 1) * gap + label_h
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for r, (row, lbl) in enumerate(zip(rows, labels)):
        y0 = r * (h + gap) + label_h
        for c, img in enumerate(row):
            x0 = label_w + c * (w + gap)
            canvas[y0:y0 + h, x0:x0 + w] = img

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    # top: column index
    for c in range(n_cols):
        x0 = label_w + c * (w + gap) + w // 2 - 8
        draw.text((x0, 4), f"#{c}", fill=(0, 0, 0), font=font)

    # left: row label
    for r, lbl in enumerate(labels):
        y0 = r * (h + gap) + label_h + h // 2 - 8
        draw.text((6, y0), lbl, fill=(0, 0, 0), font=font)

    img.save(out_path)
    print(f"saved grid → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/tmp/perturb_grid.png")
    ap.add_argument("--n", type=int, default=8, help="frames per row")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--perturb", type=str, action="append", default=None,
        help='Add a row for the given perturb. Format "rx,ry,rz" (meters) or '
             '"none"/"null" for robosuite default. Repeat the flag for multiple '
             'rows. If omitted, uses the default demo/0.03/0.05 triple.',
    )
    args = ap.parse_args()

    # Match the launch convention from scripts/. EGL is faster; fall back to
    # osmesa via env var if EGL fails on your machine.
    os.environ.setdefault("MUJOCO_GL", "egl")

    def parse_perturb(s):
        # Returns (perturb_tuple_or_None, joint_sigma_or_None, label).
        if s.lower() in ("none", "null"):
            return None, None, "demo (robosuite σ=0.02)"
        if s.lower().startswith("joint:"):
            sigma = float(s.split(":", 1)[1])
            return None, sigma, f"joint σ={sigma:g}"
        vals = tuple(float(x) for x in s.split(","))
        assert len(vals) == 3, f'--perturb expects rx,ry,rz, "none", or "joint:σ"; got {s!r}'
        return vals, None, f"IK ({vals[0]:g}, {vals[1]:g}, {vals[2]:g})"

    if args.perturb:
        settings = [parse_perturb(s) for s in args.perturb]
    else:
        settings = [
            (None,                None, "demo (robosuite σ=0.02, ~22mm EE std)"),
            ((0.03, 0.03, 0.02),  None, "IK (0.03, 0.03, 0.02)"),
            ((0.05, 0.05, 0.03),  None, "IK (0.05, 0.05, 0.03)"),
        ]

    rows, labels = [], []
    for perturb, joint_sigma, label in settings:
        frames, _ = render_row(
            perturb, label, args.n, args.image_size, args.seed,
            joint_sigma=joint_sigma,
        )
        rows.append(frames)
        labels.append(label)

    make_grid(rows, labels, args.out)


if __name__ == "__main__":
    main()
