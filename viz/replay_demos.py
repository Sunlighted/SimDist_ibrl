"""Replay TARGET demo datasets to per-episode mp4 videos.

Both `threading_d0_target_dynamics` (sigma=0.02, 20 demos) and
`threading_d0_target_expert_n500` (sigma=0.1) store `obs/sim_state` — the
flattened MuJoCo state at every control step — so an episode can be replayed
exactly by `sim.set_state_from_flattened(...) + sim.forward()` and rendered
off-screen at arbitrary resolution. No policy, no physics stepping, so the
replay is bit-exact to what was collected.

The two datasets share the Threading_D0 model, hence the same sim_state layout,
so one env replays both. The domain shift is irrelevant for pure replay (it only
changes mass/friction/solref and the *collision* ring geoms; the rendered `_vis`
twins are untouched — see CLAUDE.md §3), but it is applied anyway so the scene
matches the collection env exactly.

Filenames carry the per-episode quantities from the data-quality audit so the
videos are sortable/greppable by what distinguishes the two sets:

    demo_007_len084_contact014_jerk0.089_eefinit078mm.mp4
                   |         |            |         `-- ||eef(t=0) - mean eef(t=0)||
                   |         |            `------------ mean |a_t - a_{t-1}| (smoothness)
                   |         `------------------------- steps within +25mm of the
                   |                                    episode's terminal needle->tripod
                   |                                    distance (contact phase length)
                   `----------------------------------- episode length

Usage:
    python viz/replay_demos.py                      # both sets (20 + 50), agentview
    python viz/replay_demos.py --n_sigma01 20       # only the first 20 of the sigma=0.1 set
    python viz/replay_demos.py --cameras agentview,sideview --res 512
    python viz/replay_demos.py --only sigma0.1      # one dataset
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

import common_utils  # noqa: F401  (side-effect imports)
from env.domain_shift import DomainShiftConfig
from env.robosuite_wrapper import PixelRobosuite

# ---------------------------------------------------------------- datasets
DATASETS = {
    "sigma0.02": dict(
        path="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5",
        n=20,
        note="hand_init sigma=0.02, collect_target_demos.py, 20 demos",
    ),
    "sigma0.1": dict(
        path="release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5",
        n=50,
        note="hand_init sigma=0.1, collect_target_expert.py, first 50 of 500",
    ),
}

# the active TARGET recipe (CLAUDE.md §12, dynamics-only) — both sets were
# collected under it
TARGET_SHIFT = DomainShiftConfig(
    mode="fixed",
    friction_scale=0.5,
    mass_scale=3.0,
    solref_scale=1.0,
    solimp_scale=1.0,
    gripper_kp_scale=0.12,
    hole_offset=(0.0, 0.0, 0.0),
    mask_hole_offset_in_obs=False,
    ring_size_scale=0.85,
    ring_friction_scale=2.0,
)

# state layout (dim 37): [0:3] eef_pos  [9:12] needle_pos  [23:26] tripod_pos
NEEDLE = slice(9, 12)
TRIPOD = slice(23, 26)
EEF = slice(0, 3)


def render(env, cam, res):
    """Off-screen render; MuJoCo's offscreen buffer is bottom-left origin."""
    img = env.env.sim.render(camera_name=cam, width=res, height=res)
    return np.flipud(img).copy()


def episode_stats(state, action, eef0_mean):
    """Per-episode quantities used in the filename (mirrors the audit script)."""
    T = len(action)
    d = np.linalg.norm(state[:T, NEEDLE] - state[:T, TRIPOD], axis=1)
    contact = int((d < d[-1] + 0.025).sum())
    jerk = float(np.abs(np.diff(action, axis=0)).mean()) if T > 1 else 0.0
    eefinit = float(np.linalg.norm(state[0, EEF] - eef0_mean))
    return T, contact, jerk, eefinit


def replay_dataset(env, tag, spec, out_root, cameras, res, fps, limit):
    path = os.path.join(_REPO, spec["path"])
    with h5py.File(path, "r") as f:
        g = f["data"]
        keys = sorted(g.keys(), key=lambda k: int(k.split("_")[1]))[: spec["n"]]
        if limit:
            keys = keys[:limit]
        eps = [
            dict(
                key=k,
                sim=np.asarray(g[k]["obs"]["sim_state"]),
                state=np.asarray(g[k]["obs"]["state"]),
                action=np.asarray(g[k]["actions"]),
            )
            for k in keys
        ]

    eef0_mean = np.stack([e["state"][0, EEF] for e in eps]).mean(0)
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {tag}: {len(eps)} demos -> {out_dir}")
    print(f"    {spec['note']}")
    manifest = []
    for i, ep in enumerate(tqdm(eps, desc=tag, dynamic_ncols=True)):
        T, contact, jerk, eefinit = episode_stats(ep["state"], ep["action"], eef0_mean)
        name = (f"demo_{i:03d}_len{T:03d}_contact{contact:03d}"
                f"_jerk{jerk:.3f}_eefinit{eefinit*1000:03.0f}mm")
        frames = []
        # sim_state has T+1 rows when the terminal obs is stored; render what exists
        for t in range(min(len(ep["sim"]), T + 1)):
            env.env.sim.set_state_from_flattened(np.asarray(ep["sim"][t], dtype=np.float64))
            env.env.sim.forward()
            frames.append(np.concatenate([render(env, c, res) for c in cameras], axis=1))
        imageio.mimwrite(os.path.join(out_dir, name + ".mp4"), frames, fps=fps,
                         macro_block_size=1)
        manifest.append((name, T, contact, jerk, eefinit))

    with open(os.path.join(out_dir, "index.tsv"), "w") as fh:
        fh.write("file\tlen\tcontact_steps\tcontact_frac\tjerk\teef_init_mm\n")
        for name, T, c, j, e in manifest:
            fh.write(f"{name}.mp4\t{T}\t{c}\t{c/T:.3f}\t{j:.4f}\t{e*1000:.1f}\n")
    arr = np.array([[T, c, c / T, j, e * 1000] for _, T, c, j, e in manifest])
    print(f"    mean: len {arr[:,0].mean():.1f}  contact {arr[:,1].mean():.1f} steps "
          f"({arr[:,2].mean()*100:.1f}%)  |da| {arr[:,3].mean():.4f}  "
          f"eef-init {arr[:,4].mean():.1f}mm")
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_REPO, "viz", "demo_replays"))
    p.add_argument("--cameras", default="agentview,sideview",
                   help="comma-separated; multiple are tiled side by side. "
                        "available: frontview, birdview, agentview, sideview, "
                        "agentview_full, robot0_robotview, robot0_eye_in_hand. "
                        "agentview shows the threading moment most clearly; the "
                        "wrist cam (robot0_eye_in_hand) is mostly gripper.")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--fps", type=int, default=20, help="control freq is 20 Hz")
    p.add_argument("--n_sigma002", type=int, default=20)
    p.add_argument("--n_sigma01", type=int, default=50)
    p.add_argument("--only", default="", choices=["", "sigma0.02", "sigma0.1"])
    p.add_argument("--limit", type=int, default=0, help="debug: cap demos per set")
    args = p.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    DATASETS["sigma0.02"]["n"] = args.n_sigma002
    DATASETS["sigma0.1"]["n"] = args.n_sigma01

    env = PixelRobosuite(
        env_name="Threading_D0",
        robots="Panda",
        episode_length=400,
        device="cuda",
        camera_names=cameras,
        rl_cameras=[cameras[0]],
        image_size=args.res,
        rl_image_size=96,
        use_state=1,
        domain_shift=TARGET_SHIFT,
    )
    env.reset()

    todo = [args.only] if args.only else list(DATASETS)
    for tag in todo:
        replay_dataset(env, tag, DATASETS[tag], args.out, cameras,
                       args.res, args.fps, args.limit)

    print(f"\ndone -> {args.out}")
    os._exit(0)  # EGL teardown hangs otherwise


if __name__ == "__main__":
    main()
