"""Re-render camera images for an EXISTING SimDist dataset, from stored state.

Why this exists: rollouts (policy + physics, the expensive part) and rendering
(cheap-ish, and dependent on choices that are still in flux -- wrist extrinsics,
intrinsics, resolution, which cameras at all) must not be coupled. A dataset
collected with `--store_sim_state 1` carries

    obs/sim_state    (T, nq+nv+1) float64   per step: the full MuJoCo state
    model_state      (D,)         float64   per episode: every mutable model
                                            field (the episode's DR draw --
                                            base pose, lights, colours, cameras)

and rendering a frame is a pure function of those two, so this script can
reproduce any camera BYTE FOR BYTE, index-aligned with the same transitions.
That alignment is the whole point: the pixel world model consumes BOTH cameras
of the SAME transition, so a second collection pass (different rollouts) could
never be paired with the first.

Gate: scripts/test_rerender.py (re-renders a smoke set and asserts equality,
AND asserts that skipping the model_state restore changes the frames -- i.e.
that the restore is load-bearing rather than decorative).

Usage (add the wrist camera to an existing dataset, in place of a re-collect):
  python scripts/rerender_simdist_data.py \
      --dataset  <in>.hdf5 --out <out>.hdf5 \
      --expert   exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt \
      --cameras  robot0_eye_in_hand --rl_image_size 224 --jpeg_quality 90 \
      --copy_cameras agentview        # carried over from the input unchanged
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pyrallis  # noqa: E402

import train_rl  # noqa: E402
from env.sim2real import (restore_model_state, mat2quat_wxyz,  # noqa: E402
                         apply_background_image, apply_sim2real, background_files)
from scripts.generate_simdist_data import (  # noqa: E402
    _RES_OVERRIDE, _cams, _write_image_dataset, build_source_env,
    resolve_domain_shift,
)


def render_episode(env, sim_state, model_state, cams, wrist_calibration=None,
                   visual_cfg=None, visual_nominal=None, visual_rng=None,
                   bg_index=None, repaint_cfg=None):
    """(T, ...) uint8 CHW frames per camera, from stored state alone.

    `visual_cfg` re-draws the PURELY VISUAL randomisation (background photo,
    lights, colours, camera jitter) for this episode. It must have every
    dr_base_* zeroed: the rollout happened at one specific base pose, which
    model_state restores, and re-drawing it would desync the frames from the
    qpos they depict. Both cameras get the same draw, so the pair stays
    physical -- which is also why agentview must be RE-RENDERED here rather
    than copied across when this is on.
    """
    wrapper = env
    sim = env.env.sim
    if model_state is not None:
        n_restored = restore_model_state(env.env, model_state)
        # Fields that restore_model_state could not take from this (older)
        # vector keep whatever the PREVIOUS episode left in the model. With
        # visual DR on that is harmless -- apply_sim2real overwrites them below
        # -- but without it they leak across episodes and the frames silently
        # depend on iteration order. Measured on part3: light_castshadow after
        # the restore was 5,5,3,5,7,8 in one run and 3,5,3,5,7,8 in another,
        # i.e. episode 0 inherited a different value purely from process state.
        if n_restored and not visual_cfg:
            raise SystemExit(
                f"This dataset's model_state predates {sorted(n_restored)}, so "
                f"those fields are not restorable from it and would carry over "
                f"from the previous episode. Re-render it with --visual_dr 1 "
                f"(which re-draws them per episode from a recorded seed), or "
                f"re-render a dataset whose model_state covers them.")
    if visual_cfg is not None:
        # ⚠️ visual_nominal must be the PRISTINE pre-DR snapshot taken at build
        # time. Snapshotting after restore_model_state captures this episode's
        # already-randomised values -- including the geom_matid that the old
        # dr_background_gray stripped to -1, which is exactly what stops the
        # photo background from resolving its target.
        apply_sim2real(env.env, visual_cfg, rng=visual_rng,
                       nominal=visual_nominal, bg_index=bg_index)
    elif repaint_cfg is not None and bg_index is not None:
        # Not re-drawing the visual DR, but the episode's photo background still
        # has to be put back: it lives in tex_rgb, which model_state does not
        # carry, so without this the wrist pass renders whatever photo this
        # process happened to draw at its own reset -- a different room from the
        # agentview frames it is being paired with, and the byte-exact replay
        # gate fails outright (measured: max |diff| 212).
        assert apply_background_image(env.env, repaint_cfg, index=bg_index) is not None, (
            "bg_index is recorded but the photo could not be painted "
            "(no render context, or the background material was stripped)")
    wrapper.past_obses.clear()
    if wrist_calibration is not None:
        sim.set_state_from_flattened(np.asarray(sim_state[0], dtype=np.float64))
        sim.forward()
        cid = sim.model.camera_name2id('robot0_eye_in_hand')
        hand = sim.model.body_name2id('robot0_right_hand')
        parent = int(sim.model.cam_bodyid[cid])
        hr = sim.data.body_xmat[hand].reshape(3, 3)
        pr = sim.data.body_xmat[parent].reshape(3, 3)
        mount = np.asarray(wrist_calibration['T_sim_hand_camera_opencv'])
        sim.model.cam_pos[cid] = pr.T @ (sim.data.body_xpos[hand] + hr @ mount[:3, 3] - sim.data.body_xpos[parent])
        sim.model.cam_quat[cid] = mat2quat_wxyz(pr.T @ hr @ mount[:3, :3] @ np.diag([1, -1, -1]))
        sim.model.cam_fovy[cid] = wrist_calibration['fovy']
    out = {c: [] for c in cams}
    for t in range(sim_state.shape[0]):
        sim.set_state_from_flattened(np.asarray(sim_state[t], dtype=np.float64))
        sim.forward()
        # Drive the env's OWN observation path so the frames go through exactly
        # the pipeline collection used (robosuite render -> flip -> resize).
        obs = env.env._get_observations(force_update=True)
        rl_obs = wrapper._extract_images(obs)
        rl_obs = rl_obs[0] if isinstance(rl_obs, tuple) else rl_obs
        for c in cams:
            out[c].append(rl_obs[c].cpu().numpy())
        keep = max(wrapper.obs_stack, wrapper.state_stack, wrapper.prop_stack)
        for history in wrapper.past_obses.values():
            del history[:-keep]
    return {c: np.asarray(v, dtype=np.uint8) for c, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expert", required=True,
                   help="the collection expert ckpt; its cfg.yaml supplies the "
                        "task, the sim2real cell and the render size")
    p.add_argument("--cameras", required=True,
                   help='cameras to RENDER, "+"/","-separated')
    p.add_argument("--copy_cameras", default="",
                   help="cameras to COPY from the input unchanged (they were "
                        "rendered from the same states, so they stay aligned)")
    p.add_argument("--rl_image_size", type=int, default=0)
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--limit", type=int, default=0, help="first N episodes (debug)")
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--wrist_calibration', default='')
    p.add_argument("--cell", default="",
                   help="sim2real cell to render WITH, overriding the expert's. "
                        "This is how the wrist camera gets added: the collection "
                        "cell has agentview only, while the *_calibrated_wrist "
                        "cell carries the hand-eye pose AND the chosen fovy. "
                        "Preferred over --wrist_calibration, which hard-codes a "
                        "pose outside the cell and so can drift from it.")
    p.add_argument("--visual_dr", type=int, default=0,
                   help="1 = re-draw the purely visual randomisation per episode "
                        "(photo background, lights, colours, camera jitter) from "
                        "the expert's cell. DEFAULT OFF so the byte-for-byte "
                        "re-render gate (scripts/test_rerender.py) keeps passing. "
                        "Requires --copy_cameras to be empty: every camera must "
                        "see the SAME draw or the pair is physically impossible.")
    p.add_argument("--visual_dr_seed", type=int, default=0,
                   help="base seed; episode i uses seed + i, and the photo index "
                        "is recorded per episode so the result is reproducible")
    p.add_argument("--no_model_state", type=int, default=0,
                   help="DEBUG/gate only: skip the per-episode model_state "
                        "restore, i.e. render the episodes under a FRESH DR draw")
    args = p.parse_args()
    started = time.monotonic()
    calibration = None
    if args.wrist_calibration:
        with open(args.wrist_calibration) as f:
            calibration = json.load(f)['camera']
        assert 'robot0_eye_in_hand' in args.cameras
    if os.path.exists(args.out):
        raise FileExistsError(args.out)

    cams = _cams(argparse.Namespace(rl_cameras=args.cameras))
    copy_cams = _cams(argparse.Namespace(rl_cameras=args.copy_cameras))
    if args.image_size:
        _RES_OVERRIDE["image_size"] = args.image_size
    if args.rl_image_size:
        _RES_OVERRIDE["rl_image_size"] = args.rl_image_size

    cfg = pyrallis.load(  # type: ignore
        train_rl.MainConfig,
        open(os.path.join(os.path.dirname(args.expert), "cfg.yaml"), "r"))
    env, env_params = build_source_env(
        cfg, 0.0, resolve_domain_shift(argparse.Namespace(domain="expert"), cfg),
        rl_cameras=tuple(cams))
    if args.cell:
        from env.sim2real import load_sim2real
        _c = load_sim2real(args.cell)
        env_params["sim2real"] = _c
        env.sim2real = _c
        env._s2r_active = bool(_c.enabled)
        print(f"[rerender] cell override -> {args.cell}  cameras="
              f"{[(c.name, round(c.resolved_fovy(), 3)) for c in _c.cameras]}", flush=True)
    env.reset()   # build the model once; per-episode state is restored below

    # The cell the env was built with -- used to repaint an episode's recorded
    # background photo when we are NOT re-drawing the visual DR.
    cell_for_repaint = env_params.get("sim2real")
    if cell_for_repaint is not None and not cell_for_repaint.dr_background_image:
        cell_for_repaint = None
    visual_cfg = visual_nominal = None
    n_bg = 0
    if args.visual_dr:
        import dataclasses
        assert not copy_cams, (
            "--visual_dr with --copy_cameras: the copied camera would keep the "
            "background/lighting of the ORIGINAL draw while the re-rendered one "
            "gets a new draw, i.e. two cameras of the same instant disagreeing "
            "about the room. Re-render every camera instead.")
        cell = env_params.get("sim2real")
        assert cell is not None and cell.enabled, "--visual_dr needs a sim2real cell"
        visual_cfg = dataclasses.replace(cell, dr_base_height=0.0, dr_base_xy=0.0,
                                         dr_base_yaw_deg=0.0, dr_base_tilt_deg=0.0)
        visual_nominal = getattr(env, "_s2r_nominal", None)
        assert visual_nominal is not None, "wrapper did not snapshot a nominal"
        n_bg = len(background_files(cell.dr_background_image)) if cell.dr_background_image else 0
        print(f"[rerender] visual DR on; photo pool = {n_bg} images; "
              f"base-pose DR disabled (it would desync frames from qpos)", flush=True)

    frames_total = 0
    with h5py.File(args.dataset, "r") as fin, h5py.File(args.out + '.partial', "w") as fout:
        gi, go = fin["data"], fout.create_group("data")
        for k, v in gi.attrs.items():
            go.attrs[k] = v
        keys = sorted(gi.keys(), key=lambda x: int(x.split("_")[1]))
        keys = keys[args.start:]
        if args.limit:
            keys = keys[: args.limit]
        for i, k in enumerate(keys):
            src = gi[k]
            assert "sim_state" in src, (
                f"{args.dataset}:{k} has no sim_state -- it was collected without "
                "--store_sim_state 1 and cannot be re-rendered")
            sim_state = np.asarray(src["sim_state"])
            ms = None if args.no_model_state else np.asarray(src["model_state"])
            # Default: repaint the episode's OWN photo (recorded at collection).
            bg_idx = (int(src.attrs["bg_index"]) if "bg_index" in src.attrs else None)
            repaint_cfg = cell_for_repaint if bg_idx is not None else None
            vrng = None
            if args.visual_dr:
                repaint_cfg = None          # a fresh draw replaces it entirely
                vrng = np.random.default_rng(args.visual_dr_seed + args.start + i)
                if n_bg:
                    bg_idx = int(vrng.integers(n_bg))
            frames = render_episode(env, sim_state, ms, cams, calibration,
                                    visual_cfg=visual_cfg,
                                    visual_nominal=visual_nominal,
                                    visual_rng=vrng, bg_index=bg_idx,
                                    repaint_cfg=repaint_cfg)
            frames_total += len(sim_state)

            dst = go.create_group(k)
            src.copy("obs/state", dst.require_group("obs"), name="state")
            for c in copy_cams:
                src.copy(f"obs/{c}_image", dst["obs"], name=f"{c}_image")
            for name in ("actions", "rewards", "expert_flags", "values",
                         "sim_state", "model_state"):
                if name in src:
                    src.copy(name, dst, name=name)
            for c in cams:
                _write_image_dataset(dst, f"obs/{c}_image", frames[c],
                                     args.jpeg_quality)
            for a, v in src.attrs.items():
                dst.attrs[a] = v
            if args.visual_dr:
                # index, not pixels: background_files() is sorted, so this is a
                # stable identifier for "which photo" (docs/infra.md 14f).
                dst.attrs["visual_dr_seed"] = args.visual_dr_seed + args.start + i
                if bg_idx is not None:
                    dst.attrs["bg_index"] = bg_idx
            if (i + 1) % 50 == 0:
                print(f"[rerender] {i + 1}/{len(keys)} episodes", flush=True)
        go.attrs["rerendered_cameras"] = json.dumps(list(cams))
        go.attrs['num_demos'] = len(keys)
        go.attrs['num_transitions'] = frames_total
        go.attrs['render_complete'] = True
        go.attrs['visual_dr'] = int(args.visual_dr)
        go.attrs['visual_dr_seed'] = args.visual_dr_seed
        go.attrs['wrist_calibration'] = json.dumps(calibration, sort_keys=True)
        fout.flush()
    os.replace(args.out + '.partial', args.out)
    env.env.close()
    print(json.dumps(dict(episodes=len(keys), frames=frames_total, seconds=time.monotonic()-started)), flush=True)
    print(f"[rerender] wrote {args.out}")


if __name__ == "__main__":
    main()
