"""Collect successful demos on the TARGET domain with a trained (IBRL) agent.

Rolls out a checkpoint under a fixed contact-phase shift (with an overridable
hole_offset), keeps only episodes that succeed within `--max_demo_len` steps,
and writes them in the same HDF5 layout produced by `rl/replay.py::_save_replay`
(so the resulting dataset is loadable by `rl.replay.add_demos_to_replay` and the
BC dataset tooling). A per-demo mp4 video is also saved.

Example:
    python scripts/collect_target_demos.py \
        --weight exps/mimicgen/threading_ibrl_dr_oracle_seed1/model0.pt \
        --num_demo 20 --hole_offset 0.01 --max_demo_len 100 \
        --hand_init_perturb_joint_sigma 0.02
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import json
from collections import defaultdict

import numpy as np
import torch
import h5py

import common_utils
from common_utils import Recorder
from common_utils import ibrl_utils as utils
import train_rl
from env.robosuite_wrapper import PixelRobosuite, DEFAULT_STATE_KEYS
from env.domain_shift import DomainShiftConfig


# obs keys that are stored verbatim; everything else is treated as a camera
# image and gets an `_image` suffix (mirrors rl/replay.py::_save_replay).
_NON_IMAGE_KEYS = ["prop", "state", "sim_state"] + DEFAULT_STATE_KEYS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str,
                   default="exps/mimicgen/threading_ibrl_dr_oracle_seed1/model0.pt")
    p.add_argument("--mode", type=str, default="rl", choices=["rl", "bc"])
    p.add_argument("--num_demo", type=int, default=20,
                   help="number of successful demos to collect")
    p.add_argument("--max_demo_len", type=int, default=100,
                   help="reject any episode that has not succeeded by this step")
    p.add_argument("--hole_offset", type=float, default=0.01,
                   help="per-axis (x,y) hole offset; z stays 0")
    p.add_argument("--solref_scale", type=float, default=0.8,
                   help="contact solref scale applied over the ckpt's shift. "
                        "1.0 = nominal; target uses 0.8 (0.02->0.016s, stiffer = "
                        "mildly harder contact, clip-through-safe).")
    p.add_argument("--solimp_scale", type=float, default=1.0,
                   help="contact solimp scale. Keep 1.0 to avoid clip-through "
                        "(lowering solimp directly allows mesh interpenetration).")
    p.add_argument("--hand_init_perturb_joint_sigma", type=float, default=0.02,
                   help="joint-space gaussian sigma for robot init (0.02 = demo default)")
    # Explicit shift overrides (any task). Default None = leave the ckpt/base
    # value. Use these to collect on a specific fixed target (e.g. Square_D0:
    # --gripper_kp_scale 0.25 --peg_size_scale 1.25 --peg_friction_scale 2.0).
    p.add_argument("--gripper_kp_scale", type=float, default=None)
    p.add_argument("--peg_size_scale", type=float, default=None,
                   help="[Square] grow the square peg (>1 = tighter clearance)")
    p.add_argument("--peg_friction_scale", type=float, default=None,
                   help="[Square] scale the square peg friction (>1 = grippier)")
    p.add_argument("--ring_size_scale", type=float, default=None,
                   help="[Threading] shrink the ring opening (<1 = tighter; "
                        "the SS12 target uses 0.85 -> 13.6 mm)")
    p.add_argument("--ring_friction_scale", type=float, default=None,
                   help="[Threading] scale the ring rim friction (>1 = grippier)")
    p.add_argument("--needle_size_scale", type=float, default=None,
                   help="[Threading] THICKEN the needle shaft (>1 = tighter); the "
                        "mirror of --ring_size_scale, closing the same clearance "
                        "from the needle side. 1.2 -> 12 mm (nominal 10 mm).")
    p.add_argument("--lid_damping_scale", type=float, default=None,
                   help="[Coffee] scale the lid hinge viscous damping (nominal "
                        "0.005). The stage-2 knob; the SS40 targets use 100 "
                        "(sharp) / 75 (moderate).")
    p.add_argument("--lid_frictionloss", type=float, default=None,
                   help="[Coffee] ABSOLUTE dry friction on the lid hinge (N*m; "
                        "nominal 0, so a scale would be a no-op). Weak/non-"
                        "monotonic in practice -- prefer --lid_damping_scale.")
    p.add_argument("--pod_holder_size_scale", type=float, default=None,
                   help="[Coffee] shrink the pod holder bore. MEASURED NO-OP "
                        "(100%% at every level to 0.94) -- the pod's widest ring "
                        "never passes the wall. Kept for completeness; do not "
                        "use it to build a target.")
    p.add_argument("--pod_holder_friction_scale", type=float, default=None,
                   help="[Coffee] scale the pod holder cup-wall friction "
                        "(nominal 1.0). Measured inert.")
    p.add_argument("--friction_scale", type=float, default=None)
    p.add_argument("--mass_scale", type=float, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    # --- optional CONTIGUOUS action noise during collection (default OFF) ---
    # Mirrors generate_simdist_data.py's scheme. sigma_max<=0 => byte-identical
    # to the old deterministic path (no RNG draws, no branch taken).
    p.add_argument("--noise_sigma_min", type=float, default=0.0,
                   help="per-dim action-noise std lower bound; noise is OFF unless "
                        "--noise_sigma_max > 0")
    p.add_argument("--noise_sigma_max", type=float, default=0.0)
    p.add_argument("--noise_min", type=int, default=1,
                   help="noisy interval length ~ U[noise_min, noise_max] steps")
    p.add_argument("--noise_max", type=int, default=5)
    p.add_argument("--nonoise_min", type=int, default=5,
                   help="noise-free interval length ~ U[nonoise_min, nonoise_max]")
    p.add_argument("--nonoise_max", type=int, default=10)
    p.add_argument("--max_episodes", type=int, default=2000,
                   help="safety cap on total rollouts before giving up")
    p.add_argument("--cameras", type=str, default="agentview,robot0_eye_in_hand",
                   help="comma-separated cameras to render (first two used for video)")
    p.add_argument("--interactive", type=int, default=1,
                   help="1 = human-in-the-loop: review each candidate's video and "
                        "keep/discard by hand (filters clip-through false successes). "
                        "0 = auto-accept every success within max_demo_len.")
    p.add_argument("--shift_mode", type=str, default="fixed", choices=["none", "fixed"],
                   help="'fixed' = collect on the target shift (default). "
                        "'none' = pure SOURCE domain (no contact/hole shift); "
                        "hole_offset/solref/solimp/mask flags are then ignored.")
    p.add_argument("--record_video", type=int, default=1,
                   help="Save a video every Nth accepted demo (0 = no videos). "
                        "With 0 only the rl camera is rendered (faster for large "
                        "collections); >0 renders the full --cameras set.")
    p.add_argument("--collect_mask_in_obs", type=int, default=0,
                   help="Does the POLICY see masked obs during collection? "
                        "0 = policy sees the TRUE (shifted) hole so it threads "
                        "cleanly (no stabbing). 1 = policy is blind (nominal hole). "
                        "Ignored when --shift_mode none.")
    p.add_argument("--save_masked_state", type=int, default=1,
                   help="1 = subtract the hole offset from the SAVED object-state/"
                        "state so the stored demo matches the source (nominal-hole) "
                        "convention, even though the policy saw the true hole. "
                        "0 = store the raw (true, offset) state.")
    p.add_argument("--write_every", type=int, default=250,
                   help="flush the full hdf5 every N accepted demos (plus once "
                        "at the very end). write_dataset rewrites the WHOLE file "
                        "each flush, so writing after every demo is O(N^2) and "
                        "kills large collections; batching keeps it fast.")
    p.add_argument("--drop_images", type=int, default=0,
                   help="1 = do not store rendered camera *_image keys in the "
                        "demo obs (state-based downstream ignores them; avoids "
                        "~2.4 MB/demo of eye-in-hand frames AND the matching RAM "
                        "growth). Video recording (--record_video) is unaffected.")
    return p.parse_args()


def build_env(args):
    if args.mode == "rl":
        agent, _, env_params = train_rl.load_model(args.weight, "cuda")
    else:
        import train_bc
        agent, _, env_params = train_bc.load_model(args.weight, "cuda")

    # Target shift: start from the ckpt's eval_domain_shift (the trained target)
    # and force fixed mode + the requested hole_offset. solref/solimp are
    # overridden to the (milder) collection target so demos don't clip through
    # the ring mesh — the ckpt's stored shift may still carry the old soft
    # values (solref 3.0 / solimp 0.7) that produced false-positive successes.
    base_shift = env_params.get("domain_shift")
    hole = (args.hole_offset, args.hole_offset, 0.0)
    mask_in_obs = bool(args.collect_mask_in_obs)
    if args.shift_mode == "none":
        # Pure source domain — no contact/hole shift at all.
        shift = DomainShiftConfig(mode="none")
    elif isinstance(base_shift, DomainShiftConfig):
        shift = dataclasses.replace(
            base_shift, mode="fixed", hole_offset=hole,
            solref_scale=args.solref_scale, solimp_scale=args.solimp_scale,
            mask_hole_offset_in_obs=mask_in_obs)
    else:
        # Fallback: the documented Threading target defaults.
        shift = DomainShiftConfig(
            mode="fixed", friction_scale=0.3, mass_scale=3.0,
            solref_scale=args.solref_scale, solimp_scale=args.solimp_scale,
            gripper_kp_scale=0.5, hole_offset=hole,
            mask_hole_offset_in_obs=mask_in_obs,
        )

    # Explicit per-field overrides (task-agnostic; used for Square_D0 target
    # which needs peg_size/peg_friction/gripper_kp the ckpt shift doesn't carry).
    _ov = {}
    for _f in ("gripper_kp_scale", "peg_size_scale", "peg_friction_scale",
               "ring_size_scale", "ring_friction_scale", "needle_size_scale",
               "lid_damping_scale", "lid_frictionloss",
               "pod_holder_size_scale", "pod_holder_friction_scale",
               "friction_scale", "mass_scale"):
        _v = getattr(args, _f)
        if _v is not None:
            _ov[_f] = _v
    if _ov:
        shift = dataclasses.replace(shift, mode="fixed", **_ov)
        print(f"[override] applied shift fields: {_ov}")

    env_params["domain_shift"] = shift
    env_params["hand_init_perturb"] = None
    env_params["hand_init_perturb_joint_sigma"] = args.hand_init_perturb_joint_sigma
    if args.record_video > 0:
        env_params["camera_names"] = [c for c in args.cameras.split(",") if c]
    else:
        # No videos: render only the rl camera (kept in the demo obs), faster.
        env_params["camera_names"] = list(
            env_params.get("rl_cameras") or ["robot0_eye_in_hand"])
    env_params["record_sim_state"] = True

    print(common_utils.wrap_ruler("collection env"))
    print(f"shift_mode = {args.shift_mode}; domain_shift = {shift}")
    print(f"policy sees masked obs = {mask_in_obs}; save nominal (masked) "
          f"state = {bool(args.save_masked_state)}; record_video = {args.record_video}")
    print(f"hand_init_perturb_joint_sigma = {args.hand_init_perturb_joint_sigma}")
    print(f"cameras = {env_params['camera_names']}")
    print(f"episode_length = {env_params.get('episode_length')}")
    print(common_utils.wrap_ruler(""))

    env = PixelRobosuite(**env_params)
    return agent, env, env_params


def _noise_cfg(args):
    """None when disabled, so the no-noise path stays byte-identical."""
    if args.noise_sigma_max <= 0:
        return None
    return dict(sigma_min=args.noise_sigma_min, sigma_max=args.noise_sigma_max,
                noise_min=args.noise_min, noise_max=args.noise_max,
                nonoise_min=args.nonoise_min, nonoise_max=args.nonoise_max)


def rollout_once(agent, env, seed, max_len, drop_images=False, noise=None):
    """Run one episode (capped at max_len). Returns (ok, ep_obs, actions, rewards, frames).

    `noise` (optional dict, see `_noise_cfg`) injects CONTIGUOUS action-noise
    intervals, the scheme `generate_simdist_data.py` uses for the stage-1
    distillation set: per-episode Sigma=diag(sigma) with sigma_i ~ U[min,max],
    noisy intervals of ~U[noise_min,noise_max] steps alternating with clean ones
    of ~U[nonoise_min,nonoise_max]. Contiguous (not per-step iid) on purpose:
    iid perturbations are largely cancelled by the OSC controller and the next
    action, so they only corrupt the BC target without buying state coverage,
    whereas a sustained interval produces a real detour (the same
    persistent-vs-iid distinction CLAUDE.md 27 measured for oracle-dynamics noise).
    The STORED action is the executed (noisy, clamped) one -- that is what
    produced the transition, and what BC must reproduce.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sigma = None
    if noise is not None:
        sigma = rng.uniform(noise["sigma_min"], noise["sigma_max"],
                            size=env.action_dim).astype(np.float32)

        def _interval(on):
            lo, hi = ((noise["noise_min"], noise["noise_max"]) if on
                      else (noise["nonoise_min"], noise["nonoise_max"]))
            return int(rng.integers(lo, hi + 1))

        noise_on = bool(rng.integers(0, 2))
        steps_left = _interval(noise_on)
    obs, image_obs = env.reset()

    ep_obs: dict = defaultdict(list)
    actions: list = []
    rewards: list = []
    frames: list = [image_obs]  # high-res dicts for video

    terminal = False
    succ = False
    steps = 0
    with torch.no_grad(), utils.eval_mode(agent):
        while not terminal and steps < max_len:
            # Record the obs the agent acts on BEFORE calling act(): the
            # agent's _maybe_unsqueeze_ mutates obs in place (adds a batch
            # dim), which would otherwise leak a spurious (1, dim) axis into
            # the saved arrays and break add_demos_to_replay's concat.
            for k, v in obs.items():
                # Skip rendered camera frames entirely (not just at save time)
                # so they never accumulate in RAM over a large collection.
                if drop_images and k not in _NON_IMAGE_KEYS:
                    continue
                ep_obs[k].append(v.cpu().numpy())

            action = agent.act(obs, eval_mode=True)
            if sigma is not None:
                if steps_left <= 0:
                    noise_on = not noise_on
                    steps_left = _interval(noise_on)
                if noise_on:
                    eps = torch.from_numpy(rng.normal(0.0, sigma).astype(np.float32))
                    action = (action + eps.to(action.device)).clamp(-1.0, 1.0)
                steps_left -= 1
            actions.append(action.cpu().numpy())

            obs, reward, terminal, succ, image_obs = env.step(action)
            rewards.append(float(reward))
            frames.append(image_obs)
            steps += 1

    success = bool(succ) or (np.sum(rewards) > 0)
    ok = success and terminal and steps <= max_len
    return ok, ep_obs, actions, rewards, frames, steps


def store_nominal_state(ep_obs, env):
    """Subtract the hole offset from the SAVED object-state / state so the
    stored demo matches the source (nominal-hole) convention, even though the
    policy acted on the true (unmasked) hole. This mirrors
    `mask_hole_offset_from_obs` but is applied only to the saved copy — net
    result == what mask=True would have stored, with clean trajectories.

    `sim_state` is intentionally left at the true physical configuration so the
    demo can still be replayed/reset exactly; only the BC-facing state channels
    are nominalized.
    """
    shift = getattr(env, "_shift_cfg", None)
    if shift is None:
        return
    offset = np.asarray(shift.hole_offset, dtype=np.float64)
    if not np.any(offset != 0):
        return
    slices = (getattr(env, "_shift_bindings", None) or {}).get("obj_state_slices", {})
    if not slices:
        return

    if "object" in ep_obs:  # object == object-state, shape (T, 28)
        obj = np.asarray(ep_obs["object"], dtype=np.float64)
        for key in ("tripod_pos", "tripod_to_robot0_eef_pos"):
            if key in slices:
                s, e = slices[key]
                obj[:, s:e] -= offset
        ep_obs["object"] = obj

    if "state" in ep_obs:  # object-state occupies the trailing 28 dims of state
        st = np.asarray(ep_obs["state"], dtype=np.float64)
        base = st.shape[1] - 28
        if base >= 0:
            for key in ("tripod_pos", "tripod_to_robot0_eef_pos"):
                if key in slices:
                    s, e = slices[key]
                    st[:, base + s:base + e] -= offset
            ep_obs["state"] = st


def save_video(frames, video_dir, name):
    rec = Recorder(video_dir)
    for img_dict in frames:
        rec.add(img_dict)
    return rec.save(name)


def write_dataset(accepted, out_dir, env_params, args):
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "dataset.hdf5")
    with h5py.File(h5_path, "w") as hf:
        data_grp = hf.create_group("data")
        data_grp.attrs["num_demos"] = len(accepted)
        for i, ep in enumerate(accepted):
            g = data_grp.create_group(f"demo_{i}")
            for k, arr_list in ep["obs"].items():
                key = k if k in _NON_IMAGE_KEYS else f"{k}_image"
                g.create_dataset(f"obs/{key}", data=np.asarray(arr_list))
            g.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
            g.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
            g.attrs["num_samples"] = len(ep["actions"])
            # The rollout seed IS the episode's initial state (`rollout_once`
            # does np.random.seed(seed) before env.reset()), so storing it lets
            # an A/B pair be aligned by SEED INTERSECTION afterwards. Necessary
            # because `seed = args.seed + attempt`: one differing accept/reject
            # shifts every later seed, so two arms of the "same" collection can
            # silently end up on disjoint initial states -- and the resulting
            # 20-demo draw noise (adapt std +-0.109) dwarfs the effect under test.
            if "seed" in ep:
                g.attrs["seed"] = int(ep["seed"])

    # env_cfg.json — add_demos_to_replay reads env_name from here.
    env_cfg = {
        "env_name": env_params["env_name"],
        "domain_shift": dataclasses.asdict(env_params["domain_shift"]),
        "hand_init_perturb_joint_sigma": args.hand_init_perturb_joint_sigma,
        # Inherited SILENTLY from the collecting ckpt's cfg (eval_init_dist) and
        # not settable from the CLI, so record it or a set collected under a
        # narrowed reset window (Square's rotw1 120-180 deg nut yaw) is
        # indistinguishable afterwards from a full-circle one.
        "init_distribution": (
            dataclasses.asdict(env_params["init_distribution"])
            if env_params.get("init_distribution") is not None else None),
        "rl_image_size": env_params.get("rl_image_size"),
        "use_state": env_params.get("use_state"),
        "source_weight": args.weight,
        "collect_action_noise": (None if args.noise_sigma_max <= 0 else {
            "sigma": [args.noise_sigma_min, args.noise_sigma_max],
            "noise_interval": [args.noise_min, args.noise_max],
            "nonoise_interval": [args.nonoise_min, args.nonoise_max],
            "note": "contiguous intervals; STORED action is the executed (noisy) one"}),
    }
    with open(os.path.join(out_dir, "env_cfg.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)
    return h5_path


def prompt_keep(video_path):
    """Ask the human to keep/discard a candidate. Returns 'y' / 'n' / 'q'."""
    print(f"  video: {os.path.abspath(video_path)}")
    while True:
        try:
            ans = input("  keep this demo? [y]es / [n]o / [q]uit & save: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        if ans in ("y", "yes"):
            return "y"
        if ans in ("n", "no", "r"):
            return "n"
        if ans in ("q", "quit"):
            return "q"
        print("  please type y (keep), n (discard), or q (quit & save).")


def collect_interactive(args, agent, env, env_params, out_dir, video_dir):
    """Human-in-the-loop: roll out one success candidate at a time, save its
    video, and let the human keep/discard it before it enters the dataset.
    Filters clip-through false positives that pass the automatic success check.
    """
    cand_dir = os.path.join(video_dir, "candidates")
    rej_dir = os.path.join(video_dir, "rejected")
    os.makedirs(cand_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)

    accepted: list = []
    lens: list = []
    attempt = 0
    n_skip = 0
    cand_id = 0
    quit_requested = False

    while len(accepted) < args.num_demo and attempt < args.max_episodes:
        seed = args.seed + attempt
        ok, ep_obs, actions, rewards, frames, steps = rollout_once(
            agent, env, seed, args.max_demo_len, drop_images=bool(args.drop_images),
            noise=_noise_cfg(args))
        attempt += 1

        if not ok:
            n_skip += 1
            print(f"  ... rolling (skipped {n_skip}: seed {seed} no clean "
                  f"success within {args.max_demo_len} steps)")
            continue

        cand_name = f"candidate_{cand_id}_seed{seed}_len{steps}"
        vid_path = save_video(frames, cand_dir, cand_name)
        cand_id += 1

        print("\n" + "=" * 60)
        print(f"CANDIDATE  (kept {len(accepted)}/{args.num_demo}, "
              f"skipped {n_skip} non-success)")
        print(f"  seed={seed}  len={steps}  (auto-success within {args.max_demo_len} steps)")
        choice = prompt_keep(vid_path)

        if choice == "q":
            os.rename(vid_path, os.path.join(rej_dir, os.path.basename(vid_path)))
            quit_requested = True
            break
        if choice == "n":
            os.rename(vid_path, os.path.join(rej_dir, os.path.basename(vid_path)))
            print("  -> discarded (clip-through / not clean)")
            continue

        # keep: rename video, append, and persist the dataset incrementally so
        # progress survives an interrupt mid-collection.
        idx = len(accepted)
        final_vid = os.path.join(video_dir, f"demo_{idx}_seed{seed}_len{steps}.mp4")
        os.rename(vid_path, final_vid)
        if args.save_masked_state:
            store_nominal_state(ep_obs, env)
        accepted.append({"obs": ep_obs, "actions": actions, "rewards": rewards,
                         "seed": seed, "len": steps})
        lens.append(steps)
        if args.write_every > 0 and len(accepted) % args.write_every == 0:
            write_dataset(accepted, out_dir, env_params, args)
        print(f"  -> KEPT as demo_{idx}  ({len(accepted)}/{args.num_demo} done)")

    return accepted, lens, attempt, quit_requested


def collect_auto(args, agent, env, env_params, out_dir, video_dir):
    """Non-interactive: accept every success within max_demo_len."""
    accepted: list = []
    lens: list = []
    attempt = 0
    while len(accepted) < args.num_demo and attempt < args.max_episodes:
        seed = args.seed + attempt
        ok, ep_obs, actions, rewards, frames, steps = rollout_once(
            agent, env, seed, args.max_demo_len, drop_images=bool(args.drop_images),
            noise=_noise_cfg(args))
        attempt += 1
        if not ok:
            print(f"[reject] attempt {attempt} (seed {seed}): "
                  f"timeout/no-success within max_demo_len, steps={steps}")
            continue
        idx = len(accepted)
        if args.record_video > 0 and idx % args.record_video == 0:
            save_video(frames, video_dir, f"demo_{idx}_seed{seed}_len{steps}")
        if args.save_masked_state:
            store_nominal_state(ep_obs, env)
        accepted.append({"obs": ep_obs, "actions": actions, "rewards": rewards,
                         "seed": seed, "len": steps})
        lens.append(steps)
        if args.write_every > 0 and len(accepted) % args.write_every == 0:
            write_dataset(accepted, out_dir, env_params, args)
        print(f"[accept] demo {idx + 1}/{args.num_demo} (seed {seed}): len={steps}")
    return accepted, lens, attempt, False


def main():
    args = parse_args()
    common_utils.set_all_seeds(args.seed)

    if args.out_dir:
        out_dir = args.out_dir
    elif args.shift_mode == "none":
        out_dir = f"release/data/mimicgen/threading_d0_source_sigma{args.hand_init_perturb_joint_sigma}"
    else:
        out_dir = f"release/data/mimicgen/threading_d0_target_hole{args.hole_offset}"
    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    agent, env, env_params = build_env(args)

    if args.interactive:
        print("HUMAN-IN-THE-LOOP mode: review each candidate's video, then keep/discard.")
        accepted, lens, attempt, quit_requested = collect_interactive(
            args, agent, env, env_params, out_dir, video_dir)
    else:
        accepted, lens, attempt, quit_requested = collect_auto(
            args, agent, env, env_params, out_dir, video_dir)

    h5_path = write_dataset(accepted, out_dir, env_params, args)

    print(common_utils.wrap_ruler("summary"))
    print(f"collected {len(accepted)}/{args.num_demo} demos in {attempt} attempts")
    if lens:
        print(f"demo length: mean={np.mean(lens):.1f}, min={min(lens)}, max={max(lens)}")
    print(f"dataset: {h5_path}")
    print(f"videos:  {video_dir}/  (kept demos as demo_*.mp4)")
    if quit_requested:
        print("(quit early at your request; dataset holds what you kept so far.)")
    elif len(accepted) < args.num_demo:
        print(f"WARNING: only got {len(accepted)} demos; raise --max_episodes "
              f"or --max_demo_len if you need more.")


if __name__ == "__main__":
    import rich.traceback
    rich.traceback.install()
    main()
