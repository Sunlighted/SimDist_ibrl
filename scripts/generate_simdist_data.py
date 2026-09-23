"""Diverse Data Generation for SimDist (方案 A / CLAUDE.md §17).

Faithful port of SimDist's Algorithm 2 (Levy et al., RSS 2026) to this repo's
state-based Threading setup. Given an EXPERT policy pi_e (+ its critic V_e) and a
set of sub-optimal policy checkpoints {pi_k} harvested during IBRL training
(train_rl.py --snapshot_period), roll out in the SOURCE sim with:

  * per-episode policy selection: pi_e with prob `p_expert`, else a uniformly
    random sub-optimal checkpoint (the "reset with sub-optimal policies" step);
  * Gaussian action perturbations injected in CONTIGUOUS noise intervals
    (length ~ U[noise_min, noise_max]) interleaved with noise-free intervals
    (~ U[nonoise_min, nonoise_max]); per-episode noise covariance Sigma =
    diag(sigma), sigma_i ~ U[sigma_min, sigma_max];
  * each step records (obs/state, action, reward, V_e(s), expert_flag), where
    expert_flag = 1 iff acting with pi_e AND no noise this step.

The whole (non-truncated) trajectory is kept — incl. sub-optimal / failed
exploration — which is the entire point: broad state-action coverage for the
world model's dynamics head.

Parallelism: `--num_proc N` runs N self-contained worker processes (each loads
its own policies + SOURCE env and rolls out its share of trajectories to a
shard hdf5); the parent then merges the shards into `<out_dir>/dataset.hdf5`.
This is embarrassingly parallel (no central agent), so ~Nx throughput; needed
for the 50k-trajectory regime.

Output is the SAME hdf5 layout as the IBRL replay dump
(`data/demo_N/{obs/state, actions, rewards}` + `values`/`expert_flags`), so
`rl/sequence_replay.py::load_source_replay_to_sequence_buffer` (SimDist 方案 B
loader) reads it directly:

  python train_simdist.py ... --source_replay_path <out_dir>/dataset.hdf5

Usage:
  python scripts/generate_simdist_data.py \
      --expert exps/mimicgen/threading_ibrl_snapshots_seed1/snapshot_187500.pt \
      --checkpoints_glob 'exps/mimicgen/threading_ibrl_snapshots_seed1/snapshot_*.pt' \
      --out_dir release/data/mimicgen/threading_d0_source_diverse \
      --num_traj 50000 --num_proc 10
"""

import argparse
import contextlib
import dataclasses
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyrallis  # noqa: E402

import train_rl  # noqa: E402
import train_bc  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.domain_shift import DomainShiftConfig  # noqa: E402
from common_utils import ibrl_utils as utils  # noqa: E402
# Repo-relative splat defaults, so another machine needs only the repo +
# submodules. Module-level imports here are cheap (json/os/sys/numpy); the
# renderer is only pulled in when a SplatCompositor is constructed.
from real2sim.splat_obs import (  # noqa: E402
    DEFAULT_CALIB, DEFAULT_SEG_DIR, DEFAULT_SCENE_PLY, DEFAULT_GROUND)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", type=str, required=True,
                   help="expert ckpt; also the source of V_e (optimal value labels)")
    p.add_argument("--checkpoints", type=str, default="",
                   help="comma-separated sub-optimal ckpt paths (e.g. snapshot_*.pt)")
    p.add_argument("--checkpoints_glob", type=str, default="",
                   help="glob for sub-optimal ckpts (alternative to --checkpoints)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_traj", type=int, default=2000)
    p.add_argument("--num_proc", type=int, default=1,
                   help="parallel worker processes (each self-contained); >1 => shard+merge")
    p.add_argument("--p_expert", type=float, default=0.5,
                   help="prob of using the expert policy for an episode (else a sub-optimal)")
    p.add_argument("--sigma_min", type=float, default=0.05,
                   help="per-dim action-noise std lower bound (Sigma=diag(sigma), sigma_i~U[min,max])")
    p.add_argument("--sigma_max", type=float, default=0.2)
    p.add_argument("--noise_min", type=int, default=1, help="noisy interval length ~ U[noise_min,noise_max]")
    p.add_argument("--noise_max", type=int, default=5)
    p.add_argument("--nonoise_min", type=int, default=5, help="noise-free interval length ~ U[..]")
    p.add_argument("--nonoise_max", type=int, default=10)
    p.add_argument("--max_len", type=int, default=400)
    p.add_argument("--hand_init_sigma", type=float, default=0.1,
                   help="joint-space init gaussian sigma for the source env")
    p.add_argument("--save_values", type=int, default=1)
    p.add_argument("--value_full_ensemble", type=int, default=0,
                   help="if 1, V_e(s) is the min over ALL num_q critic heads instead of a "
                        "RANDOM num_k subset (MultiFcQ.forward_k is RED-Q's random-subset "
                        "min, so the default value labels carry per-call sampling noise, "
                        "std ~0.05 on this ckpt). Only the value LABELS are affected; the "
                        "rollout's action selection keeps the configured num_k.")
    p.add_argument("--value_use_ibrl", type=int, default=0,
                   help="if 1, the value labels V_e(s) are the IBRL HYBRID value "
                        "max(Q(s, pi_rl(s)), Q(s, pi_prior(s))) — i.e. the value of the "
                        "action the hybrid expert ACTUALLY executes — instead of the bare "
                        "RL actor's Q(s, pi_rl(s)). Requires --expert_hybrid 1 (the prior "
                        "must be attached).")
    p.add_argument("--expert_hybrid", type=int, default=0,
                   help="if 1, roll out the EXPERT AND the sub-optimal snapshots as their "
                        "IBRL hybrid (RL actor + BC prior, act_method=ibrl) instead of the "
                        "bare RL actor. The prior is the expert cfg's bc_policy (override with "
                        "--expert_bc_policy). V_e (values) is unaffected (it uses the critic + "
                        "RL actor either way).")
    p.add_argument("--expert_bc_policy", type=str, default="",
                   help="override the BC-prior ckpt for the hybrid expert (default: read "
                        "bc_policy from the expert's cfg.yaml). Only used when --expert_hybrid 1.")
    p.add_argument("--snapshot_hybrid", type=int, default=1,
                   help="only used when --expert_hybrid 1: if 0, ONLY the expert gets the "
                        "IBRL hybrid treatment; the sub-optimal snapshots roll out as bare "
                        "RL actors (act_method=rl). Default 1 = legacy behavior (all hybrid).")
    p.add_argument("--domain", type=str, default="expert", choices=["expert", "source"],
                   help="rollout domain: 'expert' = the expert's own training env "
                        "(cfg.train_domain_shift; source expert=>source, target "
                        "expert=>its target shift), 'source' = force nominal (mode=none).")
    # PIXEL arm: render + record camera frames alongside the state. Empty (default)
    # = state-only, byte-for-byte the legacy behaviour. The state vector is ALWAYS
    # recorded even with cameras on, because stage-1's value distillation queries
    # the frozen STATE critic (train_simdist.py:354-367).
    p.add_argument("--rl_cameras", type=str, default="",
                   help='"+"- or comma-separated, e.g. '
                        '"agentview+robot0_eye_in_hand". Prefer "+": sbatch '
                        '--export splits values on commas.')
    # PIXEL resolution overrides. Both default to 0 = "use the expert cfg's value"
    # (byte-for-byte legacy). The expert cfg carries image_size=224 / rl_image_size=96,
    # i.e. we ALREADY render at 224 and throw it away in the wrapper's resize; setting
    # --rl_image_size 224 stores the native render at NO extra render cost.
    p.add_argument("--image_size", type=int, default=0,
                   help="0 = use expert cfg.image_size (offscreen render size)")
    p.add_argument("--rl_image_size", type=int, default=0,
                   help="0 = use expert cfg.rl_image_size (STORED frame size). "
                        "Set 224 for the ResNet-18 arm.")
    # PHOTOREAL obs: replace the MuJoCo camera frames with 3DGS composites
    # (splat scene + splat board; MuJoCo arm + peg -- real2sim/splat_obs.py).
    # Needs a GPU per worker: gsplat JIT-compiles on first use and each worker
    # holds its own copy of the gaussians (~570k => ~150 MB of VRAM each).
    p.add_argument("--splat", type=int, default=0)
    # BOARD RESET JITTER (peg env). The real board is hand-placed and will not
    # land identically twice; a fixed board teaches a constant the cell cannot
    # honour. 0.0 = the original fixed board, byte-identical.
    p.add_argument("--board_pos_noise", type=float, default=0.0)   # metres, uniform +-
    p.add_argument("--board_yaw_noise", type=float, default=0.0)   # degrees, uniform +-
    # CAMERA EXTRINSIC + LIGHTING DR (arm 3). §4 B calls camera-extrinsic DR
    # mandatory: the external ZED is re-mounted between sessions. Lighting DR
    # covers the strip being bumped / dimmed.
    p.add_argument("--cam_pos_noise", type=float, default=0.0)     # metres, ball radius
    p.add_argument("--cam_rot_noise", type=float, default=0.0)     # degrees, per-axis
    p.add_argument("--light_noise", type=float, default=0.0)       # 0 = off, 1 = nominal band
    p.add_argument("--splat_calib", type=str,
                   default=DEFAULT_CALIB)
    p.add_argument("--splat_seg_dir", type=str,
                   default=DEFAULT_SEG_DIR)
    p.add_argument("--splat_scene_ply", type=str,
                   default=DEFAULT_SCENE_PLY)
    p.add_argument("--splat_ground", type=str,
                   default=DEFAULT_GROUND)
    p.add_argument("--jpeg_quality", type=int, default=0,
                   help="0 = store raw uint8 arrays (legacy). 1-100 = JPEG-encode each "
                        "frame and store variable-length bytes; the loader decodes. "
                        "224x224x3x2cam raw is 301,056 B/transition (6.1 TB at 100k "
                        "traj), which neither the quota nor RAM can hold.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--flush_mb", type=int, default=2000,
                   help="flush a worker's buffer once it holds this many MB, in "
                        "ADDITION to --flush_every. Trajectory count is the wrong "
                        "unit for pixels: 500 traj at 224^2x2cam is ~31 GB/worker, "
                        "x10 workers = OOM. State runs never reach 2 GB, so this is "
                        "inert for the legacy arm.")
    p.add_argument("--flush_every", type=int, default=500,
                   help="每个 worker 每攒够这么多条轨迹就 append 写盘并释放内存 "
                        "(峰值 RAM 与总轨迹数无关; 防 150k OOM)")
    return p.parse_args()


def resolve_checkpoints(args):
    ckpts = [c for c in args.checkpoints.split(",") if c]
    if args.checkpoints_glob:
        ckpts += sorted(glob.glob(args.checkpoints_glob))
    # de-dup (keep order); drop the expert itself if it sneaks into the glob
    ckpts = [c for c in dict.fromkeys(ckpts) if os.path.abspath(c) != os.path.abspath(args.expert)]
    return ckpts


def _cams(args):
    """Parse --rl_cameras into a list. () => state-only (legacy behaviour)."""
    # Accept BOTH separators. "+" is the repo convention (MainConfig.rl_camera is
    # "a+b") and is also the only one safe through `sbatch --export=ALL,K=V`,
    # which itself splits on commas -- passing "a,b" there silently delivers just
    # "a" (this cost one smoke-test run).
    raw = str(getattr(args, "rl_cameras", "") or "").replace("+", ",")
    return [c for c in raw.split(",") if c]


# Set once per process from argparse before any env is built. Empty => legacy.
_RES_OVERRIDE: dict = {}
_JPEG_Q = [0]   # list so worker processes can mutate it after argparse


def build_source_env(cfg, hand_init_sigma, domain_shift=None, rl_cameras=(),
                     splat_obs=None, env_kwargs=None):
    """Build a NO-RENDER state env directly from cfg.

    Two reasons this does NOT go through train_rl.load_model (the old path):
      1. load_model builds a full throwaway Workspace per worker — an EGL-rendered
         eval_env AND a QAgent loaded onto cuda — just to read env_params, then
         discards both. ×10 workers = 10 wasted rendered envs + 10 wasted agents.
      2. use_state=1 makes the rendered image PURE WASTE, yet PixelRobosuite still
         creates an offscreen EGL context unless state_only_no_render=True. ~20
         concurrent EGL contexts on one GPU was crashing the Blackwell driver.
    So we build env_params here (mirroring Workspace.eval_env_params) and turn
    rendering OFF entirely.

    `domain_shift`: which domain to roll out in. `None` => SOURCE (mode="none",
    backward-compatible with the source diverse-data runs). Pass a
    DomainShiftConfig (e.g. the expert's own `cfg.train_domain_shift`) to collect
    on the TARGET domain — the env is then the same one the expert was trained on.
    """
    assert cfg.use_state, "generate_simdist_data assumes a state-based source agent"
    if domain_shift is None:
        domain_shift = DomainShiftConfig(mode="none")
    env_params = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        episode_length=cfg.episode_length,
        reward_shaping=False,
        image_size=_RES_OVERRIDE.get("image_size") or cfg.image_size,
        rl_image_size=_RES_OVERRIDE.get("rl_image_size") or cfg.rl_image_size,
        camera_names=list(rl_cameras),   # [] => state-only (legacy)
        rl_cameras=list(rl_cameras),
        splat_obs=splat_obs,
        env_kwargs=env_kwargs,
        use_state=cfg.use_state,
        obs_stack=cfg.obs_stack,
        state_stack=cfg.state_stack,
        prop_stack=cfg.prop_stack,
        domain_shift=domain_shift,
        strict_success=cfg.strict_success,
        hand_init_perturb=None,
        hand_init_perturb_joint_sigma=hand_init_sigma,
        init_distribution=cfg.eval_init_dist,
        # state_only_no_render kills the per-worker EGL context, which is a large
        # speedup -- but it is only legal with no cameras (it asserts use_state and
        # renders nothing). Must be OFF for the pixel arm.
        state_only_no_render=not rl_cameras,
    )
    # SIM-TO-REAL cell (env/sim2real.py), exactly as train_rl.Workspace applies
    # it. Without this the rollouts silently happen on the ORIGINAL arena (0.8 m
    # table, base at (-0.56,0), robosuite reset pose) instead of the cell the
    # expert was trained on -- no error, just the wrong data.
    env_params["sim2real"] = load_cell(cfg)
    env = PixelRobosuite(**env_params)  # type: ignore
    return env, env_params


def load_cell(cfg):
    """The expert cfg's sim2real cell (None if it has none). Read from the yaml
    at collection time, so the cell's DR ranges are the CURRENT file's."""
    path = getattr(cfg, "sim2real_cell", "") or ""
    if not path:
        return None
    from env.sim2real import load_sim2real
    return load_sim2real(path)


def cell_record(cfg):
    """Provenance for env_cfg.json: path + sha256 + the resolved config."""
    path = getattr(cfg, "sim2real_cell", "") or ""
    if not path:
        return None
    import hashlib
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"path": path, "sha256": sha, "resolved": dataclasses.asdict(load_cell(cfg))}


@contextlib.contextmanager
def full_ensemble_value(agent, enabled):
    """Temporarily make `agent.value()` deterministic.

    `MultiFcQ.forward_k` (which `QAgent.value` calls) is RED-Q's RANDOM-subset min:
    it draws `num_k` of the `num_q` heads per call, so V labels are stochastic
    (std ~0.05 here). Widening num_k to num_q for the label computation only —
    the rollout's own action selection is left untouched.
    """
    critic = getattr(agent, "critic", None)
    cfg = getattr(critic, "cfg", None)
    if not enabled or cfg is None or not hasattr(cfg, "num_k"):
        yield
        return
    old_k = cfg.num_k
    cfg.num_k = cfg.num_q
    try:
        yield
    finally:
        cfg.num_k = old_k


def rollout_trajectories(env, expert_agent, subopt_agents, value_agent, rng,
                         args, num_traj, prefix="", pbar=None, counter=None,
                         writer=None, flush_every=500, flush_mb=2000):
    """Run `num_traj` episodes (Algorithm 2). Returns a list of (ep_dict, success).

    Progress reporting (mutually preferred, in order):
      * `pbar`    — an in-process tqdm bar updated per trajectory (serial path);
      * `counter` — a shared mp.Value incremented per trajectory so a parent
                    process can drive a single tqdm bar over all workers;
      * otherwise — a periodic textual print every 50 trajectories (fallback).
    """
    cams = _cams(args)   # [] => state-only
    action_dim = env.action_dim
    p_expert = 1.0 if not subopt_agents else args.p_expert

    def sample_interval(noise_on):
        lo, hi = (args.noise_min, args.noise_max) if noise_on else (args.nonoise_min, args.nonoise_max)
        return int(rng.integers(lo, hi + 1))

    accepted = []
    _pending_mb = 0.0
    n_success = 0
    for j in range(num_traj):
        use_expert_ep = bool(rng.random() < p_expert)
        pol = expert_agent if use_expert_ep else subopt_agents[int(rng.integers(len(subopt_agents)))]
        sigma = rng.uniform(args.sigma_min, args.sigma_max, size=action_dim).astype(np.float32)
        noise_on = bool(rng.integers(0, 2))
        steps_left = sample_interval(noise_on)

        obs, _img = env.reset()
        states, actions, rewards, eflags = [], [], [], []
        frames = {c: [] for c in cams}   # index-aligned with `states`
        terminal = succ = False
        steps = 0
        with torch.no_grad(), utils.eval_mode(pol):
            while not terminal and steps < args.max_len:
                if steps_left <= 0:
                    noise_on = not noise_on
                    steps_left = sample_interval(noise_on)
                states.append(obs["state"].cpu().numpy().astype(np.float32))
                for c in cams:
                    # uint8 CHW straight from the wrapper; appended in the SAME
                    # iteration as `states`, so frame[t] and state[t] are the same
                    # timestep (the alignment the world model depends on).
                    frames[c].append(obs[c].cpu().numpy())
                a = pol.act(obs, eval_mode=True)
                if noise_on:
                    eps = torch.from_numpy(rng.normal(0.0, sigma).astype(np.float32)).to(a.device)
                    a = (a + eps).clamp(-1.0, 1.0)
                actions.append(a.detach().cpu().numpy().astype(np.float32))
                eflags.append(1 if (use_expert_ep and not noise_on) else 0)
                obs, reward, terminal, succ, _img = env.step(a)
                rewards.append(float(reward))
                steps_left -= 1
                steps += 1

        ep = {
            "state": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "expert_flags": np.asarray(eflags, dtype=np.int8),
        }
        for c in cams:
            ep[f"img::{c}"] = np.asarray(frames[c], dtype=np.uint8)
        if args.save_values:
            with torch.no_grad(), full_ensemble_value(value_agent, args.value_full_ensemble):
                sv = torch.from_numpy(ep["state"]).to("cuda")
                ep["values"] = (
                    value_agent.value({"state": sv}, use_ibrl=bool(args.value_use_ibrl))
                    .cpu().numpy().astype(np.float32)
                )
        success = bool(succ) or (ep["rewards"].sum() > 0)
        n_success += int(success)
        accepted.append((ep, success))
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(success=n_success, refresh=False)
        elif counter is not None:
            with counter.get_lock():
                counter.value += 1
        elif (j + 1) % 50 == 0 or j == num_traj - 1:
            print(f"  {prefix}[{j + 1}/{num_traj}] success={n_success} (running)", flush=True)
        # periodic flush -> bound peak RAM (else 150k traj OOMs, see ShardWriter)
        _pending_mb += sum(v.nbytes for v in ep.values()
                           if isinstance(v, np.ndarray)) / 1e6
        if writer is not None and (len(accepted) >= flush_every
                                   or _pending_mb >= flush_mb):
            writer.add_batch(accepted)
            accepted = []
            _pending_mb = 0.0
    if writer is not None:
        writer.add_batch(accepted)   # tail
        return []                    # already persisted incrementally
    return accepted


def _encode_jpeg_stack(arr, quality: int):
    """[T,3,H,W] uint8 -> list of 1-D uint8 byte arrays (one JPEG per frame).

    Lossy, so it is OPT-IN (`--jpeg_quality 0` keeps raw). At 224x224x3x2cam the
    raw store is 301,056 B/transition = 6.1 TB for the 100k-trajectory reference
    recipe, which exceeds both the /scratch quota and any node's RAM."""
    import torchvision.io as tvio
    out = []
    for t in range(arr.shape[0]):
        buf = tvio.encode_jpeg(torch.from_numpy(arr[t]), quality=int(quality))
        out.append(buf.numpy())
    return out


def _write_image_dataset(d, name, arr, quality: int):
    """Store frames either raw (legacy) or as per-frame JPEG byte strings."""
    if not quality:
        d.create_dataset(name, data=arr, dtype="uint8")
        return
    enc = _encode_jpeg_stack(arr, quality)
    vl = h5py.vlen_dtype(np.dtype("uint8"))
    ds = d.create_dataset(name, shape=(len(enc),), dtype=vl)
    for i, b in enumerate(enc):
        ds[i] = b
    # the loader needs the decoded shape + a codec flag; keep them ON the dataset
    ds.attrs["codec"] = "jpeg"
    ds.attrs["frame_shape"] = np.asarray(arr.shape[1:], dtype=np.int32)  # (3,H,W)
    ds.attrs["jpeg_quality"] = int(quality)


def write_episodes(accepted, h5_path):
    """Write a list of (ep, success) to an hdf5 (data/demo_N/... layout)."""
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    n_success = 0
    with h5py.File(h5_path, "w") as hf:
        g = hf.create_group("data")
        for i, (ep, success) in enumerate(accepted):
            d = g.create_group(f"demo_{i}")
            d.create_dataset("obs/state", data=ep["state"])
            for k, v in ep.items():
                if k.startswith("img::"):
                    # `obs/<camera>_image`, matching the layout
                    # rl/replay.py::add_demos_to_replay and the collector use.
                    _write_image_dataset(d, f"obs/{k[5:]}_image", v, _JPEG_Q[0])
            d.create_dataset("actions", data=ep["actions"])
            d.create_dataset("rewards", data=ep["rewards"])
            d.create_dataset("expert_flags", data=ep["expert_flags"])
            if "values" in ep:
                d.create_dataset("values", data=ep["values"])
            d.attrs["num_samples"] = int(ep["actions"].shape[0])
            d.attrs["success"] = bool(success)
            n_success += int(success)
        g.attrs["num_demos"] = len(accepted)
        g.attrs["num_success"] = n_success
    return n_success


class ShardWriter:
    """Incremental hdf5 writer: append a batch of episodes and free them, so a
    worker's peak RAM is bounded by `flush_every` trajectories instead of ALL of
    them. Same `data/demo_N` layout as `write_episodes`, so `merge_shards` is
    unchanged. (The old accumulate-everything-then-write path OOM'd at ~7-12k
    trajectories with 20 workers, especially with value_full_ensemble.)"""

    def __init__(self, path, jpeg_quality: int = 0):
        self.path = path
        self.jpeg_quality = int(jpeg_quality)
        self.n = 0
        self.n_success = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with h5py.File(path, "w") as hf:
            hf.create_group("data")

    def add_batch(self, batch):
        if not batch:
            return
        with h5py.File(self.path, "a") as hf:
            g = hf["data"]
            for ep, success in batch:
                d = g.create_group(f"demo_{self.n}")
                d.create_dataset("obs/state", data=ep["state"])
                for k, v in ep.items():
                    if k.startswith("img::"):
                        _write_image_dataset(d, f"obs/{k[5:]}_image", v, self.jpeg_quality)
                d.create_dataset("actions", data=ep["actions"])
                d.create_dataset("rewards", data=ep["rewards"])
                d.create_dataset("expert_flags", data=ep["expert_flags"])
                if "values" in ep:
                    d.create_dataset("values", data=ep["values"])
                d.attrs["num_samples"] = int(ep["actions"].shape[0])
                d.attrs["success"] = bool(success)
                self.n += 1
                self.n_success += int(success)
            g.attrs["num_demos"] = self.n
            g.attrs["num_success"] = self.n_success


def merge_shards(shard_paths, out_path):
    """Concatenate shard hdf5s into one, re-indexing demo_N. Returns (n, n_success)."""
    total = n_success = 0
    with h5py.File(out_path, "w") as out:
        g = out.create_group("data")
        for sp in shard_paths:
            if not os.path.exists(sp):
                continue
            with h5py.File(sp, "r") as sf:
                sg = sf["data"]
                for k in sorted(sg.keys(), key=lambda x: int(x.split("_")[1])):
                    src = sg[k]
                    # h5py's native copy preserves DATASET attrs (the jpeg `codec` /
                    # `frame_shape` flags live there) and streams instead of pulling
                    # every frame through Python. The old manual walk dropped both.
                    sg.copy(k, g, name=f"demo_{total}")
                    n_success += int(bool(src.attrs.get("success", False)))
                    total += 1
        g.attrs["num_demos"] = total
        g.attrs["num_success"] = n_success
    return total, n_success


def write_env_cfg(out_dir, env_params, args):
    env_cfg = {
        "env_name": env_params["env_name"],
        "domain_shift": dataclasses.asdict(env_params["domain_shift"]),
        "hand_init_perturb_joint_sigma": args.hand_init_sigma,
        # The reset distribution is inherited from the expert ckpt's eval_init_dist
        # (build_source_env), so it is NOT visible from the CLI -- record it or a
        # dataset collected under a narrowed window (e.g. Square rotw1's 120-180
        # deg nut yaw) is indistinguishable from a full-circle one after the fact.
        "init_distribution": (
            dataclasses.asdict(env_params["init_distribution"])
            if env_params.get("init_distribution") is not None else None),
        "use_state": env_params.get("use_state"),
        "sim2real_cell": cell_record(pyrallis.load(  # type: ignore
            train_rl.MainConfig,
            open(os.path.join(os.path.dirname(args.expert), "cfg.yaml"), "r"))),
        "expert_weight": args.expert,
        "expert_hybrid": args.expert_hybrid,
        "snapshot_hybrid": args.snapshot_hybrid,
        "value_use_ibrl": args.value_use_ibrl,
        "value_full_ensemble": args.value_full_ensemble,
        "p_expert": args.p_expert,
        "sigma_range": [args.sigma_min, args.sigma_max],
        "noise_interval": [args.noise_min, args.noise_max],
        "nonoise_interval": [args.nonoise_min, args.nonoise_max],
        "num_traj": args.num_traj,
    }
    with open(os.path.join(out_dir, "env_cfg.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)


def _make_hybrid(agent, bc_policy):
    """Turn a bare RL agent into its IBRL hybrid: attach the (shared, frozen) BC
    prior and switch act_method back to "ibrl" (load_frozen_agent forced "rl").
    The bare RL actor is then complemented by max-Q over {actor, prior} — much
    stronger when the actor leaned on the prior during training. V_e is unaffected
    (value() = min-ensemble Q(s, RL-actor(s)), independent of act_method)."""
    agent.add_bc_policy(bc_policy)
    agent.cfg.act_method = "ibrl"


def resolve_domain_shift(args, cfg):
    """Which domain to roll out in. `--domain`:
      * "expert" (default): use the expert's OWN training env
        (`cfg.train_domain_shift`) — source expert => source (mode=none, the
        original v2 behavior), target expert => its §12 target shift. Zero config.
      * "source": force the nominal env (mode=none), whatever the expert.
    """
    if args.domain == "source":
        return DomainShiftConfig(mode="none")
    return cfg.train_domain_shift  # "expert"


def _load_all(args, ckpts):
    """Build the env + load expert and sub-optimal agents. Returns
    (env, env_params, expert_agent, subopt_agents)."""
    cfg = pyrallis.load(  # type: ignore
        train_rl.MainConfig, open(os.path.join(os.path.dirname(args.expert), "cfg.yaml"), "r"))
    # spawn-safe: `args` is pickled into each worker, so每个进程各自设置一次
    if getattr(args, "image_size", 0):
        _RES_OVERRIDE["image_size"] = int(args.image_size)
    if getattr(args, "rl_image_size", 0):
        _RES_OVERRIDE["rl_image_size"] = int(args.rl_image_size)
    _JPEG_Q[0] = int(getattr(args, "jpeg_quality", 0) or 0)
    if _RES_OVERRIDE:
        print(f"[res] override {_RES_OVERRIDE} (cfg had image_size={cfg.image_size}, "
              f"rl_image_size={cfg.rl_image_size})", flush=True)
    splat_cfg = None
    if int(getattr(args, "splat", 0) or 0):
        # one compositor PER WORKER (each has its own env + CUDA context)
        splat_cfg = dict(calib_path=args.splat_calib, seg_dir=args.splat_seg_dir,
                         scene_ply=args.splat_scene_ply, ground_json=args.splat_ground)
        print(f"[splat] photoreal obs ON: {splat_cfg}", flush=True)
    ek = None
    _dr = {k: float(getattr(args, k, 0) or 0) for k in
           ("board_pos_noise", "board_yaw_noise", "cam_pos_noise",
            "cam_rot_noise", "light_noise")}
    if any(_dr.values()):
        ek = dict(_dr)
        print(f"[env-DR] {ek}", flush=True)
    env, env_params = build_source_env(cfg, args.hand_init_sigma,
                                       resolve_domain_shift(args, cfg),
                                       rl_cameras=_cams(args),
                                       splat_obs=splat_cfg, env_kwargs=ek)
    expert_agent = train_rl.load_frozen_agent(args.expert, env, cfg, "cuda")
    subopt_agents = [train_rl.load_frozen_agent(c, env, cfg, "cuda") for c in ckpts]
    if args.expert_hybrid:
        # Roll out BOTH the expert AND the sub-optimal snapshots as their IBRL
        # hybrid. One shared frozen prior is attached to every agent (held in
        # each agent's bc_policies list — not a submodule, so sharing is safe).
        if not args.expert_bc_policy and cfg.sgft_source_agent:
            # Prior is a FROZEN RL QAgent (trained with --sgft_source_agent), NOT
            # a BC head — mirror train_rl.load_model. Both the expert and the prior
            # are then RL MLP actors, so the hybrid rollout is 2 cheap MLP forwards
            # per step (no flow-BC ODE) => much faster data collection.
            source_agent = train_rl.load_frozen_agent(
                cfg.sgft_source_agent, env, cfg, "cuda")
            bc_policy = train_rl.FrozenAgentBCPolicy(source_agent)
            print(f"[hybrid] RL-actor prior (sgft_source_agent) = {cfg.sgft_source_agent}")
        else:
            bc_path = args.expert_bc_policy or cfg.bc_policy
            assert bc_path, (
                "--expert_hybrid 1 needs a prior, but the expert cfg has no "
                "bc_policy / sgft_source_agent and --expert_bc_policy was not given.")
            bc_policy = train_bc._load_model(bc_path, env, "cuda")
            for p in bc_policy.parameters():
                p.requires_grad_(False)
            bc_policy.train(False)
            print(f"[hybrid] BC-head prior = {bc_path}")
        _make_hybrid(expert_agent, bc_policy)
        if args.snapshot_hybrid:
            for ag in subopt_agents:
                _make_hybrid(ag, bc_policy)
            print(f"[hybrid] IBRL rollout (act_method=ibrl) for expert + "
                  f"{len(subopt_agents)} snapshot(s)")
        else:
            print(f"[hybrid] IBRL rollout for the EXPERT only; "
                  f"{len(subopt_agents)} snapshot(s) stay bare RL actors")
    return env, env_params, expert_agent, subopt_agents


def _worker(rank, args, ckpts, n_traj, shard_path, counter=None):
    """Self-contained worker: load policies + SOURCE env, roll out n_traj, write shard."""
    os.environ["MUJOCO_GL"] = "egl"
    import common_utils
    common_utils.set_all_seeds(args.seed * 100000 + rank)
    rng = np.random.default_rng(args.seed * 100000 + rank)
    env, _env_params, expert_agent, subopt_agents = _load_all(args, ckpts)
    writer = ShardWriter(shard_path, jpeg_quality=args.jpeg_quality)  # bounded RAM
    rollout_trajectories(env, expert_agent, subopt_agents, expert_agent,
                         rng, args, n_traj, prefix=f"[w{rank}] ", counter=counter,
                         writer=writer, flush_every=args.flush_every,
                         flush_mb=args.flush_mb)
    print(f"[w{rank}] done: {writer.n} traj, {writer.n_success} success -> {shard_path}", flush=True)


def main():
    args = parse_args()
    assert not (args.value_use_ibrl and not args.expert_hybrid), (
        "--value_use_ibrl 1 needs --expert_hybrid 1 (the hybrid value is max-Q over "
        "{RL actor, prior}, so the prior must be attached).")
    ckpts = resolve_checkpoints(args)
    if not ckpts:
        print("[warn] no sub-optimal checkpoints; ALL episodes use the expert "
              "(p_expert forced to 1.0). Pass --checkpoints/--checkpoints_glob for diversity.")
    print(f"expert={args.expert}")
    print(f"{len(ckpts)} sub-optimal checkpoints, num_traj={args.num_traj}, "
          f"num_proc={args.num_proc}, p_expert={args.p_expert}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "dataset.hdf5")

    if args.num_proc <= 1:
        # serial path
        import common_utils
        common_utils.set_all_seeds(args.seed)
        rng = np.random.default_rng(args.seed)
        env, env_params, expert_agent, subopt_agents = _load_all(args, ckpts)
        with tqdm(total=args.num_traj, desc="generate", unit="traj", dynamic_ncols=True) as bar:
            acc = rollout_trajectories(env, expert_agent, subopt_agents, expert_agent,
                                       rng, args, args.num_traj, pbar=bar)
        n_success = write_episodes(acc, out_path)
        write_env_cfg(args.out_dir, env_params, args)
        total = len(acc)
    else:
        # parallel: spawn N self-contained workers, then merge shards
        import torch.multiprocessing as mp
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        per = [args.num_traj // args.num_proc + (1 if i < args.num_traj % args.num_proc else 0)
               for i in range(args.num_proc)]
        shard_paths = [os.path.join(args.out_dir, f"_shard_{i}.hdf5") for i in range(args.num_proc)]
        counter = mp.Value("i", 0)  # trajectories completed across all workers
        procs = [mp.Process(target=_worker, args=(i, args, ckpts, per[i], shard_paths[i], counter))
                 for i in range(args.num_proc) if per[i] > 0]
        for p in procs:
            p.start()
        # single progress bar for the whole fleet, driven by the shared counter
        with tqdm(total=args.num_traj, desc="generate", unit="traj", dynamic_ncols=True) as bar:
            while any(p.is_alive() for p in procs):
                bar.n = counter.value
                bar.refresh()
                time.sleep(0.5)
            bar.n = counter.value
            bar.refresh()
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode != 0]
        if failed:
            print(f"[error] {len(failed)} worker(s) exited non-zero: {failed}")
        total, n_success = merge_shards(shard_paths, out_path)
        # env_cfg without building an env (read task_name from the expert cfg)
        ecfg = pyrallis.load(  # type: ignore
            train_rl.MainConfig, open(os.path.join(os.path.dirname(args.expert), "cfg.yaml"), "r"))
        env_params = {"env_name": ecfg.task_name,
                      "domain_shift": resolve_domain_shift(args, ecfg),
                      "init_distribution": ecfg.eval_init_dist,
                      "use_state": ecfg.use_state}
        write_env_cfg(args.out_dir, env_params, args)
        for sp in shard_paths:
            if os.path.exists(sp):
                os.remove(sp)

    print(f"DONE: {total} trajectories, {n_success} success "
          f"({100 * n_success / max(total, 1):.1f}%) -> {out_path}")


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()
