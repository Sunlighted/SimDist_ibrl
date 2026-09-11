"""Offscreen-render throughput benchmark (pixel-mode go/no-go).

Every pixel-mode cost estimate depends on how fast this cluster can render two
96x96 camera views per env step. MuJoCo silently falls back to OSMesa (software
GL) when the EGL vendor JSON / MUJOCO_EGL_DEVICE_ID is wrong, which is ~10x
slower and would triple the eval budget -- so measure before queueing GPU-days.

Reports steps/s for: state-only (no render, the current baseline), 1 camera, and
2 cameras; then the same under N spawned processes to check that mp-eval scales.

Run on a GPU node:
    salloc --account=aip-gigor --partition=gpubase_interac --gres=gpu:l40s:1 \
           --cpus-per-task=16 --mem=64G --time=1:00:00
    source set_env.sh && python scripts/bench_render.py --n_steps 300 --n_proc 10
"""

import argparse
import os
import time

# must precede any mujoco import (mirrors evaluate/eval.py:102)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def _build(cameras: list[str], image_size: int, rl_image_size: int, no_render: bool):
    from env.robosuite_wrapper import PixelRobosuite
    from env.strict_success import StrictSuccessConfig

    return PixelRobosuite(
        env_name="Threading_D0",
        robots=["Panda"],
        episode_length=400,
        image_size=image_size,
        rl_image_size=rl_image_size,
        camera_names=list(cameras),
        rl_cameras=list(cameras),
        use_state=no_render or not cameras,
        obs_stack=1,
        state_stack=1,
        prop_stack=1,
        end_on_success=True,
        strict_success=StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04),
        state_only_no_render=no_render,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _timed_rollout(env, n_steps: int, action_dim: int, seed: int = 0):
    """Random-action stepping. Returns (steps_per_sec, obs_key_report)."""
    rng = np.random.RandomState(seed)
    env.reset()
    # warm up: first steps pay lazy GL context + cuDNN autotune costs
    for _ in range(10):
        _, _, terminal, _, _ = env.step(torch.from_numpy(
            rng.uniform(-0.2, 0.2, action_dim).astype(np.float32)))
        if terminal:
            env.reset()

    t0 = time.perf_counter()
    done_count = 0
    for _ in range(n_steps):
        obs, _, terminal, _, _ = env.step(torch.from_numpy(
            rng.uniform(-0.2, 0.2, action_dim).astype(np.float32)))
        if terminal:
            env.reset()
            done_count += 1
    dt = time.perf_counter() - t0

    report = {k: (tuple(v.shape), str(v.dtype)) for k, v in obs.items()}
    return n_steps / dt, report, done_count


def _worker(rank: int, cameras, image_size, rl_image_size, n_steps, q):
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        env = _build(cameras, image_size, rl_image_size, no_render=False)
        sps, _, _ = _timed_rollout(env, n_steps, env.action_dim, seed=rank)
        q.put((rank, sps, None))
    except Exception as e:  # a worker crash must not hang the parent
        q.put((rank, 0.0, repr(e)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps", type=int, default=300)
    p.add_argument("--n_proc", type=int, default=10, help="0 = skip the mp scaling test")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--rl_image_size", type=int, default=96)
    args = p.parse_args()

    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    single = {}
    for label, cameras, no_render in [
        ("state-only (no render)", [], True),
        ("1 cam  (eye_in_hand)", ["robot0_eye_in_hand"], False),
        ("2 cams (agentview+wrist)", ["agentview", "robot0_eye_in_hand"], False),
    ]:
        env = _build(cameras, args.image_size, args.rl_image_size, no_render)
        sps, report, dones = _timed_rollout(env, args.n_steps, env.action_dim)
        single[label] = sps
        print(f"\n--- {label} ---")
        print(f"  {sps:7.1f} steps/s   ({1000.0 / sps:6.2f} ms/step, {dones} resets)")
        for k, (shape, dtype) in sorted(report.items()):
            print(f"    obs[{k!r}]: {shape} {dtype}")
        del env

    base = single["state-only (no render)"]
    two = single["2 cams (agentview+wrist)"]
    print(f"\nrender overhead (2 cams): {1000.0 / two - 1000.0 / base:.2f} ms/step "
          f"({base / two:.2f}x slowdown vs state-only)")

    if args.n_proc > 0:
        import torch.multiprocessing as mp
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn")  # matches evaluate/multi_process_eval.py:6-9
        q = mp.Queue()
        cams = ["agentview", "robot0_eye_in_hand"]
        procs = [mp.Process(target=_worker,
                            args=(r, cams, args.image_size, args.rl_image_size,
                                  args.n_steps, q))
                 for r in range(args.n_proc)]
        t0 = time.perf_counter()
        for pr in procs:
            pr.start()
        results = [q.get() for _ in range(args.n_proc)]
        for pr in procs:
            pr.join()
        wall = time.perf_counter() - t0

        errs = [(r, e) for r, _, e in results if e]
        rates = [s for _, s, e in results if not e]
        print(f"\n--- {args.n_proc} spawned procs, 2 cams ---")
        if errs:
            print(f"  {len(errs)} FAILED, e.g. rank {errs[0][0]}: {errs[0][1]}")
        if rates:
            agg = sum(rates)
            print(f"  per-proc {np.mean(rates):7.1f} steps/s (min {min(rates):.1f}, "
                  f"max {max(rates):.1f})")
            print(f"  aggregate {agg:7.1f} steps/s   "
                  f"scaling {agg / two:.2f}x vs 1 proc (ideal {args.n_proc}x)")
            print(f"  wall {wall:.1f}s for {args.n_proc * args.n_steps} steps")

    # The right EGL diagnostic is the render OVERHEAD, not total throughput.
    # Total steps/s is dominated by physics (robosuite runs ~25 sim substeps per
    # control step), so a "<100 steps/s => OSMesa" rule -- which is what an
    # earlier version of this script printed -- fires on a perfectly healthy
    # GPU setup. Measured on an L40S: 11.8 ms/step with NO rendering, and only
    # +2.2 ms for two 224x224 renders + GPU resize. Software GL would add tens
    # of ms, so overhead is the quantity that separates the two.
    overhead_ms = 1000.0 / two - 1000.0 / base
    if overhead_ms < 6.0:
        verdict = "EGL hardware rendering OK (render is a minor cost; physics dominates)"
    elif overhead_ms < 20.0:
        verdict = "render cost noticeable -- budget for it, but not a software fallback"
    else:
        verdict = ("SUSPECT OSMesa/software GL -- check MUJOCO_GL=egl and "
                   "MUJOCO_EGL_DEVICE_ID before spending GPU-days")
    print(f"\nverdict: render overhead {overhead_ms:.2f} ms/step -> {verdict}")
    print("mp scaling: near-linear aggregate throughput means mp_eval workers are "
          "not contending on the GPU/EGL; well short of that, reduce num_eval_proc.")


if __name__ == "__main__":
    main()
