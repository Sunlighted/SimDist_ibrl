"""Targeted SOURCE collection: roll the ADAPTED SimDist policy in SOURCE physics.

WHY. On TPA the planner's ranking regret correlates with how far the executed
(s, a) sits from the stage-1 source dataset: Spearman(regret, d_src(s,a)) = +0.381
(p<0.001, n=120) and the worst (s,a)-distance quartile has 2.6x the regret of the
best (0.88 -> 2.25). The aggregate distance hides this -- on-policy states sit at
1.01x a held-out source set's nearest-neighbour distance, which is why an earlier
pass concluded "coverage is fine". The source set was collected with the source
EXPERT + 16 of its training snapshots; the deployment policy is a different animal
(stage-2 BC'd SimDist policy head, executed through MPPI), so part of its (s,a)
support was never distilled on. This script fills exactly that region -- same
policy, SOURCE physics (so the recorded s' is valid stage-1 data).

Output is the `rl/replay.py::_save_replay` layout that
`rl.sequence_replay.load_source_replay_to_sequence_buffer` reads:
    data/demo_N/{obs/state[T,obs], actions[T,act], rewards[T], expert_flags[T]}
so it can be merged with tpa_d0_source_diverse_v1 and re-distilled with no code
changes. expert_flags mirrors generate_simdist_data.py's 1_e(a_t): 1 on steps
executed WITHOUT injected noise (stage-1's policy BC uses bc_expert_only=1 and
clones only those steps).

VRAM. One worker holds ONE SimDist agent (~29 MB of weights) plus the MPPI
candidate batch (num_samples x horizon x action_dim), i.e. far less than
generate_simdist_data.py's 2706 MiB/worker -- that figure comes from keeping all
17 QAgent snapshots resident (generate_simdist_data.py:430). Measure before
scaling: --num_proc 1 --num_demo 2 and read `[vram]` in the log, then size the job
to HALF of what nvidia-smi reports free on the chosen GPU (CLAUDE.md's GPU-density
rule: a full card can wedge the driver's rwsem, and an OOM takes the whole machine
down, not just the job).

Usage:
    python scripts/collect_source_targeted.py \
        --agent exps/mimicgen/_evalshim/tpa_s3_anch3/latest.pt \
        --num_demo 2 --num_proc 1 --out_dir /tmp/probe        # measure first
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_simdist
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from common_utils import ibrl_utils as utils
from rl.simdist_planner import MppiPlanner, apply_original_recipe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True,
                   help="SimDist ckpt; its DIRECTORY must hold the stage-1 cfg.yaml "
                        "(use an _evalshim dir -- an adapt run's own cfg.yaml is an "
                        "AdaptConfig dump and train_simdist.load_model cannot parse it)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_demo", type=int, default=100, help="trajectories PER WORKER")
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_len", type=int, default=1000,
                   help="TPA runs at episode_length 1000; 400 would drop ~47% of "
                        "transitions (see the v1 dataset's PROVENANCE)")
    p.add_argument("--use_planner", type=int, default=1,
                   help="1 = execute MPPI (the deployment distribution, ~35 it/s). "
                        "0 = the bare policy head (~10x faster, but that is the "
                        "PROPOSAL distribution, not what regret was measured on)")
    p.add_argument("--planner_set", type=str, default="init_std=0.15",
                   help="planner overrides applied AFTER apply_original_recipe; "
                        "must match how the agent is actually deployed (rule 9: "
                        "the recipe's init_std=1.0 is not portable -- on TPA 0.15 "
                        "scored 0.42 vs 0.35/0.40)")
    # contiguous action-noise intervals, same scheme as generate_simdist_data.py:
    # widening the (s,a) support is the whole point -- a purely greedy rollout would
    # trace one thin tube and re-add the coverage hole we are trying to fill.
    p.add_argument("--noise_sigma_min", type=float, default=0.05)
    p.add_argument("--noise_sigma_max", type=float, default=0.2)
    p.add_argument("--noise_min", type=int, default=1)
    p.add_argument("--noise_max", type=int, default=5)
    p.add_argument("--nonoise_min", type=int, default=5)
    p.add_argument("--nonoise_max", type=int, default=10)
    p.add_argument("--flush_every", type=int, default=50)
    # ---- seed-from-target (DEFAULT, and the whole point) ----
    # Rolling this policy in source physics from a normal reset does NOT reach the
    # region regret was measured on: probed 4/4 successes at length 135-153 in
    # SOURCE, while on TARGET the same agent fails ~60% and the high-regret states
    # are the struggling / jammed contact ones. So seed each source segment from a
    # sim_state the agent actually visited on TARGET: reset the SOURCE env there and
    # roll on. The recorded s' is then source physics (valid stage-1 data) at a
    # TARGET-visited (s, a) -- exactly the (s,a) quartile with 2.6x the regret.
    p.add_argument("--seed_from_target", type=int, default=1)
    p.add_argument("--seeds_per_episode", type=int, default=24,
                   help="sim_states sampled per TARGET episode to seed segments")
    p.add_argument("--segment_len", type=int, default=24,
                   help="source-physics steps per seed. Must exceed the world "
                        "model's horizon (H=5) or load_source_replay drops it")
    return p.parse_args()


def build(args, device):
    agent, _, ep = train_simdist.load_model(args.agent, device)
    if args.use_planner:
        apply_original_recipe(agent.cfg.planner)
        for kv in (x for x in args.planner_set.split(",") if x.strip()):
            k, v = kv.split("=")
            k = k.strip()
            old = getattr(agent.cfg.planner, k)
            setattr(agent.cfg.planner, k, type(old)(v))
            print(f"[planner] {k}: {old} -> {getattr(agent.cfg.planner, k)}", flush=True)
        agent.cfg.use_planner = 1
        agent.planner = MppiPlanner(agent.model, agent.cfg.planner,
                                    agent.action_dim, device=device)
    else:
        agent.cfg.use_planner = 0
    # SOURCE physics: this is stage-1 distillation data, so s' must be the source's.
    ep["domain_shift"] = DomainShiftConfig(mode="none")
    ep["record_sim_state"] = False
    ep["episode_length"] = args.max_len
    # Match the v1 dataset: collection is deliberately WIDER than deployment
    # (v1 used 0.1 while TPA trains/evals at null).
    ep["hand_init_perturb_joint_sigma"] = 0.1
    src_ep = dict(ep)
    src_ep["record_sim_state"] = bool(args.seed_from_target)
    src_env = PixelRobosuite(**src_ep)
    tgt_env = None
    if args.seed_from_target:
        # TARGET = the stage-1 cfg's eval_domain_shift, the single source of truth
        # for this task's target (rule 19b). Read it from the SAME directory the
        # agent came from so the two can never drift apart.
        import yaml
        raw = yaml.safe_load(open(os.path.join(os.path.dirname(args.agent), "cfg.yaml")))
        sk = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in (raw.get("eval_domain_shift") or {}).items()}
        print(f"[env] target shift = {sk}", flush=True)
        tgt_ep = dict(ep)
        tgt_ep["domain_shift"] = DomainShiftConfig(**sk)
        tgt_ep["record_sim_state"] = True
        tgt_env = PixelRobosuite(**tgt_ep)
    return agent, src_env, tgt_env


def restore(env, sim_state):
    """Cold-start reset to a recorded sim_state. Copied field-for-field from
    scripts/eval_oracle_dyn_mppi.py:49 -- omitting done/timestep/qacc_warmstart
    raises "executing action in terminated episode" on the second use (bitten)."""
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    # ...plus the WRAPPER's own episode state, which restore() in the eval script
    # never had to touch (it only ever rolled H=3 steps). step() reads terminal /
    # time_step / episode_reward, so a stale terminal=True would end the segment
    # after one step.
    env.time_step = 0
    env.terminal = False
    env.episode_reward = 0
    env.episode_extra_reward = 0
    env._stage1_fired = False
    env.past_obses.clear()
    if getattr(env, "_success_validator", None) is not None:
        env._success_validator.reset(raw, env._bindings)
    # env.reset() derives the policy obs via _extract_images; after set_state we
    # must pull a FRESH observation the same way (force_update, the sim moved).
    rl_obs, _ = env._extract_images(raw._get_observations(force_update=True))
    return rl_obs


@torch.no_grad()
def target_seeds(agent, env, args):
    """One TARGET episode -> sim_states the agent actually visits there."""
    obs, _ = env.reset()
    out = []
    with utils.eval_mode(agent):
        for _ in range(args.max_len):
            out.append(env.env.sim.get_state().flatten().copy())
            a = agent.act(obs, eval_mode=True)
            if a.dim() > 1:
                a = a.view(-1)
            obs, _, term, _, _ = env.step(a.cpu())
            if term:
                break
    if len(out) > args.seeds_per_episode:
        idx = np.linspace(0, len(out) - 1, args.seeds_per_episode).astype(int)
        out = [out[i] for i in idx]
    return out


def rollout_from(agent, env, rng, args, sim_state, n_steps):
    """Roll `n_steps` of SOURCE physics starting from a TARGET-visited sim_state."""
    obs = restore(env, sim_state)
    S, A, R, E = [], [], [], []
    noisy, left, sigma = False, int(rng.integers(args.nonoise_min, args.nonoise_max + 1)), 0.0
    succ = False
    with utils.eval_mode(agent):
        for _ in range(n_steps):
            a = agent.act(obs, eval_mode=True)
            if a.dim() > 1:
                a = a.view(-1)
            S.append(obs["state"].cpu().numpy().copy())
            if left <= 0:
                noisy = not noisy
                if noisy:
                    sigma = float(rng.uniform(args.noise_sigma_min, args.noise_sigma_max))
                    left = int(rng.integers(args.noise_min, args.noise_max + 1))
                else:
                    sigma, left = 0.0, int(rng.integers(args.nonoise_min, args.nonoise_max + 1))
            left -= 1
            a_exec = a
            if noisy and sigma > 0:
                a_exec = (a + torch.as_tensor(
                    rng.normal(0.0, sigma, size=a.shape), dtype=a.dtype,
                    device=a.device)).clamp(-1.0, 1.0)
            A.append(a_exec.cpu().numpy().copy())
            E.append(0.0 if noisy else 1.0)
            obs, r, term, s_, _ = env.step(a_exec.cpu())
            R.append(float(r))
            succ = succ or bool(s_)
            if term:
                break
    return (np.asarray(S, np.float32), np.asarray(A, np.float32),
            np.asarray(R, np.float32), np.asarray(E, np.float32), succ)


def rollout(agent, env, rng, args):
    """One episode. Returns (states[T,obs], actions[T,act], rewards[T], eflag[T], success)."""
    obs, _ = env.reset()
    S, A, R, E = [], [], [], []
    # contiguous noise schedule
    noisy, left = False, int(rng.integers(args.nonoise_min, args.nonoise_max + 1))
    sigma = 0.0
    with torch.no_grad(), utils.eval_mode(agent):
        for _ in range(args.max_len):
            a = agent.act(obs, eval_mode=True)
            if a.dim() > 1:
                a = a.view(-1)
            S.append(obs["state"].cpu().numpy().copy())
            if left <= 0:   # flip phase
                noisy = not noisy
                if noisy:
                    sigma = float(rng.uniform(args.noise_sigma_min, args.noise_sigma_max))
                    left = int(rng.integers(args.noise_min, args.noise_max + 1))
                else:
                    sigma = 0.0
                    left = int(rng.integers(args.nonoise_min, args.nonoise_max + 1))
            left -= 1
            a_exec = a
            if noisy and sigma > 0:
                a_exec = (a + torch.as_tensor(
                    rng.normal(0.0, sigma, size=a.shape), dtype=a.dtype,
                    device=a.device)).clamp(-1.0, 1.0)
            A.append(a_exec.cpu().numpy().copy())
            E.append(0.0 if noisy else 1.0)     # 1_e(a_t): un-noised steps only
            obs, r, term, succ, _ = env.step(a_exec.cpu())
            R.append(float(r))
            if term:
                break
    return (np.asarray(S, np.float32), np.asarray(A, np.float32),
            np.asarray(R, np.float32), np.asarray(E, np.float32), bool(succ))


def worker(rank, args):
    dev = "cuda"
    torch.manual_seed(args.seed * 100000 + rank)
    rng = np.random.default_rng(args.seed * 100000 + rank)
    agent, env, tgt_env = build(args, dev)
    path = os.path.join(args.out_dir, f"_shard{rank}.hdf5")
    f = h5py.File(path, "w")
    g = f.create_group("data")
    n_succ = 0
    seeds: list = []
    for i in range(args.num_demo):
        if args.seed_from_target:
            if not seeds:
                seeds = target_seeds(agent, tgt_env, args)
                if rank == 0:
                    print(f"[w0] new TARGET episode -> {len(seeds)} seeds", flush=True)
            S, A, R, E, succ = rollout_from(agent, env, rng, args, seeds.pop(0),
                                            args.segment_len)
        else:
            S, A, R, E, succ = rollout(agent, env, rng, args)
        if len(A) < 6:          # needs >= horizon+1 for the H=5 consistency loss
            continue
        d = g.create_group(f"demo_{i}")
        d.create_dataset("obs/state", data=S, compression="gzip", compression_opts=1)
        d.create_dataset("actions", data=A, compression="gzip", compression_opts=1)
        d.create_dataset("rewards", data=R)
        d.create_dataset("expert_flags", data=E)
        d.attrs["success"] = succ
        n_succ += int(succ)
        if (i + 1) % args.flush_every == 0:
            f.flush()
        if rank == 0 and (i + 1) % 2 == 0:
            print(f"[w0] {i+1}/{args.num_demo}  len={len(A)}  succ={n_succ}/{i+1}  "
                  f"[vram] alloc={torch.cuda.memory_allocated()/2**20:.0f} MiB "
                  f"peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB "
                  f"reserved={torch.cuda.memory_reserved()/2**20:.0f} MiB", flush=True)
    n_ep = len(g.keys())
    g.attrs["num_demos"] = n_ep
    g.attrs["num_success"] = n_succ
    f.close()
    print(f"[w{rank}] done: {n_ep} episodes -> {path} succ={n_succ}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "collect_args.json"), "w"),
              indent=1)
    if args.num_proc == 1:
        worker(0, args)
    else:
        mp.set_start_method("spawn", force=True)
        ps = [mp.Process(target=worker, args=(r, args)) for r in range(args.num_proc)]
        [p.start() for p in ps]
        [p.join() for p in ps]
    print("ALL WORKERS DONE")


if __name__ == "__main__":
    main()
