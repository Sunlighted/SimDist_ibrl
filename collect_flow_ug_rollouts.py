"""Collect SUCCESSFUL flow+UG rollouts on the TARGET domain, for self-distillation
back into the flow policy (CLAUDE.md §24 follow-up / project_flow_ug_simdist TODO
"UG 高质量 rollout self-distill 回 flow").

Rolls out the TEACHER = flow.pt (target-FT'd) + Universal-Guidance (adapt-WM,
coeff 0.3, Ta=1 — the config that scored 79% target strict-success) in the §12
dynamics-only TARGET env, and keeps only the trajectories that ACTUALLY succeed
(strict success). Success filtering by the REAL env — NOT the world-model belief —
so the collected data does NOT clone UG's world-model exploitation, only what
truly threaded the needle in the target.

Each success trajectory ends at the first success step (end_on_success=True), so
it is already truncated like the 20 hand demos. It is sliced into (state[t],
action_chunk[t:t+Tp]) pairs (padded at the end with the last action) — the exact
format extract_target_demo_chunks() produces — and saved as a pre-chunked tensor
file that finetune_flow_selfdistill.py consumes directly.

Vectorized: `num_envs` target envs stepped in lockstep so the (GPU-bound) UG
sampling is batched; MuJoCo steps run sequentially on CPU. Envs auto-reset with a
fresh seed on termination and collection runs until `num_success` successes (or
`max_episodes` episodes) are gathered.

Run:
  python collect_flow_ug_rollouts.py \
      --flow exps/mimicgen/threading_flow_ug_seed1/flow.pt \
      --wm_ckpt exps/mimicgen/threading_simdist_adapt_seed1/model0.pt \
      --steering_coeff 0.3 --num_success 300 --num_envs 10 \
      --out_dir exps/mimicgen/threading_flow_ug_selfdistill_seed1
"""

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.flow_ug_agent import FlowUGAgent, FlowUGConfig
from eval_flow_ug import load_flow, load_world_model, TARGET_SHIFT


def build_chunks(states, actions, Tp):
    """states: list[T] of [obs] np; actions: list[T] of [act] np (executed).
    -> (list of [obs], list of [Tp,act]) with last-action padding at the end."""
    A = np.clip(np.stack(actions).astype(np.float32), -1.0, 1.0)   # [T, act]
    S = np.stack(states).astype(np.float32)                        # [T, obs]
    T = A.shape[0]
    out_s, out_c = [], []
    for t in range(T):
        end = min(t + Tp, T)
        ck = A[t:end]
        if ck.shape[0] < Tp:
            pad = np.repeat(ck[-1:], Tp - ck.shape[0], axis=0)
            ck = np.concatenate([ck, pad], axis=0)
        out_s.append(S[t])
        out_c.append(ck)
    return out_s, out_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", type=str,
                   default="exps/mimicgen/threading_flow_ug_seed1/flow.pt",
                   help="TEACHER flow (target-FT'd) to roll out with UG")
    p.add_argument("--wm_ckpt", type=str,
                   default="exps/mimicgen/threading_simdist_adapt_seed1/model0.pt",
                   help="target-adapted world model for UG steering")
    p.add_argument("--out_dir", type=str,
                   default="exps/mimicgen/threading_flow_ug_selfdistill_seed1")
    p.add_argument("--num_success", type=int, default=300,
                   help="stop after this many SUCCESSFUL trajectories")
    p.add_argument("--max_episodes", type=int, default=4000,
                   help="hard cap on episodes (guards a low success rate)")
    p.add_argument("--num_envs", type=int, default=10)
    # UG config (mirror the 79% eval arm B1)
    p.add_argument("--use_ug", type=int, default=1)
    p.add_argument("--steering_coeff", type=float, default=0.3)
    p.add_argument("--ug_num_steps", type=int, default=0)
    p.add_argument("--ug_rollout_H", type=int, default=0)
    p.add_argument("--use_reward", type=int, default=1)
    p.add_argument("--ta_eval", type=int, default=1,
                   help="executed chunk length; 1 = re-query every step (= 79% arm)")
    # env
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    env_params = dict(
        env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        strict_success=StrictSuccessConfig(enabled=bool(args.strict_success),
                                           dwell_steps=3, max_tripod_disp=0.04),
        domain_shift=DomainShiftConfig(**TARGET_SHIFT),
        hand_init_perturb_joint_sigma=(args.hand_init_sigma if args.hand_init_sigma > 0 else None),
    )

    # ---- teacher agent (flow.pt EMA + UG on target-adapted WM) ----
    flow = load_flow(args.flow, device)
    Tp = flow.Tp
    if args.ta_eval > 0:
        assert args.ta_eval <= flow.Tp
        flow.Ta = args.ta_eval

    # build one env first to read obs/action dims for the world model
    envs = [PixelRobosuite(**env_params) for _ in range(args.num_envs)]
    obs_dim = int(envs[0].observation_shape[0])
    action_dim = int(envs[0].action_dim)
    model = load_world_model(args.wm_ckpt, obs_dim, action_dim, device) if args.use_ug else None

    ug_cfg = FlowUGConfig(
        use_ug=args.use_ug, steering_coeff=args.steering_coeff,
        ug_num_steps=args.ug_num_steps, ug_rollout_H=args.ug_rollout_H,
        use_reward=args.use_reward, discount=0.99)
    agent = FlowUGAgent(flow, model, ug_cfg, device=device)
    agent.eval()
    print(f"[setup] teacher={os.path.basename(os.path.dirname(args.flow))} "
          f"UG(coeff={args.steering_coeff}) Ta={flow.Ta} Tp={Tp} | "
          f"target success collection to n={args.num_success}", flush=True)

    # ---- vectorized rollout with auto-reset ----
    ep_seed = [0] * args.num_envs
    seed_counter = args.seed * 100000
    obs_list = [None] * args.num_envs
    traj_s = [[] for _ in range(args.num_envs)]
    traj_a = [[] for _ in range(args.num_envs)]

    def reset_env(i):
        nonlocal seed_counter
        np.random.seed(seed_counter)
        ep_seed[i] = seed_counter
        seed_counter += 1
        obs, _ = envs[i].reset()
        obs_list[i] = obs
        traj_s[i] = []
        traj_a[i] = []

    for i in range(args.num_envs):
        reset_env(i)

    all_S, all_C = [], []
    n_success = 0
    n_episodes = 0
    n_success_transitions = 0
    ep_lens = []

    while n_success < args.num_success and n_episodes < args.max_episodes:
        # batch UG act across all envs (Ta=1 -> chunk[:, 0] is executed)
        states = torch.stack([obs_list[i]["state"].to(device).float()
                              for i in range(args.num_envs)])       # [B, obs]
        chunk = agent.act_chunk({"state": states}, eval_mode=True)  # [B, Ta, act] cpu
        for i in range(args.num_envs):
            a = chunk[i, 0]                                          # (act,)
            traj_s[i].append(obs_list[i]["state"].detach().cpu().numpy().copy())
            traj_a[i].append(a.detach().cpu().numpy().copy())
            nobs, reward, terminal, success, _ = envs[i].step(a)
            obs_list[i] = nobs
            if terminal:
                n_episodes += 1
                succ = bool(success) or float(reward) > 0
                if succ:
                    n_success += 1
                    ep_lens.append(len(traj_a[i]))
                    s_chunks, c_chunks = build_chunks(traj_s[i], traj_a[i], Tp)
                    all_S.extend(s_chunks)
                    all_C.extend(c_chunks)
                    n_success_transitions += len(c_chunks)
                if (n_episodes % 20 == 0) or (succ and n_success % 25 == 0):
                    sr = n_success / max(n_episodes, 1)
                    print(f"[collect] episodes={n_episodes} success={n_success}/"
                          f"{args.num_success} (rate={sr:.2f}) "
                          f"chunks={n_success_transitions}", flush=True)
                reset_env(i)

    states_t = torch.from_numpy(np.stack(all_S)).float()            # [M, obs]
    chunks_t = torch.from_numpy(np.stack(all_C)).float()            # [M, Tp, act]
    out_path = os.path.join(args.out_dir, "collected_chunks.pt")
    torch.save({"states": states_t, "chunks": chunks_t,
                "obs_dim": obs_dim, "action_dim": action_dim, "Tp": Tp}, out_path)

    meta = {
        "teacher_flow": args.flow, "wm_ckpt": args.wm_ckpt,
        "steering_coeff": args.steering_coeff, "ta_eval": flow.Ta,
        "num_success": n_success, "num_episodes": n_episodes,
        "collection_success_rate": n_success / max(n_episodes, 1),
        "num_chunks": int(chunks_t.shape[0]),
        "mean_success_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "hand_init_sigma": args.hand_init_sigma, "strict_success": args.strict_success,
        "Tp": Tp, "obs_dim": obs_dim, "action_dim": action_dim,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "collect_meta.json"), "w"), indent=2)
    print(f"\n[done] collected {n_success} successes / {n_episodes} episodes "
          f"(rate={meta['collection_success_rate']:.3f}); "
          f"{chunks_t.shape[0]} chunks -> {out_path}", flush=True)
    print(f"       mean success len={meta['mean_success_len']:.1f} steps", flush=True)

    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
