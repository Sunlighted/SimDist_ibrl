"""Evaluate "dynamics-only SimDist" = MPPI on a SOURCE-trained dynamics model,
borrowing value + policy from the frozen source agent and a heuristic reward
(CLAUDE.md §17). NO target adaptation — the dynamics is source-only, so the
meaningful sanity check is on SOURCE: does MPPI-on-learned-dynamics match/beat
the source policy itself?

Pieces:
  dynamics : the raw-state WorldModel trained by train_dynamics.py (only the
             dynamics head is meaningful; value/reward/policy heads are random
             and are OVERRIDDEN below).
  value    : source agent's critic  V_src(s) = min-ensemble Q(s, pi_src(s)).
  policy   : source agent's RL actor, as the MPPI candidate proposal.
  reward   : heuristic shaped -‖needle - ring‖ from the predicted state (off by
             default: --use_reward 1 to enable). Note: needle/ring here are BODY
             positions (state[9:12] / [23:26]); the true success uses geom_xpos,
             so this is a planning proxy, not the success metric.

Single-process eval (the value/policy closures wrap a cuda source agent; mp eval
would need them picklable).

Run:
  python eval_simdist_mpc.py \
      --dyn_ckpt exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt \
      --source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
      --num_game 50 --shift source
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluate import run_eval
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.world_model import WorldModelConfig
from rl.simdist_planner import PlannerConfig
from rl.simdist_agent import SimDistConfig, SimDistAgent
import train_simdist
import train_rl

# 37-dim Threading state layout (probed): needle/tripod BODY world positions.
NEEDLE_SL = slice(9, 12)
RING_SL = slice(23, 26)

# §12 dynamics-only TARGET (for --shift target).
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=[0.0, 0.0, 0.0],
    ring_size_scale=0.85, ring_friction_scale=2.0, mask_hole_offset_in_obs=False,
)


class VelAugAgent:
    """Eval-time wrapper for a velocity-AUGMENTED dynamics: the env emits a 37-dim
    obs (no velocity), so we finite-difference consecutive obs to build the 74-dim
    [pos, vel] state the augmented world model expects. run_eval is single-process +
    sequential, so one persistent `prev` suffices; a large jump (env.reset teleport)
    zeroes the velocity at the (single) episode-boundary step."""
    def __init__(self, inner, pos_dim, dt, jump_thresh=0.2):
        self.inner, self.pos_dim, self.dt, self.jump = inner, pos_dim, dt, jump_thresh
        self.prev = None

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        s = obs["state"].to(self.inner.device).float()
        single = s.dim() == 1
        pos = (s.view(1, -1) if single else s)[:, :self.pos_dim]
        if self.prev is None or self.prev.shape != pos.shape:
            vel = torch.zeros_like(pos)
        else:
            vel = (pos - self.prev) / self.dt
            vel[(pos - self.prev).norm(dim=-1) > self.jump] = 0.0   # episode boundary
        self.prev = pos.detach().clone()
        aug = torch.cat([pos, vel], -1)
        return self.inner.act({"state": aug[0] if single else aug},
                              eval_mode=eval_mode, stddev=stddev, cpu=cpu)

    def train(self, mode=False):
        self.inner.train(mode); return self

    def __getattr__(self, k):   # delegate .training etc. (NOT act_chunk -> inner lacks it)
        return getattr(self.__dict__["inner"], k)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dyn_ckpt", type=str, required=True,
                   help="dynamics WorldModel state_dict from train_dynamics.py")
    p.add_argument("--source_agent", type=str,
                   default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--num_game", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--shift", type=str, default="source", choices=["source", "target"])
    p.add_argument("--fxc", action="store_true",
                   help="§27 fx_C: SOURCE physics + shrunk ring (0.85) + ring friction 2.0 "
                        "-> narrow ~1.8mm window on source dynamics, so model error (and thus "
                        "the velocity-augmentation) can show in success rate (source alone is "
                        "saturated at 0.97). Overrides --shift.")
    p.add_argument("--hand_init_sigma", type=float, default=0.0)
    p.add_argument("--strict_success", type=int, default=1)
    # planner overrides (default to the trained horizon + SimDist defaults)
    p.add_argument("--use_reward", type=int, default=0,
                   help="0 = rank by terminal V_src only (cleanest first test); "
                        "1 = add the heuristic shaped reward")
    p.add_argument("--reward_scale", type=float, default=0.3)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--num_elites", type=int, default=64)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--mixture_coef", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--init_std", type=float, default=0.15)
    p.add_argument("--no_planner", action="store_true",
                   help="diagnostic: act with the world model's OWN policy head (no "
                        "MPPI). In a dynamics-only ckpt that head is UNTRAINED — the "
                        "source-policy baseline is always printed separately.")
    return p.parse_args()


def main():
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dcfg = json.load(open(os.path.join(os.path.dirname(args.dyn_ckpt), "dyn_cfg.json")))
    H = dcfg["horizon"]
    # velocity-augmented dynamics? obs_dim=74 (=[pos37, vel37]); pos block = first 37.
    augment = int(dcfg.get("augment_velocity", 0))
    model_obs_dim = int(dcfg["obs_dim"])
    pos_dim = model_obs_dim // 2 if augment else model_obs_dim   # =37 either way
    ctrl_dt = float(dcfg.get("ctrl_dt", 0.05))

    # ---- eval env ----
    if args.fxc:
        shift = DomainShiftConfig(mode="fixed", ring_size_scale=0.85, ring_friction_scale=2.0)
    elif args.shift == "source":
        shift = DomainShiftConfig(mode="none")
    else:
        shift = DomainShiftConfig(**TARGET_SHIFT)
    env_params = dict(
        env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        strict_success=StrictSuccessConfig(enabled=bool(args.strict_success),
                                           dwell_steps=3, max_tripod_disp=0.04),
        domain_shift=shift,
        hand_init_perturb_joint_sigma=(args.hand_init_sigma if args.hand_init_sigma > 0 else None),
    )
    env = PixelRobosuite(**env_params)  # type: ignore

    # ---- world model (raw-state dynamics; value/reward/policy heads overridden) ----
    sd_cfg = SimDistConfig(
        world_model=WorldModelConfig(
            encoder_type="identity", mlp_hidden_dim=dcfg["hidden_dim"],
            num_layer=dcfg["num_layer"], layer_norm=dcfg["layer_norm"],
            horizon=H, activation=dcfg.get("activation", "relu")),
        planner=PlannerConfig(
            horizon=H, num_samples=args.num_samples, num_elites=args.num_elites,
            iterations=args.iterations, mixture_coef=args.mixture_coef,
            temperature=args.temperature, init_std=args.init_std,
            use_reward=bool(args.use_reward), discount=0.99),
        use_planner=0 if args.no_planner else 1,
    )
    agent = SimDistAgent(True, (model_obs_dim,), env.prop_shape, env.action_dim,
                         "robot0_eye_in_hand", sd_cfg, device=dev)
    agent.model.load_state_dict(torch.load(args.dyn_ckpt, map_location=dev))
    agent.eval()

    # ---- frozen source agent: value + policy ----
    sim_cfg = train_simdist.MainConfig()  # carries use_state=1/state_stack=1/rl_camera for the loader
    source = train_rl.load_frozen_agent(args.source_agent, env, sim_cfg, dev)

    # value / proposal / reward operate on the source agent's 37-dim state; the
    # (possibly augmented 74-dim) decoded planner state is sliced to its pos block.
    @torch.no_grad()
    def value_fn(state):                       # [N,obs] -> [N]
        return source.value({"state": state[:, :pos_dim]})

    @torch.no_grad()
    def proposal_policy(state):                # [N,obs] -> [N,act]
        return source.act({"state": state[:, :pos_dim]}, eval_mode=True, stddev=0.0, cpu=False)

    @torch.no_grad()
    def reward_fn(state, action):              # [N,obs],[N,act] -> [N]
        dist = (state[:, NEEDLE_SL] - state[:, RING_SL]).norm(dim=-1)
        return -args.reward_scale * dist

    agent.set_planning_sources(
        value_fn=value_fn,
        reward_fn=reward_fn if args.use_reward else None,
        proposal_policy=proposal_policy,
    )

    # velocity-augmented dynamics: wrap so eval builds [pos, finite-diff vel]
    eval_agent = VelAugAgent(agent, pos_dim, ctrl_dt) if augment else agent

    # ---- baseline: source policy alone, on the SAME env ----
    shift_tag = "fx_C(source+shrunk-ring)" if args.fxc else args.shift
    print(f"[setup] dynamics={args.dyn_ckpt} | augment_vel={augment} obs_dim={model_obs_dim} | "
          f"shift={shift_tag} | planner={'OFF' if args.no_planner else 'MPPI'} | "
          f"use_reward={args.use_reward} | H={H} | games={args.num_game}")
    src_scores = run_eval(env_params=env_params, agent=source, num_game=args.num_game,
                          seed=args.seed, record_dir=None, verbose=False)
    print(f"[baseline] source policy alone: {np.mean(src_scores):.3f}")

    scores = run_eval(env_params=env_params, agent=eval_agent, num_game=args.num_game,
                      seed=args.seed, record_dir=None, verbose=False)
    tag = "source-policy-direct" if args.no_planner else "MPPI(dyn+V_src+proposal)"
    print(f"[result]  {tag}: {np.mean(scores):.3f}")
    print(f"[compare] MPPI - source_policy = {np.mean(scores) - np.mean(src_scores):+.3f}")

    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()
