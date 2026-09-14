"""Measure the IBRL warm-up success rate: each arm's BC prior, on its target.

Why this exists
---------------
In the main sim2sim figure the model-free arms need a value at x=0, and for
IBRL that value is its WARM-UP success rate: warm_up() rolls out `bc_policy`
with eval_mode=True for 30 episodes before a single gradient step
(train_rl.py:703-746), so the warm-up rate IS the prior's success rate on the
target. The figure previously reconstructed it from the replay counters W&B
keeps, which only bounds it (Threading [37.3, 69.0], Square [31.5, 51.1],
Coffee [28.3, 52.3]) because early training episodes share those counters.
This script measures it directly instead, at 10-20x the episode count.

The two traps this script exists to avoid
-----------------------------------------
1. The prior checkpoints' own cfg.yaml says `domain_shift.mode: none` and
   carries no init_distribution, so their in-training eval number (Threading
   69.6, Coffee 50.6, Square 15.2 over 10 seeds) is measured on SOURCE physics
   with the DEFAULT reset distribution -- for Square that is the full 360 deg
   nut yaw rather than rotw1's 120-180 deg, which is why 15.2 looks nothing
   like the warm-up estimate. Those numbers are not target numbers.
   CLAUDE.md rule 17.
2. Assembling the target by hand on the eval.py CLI risks a PARTIAL shift
   (every scale you forget silently becomes 1.0, infra.md section 5).

So the env is taken WHOLESALE from the authoritative cfg -- the same
release/cfgs/mimicgen/*_ibrl_m001.yaml that the IBRL arms themselves ran, which
scripts/make_ibrl_m001_cfg.py generated from the SimDist stage-1 cfg
(infra.md section 13: "target 就是 SimDist adapt 实际跑的那个 env"). Only the
observation-shaping keys (stacking, cameras, use_state) come from the prior's
own cfg, because those describe the policy, not the task.

warm_up() runs on train_env, so the TRAIN-side env fields are used; the script
asserts the train and eval sides agree, which they do on all three tasks.

    python scripts/eval_ibrl_prior_warmup.py --num_game 200 --mp 10
    python scripts/eval_ibrl_prior_warmup.py --task square --num_game 20 --mp 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pyrallis
import torch

import train_bc
import train_rl
from evaluate.multi_process_eval import run_eval as mp_run_eval

SEEDS = [1, 101, 201, 301, 401, 501, 601, 701, 801, 901]

# task -> (authoritative cfg, prior checkpoint dir template)
# The prior path template is the `bc_policy` field of the matching W&B runs,
# with the seed substituted -- each adapt seed warmed up with its OWN prior.
TASKS = {
    "threading": ("release/cfgs/mimicgen/threading_ibrl_m001.yaml",
                  "exps/mimicgen/il_ft_target_s160_seed{seed}/model0.pt"),
    "square":    ("release/cfgs/mimicgen/square_rotw1_ibrl_m001.yaml",
                  "exps/mimicgen/square_il_ft_target_rotw1_seed{seed}/model0.pt"),
    "coffee":    ("release/cfgs/mimicgen/coffee_g5x_ibrl_m001.yaml",
                  "exps/mimicgen/coffee_il_ft_target_g5x_seed{seed}/model0.pt"),
}

# Env fields that define the TASK. Everything here is copied from the
# authoritative cfg and nothing here may come from the prior's cfg.
TASK_FIELDS = ("env_name", "robots", "episode_length", "domain_shift",
               "strict_success", "hand_init_perturb",
               "hand_init_perturb_joint_sigma", "init_distribution")


def target_env_fields(cfg_path):
    """The task half of eval_env_params, straight from the authoritative cfg."""
    cfg = pyrallis.load(train_rl.MainConfig, open(cfg_path, "r"))  # type: ignore

    # warm_up() runs on train_env; the figure's later points are eval_env. If
    # the two ever diverge, the x=0 point stops being comparable to the rest of
    # the curve and the caller has to decide which one it wants.
    for train_side, eval_side, name in (
            (cfg.train_domain_shift, cfg.eval_domain_shift, "domain_shift"),
            (cfg.train_init_dist, cfg.eval_init_dist, "init_dist"),
            (cfg.train_hand_init_perturb, cfg.eval_hand_init_perturb, "perturb"),
            (cfg.train_hand_init_perturb_joint_sigma,
             cfg.eval_hand_init_perturb_joint_sigma, "sigma")):
        assert train_side == eval_side, (
            f"{cfg_path}: train/eval {name} differ; warm-up runs on the TRAIN "
            f"env, so pick one deliberately instead of letting this pass")

    return dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        episode_length=cfg.episode_length,
        domain_shift=cfg.train_domain_shift,
        strict_success=cfg.strict_success,
        hand_init_perturb=cfg.train_hand_init_perturb,
        hand_init_perturb_joint_sigma=cfg.train_hand_init_perturb_joint_sigma,
        init_distribution=cfg.train_init_dist,
    )


def describe(p):
    ds, idc = p["domain_shift"], p["init_distribution"]
    scales = {k: v for k, v in vars(ds).items()
              if k.endswith("_scale") and v != 1.0}
    rot = getattr(idc, "nut_z_rot", None)
    rot = (f"{np.degrees(rot[0]):.0f}-{np.degrees(rot[1]):.0f}deg"
           if rot is not None and getattr(idc, "enabled", False) else "FULL")
    return (f"task={p['env_name']} shift={ds.mode} {scales} "
            f"init_dist={getattr(idc, 'enabled', None)} nut_z_rot={rot} "
            f"sigma={p['hand_init_perturb_joint_sigma']} "
            f"strict={getattr(p['strict_success'], 'enabled', None)} "
            f"ep_len={p['episode_length']}")


def eval_one(weight, task_fields, num_game, mp, seed, no_render):
    """Success rate of one prior checkpoint on the target env."""
    agent, _, env_params = train_bc.load_model(weight, "cuda", verbose=False)
    # env_params from the BC cfg describes the OBSERVATION (stacking, cameras,
    # use_state); the task half is replaced wholesale. Anything the BC loader
    # does not set (domain_shift, init_distribution, ...) is simply absent
    # there, which is exactly why it must be injected rather than patched.
    for k in TASK_FIELDS:
        env_params[k] = task_fields[k]
    if no_render:
        assert env_params.get("use_state"), "--no_render needs a state policy"
        env_params["state_only_no_render"] = True
    env_params.pop("device", None)

    scores = mp_run_eval(env_params, agent, num_game, mp, seed, verbose=False)
    del agent
    torch.cuda.empty_cache()
    return float(np.mean([s > 0 for s in scores])), env_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", *TASKS])
    ap.add_argument("--num_game", type=int, default=200)
    ap.add_argument("--mp", type=int, default=10)
    ap.add_argument("--seed", type=int, default=100_000,
                    help="episode-seed offset; kept clear of the in-training "
                         "eval seeds (which are step//log_per_step * 100)")
    ap.add_argument("--seeds", default="", help="comma list, default all 10")
    ap.add_argument("--no_render", type=int, default=1)
    args = ap.parse_args()
    assert args.num_game % args.mp == 0, "num_eval_proc must divide num_game"

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS)
    tasks = list(TASKS) if args.task == "all" else [args.task]

    for task in tasks:
        cfg_path, weight_tmpl = TASKS[task]
        fields = target_env_fields(cfg_path)
        print(f"\n===== {task}  ({cfg_path})")

        rates, shown = {}, False
        for seed in seeds:
            weight = weight_tmpl.format(seed=seed)
            if not os.path.exists(weight):
                print(f"  seed{seed:<4d} MISSING {weight}")
                continue
            rate, env_params = eval_one(weight, fields, args.num_game, args.mp,
                                        args.seed, args.no_render)
            if not shown:
                print(f"  [EVAL ENV] {describe(env_params)}", flush=True)
                shown = True
            rates[seed] = rate
            print(f"  seed{seed:<4d} {100 * rate:5.1f}%  ({weight})", flush=True)

        if rates:
            v = np.array(list(rates.values())) * 100
            print(f"  ---- {task}: mean {v.mean():.1f}%  sd {v.std(ddof=1):.1f}"
                  f"  n={v.size}  ({args.num_game} episodes/seed)")
            print(f"  IBRL_WARMUP entry: {task!r}: {v.mean() / 100:.4f},")


if __name__ == "__main__":
    main()
