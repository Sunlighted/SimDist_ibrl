from collections import deque

import torch
import numpy as np
from common_utils import Recorder, Stopwatch
from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite


def _env_cls_for(env_name):
    """The registered robosuite class behind an env_name, or None.

    Used only so the [EVAL ENV] banner can report the reset distribution
    honestly (see its call site). Never raises -- a missing class must degrade
    the PRINT, not the eval.
    """
    try:
        import env.threading_variants  # noqa: F401
        import env.square_variants  # noqa: F401
        from robosuite.environments.base import REGISTERED_ENVS
        return REGISTERED_ENVS.get(env_name)
    except Exception:
        return None


def _policy_act_chunk_single(agent, obs, eval_mode):
    """Call agent.act_chunk for a single env if available; else wrap act() as len-1."""
    if hasattr(agent, "act_chunk"):
        return agent.act_chunk(obs, eval_mode=eval_mode)  # (Ta, D) — 1-D state path
    return agent.act(obs, eval_mode=eval_mode).unsqueeze(0)  # (D,) -> (1, D)


def run_eval(
    env_params,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    eval_mode=True,
    record_every=1,
) -> list[float]:
    scores = []
    lens = []
    stopwatch = Stopwatch()
    recorder = None if record_dir is None else Recorder(record_dir)

    env = PixelRobosuite(**env_params)
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            with stopwatch.time("reset"):
                obs, image_obs = env.reset()
            if hasattr(agent, "reset_prior"):
                agent.reset_prior()                    # IBRL prior_chunk > 1

            record_this = recorder is not None and (episode_idx % record_every == 0)
            terminal = False
            # Per-episode action queue: refilled from agent.act_chunk when
            # empty. For non-chunked policies this is always length-1, so
            # eval cost is unchanged.
            queue: deque = deque()
            while not terminal:
                if record_this:
                    recorder.add(image_obs)

                if not queue:
                    with stopwatch.time("act"):
                        chunk = _policy_act_chunk_single(agent, obs, eval_mode)
                    for a in chunk:
                        queue.append(a)
                action = queue.popleft()

                with stopwatch.time("step"):
                    obs, reward, terminal, _, image_obs = env.step(action)

                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))
            if scores[-1] > 0:
                lens.append(env.time_step)

            if record_this:
                recorder.save(f"episode{episode_idx}_seed{seed + episode_idx}")

    if verbose:
        num_success = int(sum(s > 0 for s in scores))
        success_rate = num_success / len(scores) if scores else 0.0
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")
        print(f"success rate: {100 * success_rate:.2f}% ({num_success}/{len(scores)})")
        print(f"average steps for success games: {np.mean(lens)}")
        stopwatch.summary()

    return scores


if __name__ == "__main__":
    import argparse
    import os
    import time
    import math
    import common_utils
    import train_bc
    import train_rl
    from multi_process_eval import run_eval as mp_run_eval
    import rich.traceback

    # make logging more beautiful
    rich.traceback.install()

    os.environ.setdefault("MUJOCO_GL", "egl")  # see train_rl.py: an outer
    # MUJOCO_GL=osmesa must survive (MIG GPUs cannot init EGL at import time).
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--include", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl/simdist/residual_rl/residual_simdist/ibrl_simdist/tdmpc2")
    parser.add_argument("--prior_chunk", type=int, default=None,
                        help="Override QAgent prior_chunk at eval (IBRL: run the BC prior in chunks of N steps).")
    parser.add_argument("--act_method", type=str, default=None,
                        help="Override QAgent act_method at eval (rl/ibrl/ibrl_soft). "
                             "rl = pure trained actor; ibrl = max-Q hybrid over "
                             "{actor, BC/source prior}. rl/h2oplus modes only.")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--record_every", type=int, default=1,
                        help="Save one video every N episodes when --record_dir "
                             "is set. 1 = every episode (legacy). 20 = save 1 of "
                             "every 20 episodes. Ignored if --record_dir is None.")
    parser.add_argument("--mp", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    # Optional env-side overrides to re-eval a checkpoint under different
    # source/target conditions without re-training. Leaving everything default
    # preserves the env_params loaded from the checkpoint's cfg.
    parser.add_argument("--shift_mode", type=str, default=None,
                        choices=[None, "none", "fixed", "dr"],
                        help="Override domain_shift.mode (Threading_D0 only).")
    parser.add_argument("--shift_hole_offset", type=str, default=None,
                        help='Comma-separated 3-vec, e.g. "0.005,0.005,0.0".')
    parser.add_argument("--shift_friction_scale", type=float, default=None)
    parser.add_argument("--shift_mass_scale", type=float, default=None)
    parser.add_argument("--shift_solref_scale", type=float, default=None)
    parser.add_argument("--shift_solimp_scale", type=float, default=None)
    parser.add_argument("--shift_gripper_kp_scale", type=float, default=None)
    parser.add_argument("--shift_ring_size_scale", type=float, default=None,
                        help="Shrink the tripod hole opening (nominal 16mm). "
                             "<1 = smaller/harder; e.g. 0.85 -> 13.6mm. Feasible "
                             "floor ~0.78 (needle is 10mm).")
    parser.add_argument("--shift_ring_friction_scale", type=float, default=None,
                        help="Scale the ring rim friction (nominal 0.3 sliding). "
                             ">1 = grippier/harder; e.g. 3.0 -> 0.9.")
    parser.add_argument("--shift_needle_size_scale", type=float, default=None,
                        help="THICKEN the needle shaft (nominal 10mm square "
                             "cross-section) -- the mirror of --shift_ring_size_scale, "
                             "closing the same clearance from the needle side. "
                             "1.2 -> 12mm (2.0mm/side vs a 16mm ring). The success "
                             "check reads the shaft geom's CENTER, which does not "
                             "move, so the 12mm criterion is unchanged. Feasible "
                             "ceiling <1.6 at ring_size 1.0.")
    parser.add_argument("--shift_peg_size_scale", type=float, default=None,
                        help="[Square_D0] Grow the square peg x/y half-size "
                             "(nominal 32mm peg / 6.75mm clearance). >1 = bigger "
                             "peg / tighter; 1.25 -> 40mm (2.75mm/side). Floor "
                             "<1.42 (peg must fit the 45.5mm nut hole).")
    parser.add_argument("--shift_peg_friction_scale", type=float, default=None,
                        help="[Square_D0] Scale the square peg friction "
                             "(nominal 1.0 sliding). >1 = grippier/harder.")
    parser.add_argument("--shift_nut_size_scale", type=float, default=None,
                        help="[Square_D0] Shrink the HELD nut hole (nominal "
                             "45.5mm). <1 = smaller/harder; 0.85 -> 38.7mm "
                             "(3.35mm/side). Floor ~0.70 (peg is 32mm).")
    parser.add_argument("--shift_pod_holder_size_scale", type=float, default=None,
                        help="[Coffee_D0] Shrink the pod holder's INNER opening "
                             "(nominal 24.97mm radius vs 24.30mm pod = only "
                             "0.67mm/side clearance). <1 = tighter; 0.985 -> "
                             "0.30mm/side. Floor >0.973 (pod must still fit). "
                             "Success threshold (r_diff 5.2mm) is unaffected.")
    parser.add_argument("--shift_pod_holder_friction_scale", type=float, default=None,
                        help="[Coffee_D0] Scale the pod holder cup-wall friction "
                             "(nominal 1.0). >1 = grippier insertion.")
    parser.add_argument("--shift_lid_damping_scale", type=float, default=None,
                        help="[Coffee_D0] Scale the lid hinge viscous damping "
                             "(nominal 0.005). >1 resists fast closing.")
    parser.add_argument("--shift_lid_frictionloss", type=float, default=None,
                        help="[Coffee_D0] ABSOLUTE dry friction on the lid hinge "
                             "(N*m; nominal 0 so a scale would be a no-op). The "
                             "lid stops unless pushed hard enough -- the sharpest "
                             "second-stage knob.")
    parser.add_argument("--shift_mask_hole_in_obs", type=int, default=None,
                        help="1 = mask hole offset from obs, 0 = leave raw")
    parser.add_argument("--hand_init_perturb", type=str, default=None,
                        help='Comma-separated 3-vec EE per-axis half-range (m), '
                             'e.g. "0.015,0.015,0.01". "0,0,0" = deterministic.')
    parser.add_argument("--needle_scale", type=float, default=1.0,
                        help="Threading only: WIDEN the needle placement box by this "
                             "factor around the D0 center (1.0 = D0, off). Same formula "
                             "as train_simdist_adapt.py --needle_scale, so eval numbers "
                             "are comparable to those runs. Tripod stays fixed (D0).")
    parser.add_argument("--nut_z_rot", type=str, default=None,
                        help="Square only: SquareNut yaw window in DEGREES, "
                             "e.g. '0,60'. Pass 'full' to force the whole "
                             "(0,360) circle, overriding a window baked into "
                             "the ckpt's cfg.yaml (that is how a window-trained "
                             "ckpt gets scored on the original Square_D0 task). "
                             "Default None = use whatever cfg.yaml says.")
    parser.add_argument("--hand_init_perturb_joint_sigma", type=float, default=None,
                        help='Joint-space gaussian σ (rad). Mutually exclusive '
                             'with --hand_init_perturb. e.g. 0.02 = robosuite '
                             'default (demo distribution); 0.05 ≈ 2.5× demo, '
                             '0.1 ≈ 5× demo.')
    parser.add_argument("--strict_success", type=int, default=None,
                        help="1 = enable strict success gating at eval "
                             "(Threading_D0): a validated success requires the "
                             "position test to hold for --strict_dwell_steps "
                             "consecutive steps AND the ring to stay within "
                             "--strict_max_tripod_disp of its reset pose. Kills "
                             "clip-through / shove-the-ring false positives. "
                             "0 = raw upstream success.")
    parser.add_argument("--strict_dwell_steps", type=int, default=None,
                        help="Consecutive control steps the position test must "
                             "hold (default 3 @ 20Hz = 0.15s).")
    parser.add_argument("--strict_max_tripod_disp", type=float, default=None,
                        help="Max ring-center drift from reset, meters "
                             "(default 0.04).")
    parser.add_argument("--strict_max_needle_speed", type=float, default=None,
                        help="Optional needle linear-speed gate, m/s "
                             "(default 0 = disabled).")
    parser.add_argument("--sim2real_cell", type=str, default=None,
                        help="Force this sim2real cell yaml instead of the one stored in the "
                             "checkpoint. REQUIRED when comparing checkpoints trained on "
                             "DIFFERENT cells: env_params come from the ckpt (iron law #17), so "
                             "without this each arm is silently scored on its own distribution.")
    parser.add_argument("--no_render", type=int, default=0,
                        help="1 = state-based eval with ALL camera rendering "
                             "disabled (no offscreen GL context, no per-step "
                             "224x224 render). Much faster + lets --mp run "
                             "without the EGL deadlock. Requires the run to be "
                             "state-based (use_state=1); incompatible with "
                             "--record_dir.")
    parser.add_argument("--shift_from_cfg", type=int, default=0,
                        help="If 1, base the domain_shift on the ckpt's "
                             "<run_folder>/cfg.yaml `eval_domain_shift` field "
                             "(so re-eval matches the training-time target "
                             "without re-typing all six --shift_* flags). "
                             "Individual --shift_* flags still override "
                             "specific fields on top of the cfg baseline.")
    parser.add_argument("--use_planner", type=int, default=None,
                        help="(--mode simdist) override SimDistAgent.use_planner: "
                             "1 = MPPI planning, 0 = run the policy head directly "
                             "(no planning). Default None = use the ckpt cfg's "
                             "value.")
    parser.add_argument("--planner_set", type=str, default="",
                        help="(--mode simdist) comma-separated planner overrides, "
                             "e.g. 'init_std=0.3,num_samples=256'. Applied AFTER "
                             "--original_mppi and BEFORE the planner is rebuilt, so "
                             "one knob can be swept without cloning the run dir.")
    parser.add_argument("--dyn_heads", type=int, default=None,
                        help="(--mode simdist) truncate the dynamics ensemble to "
                             "this many heads AT EVAL TIME (head 0 first), from the "
                             "SAME weights. Isolates plan-time ensemble averaging "
                             "from 'the encoder was trained with N heads' -- see "
                             "docs/findings.md 31, where 5 heads scored 0.970 and 1 "
                             "head 0.120 off one checkpoint. Must be <= the ckpt's "
                             "head count; applied BEFORE the planner is rebuilt.")
    parser.add_argument("--original_mppi", type=int, default=0,
                        help="(--mode simdist, implies --use_planner 1) rebuild the "
                             "MPPI planner with the ORIGINAL SimDist recipe "
                             "(mixture 0.05 / init_std 2.0 / 8 iters / temp 0.25) "
                             "instead of the ibrl-tuned trust-the-proposal config. "
                             "horizon/discount kept (dynamics-dependent).")

    args = parser.parse_args()
    common_utils.set_all_seeds(args.seed)

    # ------------------------------------------------------------------
    # Helper: build env-param override dict for ONE weight. Per-weight
    # because --shift_from_cfg reads each ckpt's own cfg.yaml.
    # ------------------------------------------------------------------
    def _build_env_overrides(weight_path: str) -> dict:
        from env.domain_shift import DomainShiftConfig
        from env.strict_success import StrictSuccessConfig
        import dataclasses
        import yaml

        overrides: dict = {}
        # Strict-success gate (eval-time). Independent of the domain-shift
        # overrides above; only constructed when --strict_success is set.
        if args.strict_success is not None:
            strict_kwargs = {"enabled": bool(args.strict_success)}
            if args.strict_dwell_steps is not None:
                strict_kwargs["dwell_steps"] = args.strict_dwell_steps
            if args.strict_max_tripod_disp is not None:
                strict_kwargs["max_tripod_disp"] = args.strict_max_tripod_disp
            if args.strict_max_needle_speed is not None:
                strict_kwargs["max_needle_speed"] = args.strict_max_needle_speed
            overrides["strict_success"] = StrictSuccessConfig(**strict_kwargs)
            print(f"[eval override] strict_success = {overrides['strict_success']}")
        # Hand-init perturbation (same for every weight). The two knobs are
        # mutually exclusive; passing one explicitly nulls the other so the
        # wrapper's assertion isn't tripped by a leftover from cfg.yaml.
        if args.hand_init_perturb is not None and args.hand_init_perturb_joint_sigma is not None:
            raise ValueError("set at most one of --hand_init_perturb / "
                             "--hand_init_perturb_joint_sigma")
        if args.hand_init_perturb is not None:
            overrides["hand_init_perturb"] = tuple(
                float(x) for x in args.hand_init_perturb.split(","))
            overrides["hand_init_perturb_joint_sigma"] = None
        if args.hand_init_perturb_joint_sigma is not None:
            overrides["hand_init_perturb_joint_sigma"] = args.hand_init_perturb_joint_sigma
            overrides["hand_init_perturb"] = None

        # Widen the Threading needle placement box. Same formula as
        # train_simdist_adapt.py's --needle_scale and eval_oracle_dyn_mppi.py's,
        # so all three are directly comparable. Must be set BEFORE the `active`
        # early-return below, else it would be skipped when no shift is given.
        if args.needle_scale != 1.0:
            from env.domain_shift import InitDistConfig
            PI = 3.141592653589793
            s = args.needle_scale
            ncx, ncy, nhx, nhy = -0.125, 0.2, 0.075 * s, 0.05 * s
            overrides["init_distribution"] = InitDistConfig(
                enabled=True,
                needle_x=(ncx - nhx, ncx + nhx), needle_y=(ncy - nhy, ncy + nhy),
                needle_z_rot=(-2.0 * PI / 3.0, -PI / 3.0),
                tripod_x=(0.0, 0.0), tripod_y=(-0.15, -0.15),
                tripod_z_rot=(PI / 2.0, PI / 2.0))
            print(f"[eval override] init_distribution: needle box {s}x -> "
                  f"x=({ncx-nhx:.3f},{ncx+nhx:.3f}) y=({ncy-nhy:.3f},{ncy+nhy:.3f})")

        # Square: override the SquareNut yaw window. Like --needle_scale this
        # must precede the `active` early-return, since Square runs carry no
        # domain shift. 'full' disables the config entirely, which is how a
        # window-trained ckpt is scored on the original 360-deg Square_D0.
        if args.nut_z_rot is not None:
            import math
            from env.domain_shift import InitDistConfig
            if args.nut_z_rot.strip().lower() in ("full", "off", "none"):
                overrides["init_distribution"] = InitDistConfig(enabled=False)
                print("[eval override] nut_z_rot: DISABLED -> full (0,360) deg")
            else:
                lo_d, hi_d = (float(x) for x in args.nut_z_rot.split(","))
                overrides["init_distribution"] = InitDistConfig(
                    enabled=True,
                    nut_z_rot=(math.radians(lo_d), math.radians(hi_d)))
                print(f"[eval override] nut_z_rot window = "
                      f"({lo_d:.1f},{hi_d:.1f}) deg")

        # Domain shift: assemble shift_kwargs from (optional) cfg.yaml baseline
        # then layer CLI overrides on top.
        active = args.shift_mode is not None or args.shift_from_cfg
        if not active:
            return overrides

        shift_kwargs: dict = {}
        if args.shift_from_cfg:
            cfg_path = os.path.join(os.path.dirname(weight_path), "cfg.yaml")
            with open(cfg_path) as f:
                raw_cfg = yaml.safe_load(f) or {}
            cfg_shift = raw_cfg.get("eval_domain_shift", {}) or {}
            # Translate YAML types (list -> tuple where needed). Unknown keys
            # are tolerated since DomainShiftConfig will accept whatever it
            # defines.
            for k, v in cfg_shift.items():
                if isinstance(v, list):
                    v = tuple(v)
                shift_kwargs[k] = v
            print(f"[eval override] shift baseline from {cfg_path}: {cfg_shift}")

        # CLI scalar/list flags layer on top.
        if args.shift_mode is not None:
            shift_kwargs["mode"] = args.shift_mode
        elif "mode" not in shift_kwargs:
            shift_kwargs["mode"] = "none"
        if args.shift_hole_offset is not None:
            shift_kwargs["hole_offset"] = tuple(
                float(x) for x in args.shift_hole_offset.split(","))
        for k in ("friction_scale", "mass_scale", "solref_scale",
                  "solimp_scale", "gripper_kp_scale",
                  "ring_size_scale", "ring_friction_scale", "needle_size_scale",
                  "peg_size_scale", "peg_friction_scale", "nut_size_scale",
                  "pod_holder_size_scale", "pod_holder_friction_scale",
                  "lid_damping_scale", "lid_frictionloss"):
            v = getattr(args, f"shift_{k}")
            if v is not None:
                shift_kwargs[k] = v
        if args.shift_mask_hole_in_obs is not None:
            shift_kwargs["mask_hole_offset_in_obs"] = bool(args.shift_mask_hole_in_obs)

        # Filter to fields DomainShiftConfig actually has, so older cfg.yaml
        # files with extra keys won't break us.
        valid_fields = {f.name for f in dataclasses.fields(DomainShiftConfig)}
        shift_kwargs = {k: v for k, v in shift_kwargs.items() if k in valid_fields}
        overrides["domain_shift"] = DomainShiftConfig(**shift_kwargs)
        print(f"[eval override] domain_shift = {overrides['domain_shift']}")
        return overrides

    if args.folder is None:
        weights = [args.weight]
    else:
        weights = common_utils.get_all_files(args.folder, ".pt", args.include)

    print(f"files to eval:")
    eval_items = []
    for weight in weights:
        print(weight, f", repeat {args.repeat}")
        if args.mode == "bc":
            agent, _, env_params = train_bc.load_model(weight, "cuda")
        elif args.mode == "rl":
            agent, _, env_params = train_rl.load_model(weight, "cuda")
        elif args.mode == "simdist":
            import train_simdist
            agent, _, env_params = train_simdist.load_model(weight, "cuda")
        elif args.mode == "residual_rl":
            import train_residual_rl
            agent, _, env_params = train_residual_rl.load_model(weight, "cuda")
        elif args.mode == "residual_simdist":
            import train_residual_simdist
            agent, _, env_params = train_residual_simdist.load_model(weight, "cuda")
        elif args.mode == "ibrl_simdist":
            import train_ibrl_simdist
            agent, _, env_params = train_ibrl_simdist.load_model(weight, "cuda")
        elif args.mode == "tdmpc2":
            import train_tdmpc2
            agent, _, env_params = train_tdmpc2.load_model(weight, "cuda")
        else:
            assert False, f"unsupported mode: {args.mode}"

        # Optional: override the QAgent act_method to decompose a hybrid eval
        # (e.g. --act_method rl = pure trained actor, no BC/source prior).
        if args.prior_chunk is not None and hasattr(getattr(agent, "cfg", None), "prior_chunk"):
            print(f"[eval override] prior_chunk {agent.cfg.prior_chunk} -> {args.prior_chunk}")
            agent.cfg.prior_chunk = args.prior_chunk
        if args.act_method is not None and hasattr(getattr(agent, "cfg", None), "act_method"):
            print(f"[eval override] act_method {agent.cfg.act_method} -> {args.act_method}")
            agent.cfg.act_method = args.act_method

        # SimDist planner overrides: toggle MPPI vs the bare policy head, and/or
        # swap in the ORIGINAL SimDist MPPI recipe (broad search). Only meaningful
        # for --mode simdist.
        if args.mode == "simdist":
            if args.original_mppi:
                from rl.simdist_planner import MppiPlanner, apply_original_recipe
                apply_original_recipe(agent.cfg.planner)
                agent.cfg.use_planner = 1
                agent.planner = MppiPlanner(
                    agent.model, agent.cfg.planner, agent.action_dim, device="cuda")
                p = agent.cfg.planner
                _nb = p.num_base_override if p.num_base_override >= 0 else int(p.num_samples * p.mixture_coef)
                print(f"[eval override] ORIGINAL SimDist MPPI recipe "
                      f"(base={_nb}+rand={p.num_samples - _nb}, init_std={p.init_std}, "
                      f"min_std={p.min_std}, iters={p.iterations}, temp={p.temperature}, "
                      f"elites={p.num_elites}, base_policy_std={p.base_policy_std}, "
                      f"horizon={p.horizon} [kept])")
            if args.dyn_heads is not None:
                import torch.nn as _nn
                _m = agent.model
                _tot = _m.num_dynamics_heads()
                assert 1 <= args.dyn_heads <= _tot, \
                    f"--dyn_heads must be in [1,{_tot}] for this ckpt"
                _m.dynamics_extra = _nn.ModuleList(
                    list(_m.dynamics_extra)[: args.dyn_heads - 1])
                if len(_m.dynamics_prior) > 0:   # keep priors aligned with kept heads
                    _m.dynamics_prior = _nn.ModuleList(
                        list(_m.dynamics_prior)[: args.dyn_heads])
                print(f"[eval override] dynamics heads: {_tot} -> "
                      f"{_m.num_dynamics_heads()}")
            if args.planner_set:
                from rl.simdist_planner import MppiPlanner
                for kv in args.planner_set.split(","):
                    k, v = kv.split("=")
                    k = k.strip()
                    old = getattr(agent.cfg.planner, k)
                    setattr(agent.cfg.planner, k, type(old)(v))
                    print(f"[eval override] planner.{k}: {old} -> "
                          f"{getattr(agent.cfg.planner, k)}")
                agent.cfg.use_planner = 1
                agent.planner = MppiPlanner(
                    agent.model, agent.cfg.planner, agent.action_dim, device="cuda")
            if args.use_planner is not None:
                print(f"[eval override] use_planner {agent.cfg.use_planner} "
                      f"-> {args.use_planner}")
                agent.cfg.use_planner = args.use_planner

        # Apply CLI overrides (domain_shift / hand_init_perturb) on top of
        # the env_params loaded from the checkpoint's stored cfg.
        env_overrides = _build_env_overrides(weight)
        for k, v in env_overrides.items():
            env_params[k] = v

        # Force one cell across checkpoints. Without it, `train_rl.load_model`
        # hands back the env_params stored in EACH checkpoint's cfg, so an A/B
        # over two cells scores each arm on its own reset distribution and the
        # comparison is meaningless (iron law #17 -- this exact trap voided a
        # whole 2x2 once).
        if args.sim2real_cell is not None:
            from env.sim2real import load_sim2real
            _cell = load_sim2real(args.sim2real_cell)
            _old = env_params.get("sim2real")
            env_params["sim2real"] = _cell
            print(f"[eval override] sim2real cell -> {args.sim2real_cell}  "
                  f"(layout_tripod_z_rot_deg {getattr(_old, 'layout_tripod_z_rot_deg', None)} "
                  f"-> {_cell.layout_tripod_z_rot_deg})")

        # Optional: disable all camera rendering for state-based eval (big
        # speedup + no EGL deadlock under --mp). Guarded against recording.
        if args.no_render:
            assert args.record_dir is None, \
                "--no_render is incompatible with --record_dir (no images rendered)"
            assert env_params.get("use_state", False), \
                "--no_render requires a state-based run (use_state=1)"
            env_params["state_only_no_render"] = True

        for _ in range(args.repeat):
            eval_items.append((weight, agent, env_params))
    print(common_utils.wrap_ruler(""))

    if args.record_dir:
        assert len(eval_items) == 1
        if env_params["env_name"] == "TwoArmTransport":
            env_params["camera_names"] = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]
        else:
            env_params["camera_names"] = ["agentview", "robot0_eye_in_hand"]

    weight_scores = []
    all_scores = []
    for weight, agent, env_params in eval_items:
        # Print the FULL evaluation env identity every time. Two numbers are only
        # comparable if these lines match, and the reset distribution in
        # particular is inherited silently from the ckpt's cfg (Square's rotw1
        # nut-yaw window lives in eval_init_dist, invisible on the CLI), so a
        # mismatch is otherwise undetectable after the fact.
        _idc = env_params.get("init_distribution")
        _nz = getattr(_idc, "nut_z_rot", None)
        if _nz is not None and getattr(_idc, "enabled", False):
            _nzd = f"{math.degrees(_nz[0]):.0f}-{math.degrees(_nz[1]):.0f}deg"
        else:
            # ⚠️ init_dist DISABLED does NOT mean the full 360 deg: a registered
            # env class can carry the window itself. Square_RealCell_RotW1 does,
            # precisely so MimicGen generation and train_rl.py cannot disagree
            # about it (docs/infra.md §15d) -- and reading the knob alone
            # printed "FULL(0-360)" for a run that was actually 120-180.
            _cls_nz = getattr(_env_cls_for(env_params["env_name"]), "NUT_Z_ROT", None)
            _nzd = (f"{math.degrees(_cls_nz[0]):.0f}-{math.degrees(_cls_nz[1]):.0f}deg(class)"
                    if _cls_nz is not None else "FULL(0-360)")
        _ds = env_params.get("domain_shift")
        print(f"[EVAL ENV] task={env_params['env_name']} "
              f"shift={getattr(_ds, 'mode', None)} "
              f"nut_z_rot={_nzd} "
              f"sigma={env_params.get('hand_init_perturb_joint_sigma')} "
              f"strict={getattr(env_params.get('strict_success'), 'enabled', None)} "
              f"ep_len={env_params.get('episode_length')}", flush=True)
        t = time.time()
        if args.mp >= 1:
            scores = mp_run_eval(
                env_params, agent, args.num_game, args.mp, args.seed,
                verbose=args.verbose,
                record_dir=args.record_dir,
                record_every=args.record_every,
            )
        else:
            scores = run_eval(
                env_params,
                agent,
                args.num_game,
                args.seed,
                args.record_dir,
                verbose=args.verbose,
                record_every=args.record_every,
            )
        all_scores.append(scores)
        num_success = int(sum(s > 0 for s in scores))
        success_rate = num_success / len(scores) if scores else 0.0
        print(f"weight: {weight}")
        print(f"score: {np.mean(scores)}, time: {time.time() - t:.1f}")
        print(f"success rate: {100 * success_rate:.2f}% ({num_success}/{len(scores)})")
        weight_scores.append((weight, np.mean(scores), success_rate))

    if len(weight_scores) > 1:
        weight_scores = sorted(weight_scores, key=lambda x: -x[1])
        scores = []
        success_rates = []
        for weight, score, sr in weight_scores:
            print(f"{weight} -> score {score:.4f}, success {100 * sr:.2f}%")
            scores.append(score)
            success_rates.append(sr)
        print(f"average score: {100 * np.mean(scores):.2f}")
        print(f"max score: {100 * scores[0]:.2f}")
        print(f"average success rate: {100 * np.mean(success_rates):.2f}%")
        print(f"max success rate: {100 * max(success_rates):.2f}%")
        max_score_per_seed = np.array(all_scores).max(0)
        print(f"max over seed: {100 * np.mean(max_score_per_seed):.2f}")
