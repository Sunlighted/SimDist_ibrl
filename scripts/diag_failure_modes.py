"""Diagnose Threading_D0 failure modes under a domain shift.

Runs a trained policy single-process so we can read the raw MuJoCo sim every
control step, then classifies each FAILED episode into:

The PRIMARY axis is how close the needle's tip ever got to the ring center
(min_d_nr) — the same quantity the task's success test uses. "Did it still
hold the needle at the end" is deliberately NOT the classifier, because a
needle that reaches the ring at 1.2 mm and then slips out is an
insertion-phase failure, not a grasp failure.

  GRASP / TRANSPORT phase (needle never reached the ring vicinity):
  - never_picked_up     : needle never lifted off the table   ("一开始没夹住")
  - lost_before_ring    : lifted, but never brought within REACH of the ring
                          (dropped / wandered en route)

  ALIGN / INSERT phase (needle DID reach the ring vicinity):
  - near_miss           : got within REACH but never inside the 12 mm ball
                          ("送到了环口但对不准")
  - strict_rejected      : tip DID enter the 12 mm ball but no accepted success
                          — strict gating rejected it (clip-through / shove /
                          dwell not held) or it slipped back out

This lets you compare against the ~40% grasp / ~60% alignment split you
estimated visually. `grasped_at_end` is still printed per-episode as color.

Usage (from repo root, after `source set_env.sh`):

  python scripts/diag_failure_modes.py \
    --weight exps/mimicgen/threading_ibrl_source_seed1/model0.pt --mode rl \
    --num_game 100 --seed 1 \
    --shift_friction_scale 0.05 --shift_mass_scale 10.0 \
    --shift_solref_scale 1.0 --shift_solimp_scale 1.0 \
    --shift_gripper_kp_scale 0.15 --shift_hole_offset 0,0,0 \
    --shift_ring_size_scale 0.9 --shift_ring_friction_scale 1.5 \
    --strict_success 1

All --shift_* / --strict_* defaults already match the command above, so the
bare `--weight ... --mode rl --num_game 100` reproduces that exact target.
"""

import argparse
import os
import sys

# repo root on path (script lives in scripts/)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import common_utils  # noqa: F401  (sets up nice paths)
import train_rl
import train_bc
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig


# ---- classification thresholds (meters) -----------------------------------
LIFT_THRESH = 0.04   # needle z must rise this far above its rest height to
                     # count as "picked up off the table"
END_WINDOW = 10      # look at the last N control steps to judge end-of-episode
END_HELD_FRAC = 0.5  # >= this fraction of the window grasped => held at end
HOLD_Z = 0.03        # needle still this far above rest at the end also => held
                     # (backstop when _check_grasp is unavailable)
REACH_THRESH = 0.05  # closest needle-ring approach under this => "reached" ring


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str, required=True)
    p.add_argument("--mode", type=str, default="rl", choices=["rl", "bc"])
    p.add_argument("--num_game", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    # shift defaults == the command under investigation
    p.add_argument("--shift_friction_scale", type=float, default=0.05)
    p.add_argument("--shift_mass_scale", type=float, default=10.0)
    p.add_argument("--shift_solref_scale", type=float, default=1.0)
    p.add_argument("--shift_solimp_scale", type=float, default=1.0)
    p.add_argument("--shift_gripper_kp_scale", type=float, default=0.15)
    p.add_argument("--shift_hole_offset", type=str, default="0,0,0")
    p.add_argument("--shift_ring_size_scale", type=float, default=0.9)
    p.add_argument("--shift_ring_friction_scale", type=float, default=1.5)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--strict_dwell_steps", type=int, default=3)
    p.add_argument("--strict_max_tripod_disp", type=float, default=0.04)
    p.add_argument("--strict_max_needle_speed", type=float, default=0.0)
    return p.parse_args()


def build_env_params(args):
    if args.mode == "rl":
        agent, _, env_params = train_rl.load_model(args.weight, "cuda")
    else:
        agent, _, env_params = train_bc.load_model(args.weight, "cuda")

    shift = DomainShiftConfig(
        mode="fixed",
        friction_scale=args.shift_friction_scale,
        mass_scale=args.shift_mass_scale,
        solref_scale=args.shift_solref_scale,
        solimp_scale=args.shift_solimp_scale,
        gripper_kp_scale=args.shift_gripper_kp_scale,
        hole_offset=tuple(float(x) for x in args.shift_hole_offset.split(",")),
        ring_size_scale=args.shift_ring_size_scale,
        ring_friction_scale=args.shift_ring_friction_scale,
    )
    env_params["domain_shift"] = shift
    if args.strict_success:
        env_params["strict_success"] = StrictSuccessConfig(
            enabled=True,
            dwell_steps=args.strict_dwell_steps,
            max_tripod_disp=args.strict_max_tripod_disp,
            max_needle_speed=args.strict_max_needle_speed,
        )
    print(f"[diag] domain_shift  = {shift}")
    print(f"[diag] strict_success = {env_params.get('strict_success')}")
    assert env_params["env_name"] == "Threading_D0", \
        f"this diagnostic is Threading_D0 only, got {env_params['env_name']}"
    return agent, env_params


class SimProbe:
    """Reads needle / ring / grasp state from the raw robosuite env.

    robosuite uses hard_reset=True, so the MjSim object is REBUILT on every
    env.reset(). We therefore never cache the sim/data — always read it live
    via self.raw.sim — and re-resolve geom ids each episode in refresh().
    """

    def __init__(self, pixel_env: PixelRobosuite):
        self.raw = pixel_env.env  # robosuite Threading_D0
        self.radius = float(self.raw.tripod.ring_size[1])  # success radius (~12 mm)
        self.needle_gid = None
        self.ring_gids = None
        # grasp check: env's own definition (gripper fingerpads vs needle geoms)
        self._can_grasp = True
        try:
            _ = self.raw.needle.contact_geoms
        except Exception:
            self._can_grasp = False

    def refresh(self):
        """Re-resolve geom ids after a (hard) reset rebuilt the sim."""
        m = self.raw.sim.model
        self.needle_gid = m.geom_name2id("needle_obj_needle")
        self.ring_gids = [
            m.geom_name2id(f"tripod_obj_ring_{i}")
            for i in range(self.raw.tripod.num_ring_geoms)
        ]

    def needle_pos(self):
        return np.array(self.raw.sim.data.geom_xpos[self.needle_gid])

    def ring_center(self):
        return np.mean(self.raw.sim.data.geom_xpos[self.ring_gids], axis=0)

    def dist_needle_ring(self):
        return float(np.linalg.norm(self.needle_pos() - self.ring_center()))

    def grasped(self) -> bool:
        if not self._can_grasp:
            return False
        try:
            return bool(self.raw._check_grasp(
                gripper=self.raw.robots[0].gripper,
                object_geoms=self.raw.needle.contact_geoms,
            ))
        except Exception:
            return False


def classify(ep, radius):
    """ep: dict of per-episode aggregates. Returns a bucket name.

    Primary axis = min_d_nr (closest tip-to-ring approach). Whether the needle
    is still held at the end is intentionally NOT used here.
    """
    if ep["success"]:
        return "success"
    if ep["min_d_nr"] < radius:
        # tip entered the 12 mm ball at some frame but no accepted success:
        # strict gating rejected it / it slipped back out.
        return "strict_rejected"
    if ep["min_d_nr"] <= REACH_THRESH:
        # got into the ring vicinity but never inside the ball.
        return "near_miss"
    # never reached the ring vicinity => grasp / transport phase failure.
    if not ep["ever_lifted"]:
        return "never_picked_up"
    return "lost_before_ring"


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    common_utils.set_all_seeds(args.seed)

    agent, env_params = build_env_params(args)
    env = PixelRobosuite(**env_params)
    probe = SimProbe(env)

    buckets = {
        "success": 0, "never_picked_up": 0, "lost_before_ring": 0,
        "near_miss": 0, "strict_rejected": 0,
    }
    episodes = []

    with torch.no_grad(), common_utils.ibrl_utils.eval_mode(agent):
        for g in range(args.num_game):
            np.random.seed(args.seed + g)
            obs, _ = env.reset()
            probe.refresh()

            z0 = probe.needle_pos()[2]
            max_z = z0
            min_d_nr = probe.dist_needle_ring()
            grasp_hist = []
            z_hist = []

            terminal = False
            success = False
            while not terminal:
                action = agent.act(obs, eval_mode=True)
                obs, _reward, terminal, success, _ = env.step(action)
                npos = probe.needle_pos()
                max_z = max(max_z, npos[2])
                min_d_nr = min(min_d_nr, probe.dist_needle_ring())
                grasp_hist.append(probe.grasped())
                z_hist.append(npos[2])

            tail = grasp_hist[-END_WINDOW:] if grasp_hist else [False]
            tail_z = z_hist[-END_WINDOW:] if z_hist else [z0]
            held_at_end = (
                (sum(tail) / len(tail) >= END_HELD_FRAC)
                or (np.mean(tail_z) - z0 > HOLD_Z)
            )
            ep = {
                "success": bool(success),
                "ever_lifted": (max_z - z0) > LIFT_THRESH,
                "held_at_end": bool(held_at_end),
                "min_d_nr": min_d_nr,
                "max_lift": max_z - z0,
                "len": env.time_step,
            }
            bucket = classify(ep, probe.radius)
            ep["bucket"] = bucket
            buckets[bucket] += 1
            episodes.append(ep)
            print(f"  game {g:3d} seed {args.seed+g:3d}: {bucket:18s} "
                  f"lift={ep['max_lift']*100:+5.1f}cm  "
                  f"min|n-ring|={min_d_nr*1000:6.1f}mm  "
                  f"held_end={'Y' if held_at_end else 'n'}  len={ep['len']}")

    # ---- summary ----------------------------------------------------------
    n = args.num_game
    n_succ = buckets["success"]
    n_fail = n - n_succ
    print("\n" + "=" * 60)
    print(f"success radius = {probe.radius*1000:.1f} mm | grasp_check = "
          f"{'ON' if probe._can_grasp else 'OFF (z-only fallback)'}")
    print(f"games: {n}   success: {n_succ} ({100*n_succ/n:.1f}%)   "
          f"fail: {n_fail} ({100*n_fail/n:.1f}%)")
    print("-" * 60)
    print("failure breakdown (% of FAILURES):")

    def pct_fail(k):
        return 100 * buckets[k] / n_fail if n_fail else 0.0

    grasp_phase = buckets["never_picked_up"] + buckets["lost_before_ring"]
    align_phase = buckets["near_miss"] + buckets["strict_rejected"]
    print(f"  REACH threshold = {REACH_THRESH*1000:.0f} mm")
    for k in ("never_picked_up", "lost_before_ring",
              "near_miss", "strict_rejected"):
        print(f"  {k:22s} {buckets[k]:4d}   {pct_fail(k):5.1f}%")
    print("-" * 60)
    print(f"  GRASP/TRANSPORT (never_picked_up + lost_before_ring): "
          f"{grasp_phase:4d}   {100*grasp_phase/n_fail if n_fail else 0:5.1f}%")
    print(f"  ALIGN/INSERT    (near_miss + strict_rejected)       : "
          f"{align_phase:4d}   {100*align_phase/n_fail if n_fail else 0:5.1f}%")
    print(f"    of which reached the 12mm ball (strict_rejected)  : "
          f"{buckets['strict_rejected']:4d}   "
          f"{pct_fail('strict_rejected'):5.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
