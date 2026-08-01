"""Phase-resolved contact diagnostic for a Threading_D0 BC policy.

Splits the task into the two contact phases the user cares about and reports a
success rate for each, IN ISOLATION:

  PHASE 1 — GRASP / PICK-UP (from the normal initial state):
    reset() the env (target domain shift applied), roll the policy out, and ask
    "did it cleanly pick the needle up and hold it in the air?". Success =
    the needle is BOTH grasped (env._check_grasp) AND lifted > LIFT_THRESH off
    its rest height, held for HOLD_STREAK consecutive control steps.

  PHASE 2 — INSERT (from a 'needle already grasped, just reached the ring'
  state restored from a TARGET demo trajectory):
    for each of the 20 target demos we scan the recorded sim_state frames,
    pick the first frame where the needle is grasped AND within APPROACH_DIST
    of the ring center (the pre-insertion pose), restore that exact MuJoCo
    state into the (shifted) env, hand control to the policy, and ask "did it
    thread the needle into the ring?". Success = raw task success (needle
    within the ~12 mm ball). Each demo state is rolled out N_PER_STATE times
    (the flow policy samples z~N(0,I), so it is stochastic) to get a rate.

Both phases use the SAME target domain shift the policy was evaluated under
(read from the run's cfg.yaml `eval_domain_shift`), EMA weights applied, and
raw success (strict_success OFF — matching the training-eval metric).

Usage (from repo root):
  source set_env.sh   # or: conda activate ibrl; export PYTHONPATH=$PWD
  python scripts/diag_contact_phases.py \
      --run exps/mimicgen/threading_flow_lowdata_target_seed1 \
      --grasp_games 50 --insert_per_state 5
"""

import argparse
import os
import sys
from collections import deque

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np
import torch
import pyrallis

import common_utils
import train_bc
from train_bc import MainConfig
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig


# ---- thresholds (meters) --------------------------------------------------
LIFT_THRESH = 0.04      # needle rises this far above rest => "picked up"
HOLD_STREAK = 5         # consecutive (grasped & lifted) steps => stable pick
GRASP_HORIZON = 200     # cap phase-1 rollout (grasp happens early)
APPROACH_DIST = 0.06    # needle-ring dist of the restored pre-insertion frame
INSERT_HORIZON = 150    # cap phase-2 rollout (only the insertion remains)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str,
                   default="exps/mimicgen/threading_flow_lowdata_target_seed1")
    p.add_argument("--weight", type=str, default="model0.pt",
                   help="checkpoint filename inside --run")
    p.add_argument("--demos", type=str,
                   default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--grasp_games", type=int, default=50)
    p.add_argument("--insert_per_state", type=int, default=5)
    p.add_argument("--max_demos", type=int, default=0,
                   help="cap the number of demo states used for the insert "
                        "phase; 0 = all. >0 subsamples evenly across the file.")
    p.add_argument("--skip_grasp", action="store_true")
    p.add_argument("--approach_dist", type=float, default=APPROACH_DIST,
                   help="needle-ring distance (m) of the restored pre-insertion "
                        "state; chosen from frames before insertion. Larger = "
                        "further out / needle clearly outside the ring.")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


class SimProbe:
    """Live reads of needle / ring / grasp from the raw robosuite env."""

    def __init__(self, pixel_env: PixelRobosuite):
        self.raw = pixel_env.env
        self.radius = float(self.raw.tripod.ring_size[1])  # ~12 mm success ball
        self.needle_gid = None
        self.ring_gids = None
        try:
            _ = self.raw.needle.contact_geoms
            self._can_grasp = True
        except Exception:
            self._can_grasp = False

    def refresh(self):
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

    def dist(self):
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


def policy_chunk(agent, obs):
    """Single-env act_chunk wrapper (matches evaluate/eval.py)."""
    if hasattr(agent, "act_chunk"):
        return agent.act_chunk(obs, eval_mode=True)
    return agent.act(obs, eval_mode=True).unsqueeze(0)


def load_env_and_agent(args, device="cuda"):
    weight = os.path.join(args.run, args.weight)
    cfg_path = os.path.join(args.run, "cfg.yaml")
    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

    agent, _env_throwaway, env_params = train_bc.load_model(weight, device, verbose=False)
    # apply the SAME target shift the policy was evaluated under
    env_params["domain_shift"] = cfg.eval_domain_shift
    print(f"[diag] eval_domain_shift = {cfg.eval_domain_shift}")

    # EMA weights (matches in-training eval)
    if hasattr(agent, "apply_ema"):
        applied = bool(agent.apply_ema())
        print(f"[diag] EMA applied: {applied}")
    return agent, env_params


# ===========================================================================
# PHASE 1 — grasp / pick-up
# ===========================================================================
def run_grasp_phase(agent, env_params, args):
    env = PixelRobosuite(**env_params)
    probe = SimProbe(env)
    n_success = 0
    print("\n" + "=" * 64)
    print("PHASE 1 — GRASP / PICK-UP (from normal reset)")
    print("=" * 64)
    with torch.no_grad(), common_utils.ibrl_utils.eval_mode(agent):
        for g in range(args.grasp_games):
            np.random.seed(args.seed + g)
            obs, _ = env.reset()
            probe.refresh()
            z0 = probe.needle_pos()[2]

            streak = 0
            picked = False
            max_lift = 0.0
            queue: deque = deque()
            terminal = False
            step = 0
            while not terminal and step < GRASP_HORIZON:
                if not queue:
                    for a in policy_chunk(agent, obs):
                        queue.append(a)
                action = queue.popleft()
                obs, _r, terminal, _succ, _ = env.step(action)
                step += 1
                lift = probe.needle_pos()[2] - z0
                max_lift = max(max_lift, lift)
                if probe.grasped() and lift > LIFT_THRESH:
                    streak += 1
                    if streak >= HOLD_STREAK:
                        picked = True
                        break
                else:
                    streak = 0
            n_success += int(picked)
            print(f"  game {g:3d} seed {args.seed+g:3d}: "
                  f"{'PICKED ' if picked else 'no-pick'}  "
                  f"max_lift={max_lift*100:+5.1f}cm  steps={step}")
    rate = 100 * n_success / args.grasp_games
    print("-" * 64)
    print(f"GRASP success: {n_success}/{args.grasp_games} = {rate:.1f}%")
    return rate


# ===========================================================================
# PHASE 2 — insertion (restore pre-insertion state from each demo)
# ===========================================================================
def find_preinsertion_frames(env, probe, demos_path, max_demos=0,
                             target_dist=APPROACH_DIST):
    """For each demo pick a grasped frame whose needle-ring distance is closest
    to `target_dist`, RESTRICTED to frames BEFORE the needle ever enters the
    ring (so the chosen state is needle-still-outside, not-yet-inserted).

    Larger target_dist => reset further from the ring (more of the approach is
    left to the policy); smaller => closer to the hole.
    """
    frames = []
    with h5py.File(demos_path, "r") as f:
        grp = f["data"]
        demo_keys = sorted(grp.keys(), key=lambda k: int(k.split("_")[1]))
        if max_demos and max_demos < len(demo_keys):
            # subsample evenly across the file (not just the first N)
            idx = np.linspace(0, len(demo_keys) - 1, max_demos).round().astype(int)
            demo_keys = [demo_keys[i] for i in sorted(set(idx))]
        print(f"  scanning {len(demo_keys)} demo trajectories; target "
              f"pre-insert dist = {target_dist*1000:.0f}mm (needle outside ring)")
        # one reset so the sim exists + shift applied; restore states into it
        env.reset()
        probe.refresh()
        for dk in demo_keys:
            ss = grp[dk]["obs"]["sim_state"][:]  # (T, 45)
            # per-frame (dist, grasped); find the first insertion frame so we
            # only consider the approach portion (needle still outside).
            dists, grasped = [], []
            insert_t = ss.shape[0]
            for t in range(ss.shape[0]):
                env.env.sim.set_state_from_flattened(ss[t])
                env.env.sim.forward()
                d = probe.dist()
                dists.append(d)
                grasped.append(probe.grasped())
                if d < probe.radius and insert_t == ss.shape[0]:
                    insert_t = t  # needle entered the ring here; stop approach
            # candidate frames: grasped, before insertion, outside the ring
            cand = [t for t in range(insert_t)
                    if grasped[t] and dists[t] >= probe.radius]
            if not cand:
                cand = [t for t in range(insert_t) if grasped[t]] or \
                       list(range(insert_t)) or [0]
            # pick the one closest to target_dist
            best_t = min(cand, key=lambda t: abs(dists[t] - target_dist))
            chosen, chosen_d = ss[best_t], dists[best_t]
            frames.append((dk, chosen, chosen_d))
            print(f"  {dk}: reset dist = {chosen_d*1000:.1f}mm "
                  f"(frame {best_t}/{ss.shape[0]}, insert@{insert_t})")
    return frames


def restore_state(env, probe, sim_state):
    """Put the env at sim_state and produce a fresh policy obs."""
    env.time_step = 0
    env.terminal = False
    env.past_obses.clear()
    # the underlying robosuite env may still carry done=True / a stale
    # timestep from the previous (terminated) rollout — clear them so step()
    # doesn't raise "executing action in terminated episode".
    env.env.done = False
    env.env.timestep = 0
    env.env.sim.set_state_from_flattened(sim_state)
    env.env.sim.forward()
    raw_obs = env.env._get_observations(force_update=True)
    rl_obs, _ = env._extract_images(raw_obs)
    return rl_obs


def run_insert_phase(agent, env_params, args):
    env = PixelRobosuite(**env_params)
    probe = SimProbe(env)
    print("\n" + "=" * 64)
    print("PHASE 2 — INSERT (restored pre-insertion states from demos)")
    print(f"  APPROACH_DIST={APPROACH_DIST*100:.0f}cm  "
          f"success ball radius={probe.radius*1000:.1f}mm  "
          f"{args.insert_per_state} rollouts/demo-state")
    print("=" * 64)
    frames = find_preinsertion_frames(env, probe, args.demos, args.max_demos,
                                      target_dist=args.approach_dist)

    total, total_succ = 0, 0
    per_demo = []
    with torch.no_grad(), common_utils.ibrl_utils.eval_mode(agent):
        for dk, sim_state, d0 in frames:
            if sim_state is None:
                continue
            succ_here = 0
            min_ds = []
            grasp_start = 0
            grasp_end = 0
            for r in range(args.insert_per_state):
                np.random.seed(args.seed + 1000 + total)
                obs = restore_state(env, probe, sim_state)
                grasp_start += int(probe.grasped())  # held right after restore?
                min_d = probe.dist()
                queue: deque = deque()
                terminal = False
                success = False
                step = 0
                while not terminal and step < INSERT_HORIZON:
                    if not queue:
                        for a in policy_chunk(agent, obs):
                            queue.append(a)
                    action = queue.popleft()
                    obs, _r, terminal, success, _ = env.step(action)
                    step += 1
                    min_d = min(min_d, probe.dist())
                    if success:
                        break
                grasp_end += int(probe.grasped())
                min_ds.append(min_d)
                succ_here += int(success)
                total += 1
                total_succ += int(success)
            rate = 100 * succ_here / args.insert_per_state
            per_demo.append((dk, rate))
            n = args.insert_per_state
            print(f"  {dk}: insert {succ_here}/{n} = {rate:3.0f}%   "
                  f"start {d0*1000:.0f}mm  min_reached {np.mean(min_ds)*1000:5.1f}mm  "
                  f"grasp@start {grasp_start}/{n}  grasp@end {grasp_end}/{n}")
    overall = 100 * total_succ / total if total else 0.0
    print("-" * 64)
    print(f"INSERT success: {total_succ}/{total} = {overall:.1f}%")
    return overall


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    common_utils.set_all_seeds(args.seed)

    agent, env_params = load_env_and_agent(args)

    grasp_rate = None if args.skip_grasp else run_grasp_phase(agent, env_params, args)
    insert_rate = run_insert_phase(agent, env_params, args)

    print("\n" + "#" * 64)
    print(f"# RUN:    {args.run}")
    print(f"# DEMOS:  {args.demos}")
    if grasp_rate is not None:
        print(f"# PHASE 1  GRASP  success rate : {grasp_rate:.1f}%")
    print(f"# PHASE 2  INSERT success rate : {insert_rate:.1f}%")
    print("#" * 64)


if __name__ == "__main__":
    main()
