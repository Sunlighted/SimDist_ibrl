"""Measure the OBS-CONDITIONAL stochasticity of the Threading transition.

Premise (verified): the 37-dim obs contains NO velocities (env/robosuite_wrapper.py:
DEFAULT_STATE_KEYS = eef_pos/eef_quat/gripper_qpos/object; the *_vel keys are
commented out). So the same observed (obs, a) can hide DIFFERENT true velocities,
and over one 0.05s control step velocity*dt is mm-scale -> the SAME (obs, a) maps to
DIFFERENT s'. That makes p(s'|obs, a) a genuine mixture, which an MSE deterministic
model averages into a physically-wrong "between-modes" state (a persistent bias).

This script MEASURES that spread directly, exploiting the queryable sim:
  * roll a policy, collect (sim_state, obs, action, needle-ring distance);
  * for each probe, KEEP qpos (=> obs unchanged) but overwrite qvel with velocities
    drawn from OTHER real steps at the SAME phase (real p(qvel|obs)), and re-step the
    SAME action (the obs-only policy would pick the same action regardless of the
    hidden velocity);
  * the spread of the resulting next needle-position IS the obs-conditional
    stochasticity that any obs-only dynamics model must contend with.

Report, split by free-space vs contact/insert phase:
  * mean positional spread (mm) of the next needle pos across velocity variants,
  * fraction of probes whose spread exceeds the 1.8mm target tolerance window,
  * a crude multimodality flag (max-gap / std).

Interpretation:
  spread >> 1.8mm at contact  =>  obs is insufficient (non-Markov); an MSE model is
    fundamentally biased and even a perfect generative head leaves irreducible
    aleatoric spread > window. Cheaper real fix: ADD velocity to obs (make it Markov).
  spread << 1.8mm             =>  transition is ~deterministic-from-obs; the model's
    persistent error is function-approx/chaos, not mode-averaging => a generative head
    buys little.
  spread ~ window, multimodal =>  the sweet spot where a generative head (sample a
    mode, not the mean) strictly beats MSE.
"""
import os
import sys
import argparse

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.domain_shift import DomainShiftConfig  # noqa: E402

ERR = sys.stderr
NEEDLE = slice(9, 12)    # needle body pos in the 37-dim obs (§17)
RING = slice(23, 26)     # tripod/ring body pos in the 37-dim obs (§17)


def log(*a):
    print(*a, file=ERR, flush=True)


def restore(env, sim_state, qvel=None):
    """Reset the sim to sim_state; optionally overwrite qvel (keeping qpos => obs
    unchanged). qacc_warmstart=0 for a clean, reproducible step."""
    raw = env.env
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    if qvel is not None:
        st = raw.sim.get_state()
        st.qvel[:] = qvel
        raw.sim.set_state(st)
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    env.time_step = 0
    if getattr(env, "_success_validator", None) is not None:
        env._success_validator.reset(raw, env._bindings)


@torch.no_grad()
def step_needle(env, sim_state, action, qvel=None):
    """Restore (optionally with a swapped qvel), step the action, return next needle pos."""
    restore(env, sim_state, qvel=qvel)
    obs, _, _, _, _ = env.step(torch.as_tensor(action))
    return obs["state"].cpu().numpy()[NEEDLE].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--shift", default="target", choices=["source", "target"])
    ap.add_argument("--episodes", type=int, default=20, help="rollout episodes to gather states")
    ap.add_argument("--num_probe", type=int, default=120, help="probes per phase (contact/free)")
    ap.add_argument("--num_vel", type=int, default=16, help="velocity variants per probe")
    ap.add_argument("--contact_thresh", type=float, default=0.05, help="needle-ring dist < this = contact/insert")
    ap.add_argument("--free_thresh", type=float, default=0.12, help="needle-ring dist > this = free space")
    ap.add_argument("--vel_mode", default="resample", choices=["resample", "zero", "noise", "same"],
                    help="resample = qvel from other real steps at the same phase (honest p(qvel|obs)); "
                         "zero = zero out qvel; noise = real qvel + gaussian(scale=per-dim qvel std); "
                         "same = CONTROL: do NOT change qvel (restore the identical full state M times) "
                         "-> should give ~0 spread if the sim+restore is deterministic; any residual is "
                         "uncontrolled state (e.g. OSC controller carry-over), the noise floor of the measurement")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0
    cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/obsstoch_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt)))
    model = ws.agent.to("cuda").model

    ep = dict(ws.eval_env_params)
    if args.shift == "source":
        ep["domain_shift"] = DomainShiftConfig(mode="none")
    log(f"[setup] shift={args.shift} ({ep['domain_shift'].mode}), vel_mode={args.vel_mode}")
    env = PixelRobosuite(**ep)
    ep_len = int(ws.eval_env_params.get("episode_length", 400))

    # ---- gather (sim_state, action, qvel, phase) along policy rollouts ----
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    steps = []
    for e in range(args.episodes):
        obs, _ = env.reset()
        for t in range(ep_len):
            st = obs["state"].cpu().numpy()
            dist = float(np.linalg.norm(st[NEEDLE] - st[RING]))
            with torch.no_grad():
                a = model.policy_action(obs["state"].view(1, -1)).view(-1).clamp(-1, 1)
            a_np = a.detach().cpu().numpy()
            steps.append({
                "sim_state": env.env.sim.get_state().flatten().copy(),
                "qvel": env.env.sim.get_state().qvel.copy(),
                "action": a_np, "dist": dist, "needle": st[NEEDLE].copy(),
            })
            obs, _, term, _, _ = env.step(a.detach().cpu())
            if term:
                break
    log(f"[gather] {len(steps)} steps from {args.episodes} episodes")

    dists = np.array([s["dist"] for s in steps])
    qvel_std = np.std(np.stack([s["qvel"] for s in steps]), axis=0)
    log(f"[gather] needle-ring dist: min {dists.min():.3f} med {np.median(dists):.3f} "
        f"max {dists.max():.3f} | mean|qvel| per-dim {np.abs(np.stack([s['qvel'] for s in steps])).mean():.3f}")

    # Phase split by PERCENTILE of needle-ring dist (robust to the constant body-frame
    # offset between the obs tripod-body origin and the success ring-center geom):
    # bottom 15% = closest-approach / alignment-insertion phase (mm precision matters);
    # top 30% = free-space reaching.
    lo = np.percentile(dists, 15)
    hi = np.percentile(dists, 70)
    contact_idx = np.where(dists < lo)[0]
    free_idx = np.where(dists > hi)[0]
    log(f"[gather] phase split by percentile: contact = dist<{lo:.3f} ({len(contact_idx)} steps) | "
        f"free = dist>{hi:.3f} ({len(free_idx)} steps)")

    def vel_pool(phase_idx):
        return [steps[i]["qvel"] for i in phase_idx]

    def run_phase(name, phase_idx):
        if len(phase_idx) < 3:
            log(f"[{name}] too few steps ({len(phase_idx)}), skip")
            return
        pool = vel_pool(phase_idx)
        rng = np.random.RandomState(args.seed + 1)
        probes = phase_idx if len(phase_idx) <= args.num_probe else \
            rng.choice(phase_idx, args.num_probe, replace=False)
        spreads, over_win, multimodal = [], 0, 0
        for pi in probes:
            s = steps[pi]
            outs = [step_needle(env, s["sim_state"], s["action"], qvel=None)]  # real qvel
            for _ in range(args.num_vel):
                if args.vel_mode == "same":
                    qv = None  # CONTROL: restore the identical full state (no qvel change)
                elif args.vel_mode == "zero":
                    qv = np.zeros_like(s["qvel"])
                elif args.vel_mode == "noise":
                    qv = s["qvel"] + rng.randn(*s["qvel"].shape) * qvel_std
                else:  # resample a real qvel from the same phase
                    qv = pool[rng.randint(len(pool))]
                outs.append(step_needle(env, s["sim_state"], s["action"], qvel=qv))
            outs = np.stack(outs)                      # [M+1, 3]
            centroid = outs.mean(0)
            radii = np.linalg.norm(outs - centroid, axis=1)   # per-variant dist to centroid (m)
            spread_mm = radii.mean() * 1000.0
            spreads.append(spread_mm)
            if spread_mm > 1.8:
                over_win += 1
            # crude multimodality: max pairwise gap vs std along the principal spread
            rng_mm = (radii.max()) * 1000.0
            if rng_mm > 3.0 * (radii.std() * 1000.0 + 1e-6):
                multimodal += 1
        spreads = np.array(spreads)
        log(f"\n[{name}] n_probe={len(probes)}  num_vel={args.num_vel}  (spread = RMS of next-needle-pos across obs-consistent velocity variants)")
        log(f"[{name}]   spread mm: mean {spreads.mean():.2f}  median {np.median(spreads):.2f}  "
            f"p90 {np.percentile(spreads,90):.2f}  max {spreads.max():.2f}")
        log(f"[{name}]   fraction of probes with spread > 1.8mm (target window): {over_win/len(probes):.1%}")
        log(f"[{name}]   multimodal-ish probes: {multimodal/len(probes):.1%}")

    run_phase("CONTACT", contact_idx)
    run_phase("FREE", free_idx)

    log("\n===== VERDICT GUIDE =====")
    log("  contact spread >> 1.8mm  => obs non-Markov; MSE biased; generative head leaves")
    log("                              irreducible aleatoric > window. Cheaper fix: add velocity to obs.")
    log("  contact spread << 1.8mm  => ~deterministic-from-obs; generative head buys little.")
    os._exit(0)


if __name__ == "__main__":
    main()
