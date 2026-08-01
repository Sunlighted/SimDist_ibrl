"""Verify that mode="fixed" with all-neutral shift values produces a
*different* observation than mode="none", despite being physically
identical. Confirms the FP-non-determinism hypothesis for the 80% vs 73%
score gap on Threading_D0.

We compare three things between path A (mode=None) and path B (mode=fixed
with all defaults=1.0 / 0):
  1. The flattened obs vector right after reset.
  2. sim.data.qpos and sim.data.qvel (the canonical physics state).
  3. Obs trajectory after 5 zero-action steps (to see if the divergence
     amplifies under dynamics).
"""
import os
import sys

import numpy as np
import torch


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    def make_env(shift):
        return PixelRobosuite(
            env_name="Threading_D0",
            robots=["Panda"],
            episode_length=400,
            image_size=96, rl_image_size=96,
            camera_names=["agentview"], rl_cameras=["agentview"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_state=True, obs_stack=1, state_stack=1, prop_stack=1,
            domain_shift=shift,
        )

    # Path A: mode=none → wrapper short-circuits, no extra forward()/force_update
    print("=" * 60)
    print("Building env A (mode=none, short-circuit path)...")
    env_a = make_env(None)

    # Path B: mode=fixed with ALL DEFAULTS = 1.0 / 0 (physically identical)
    print("Building env B (mode=fixed, all-neutral shift, extra forward+force_update)...")
    env_b = make_env(DomainShiftConfig(mode="fixed"))  # defaults are 1.0 / (0,0,0)

    def state_obs(obs):
        # obs is a dict, "state" is the flattened state tensor used by the policy
        s = obs["state"]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        return np.asarray(s, dtype=np.float64)

    for seed in [1, 2, 3]:
        print()
        print(f"------------ seed={seed} ------------")
        np.random.seed(seed); obs_a, _ = env_a.reset()
        np.random.seed(seed); obs_b, _ = env_b.reset()

        s_a, s_b = state_obs(obs_a), state_obs(obs_b)
        diff = s_a - s_b
        print(f"state.shape       = {s_a.shape}")
        print(f"max abs diff      = {np.abs(diff).max():.3e}")
        print(f"L2 norm of diff   = {np.linalg.norm(diff):.3e}")
        if np.abs(diff).max() > 0:
            argmax = int(np.argmax(np.abs(diff)))
            print(f"largest-diff idx  = {argmax}  (s_a={s_a[argmax]:.10f}  s_b={s_b[argmax]:.10f})")

        # Canonical physics state (qpos / qvel directly from the sim).
        qpos_a = env_a.env.sim.data.qpos.copy()
        qpos_b = env_b.env.sim.data.qpos.copy()
        qvel_a = env_a.env.sim.data.qvel.copy()
        qvel_b = env_b.env.sim.data.qvel.copy()
        print(f"qpos max diff     = {np.abs(qpos_a - qpos_b).max():.3e}  "
              f"(shape {qpos_a.shape})")
        print(f"qvel max diff     = {np.abs(qvel_a - qvel_b).max():.3e}")

        # Roll out 5 zero-action steps and compare again — does the FP
        # divergence amplify under dynamics?
        zero_action = torch.zeros(env_a.action_dim)
        for t in range(5):
            obs_a, _, _, _, _ = env_a.step(zero_action)
            obs_b, _, _, _, _ = env_b.step(zero_action)
        s_a5, s_b5 = state_obs(obs_a), state_obs(obs_b)
        diff5 = s_a5 - s_b5
        print(f"after 5 zero-steps: max abs diff = {np.abs(diff5).max():.3e}, "
              f"L2 = {np.linalg.norm(diff5):.3e}")


if __name__ == "__main__":
    main()
