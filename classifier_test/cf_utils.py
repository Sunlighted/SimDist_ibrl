"""Shared utilities for the counterfactual-pairing dynamics-gap experiment.

The two envs are both simulators, but the gap-pairing only needs the SOURCE
(nominal) env: for a target-demo transition (s_t, a_t, s'_tgt), we reset the
SOURCE sim to the demo's sim_state_t, apply a_t, and read s'_src. Because the
(s, a) is identical, ||s'_tgt - s'_src|| is the pure dynamics gap.

`sim_state` (45-dim flattened MjSimState = [time, qpos, qvel]) is stored in the
demo obs and fully restores the physical configuration via
`sim.set_state_from_flattened`. Model parameters (mass / friction / ring geom
size) are NOT in sim_state, so the source env keeps its nominal physics — which
is exactly the dynamics difference we want to measure.
"""

import os
import sys

import numpy as np
import h5py

REPO = "/home/Chenyang/ibrl"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

# 37-dim state layout for Threading_D0 (matches DEFAULT_STATE_KEYS in the wrapper)
THREADING_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state"]


def extract_state(obs):
    return np.concatenate([obs[k] for k in THREADING_STATE_KEYS]).astype(np.float32)


# Dynamics-only target shift (matches threading_d0_target_dynamics/env_cfg.json
# and CLAUDE.md §12 — the recipe the target demos were collected under).
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
    mask_hole_offset_in_obs=False, ring_size_scale=0.85, ring_friction_scale=2.0,
)

# cache env_params per weight so source+target envs share one load_model call
_ENV_PARAMS_CACHE: dict = {}


def _env_params(weight):
    if weight not in _ENV_PARAMS_CACHE:
        import train_rl
        _, _, env_params = train_rl.load_model(weight, "cuda")
        _ENV_PARAMS_CACHE[weight] = env_params
    return dict(_ENV_PARAMS_CACHE[weight])


def get_env(weight, domain="source", hand_init_sigma=None):
    """Build a PixelRobosuite from a checkpoint's env_params.

    domain: "source" -> mode=none nominal; "target" -> dynamics-only shift.
    hand_init_sigma: override joint-space init gaussian (for wider rollouts).
    """
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    params = _env_params(weight)
    params["domain_shift"] = (DomainShiftConfig(mode="none") if domain == "source"
                              else DomainShiftConfig(**TARGET_SHIFT))
    if hand_init_sigma is not None:
        params["hand_init_perturb"] = None
        params["hand_init_perturb_joint_sigma"] = hand_init_sigma
    env = PixelRobosuite(**params)
    env.reset()  # build the sim once; set_state per-transition afterwards
    return env


def get_source_env(weight, shift_mode="none"):
    """Back-compat: nominal source env."""
    return get_env(weight, domain="source")


def get_sim_state(env):
    """Flattened MjSimState of the env's current physical configuration."""
    return env.env.sim.get_state().flatten().copy()


def counterfactual_step(env, sim_state, action):
    """Set the SOURCE sim to `sim_state`, apply `action`, return next 37-dim state."""
    raw = env.env
    # Each counterfactual is an INDEPENDENT single step from sim_state. Clear
    # robosuite's done/horizon bookkeeping so a prior success doesn't block the
    # next set_state+step ("executing action in terminated episode").
    raw.done = False
    raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    # set_state restores only (time, qpos, qvel); the constraint-solver warmstart
    # (qacc_warmstart) is NOT restored and otherwise carries over from whatever the
    # sim did before, seeding chaotic divergence in stiff contact configs (so the
    # SAME sim_state+action would NOT reproduce). Zero it for a deterministic
    # cold-start solve -> the counterfactual is reproducible and the gap is pure.
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    obs, reward, done, _ = raw.step(np.asarray(action, dtype=np.float64))
    return extract_state(obs), float(reward), bool(done)


def reconstructed_state(env, sim_state):
    """Set sim to sim_state and read back the 37-dim state (sanity check that
    set_state faithfully restores the demo state)."""
    raw = env.env
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0   # deterministic cold-start (see counterfactual_step)
    raw.sim.forward()
    obs = raw._get_observations(force_update=True)
    return extract_state(obs)


def load_demo_episodes(hdf5_path, max_episodes=None):
    """Yield per-episode dicts with state (T,37), action (T,7), sim_state (T,45)."""
    eps = []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        if max_episodes is not None:
            keys = keys[:max_episodes]
        for k in keys:
            g = data[k]
            eps.append(dict(
                state=np.asarray(g["obs"]["state"], dtype=np.float32),
                action=np.asarray(g["actions"], dtype=np.float32),
                sim_state=np.asarray(g["obs"]["sim_state"], dtype=np.float64),
                rewards=np.asarray(g["rewards"], dtype=np.float32),
            ))
    return eps
