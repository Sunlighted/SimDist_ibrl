"""Load (s, a, s') transitions from ibrl/MimicGen hdf5 demo files."""

import h5py
import numpy as np


def load_transitions(hdf5_path, state_key="state", max_episodes=None):
    """Return per-episode (s, a, s') arrays from a demo hdf5.

    Each episode contributes transitions (obs[t], action[t], obs[t+1]) for
    t in [0, T-1]. Returns (s, a, ns, episode_ids) stacked over all episodes.
    `state_key` selects which obs field to use as the state (default the
    37-dim "state" used by the RL agents).
    """
    s_list, a_list, ns_list, ep_list = [], [], [], []
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        # episode keys are demo_0, demo_1, ... — sort numerically.
        eps = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        if max_episodes is not None:
            eps = eps[:max_episodes]
        for ei, ep in enumerate(eps):
            g = data[ep]
            state = np.asarray(g["obs"][state_key], dtype=np.float32)
            action = np.asarray(g["actions"], dtype=np.float32)
            T = min(state.shape[0], action.shape[0]) - 1
            if T <= 0:
                continue
            s_list.append(state[:T])
            a_list.append(action[:T])
            ns_list.append(state[1 : T + 1])
            ep_list.append(np.full(T, ei, dtype=np.int64))
    s = np.concatenate(s_list, axis=0)
    a = np.concatenate(a_list, axis=0)
    ns = np.concatenate(ns_list, axis=0)
    ep = np.concatenate(ep_list, axis=0)
    return s, a, ns, ep


# State layout for Threading_D0 (37-dim "state"):
#   eef_pos[0:3] eef_quat[3:7] gripper_qpos[7:9] object[9:37]
# object[9:37] = needle[pos(3)quat(4)to_eef_pos(3)to_eef_quat(4)] then
#                tripod[pos(3)...]. So needle body pos = state[9:12],
# tripod (ring) body pos = state[23:26] (verified empirically).
NEEDLE_POS_SLICE = slice(9, 12)
TRIPOD_POS_SLICE = slice(23, 26)


# Quaternion blocks in the 37-dim state (double-cover: q and -q are the same
# rotation but differ by 2.0 in L2, which spuriously inflates ||Δstate||).
#   eef_quat[3:7], needle_quat[12:16], needle_to_eef_quat[19:23],
#   tripod_quat[26:30], tripod_to_eef_quat[33:37]
QUAT_BLOCKS = [(3, 7), (12, 16), (19, 23), (26, 30), (33, 37)]


def align_quats(s_src, s_tgt):
    """Flip each quaternion block of s_src to the same hemisphere as s_tgt."""
    import numpy as np
    out = s_src.copy()
    for a, b in QUAT_BLOCKS:
        dot = (out[:, a:b] * s_tgt[:, a:b]).sum(1, keepdims=True)
        out[:, a:b] = np.where(dot < 0, -out[:, a:b], out[:, a:b])
    return out


def state_gap(s_src, s_tgt):
    """Quaternion-aware ||s'_tgt - s'_src|| (the per-transition dynamics gap)."""
    import numpy as np
    return np.linalg.norm(align_quats(s_src, s_tgt) - s_tgt, axis=1)


def needle_ring_dist(s):
    """Needle-to-tripod body distance — a proxy for the contact/insertion phase
    (the only phase where the Threading dynamics gap can manifest)."""
    import numpy as np
    return np.linalg.norm(s[:, NEEDLE_POS_SLICE] - s[:, TRIPOD_POS_SLICE], axis=1)


def contact_mask(s, thresh=0.15):
    """Boolean mask of transitions in the contact phase (needle near ring)."""
    return needle_ring_dist(s) < thresh


def episode_split(episode_ids, val_ratio, seed=0):
    """Split unique episode ids into (train_mask, val_mask) at the episode level."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(episode_ids)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_ratio))
    val_eps = set(uniq[:n_val].tolist())
    val_mask = np.array([e in val_eps for e in episode_ids])
    return ~val_mask, val_mask
