"""Data pipeline for the conditional-density dynamics-gap detector.

Three transition sources, all expressed as (s, a, s') with a domain label c:
  - c=0 (source) : the suboptimal RL replay (broad coverage incl. flailing).
                   Its consecutive (state[t], action[t], state[t+1]) are real
                   source-dynamics transitions; no sim_state / counterfactual
                   needed (replay was collected under nominal dynamics).
  - c=0 (source) : counterfactual source-steps from the TARGET demo sim_states
                   (the paired anchor — same (s,a) as the target c=1 rows).
  - c=1 (target) : the target demo's own (state[t], action[t], state[t+1]).

The model conditions on (s, a) and learns p(Δs' | s, a, c) with Δs' = s' - s.
The target paired anchors are exactly the (s,a) where BOTH domains' next states
are known, so they give the cleanest gap supervision.
"""

import os
import numpy as np

# cf_utils sets sys.path/cwd/MUJOCO_GL on import, so import it first.
from cf_utils import REPO, get_env, counterfactual_step, load_demo_episodes
from data_utils import episode_split, state_gap

REPLAY_HDF5 = f"{REPO}/release/data/mimicgen/threading_d0_source_replay/replay_buffer.hdf5"
TARGET_HDF5 = f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"
SOURCE_WEIGHT = f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt"


def _episodes_from_hdf5(path, max_ep=None, need_sim=False):
    """Read per-episode arrays. Reuses load_demo_episodes when sim_state needed;
    else a light reader that tolerates the replay format (no sim_state)."""
    import h5py
    eps = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        if max_ep is not None:
            keys = keys[:max_ep]
        for k in keys:
            g = data[k]
            d = dict(state=np.asarray(g["obs"]["state"], np.float32),
                     action=np.asarray(g["actions"], np.float32))
            if need_sim:
                d["sim_state"] = np.asarray(g["obs"]["sim_state"], np.float64)
            eps.append(d)
    return eps


def build_dataset(replay_hdf5=REPLAY_HDF5, target_hdf5=TARGET_HDF5,
                  source_weight=SOURCE_WEIGHT, replay_max_ep=None,
                  val_ratio=0.2, seed=0, cache=None):
    """Return a dict of arrays for training/eval.

    Keys:
      replay : (s, a, s', ep)         c=0, suboptimal source coverage
      anchor : (s, a, s'_src, s'_tgt, ep)  paired: c=0 uses s'_src, c=1 uses s'_tgt
      plus train/val episode masks for both pools.
    The anchor pool is small (≈2k) and carries the gap supervision; it also
    yields a ground-truth gap = state_gap(s'_src, s'_tgt) for eval.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    # ---- c=0 replay transitions (no sim_state needed) ----
    rep = _episodes_from_hdf5(replay_hdf5, replay_max_ep, need_sim=False)
    rs, ra, rns, rep_id = [], [], [], []
    for ei, ep in enumerate(rep):
        T = min(ep["state"].shape[0], ep["action"].shape[0]) - 1
        if T <= 0:
            continue
        rs.append(ep["state"][:T]); ra.append(ep["action"][:T])
        rns.append(ep["state"][1:T + 1]); rep_id.append(np.full(T, ei, np.int64))
    rs, ra, rns = map(lambda x: np.concatenate(x).astype(np.float32), (rs, ra, rns))
    rep_id = np.concatenate(rep_id)

    # ---- target demos + counterfactual source-step anchors ----
    tg = load_demo_episodes(target_hdf5)
    env = get_env(source_weight, domain="source")
    as_, aa, as_src, as_tgt, aep = [], [], [], [], []
    for ei, ep in enumerate(tg):
        T = ep["state"].shape[0] - 1
        for t in range(T):
            s_src, _, _ = counterfactual_step(env, ep["sim_state"][t], ep["action"][t])
            as_.append(ep["state"][t]); aa.append(ep["action"][t])
            as_src.append(s_src); as_tgt.append(ep["state"][t + 1])
            aep.append(ei)
    as_, aa, as_src, as_tgt = map(lambda x: np.asarray(x, np.float32),
                                  (as_, aa, as_src, as_tgt))
    aep = np.asarray(aep, np.int64)

    rep_tr, rep_va = episode_split(rep_id, val_ratio, seed)
    anc_tr, anc_va = episode_split(aep, val_ratio, seed)

    out = dict(
        r_s=rs, r_a=ra, r_ns=rns, r_ep=rep_id, r_tr=rep_tr, r_va=rep_va,
        a_s=as_, a_a=aa, a_src=as_src, a_tgt=as_tgt, a_ep=aep,
        a_tr=anc_tr, a_va=anc_va,
        a_gap=state_gap(as_src, as_tgt),
    )
    if cache:
        np.savez(cache, **out)
    return out


def norm_stats(d):
    """Pooled mean/std over states and actions (train split) for input scaling."""
    s = np.concatenate([d["r_s"][d["r_tr"]], d["r_ns"][d["r_tr"]],
                        d["a_s"][d["a_tr"]], d["a_src"][d["a_tr"]], d["a_tgt"][d["a_tr"]]], 0)
    a = np.concatenate([d["r_a"][d["r_tr"]], d["a_a"][d["a_tr"]]], 0)
    # Δs' stats too (the modeled target)
    ds = np.concatenate([(d["r_ns"] - d["r_s"])[d["r_tr"]],
                         (d["a_src"] - d["a_s"])[d["a_tr"]],
                         (d["a_tgt"] - d["a_s"])[d["a_tr"]]], 0)
    return dict(s_mean=s.mean(0), s_std=s.std(0) + 1e-6,
                a_mean=a.mean(0), a_std=a.std(0) + 1e-6,
                ds_mean=ds.mean(0), ds_std=ds.std(0) + 1e-6)


if __name__ == "__main__":
    d = build_dataset(replay_max_ep=None,
                      cache=f"{os.path.dirname(__file__)}/cdm_data.npz")
    print(f"replay c=0: {len(d['r_s'])} transitions, "
          f"{int(d['r_tr'].sum())} train / {int(d['r_va'].sum())} val")
    print(f"anchor   : {len(d['a_s'])} paired (s,a), "
          f"{int(d['a_tr'].sum())} train / {int(d['a_va'].sum())} val")
    print(f"anchor GT gap: mean {d['a_gap'].mean():.4f} median {np.median(d['a_gap']):.4f} "
          f"max {d['a_gap'].max():.4f}")
