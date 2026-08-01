"""Step 7: does the early-step GT-gap peak go away if gripper_kp_scale -> 1.0?

For each demo (sim_state_t, a_t), counterfactually step three sims:
  - source (nominal)
  - target FULL  (gripper_kp_scale=0.12, the active target)
  - target NO-KP (gripper_kp_scale=1.0, gripper stiffness back to nominal;
                  mass/friction/ring shifts kept)
and compare GT gap ||s'_tgt - s'_src|| vs step index for FULL vs NO-KP.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_utils import REPO, TARGET_SHIFT, counterfactual_step, load_demo_episodes, _env_params
from data_utils import state_gap, align_quats


def build_env(env_params, shift_kwargs):
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig
    params = dict(env_params)
    params["domain_shift"] = DomainShiftConfig(**shift_kwargs) if shift_kwargs else DomainShiftConfig(mode="none")
    env = PixelRobosuite(**params); env.reset()
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--target_hdf5", default=f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--n_ep", type=int, default=8)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step7_kp_ablation.png")
    args = p.parse_args()

    env_params = _env_params(args.source_weight)
    src_env = build_env(env_params, None)
    full_env = build_env(env_params, TARGET_SHIFT)
    nokp_env = build_env(env_params, {**TARGET_SHIFT, "gripper_kp_scale": 1.0})

    eps = load_demo_episodes(args.target_hdf5, args.n_ep)

    rec = {"full": {"gap": [], "ts": [], "src": [], "tgt": []},
           "nokp": {"gap": [], "ts": [], "src": [], "tgt": []}}
    for ep in eps:
        T = ep["state"].shape[0] - 1
        for t in range(T):
            sim = ep["sim_state"][t].astype(np.float64); a = ep["action"][t]
            s_src, _, _ = counterfactual_step(src_env, sim, a)
            s_full, _, _ = counterfactual_step(full_env, sim, a)
            s_nokp, _, _ = counterfactual_step(nokp_env, sim, a)
            for name, s_tgt in [("full", s_full), ("nokp", s_nokp)]:
                rec[name]["gap"].append(float(state_gap(s_src[None], s_tgt[None])[0]))
                rec[name]["ts"].append(t)
                rec[name]["src"].append(s_src); rec[name]["tgt"].append(s_tgt)

    for name in rec:
        for k in rec[name]:
            rec[name][k] = np.asarray(rec[name][k])

    # gap vs t (mean over episodes)
    print("GT gap by step:    t      FULL(kp0.12)   NO-KP(kp1.0)")
    for t in list(range(0, 8)) + [10, 20, 40]:
        mf = rec["full"]["ts"] == t; mn = rec["nokp"]["ts"] == t
        if mf.sum():
            print(f"               t={t:3d}    {rec['full']['gap'][mf].mean():.4f}        {rec['nokp']['gap'][mn].mean():.4f}")

    # early gripper-group decomposition
    gi = slice(7, 9)
    for name in ("full", "nokp"):
        e = rec[name]["ts"] < 3
        d = np.abs(align_quats(rec[name]["src"], rec[name]["tgt"]) - rec[name]["tgt"])
        print(f"{name:5s}: early(t<3) total gap mean={rec[name]['gap'][e].mean():.4f}, "
              f"gripper |Δ| mean={d[e, gi].mean():.4f}")

    # plot gap vs step
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c in [("full", "C3"), ("nokp", "C0")]:
        ts = rec[name]["ts"]; gap = rec[name]["gap"]
        xs = sorted(set(ts.tolist()))
        ym = [gap[ts == t].mean() for t in xs]
        ax.plot(xs, ym, color=c, label=f"{name} (kp={'0.12' if name=='full' else '1.0'})")
    ax.axhline(0.006, color="k", ls=":", lw=1, label="recon noise floor")
    ax.set_xlabel("step in episode"); ax.set_ylabel("GT gap (mean)")
    ax.set_xlim(0, 60); ax.set_title("early GT-gap peak: gripper_kp 0.12 vs 1.0")
    ax.legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
