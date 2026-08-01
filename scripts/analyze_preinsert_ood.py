"""Quantify WHERE the source pre-insertion states are OOD vs the target-demo
manifold. For matched ~target_dist frames of each dataset, record the policy-
relevant pose channels and compare distributions.

Channels: eef_pos(3), eef_quat(4), gripper_qpos(2), needle_pos(3),
ring_center(3), needle-relative-to-ring(3). The big-gap channels are what a
'recovery' controller must correct to bring the state back onto the demo
distribution before handing to the policy.
"""
import argparse, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")
import h5py, numpy as np
import common_utils  # noqa
from env.robosuite_wrapper import PixelRobosuite
from scripts.diag_contact_phases import SimProbe, load_env_and_agent


def collect(env, probe, demos_path, max_demos, target_dist):
    rows = []
    with h5py.File(demos_path, "r") as f:
        grp = f["data"]
        keys = sorted(grp.keys(), key=lambda k: int(k.split("_")[1]))
        if max_demos and max_demos < len(keys):
            idx = np.linspace(0, len(keys) - 1, max_demos).round().astype(int)
            keys = [keys[i] for i in sorted(set(idx))]
        env.reset(); probe.refresh()
        for dk in keys:
            ss = grp[dk]["obs"]["sim_state"][:]
            eefp = grp[dk]["obs"]["robot0_eef_pos"][:]
            eefq = grp[dk]["obs"]["robot0_eef_quat"][:]
            grip = grp[dk]["obs"]["robot0_gripper_qpos"][:]
            dists, grasped = [], []
            insert_t = ss.shape[0]
            for t in range(ss.shape[0]):
                env.env.sim.set_state_from_flattened(ss[t]); env.env.sim.forward()
                d = probe.dist(); dists.append(d); grasped.append(probe.grasped())
                if d < probe.radius and insert_t == ss.shape[0]:
                    insert_t = t
            cand = [t for t in range(insert_t) if grasped[t] and dists[t] >= probe.radius]
            if not cand:
                cand = [t for t in range(max(insert_t, 1))]
            bt = min(cand, key=lambda t: abs(dists[t] - target_dist))
            env.env.sim.set_state_from_flattened(ss[bt]); env.env.sim.forward()
            npos = probe.needle_pos(); rc = probe.ring_center()
            rows.append(np.concatenate([
                eefp[bt], eefq[bt], grip[bt], npos, rc, npos - rc, [dists[bt]],
            ]))
    return np.array(rows)


def summarize(name, arr):
    # column layout
    cols = {
        "eef_pos": (0, 3), "eef_quat": (3, 7), "gripper": (7, 9),
        "needle_pos": (9, 12), "ring_center": (12, 15),
        "needle-ring(rel)": (15, 18), "dist": (18, 19),
    }
    print(f"\n=== {name}  (n={len(arr)}) ===")
    for k, (a, b) in cols.items():
        m = arr[:, a:b].mean(0); s = arr[:, a:b].std(0)
        print(f"  {k:18s} mean={np.array2string(m, precision=3, floatmode='fixed')}"
              f"  std={np.array2string(s, precision=3, floatmode='fixed')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="exps/mimicgen/threading_flow_lowdata_target_seed1")
    p.add_argument("--weight", default="model0.pt")
    p.add_argument("--target_dist", type=float, default=0.10)
    p.add_argument("--src_max", type=int, default=40)
    p.add_argument("--tgt_max", type=int, default=20)
    p.add_argument("--skip_grasp", action="store_true", default=True)
    p.add_argument("--grasp_games", type=int, default=0)
    p.add_argument("--insert_per_state", type=int, default=1)
    p.add_argument("--max_demos", type=int, default=0)
    p.add_argument("--approach_dist", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    agent, env_params = load_env_and_agent(args)
    env = PixelRobosuite(**env_params)
    probe = SimProbe(env)

    tgt = collect(env, probe, "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5",
                  args.tgt_max, args.target_dist)
    src = collect(env, probe, "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5",
                  args.src_max, args.target_dist)
    summarize("TARGET demo pre-insert", tgt)
    summarize("SOURCE demo pre-insert", src)

    # per-channel L2 gap between the two group means + how many source-std away
    print("\n=== gap: |mean_src - mean_tgt|  (and in units of target-std) ===")
    cols = {"eef_pos": (0, 3), "eef_quat": (3, 7), "gripper": (7, 9),
            "needle_pos": (9, 12), "needle-ring(rel)": (15, 18)}
    for k, (a, b) in cols.items():
        dmean = src[:, a:b].mean(0) - tgt[:, a:b].mean(0)
        tstd = tgt[:, a:b].std(0) + 1e-9
        l2 = np.linalg.norm(dmean)
        zsig = np.abs(dmean) / tstd
        print(f"  {k:18s} |Δmean|={l2:.4f}   per-dim z(vs tgt-std)="
              f"{np.array2string(zsig, precision=1, floatmode='fixed')}")


if __name__ == "__main__":
    main()
