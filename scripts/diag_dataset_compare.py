"""Compare the two SOURCE collection datasets (strong source-RL policy vs weak
BC policy) used for weighted BC, plus the TARGET demos as a reference.

Metrics per dataset:
  - trajectory length distribution
  - success signal from rewards (any success, terminal success, first-success
    step, post-success steps = does it keep going after the +1)
  - one-shot vs recovery: needle-ring distance trajectory analysis. A "recovery"
    episode dips toward the ring (a near-insertion attempt), backs off, then
    comes back — counted as the number of local minima of needle_ring_dist that
    drop below an approach threshold separated by a retreat.
  - action quality: saturation fraction (|a|>0.99), per-step jerk
    (mean ||a_t - a_{t-1}||), gripper open/close toggles (hesitation proxy)
  - needle-ring geometry: min distance reached, final distance
"""
import sys
import os
import numpy as np
import h5py

sys.path.insert(0, "/home/Chenyang/ibrl/classifier_test")
from data_utils import needle_ring_dist  # noqa

DATA = "/home/Chenyang/ibrl/release/data/mimicgen"
DATASETS = {
    "source_RL (strong)": f"{DATA}/threading_d0_source_sigma0.1/dataset.hdf5",
    "source_BC (weak)": f"{DATA}/threading_d0_source_bcpolicy_sigma0.1/dataset.hdf5",
    "target_demos": f"{DATA}/threading_d0_target_dynamics/dataset.hdf5",
}

def local_minima_count(x, prominence):
    """# of local minima with at least `prominence` rise on BOTH sides before
    the next minimum — a robust count of distinct 'approach then retreat' dips."""
    mins = 0
    i = 1
    n = len(x)
    last_min = None
    running_max_since_min = -np.inf
    # walk, track minima separated by a prominent rise
    cur_min = x[0]
    rose = False
    for t in range(1, n):
        if x[t] > x[t - 1]:
            if x[t] - cur_min >= prominence:
                rose = True
        else:  # descending
            if rose:
                mins += 1
                rose = False
            cur_min = min(cur_min, x[t])
        cur_min = min(cur_min, x[t])
    return max(mins, 1)  # the final approach is always >=1


def path_metrics(p):
    """tortuosity = path length / straight-line displacement; total path len."""
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    plen = steps.sum()
    disp = np.linalg.norm(p[-1] - p[0])
    return plen, (plen / disp if disp > 1e-6 else np.nan)


def analyze(path, max_ep=2000):
    f = h5py.File(path, "r")
    d = f["data"]
    demos = sorted(d.keys(), key=lambda k: int(k.split("_")[1]))[:max_ep]

    lengths = []
    n_success, n_term_success = 0, 0
    eef_tort, needle_tort = [], []        # tortuosity (detour factor)
    backtrack = []                        # approach-phase retreat ratio
    n_dips = []                           # # of distinct approach dips
    approach_frac = []                    # fraction of ep spent in final approach
    sat_frac, jerk, grip_toggles = [], [], []
    grip_changes_late = []                # gripper toggles in 2nd half (regrasp)

    for k in demos:
        g = d[k]
        T = int(g.attrs["num_samples"])
        lengths.append(T)
        rew = g["rewards"][:]
        act = g["actions"][:]
        state = g["obs"]["state"][:]

        n_success += int(rew.max() > 0.5)
        n_term_success += int(rew[-1] > 0.5)

        dist = needle_ring_dist(state)        # needle->ring body distance
        eef = state[:, 0:3]
        needle = state[:, 9:12]

        # tortuosity of eef + needle paths
        _, et = path_metrics(eef)
        _, nt = path_metrics(needle)
        eef_tort.append(et)
        needle_tort.append(nt)

        # approach-phase backtracking: after the distance starts its final
        # descent to the (terminal) minimum, how much does it go back UP?
        dmin_idx = int(np.argmin(dist))
        seg = dist[: dmin_idx + 1]
        if len(seg) > 2:
            rises = np.clip(np.diff(seg), 0, None).sum()
            drop = seg[0] - seg.min() + 1e-6
            backtrack.append(rises / drop)
            approach_frac.append((T - dmin_idx) / T)
        # distinct dips toward the ring (prominence scaled to the ep range)
        rng = dist.max() - dist.min()
        n_dips.append(local_minima_count(dist, prominence=0.25 * rng))

        # action quality
        sat_frac.append((np.abs(act[:, :6]) > 0.99).mean())
        if T > 1:
            jerk.append(np.linalg.norm(np.diff(act[:, :6], axis=0), axis=1).mean())
        gsign = np.sign(act[:, -1])
        grip_toggles.append(int((np.diff(gsign) != 0).sum()))
        half = T // 2
        grip_changes_late.append(int((np.diff(gsign[half:]) != 0).sum()))

    f.close()
    L = np.array(lengths)
    nd = np.array(n_dips)
    return dict(
        n=len(demos),
        len_mean=L.mean(), len_std=L.std(), len_med=np.median(L),
        len_p10=np.percentile(L, 10), len_p90=np.percentile(L, 90),
        len_min=L.min(), len_max=L.max(),
        succ_rate=n_success / len(demos),
        term_succ_rate=n_term_success / len(demos),
        eef_tort=np.nanmean(eef_tort), needle_tort=np.nanmean(needle_tort),
        backtrack=np.mean(backtrack), approach_frac=np.mean(approach_frac),
        oneshot=(nd == 1).mean(), multi=(nd >= 2).mean(),
        dips_mean=nd.mean(),
        sat=np.mean(sat_frac), jerk=np.mean(jerk),
        grip_tog=np.mean(grip_toggles), grip_late=np.mean(grip_changes_late),
    )


def main():
    res = {name: analyze(p) for name, p in DATASETS.items()}
    rows = [
        ("episodes", "n", "{:.0f}"),
        ("len mean", "len_mean", "{:.1f}"),
        ("len std", "len_std", "{:.1f}"),
        ("len median", "len_med", "{:.0f}"),
        ("len p10/p90", None, None),
        ("len min/max", None, None),
        ("success rate", "succ_rate", "{:.3f}"),
        ("terminal-success rate", "term_succ_rate", "{:.3f}"),
        ("--- trajectory shape ---", "_sep", None),
        ("eef tortuosity", "eef_tort", "{:.2f}"),
        ("needle tortuosity", "needle_tort", "{:.2f}"),
        ("approach backtrack ratio", "backtrack", "{:.3f}"),
        ("final-approach frac of ep", "approach_frac", "{:.3f}"),
        ("one-shot (1 dip)", "oneshot", "{:.3f}"),
        (">=2 dips (re-try)", "multi", "{:.3f}"),
        ("mean # dips", "dips_mean", "{:.2f}"),
        ("--- action quality ---", "_sep", None),
        ("action saturation frac", "sat", "{:.3f}"),
        ("action jerk (mean dA)", "jerk", "{:.4f}"),
        ("gripper toggles/ep", "grip_tog", "{:.2f}"),
        ("gripper toggles 2nd half", "grip_late", "{:.2f}"),
    ]
    names = list(res.keys())
    w = 26
    print("\n" + " " * w + "".join(f"{n:>22}" for n in names))
    for label, key, fmt in rows:
        if key == "_sep":
            print(label)
            continue
        line = f"{label:<{w}}"
        if key is None:
            if "p10" in label:
                for n in names:
                    line += f"{res[n]['len_p10']:.0f}/{res[n]['len_p90']:.0f}".rjust(22)
            else:
                for n in names:
                    line += f"{res[n]['len_min']:.0f}/{res[n]['len_max']:.0f}".rjust(22)
        else:
            for n in names:
                line += fmt.format(res[n][key]).rjust(22)
        print(line)
    print()


if __name__ == "__main__":
    main()
