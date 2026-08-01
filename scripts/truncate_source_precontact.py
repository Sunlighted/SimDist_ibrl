"""Truncate SOURCE Threading_D0 demos to the PRE-CONTACT phase only.

For the co-training experiment we want a LOT of source data but only the part
of each trajectory BEFORE the needle reaches the ring (grasp + transport). The
insertion / contact phase is where the source->target dynamics gap lives, so we
drop it from the source set and let the few TARGET demos teach insertion.

Per demo we compute the EXACT geom-based needle<->ring-center distance (the same
quantity threading.py:_check_success uses) at every step by replaying the stored
`sim_state` in a nominal source sim, then truncate the episode to keep only the
steps strictly before the needle is within `--enter_dist` of the ring center.

Conservative on purpose: `--enter_dist` is a generous 6 cm (the ring is ~16 mm),
so the last kept transition still has the needle well clear of the ring, and we
additionally back off `--backoff` extra steps. Better to drop a few good
transport steps than to leak any insertion data into the source set.

Output mirrors the input layout (HDF5 + env_cfg.json sidecar) so
bc/dataset.py reads it unchanged.
"""

import argparse
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, "classifier_test")
import cf_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="release/data/mimicgen/threading_d0_source_sigma0.1")
    ap.add_argument("--out_dir", default="release/data/mimicgen/threading_d0_source_sigma0.1_precontact")
    ap.add_argument("--weight", default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt",
                    help="any source ckpt; only used to build a nominal env for geom readout")
    ap.add_argument("--num_demos", type=int, default=1000)
    ap.add_argument("--enter_dist", type=float, default=0.06,
                    help="needle<->ring-center distance (m) defining 'at the ring'; "
                         "keep only steps before the first time d<enter_dist")
    ap.add_argument("--backoff", type=int, default=2,
                    help="extra steps to drop before the enter step (margin)")
    ap.add_argument("--min_len", type=int, default=20,
                    help="never truncate below this many steps")
    args = ap.parse_args()

    src_h5 = os.path.join(args.src_dir, "dataset.hdf5")
    os.makedirs(args.out_dir, exist_ok=True)
    out_h5 = os.path.join(args.out_dir, "dataset.hdf5")
    # sidecar carries the env config (mode=none source); copy it verbatim.
    shutil.copy(os.path.join(args.src_dir, "env_cfg.json"),
                os.path.join(args.out_dir, "env_cfg.json"))

    env = cf_utils.get_env(args.weight, domain="source")
    sim = env.env.sim
    nid = sim.model.geom_name2id("needle_obj_needle")
    ring_ids = [sim.model.geom_name2id(f"tripod_obj_ring_{i}")
                for i in range(env.env.tripod.num_ring_geoms)]

    def needle_ring_dist(sim_state):
        sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
        sim.data.qacc_warmstart[:] = 0.0
        sim.forward()
        npos = np.array(sim.data.geom_xpos[nid])
        rpos = np.mean([sim.data.geom_xpos[r] for r in ring_ids], axis=0)
        return float(np.linalg.norm(npos - rpos))

    fin = h5py.File(src_h5, "r")
    fout = h5py.File(out_h5, "w")
    gin = fin["data"]
    gout = fout.create_group("data")
    # carry over top-level data attrs (e.g. env metadata) if any
    for k, v in gin.attrs.items():
        gout.attrs[k] = v

    keys = sorted(gin.keys(), key=lambda k: int(k.split("_")[-1]))[:args.num_demos]
    kept_lens, orig_lens, no_enter = [], [], 0
    for i, k in enumerate(keys):
        g = gin[k]
        sim_states = np.asarray(g["obs"]["sim_state"], dtype=np.float64)
        T = sim_states.shape[0]
        dists = np.array([needle_ring_dist(s) for s in sim_states])
        below = np.where(dists < args.enter_dist)[0]
        if len(below) == 0:
            E = T            # never reaches the ring -> keep all (no insertion)
            no_enter += 1
        else:
            E = int(below[0])
        L = max(args.min_len, E - args.backoff)
        L = min(L, T)

        og = gout.create_group(f"demo_{i}")
        og.attrs["num_samples"] = int(L)
        for attr_k, attr_v in g.attrs.items():
            if attr_k != "num_samples":
                og.attrs[attr_k] = attr_v
        # truncate every per-step array to length L; recurse one level (obs/*)
        def copy_trunc(src, dst):
            for name in src:
                item = src[name]
                if isinstance(item, h5py.Group):
                    copy_trunc(item, dst.create_group(name))
                else:
                    dst.create_dataset(name, data=np.asarray(item)[:L])
        copy_trunc(g, og)

        kept_lens.append(L)
        orig_lens.append(T)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(keys)}] mean kept {np.mean(kept_lens):.1f} / "
                  f"orig {np.mean(orig_lens):.1f} steps", flush=True)

    fin.close()
    fout.close()
    kept_lens = np.array(kept_lens)
    orig_lens = np.array(orig_lens)
    print(f"\nWrote {len(keys)} demos -> {out_h5}")
    print(f"kept {kept_lens.sum()} / {orig_lens.sum()} transitions "
          f"({100*kept_lens.sum()/orig_lens.sum():.1f}%); "
          f"mean kept {kept_lens.mean():.1f} (orig {orig_lens.mean():.1f}); "
          f"demos never reaching the ring: {no_enter}")
    os._exit(0)  # avoid EGL teardown hang


if __name__ == "__main__":
    main()
