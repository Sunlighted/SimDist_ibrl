# Running the Square baselines on a Compute Canada cluster

Square_D0 rotw1 TARGET-B, matched **100k-transition** budget, mirroring
`exps/simdist_threading_3arm/{ibrl,iql,rlpd,dyna_h5,dyna_sgft}_seed*`
field-for-field. Five arms:

| arm | what it is | needs |
|---|---|---|
| `ibrl` | IBRL, max-Q over {RL actor, BC prior} | source MLP BC ckpt (shared) |
| `iql` | offline IQL on source → online finetune | one shared warm-start ckpt |
| `rlpd` | RLPD, 50/50 balanced sampling from a prior buffer | merged source+target set per seed |
| `dyna_h5` | Dyna/MBPO on the frozen SimDist world model | stage-1 dynens ckpt |
| `dyna_sgft` | same + SGFT potential shaping (`dyna.sgft_coef 1.0`) | stage-1 dynens ckpt |

`dyna_sgft` differs from `dyna_h5` in **exactly one field** (`dyna.sgft_coef`),
verified by diffing the two Threading cfgs.

---

## 0. The one thing to know about what GitHub gives you

`.gitignore` excludes `*.pt`, `*.hdf5`, `*.npz`, `*.pkl`, `exps/`, `release/model`.
So a fresh clone is **code only** — every checkpoint and dataset has to be
rsync'd. That is what `sync_to_cluster.sh` is for.

---

## 1. On the workstation — build the derived assets

```bash
bash scripts/prep_square_baselines.sh          # or: ... il | iql | rlpd
```

This is deliberately done **locally**, because the two steps that consume the
5.2 GB `square_d0_source_rotw1_diverse` (the IQL offline pretrain and the RLPD
merge) both happen here. The cluster then never needs that file:

```
   ship the merged/pretrained artifacts   ~1.0 GB
   ship the source diverse set instead    ~5.2 GB
```

It produces, all of which already exist for Threading and are the exact analogs:

| output | count | note |
|---|---|---|
| `exps/mimicgen/square_iql_offline_source/agent_warmstart.pt` | 1 | shared by all seeds, as Threading did |
| `release/data/mimicgen/square_rlpd_prior_s<N>/` | 10 | 2000 source rollouts + 20 target demos, ~91 MB each |

IBRL's prior needs **no build step**: it is the existing
`exps/mimicgen/square_source_mlp_bc_n1000_seed1/model0.pt`, used unmodified.

⚠️ **This diverges from Threading on purpose.** Threading's prior is that source
BC *fine-tuned 60 epochs on the seed's 20 target demos* (→ 0.40 on its target).
Doing the same on Square destroys the policy. Measured on the rotw1 init +
target shift + σ=0.1 protocol, n=100 each:

| prior | target |
|---|---:|
| source MLP BC, untouched | **0.26** (0.22 on re-check) |
| source RNN-BC, untouched | 0.17 |
| fine-tuned 3 / 6 / 12 / 22 / 40 / 60 epochs | 0.00 (all six) |
| from scratch on the 20 demos, 12 / 30 / 60 / 150 / 400 epochs | 0.00 (all five) |

Two things it is **not**: a harness artifact (the source weights re-probed under
the fine-tune cfg still scored 0.22) and a data-pipeline bug (from-scratch BC
loss fell 0.223 → 0.031, so it *did* fit the demos, and SimDist stage-2 reaches
**0.53** on the same 20 demos). 20 demos is not enough for a raw-state MLP BC on
Square, and a prior sitting at 0.26 has no margin to absorb being pulled toward
them. SimDist survives the same BC because stage-2 trains only a small head on a
**frozen** encoder distilled from 200k diverse episodes (`docs/simdist.md` §24).

**State this when comparing the two IBRL arms** — the priors are differently
defined, and Square's is the weaker of the two.

⚠️ The IQL step ends with `scripts/iql_ckpt_to_agent.py`, which is **load-bearing**:
`pretrain_iql_qagent.py` saves `{"critic":…,"actor":…}` while
`train_rl.py --load_pretrained_agent` keeps only *already-matching* keys, so
feeding it the nested dict silently warm-starts **nothing** and the run looks
like a cold start with no error at all.

⚠️ The RLPD step passes `--task Square_D0`. I extended
`scripts/merge_prior_dataset.py` for this — it used to hard-code Threading's
37-dim obs packing, and Square is 23-dim. The split
`eef_pos[0:3] | eef_quat[3:7] | gripper_qpos[7:9] | object[9:23]`, `prop = state[0:9]`
is verified exact against a file carrying both layouts.

---

## 2. Ship it

```bash
CC_HOST=narval.alliancecan.ca CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
  bash cluster/sync_to_cluster.sh --dry-run     # check, then drop --dry-run
```

Manifest and sizes:

| what | size |
|---|---|
| `square_sat_600000_s{1..901}` — 10 target demo sets | 16 MB |
| `square_rlpd_prior_s{1..901}` — 10 RLPD prior sets | **~910 MB** |
| `square_source_mlp_bc_n1000_seed1/model0.pt` — IBRL's prior (shared) | 8.5 MB |
| `square_iql_offline_source/agent_warmstart.pt` | ~10 MB |
| `square_simdist_rotw1_dynens_h5_seed1/latest.pt` — stage-1 for Dyna | 29 MB |
| **total** | **~0.97 GB** |

Put it on `$SCRATCH` (or `~/projects/def-<pi>/`), not `$HOME` (quota).
If the 910 MB bothers you: ship a single 2000-episode source subset (~91 MB)
plus the 10 target sets and run the `rlpd` step of `prep_square_baselines.sh`
on the login node instead — but note that changes the口径, since Threading
drew a **different** 2000-episode subset per seed (verified: `rlpd_prior_s1`
and `_s101` share no source episodes).

Also clone with submodules — `deps/{robosuite,robomimic,mimicgen}` are the
install source:
```bash
git clone --recursive https://github.com/Sunlighted/ibrl.git
```

---

## 3. Get the env onto the cluster — two options

The env is **not** transferable as-is, even to an H100 cluster:
`common_utils/rela` is a compiled pybind11 extension tied to the workstation's
torch ABI, and `deps/*` are *editable* installs hardcoding
`/home/Chenyang/ibrl_deps/...`. Either rebuild on the cluster (3A) or ship a
container (3B).

Target here is **H100 (sm_90)**, which both the workstation's `torch 2.7+cu128`
and the container's `torch 2.5.1+cu124` support — so the architecture-coverage
question that would matter on a V100 cluster does not apply.

### 3A — venv on the login node (default, ~15 min)

```bash
bash cluster/cc_setup.sh
```

Compute nodes have no internet, so this must run on a **login** node.

### 3B — Apptainer container (`.sif`)

Better if `cc_setup.sh` fights the cluster's module stack, or if you want the
run bit-reproducible. CC supports Apptainer natively.

```bash
# on the workstation (~20 min, ~7 GB out; --fakeroot, no sudo needed)
bash cluster/build_sif.sh
rsync -avh --progress cluster/ibrl.sif $CC_USER@$CC_HOST:$CC_ROOT/cluster/
```

The image pins `torch 2.5.1+cu124` — fine on H100 (sm_90) and it needs only a
driver >= 550, whereas cu128 wants >= 570, so it is the safer bet on a cluster
whose driver you have not checked. It installs
`deps/{robosuite,robomimic,mimicgen}` **non-editable**, so the container does
not need `deps/` at run time.

**One extra step (3b):** `common_utils/rela` is deliberately *not* baked in — it
must match the container's torch ABI and it is imported from the repo's own
`PYTHONPATH`, which is a bind mount at run time. Build it once on the cluster,
inside the container:

```bash
module load apptainer
apptainer exec -B $PWD:$PWD --pwd $PWD cluster/ibrl.sif bash -c 'cd common_utils && make -j8'
```

Then submit with `SIF=` set (see step 4).

### Which one

**Use 3A.** On H100 there is no architecture problem to route around, so the
container buys only reproducibility, at the cost of a 7 GB transfer and one
extra in-container `make`. Keep 3B as the fallback if CC's module stack fights
`cc_setup.sh`.

Either way, **verify EGL on a GPU node before submitting 21 jobs.**

Diverges from `docs/setup.md` in two places, both deliberate: torch comes from
CC's `--no-index` wheelhouse (the cu128 pin in setup.md is Blackwell-specific
and wrong here), and there's no conda. The load-bearing pins stay:
`mujoco==2.3.2` (MimicGen requirement), `numpy<2` (wandb 0.16.4 uses the
removed `np.float_`), `huggingface_hub==0.23.4` (robomimic v0.5 hard-pins it).

**Verify EGL on a GPU node before submitting 21 jobs.** `train_rl.py` does not
expose `state_only_no_render`, so robosuite builds an offscreen GL context even
though the policy only reads `obs["state"]`. The command is printed at the end
of `cc_setup.sh`. If EGL is unavailable, `MUJOCO_GL=osmesa` works but is much
slower — at which point it is worth plumbing `state_only_no_render` through
`train_rl.py` instead.

---

## 4. Submit

```bash
# 3A (venv)
sbatch --account=def-<pi> cluster/job_square_baselines.sh
# 3B (container) -- the job script switches on SIF being set
SIF=cluster/ibrl.sif sbatch --account=def-<pi> --export=ALL,SIF \
    cluster/job_square_baselines.sh
```

21 array tasks = Threading's seed counts exactly (`ibrl`/`iql`/`rlpd` × 5,
`dyna_h5`/`dyna_sgft` × 3). Per task: 1 GPU, **12 CPUs**, 32 GB, 12 h.

- 12 CPUs because `train_rl.py`'s mp eval **hard-codes `num_proc=10`**
  (`train_rl.py:565`); with fewer cores every eval serialises.
- 12 h from the measured Threading wall-clocks (3.5–5.5 h each, and those ran
  5-to-a-GPU — a dedicated CC GPU should be faster).

To go to 10 seeds everywhere, regenerate the list and widen the array:
```bash
{ for a in ibrl iql rlpd dyna_h5 dyna_sgft; do
    for s in 1 101 201 301 401 501 601 701 801 901; do echo "$a $s"; done
  done; } > cluster/jobs_square_baselines.txt      # 50 lines
sbatch --account=def-<pi> --array=1-50 cluster/job_square_baselines.sh
```

### wandb

Compute nodes have no internet, so the job sets `WANDB_MODE=offline`. Sync from
a login node afterwards:
```bash
wandb sync wandb/offline-run-*
```
Runs land in project `square_rotw1_baselines` with group = the arm, because
`common_utils/py/run_cfg.py` derives `wb_exp/wb_run/wb_group` from `save_dir`
as read-only properties — `--wb_group` / `--wb_exp` are **not** CLI arguments
and passing them makes argparse reject the whole command.

---

## 5. Sanity checks worth doing before the big submit

1. **One task, interactively.** `salloc` a GPU, run task 1 by hand with
   `--num_train_step 3000 --log_per_step 1000`, and confirm three eval points
   land and `score` is not identically zero.
2. **Print the env line.** Every run logs the resolved task / shift / sigma;
   two numbers are only comparable if that line matches. Check it says
   `Square_D0`, `gripper_kp 0.3 / peg_size 1.2 / peg_friction 2.0`,
   `nut_z_rot [2.094, 3.142]`, `sigma 0.1`.
3. **IQL warm start actually loaded.** `train_rl.py` reports how many keys
   matched. Zero matched means the remap in step 1 was skipped.
4. **IBRL prior is the BC prior, not the bare RL head.** For these baselines the
   prior comes from `--bc_policy`, so the `load_frozen_agent` /
   `source_agent_use_hybrid` trap (CLAUDE.md rule 8) does not apply — but if you
   ever switch to a frozen source *agent* as the prior, it does.

---

## 6. What is NOT here

- **Nothing in this repo was cluster-aware before this directory.** There is no
  pre-existing SLURM tooling to copy from; these scripts are new.
- The SimDist arms themselves (`plain` / `dynens` / `dynens+anchor`) are not in
  this job file — those 60 Square runs already completed on the workstation
  (`exps/square_rotw1_6arm/`).
