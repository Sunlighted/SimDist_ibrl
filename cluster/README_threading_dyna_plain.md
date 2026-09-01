# Threading_D0: `dyna_bc10` + `plain_h5` on a cluster

20 array tasks = 2 arms × 10 seeds, matched **100k target transitions**, landing in
`exps/simdist_threading_3arm/` so they share the wandb project with the 9 groups
already there and overlay on one chart by group.

| arm | what it is | stage-1 | entry point |
|---|---|---|---|
| `dyna_bc10` | Dyna/MBPO on the frozen world model, `bc_coef` 0.1 → **10** | `threading_simdist_dynens_diversev2_h5_seed1` (5 heads) | `train_simdist_dyna.py` |
| `plain_h5` | plain SimDist MPPI, **single** dynamics head, horizon 5 | `threading_simdist_diversev2_h5_seed1` (1 head) | `train_simdist_adapt.py` |

## Why these two, specifically

**`bc_coef=10`** is the best the *original* Dyna knobs reach. Measured on the
workstation (75–100k steady, paired seeds): `0.1 → 0.171`, `1 → 0.208`,
`10 → 0.386`; a later scan put the peak near 30 but with a flat top
(20–35k window: 10 → 0.444, 30 → 0.460, 100 → 0.438, 300 → 0.357). 10 sits on the
plateau and is the value with completed 100k runs behind it.

**`plain_h5`** exists because the current `plain_h3` arm is horizon **3** while
every dynens arm is horizon **5** — so "plain vs ensemble" has been confounded
with horizon all along. This arm holds the source data (`diverse_v2`), the
horizon (5) and the demo set fixed, leaving the dynamics ensemble as the only
difference. `findings.md` §31 attributes the whole dynens gain to plan-time
averaging (5 heads 0.970 vs 1 head 0.120 *from the same weights*); this is the
clean between-checkpoint version of that claim.

## The demo standard

Both arms read `threading_d0_sat_160000_s<SEED>` — demos collected from an
**unsaturated** policy. This used to be the `_slow` suffix; it is now the standing
collection standard, no longer a swept variable, and so is absent from the run
names.

⚠️ **It is not a free relabelling.** `docs/demo-collection.md` §38 measured
**+0.139** from this switch alone, and the old `dynens_h5` → `dynens_h5_slow` pair
(identical but for this one path) moved round-0 from **0.274 → 0.417**. So:

> Runs from this job are **not** comparable to the workstation's
> `dyna_bc10_seed{1,101}`, which used `sat_400000`. Compare them against
> `dynens_h5_slow` / `dynens_h5_slow_anchor`, or re-run the old cells on
> `sat_160000`.

**Name collision.** `exps/simdist_threading_3arm/dyna_bc10_seed{1,101}` already
exists locally on the workstation with the *old* demo set. Nothing collides on the
cluster (its `exps/` starts empty), but syncing results back will. Either rename
the two local dirs to `dyna_bc10_sat400_seed*` first, or submit with
`SUFFIX=_s160 sbatch --export=ALL,SUFFIX ...`.

## Two settings that are not defaults

- **`--target_num_data 20`on both arms.** The `sat_160000` sets hold 25 episodes.
  The 3-arm comparison is defined on 20, but the old `plain_h3` left this `null`
  and trained stage-2 on all 25 — a second, unintended difference. Pinned to 20 here.
- **`--collect_disagree_coef 0.0` on `plain_h5`.** The dynens arms use 0.5, but
  geometric disagreement needs more than one dynamics head, so it is undefined for
  a single-head model. `eval_disagree_coefs` stays at its `""` default for the same
  reason.

## Run it

```bash
# 1. ship assets (~66 MB: 10 demo sets + 2 stage-1 ckpts + the job files)
CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
  bash cluster/sync_threading_dyna_plain.sh --dry-run     # check, then drop it

# 2. env: cluster/README.md step 3 (3A venv is the default; 3B container as fallback)

# 3. submit
sbatch --account=def-<pi> cluster/job_threading_dyna_plain.sh
# container route:
SIF=cluster/ibrl.sif sbatch --account=def-<pi> --export=ALL,SIF \
    cluster/job_threading_dyna_plain.sh
```

Per task: 1 GPU, 12 CPUs, 24 GB, 12 h. 12 CPUs because mp eval runs
`num_eval_proc=10` worker processes; with fewer cores every eval serialises.
Workstation wall-clock was 2.5–3 h per Dyna run at 6-to-a-box, and MPPI planning
is slower, so 12 h is comfortable headroom.

wandb defaults to **offline** (fir compute nodes have no internet, and `online`
also deadlocks long runs). Sync afterwards from a login node:
`wandb sync wandb/offline-run-*`. To log live instead, both the mode *and* a key
must be exported:
`WANDB_MODE=online WANDB_API_KEY=... sbatch --export=ALL,SIF,WANDB_MODE,WANDB_API_KEY ...`
— online with an empty key falls back to offline (it used to abort the task inside
`wandb.init` with `api_key not configured (no-tty)`, after the stage-1 ckpt had
already loaded). Third option: `USE_WB=0`, then replay `train.log` with
`scripts/wandb_import_logs.py`.

## Before the big submit

1. **One task interactively.** `salloc` a GPU, run task 1 with
   `--max_transitions 3000 --eval_every_rounds 1`, confirm three eval points land
   and `score` is not identically zero.
2. **Check the env line.** Every run prints the resolved task / shift / σ.
   It must read `Threading_D0`, the §12 dynamics-only shift
   (`friction 0.5 / mass 3.0 / gripper_kp 0.12 / ring_friction 2.0 / ring_size 0.85`),
   `sigma 0.1`. Two numbers are only comparable if that line matches (CLAUDE.md §19).
3. **Confirm the stage-1 each arm loaded.** `plain_h5` must report **1** dynamics
   head and `dyna_bc10` **5**. Swapping these silently produces a plausible-looking
   but meaningless comparison.

The job script already fails the task if `train.log` never reaches
`100000: score/score`, so a truncated run (disk full, walltime) shows as FAILED
rather than as a green DONE — the failure mode that hid a whole batch of 35k-step
runs during the workstation sweep.
