# Square_D0 + Coffee_D0: Dyna-IBRL

100k target transitions per run, mirroring `cluster/job_threading_margin.sh` so
the three tasks read the same way. Six arms per task; which ones run is decided
by `$JOBS` + `--array`, not by the job file.

| arm (`sq_` / `cof_` prefix) | stage-1 | margin(t) | `sgft_coef` |
|---|---|---|---|
| `*_dynaibrl` | 5-head dynens, h=5 | annealed 0.05 → 0 over 50k | 0 |
| `*_dynaibrl_m001` | same | **constant** `SGFT_MARGIN` (0.01) | 0 |
| `*_dynaibrl_sgft` | same | constant `SGFT_MARGIN` | **1.0** |
| `*_dynaibrl_plain` | single head, h=5 | annealed 0.05 → 0 | 0 |
| `*_dynaibrl_plain_m001` | same | constant `SGFT_MARGIN` | 0 |
| `*_dynaibrl_plain_sgft` | same | constant `SGFT_MARGIN` | **1.0** |

## ⚠️ The old `--array=1-20` silently dropped every `cof_*` run

`jobs_sqcof_dynaibrl.txt` holds **40** lines (2 tasks × 2 arms × 10 seeds, arm-major)
while the `#SBATCH` header said `1-20` — so a plain `sbatch` ran only rows 1-20,
i.e. both Square arms at 10 seeds and **no Coffee at all**. The header is now
`1-40`, matching the list, and the job aborts with a message if `--array` runs
past the end of `$JOBS`. **Check `squeue`/`exps/coffee_g5x_3arm`
before assuming the dynens Coffee arms exist.**

## Job lists

| list | lines | layout |
|---|---:|---|
| `jobs_sqcof_dynaibrl.txt` | 40 | arm-major, the two **dynens** arms × 2 tasks × 10 seeds |
| `jobs_sqcof_dynaibrl_round2.txt` | 80 | **seed-major**, the four remaining arms × 2 tasks × 10 seeds |

`jobs_sqcof_dynaibrl_noanneal.txt` (80 rows) is the "everything except the
annealed arms" list — the four constant-margin arms (`_m001`, `_sgft`,
`_plain_m001`, `_plain_sgft`) on both tasks.

Do not pick arms by `--array` arithmetic; generate a list by name:

```bash
python3 cluster/make_joblist.py --seeds 10 \
  --arms sq_dynaibrl_m001,sq_dynaibrl_sgft,sq_dynaibrl_plain_m001,sq_dynaibrl_plain_sgft,\
cof_dynaibrl_m001,cof_dynaibrl_sgft,cof_dynaibrl_plain_m001,cof_dynaibrl_plain_sgft \
  --out cluster/jobs_sqcof_dynaibrl_noanneal.txt --skip-done
```

It rejects unknown arm names, refuses to mix Threading arms with `sq_`/`cof_`
ones (different job scripts), prints the exact `--array` to use, and with
`--skip-done` drops rows whose `train.log` already reached step 100000. Run it on
the cluster for `--skip-done` to see the real `exps/` tree.

⚠️ Dropping the annealed arms costs you the `_m001` vs anneal comparison, i.e.
the margin-schedule question. That is fine on Square, where `sq_dynaibrl` already
ran 10 seeds — but on Coffee **nothing** annealed has ever run (see the `--array`
bug above), so a no-anneal Coffee batch leaves `_m001` with only `score/policy_only`
as a reference.

Round 2 is seed-major on purpose: `--array=1-8` is a one-seed smoke across every
arm, `1-40` is 5 seeds of everything, `1-80` is the full batch. An arm-major list
can only be truncated to "some arms, all seeds", which is never what you want
when you are deciding how much compute to spend.

```bash
JOBS=cluster/jobs_sqcof_dynaibrl_round2.txt \
  sbatch --account=def-<pi> --array=1-80 --export=ALL,JOBS cluster/job_sqcof_dynaibrl.sh
```

`--export=ALL,JOBS` is not optional — without it the job falls back to the
round-1 list and reruns work you already have.

## The `_m001` arms fix the SGFT confound

`*_sgft` differs from `*_dynaibrl` in **two** fields (`sgft_coef` *and* constant
vs annealed margin), because the annealed shaped arm was measured to do badly and
the schedule was dropped for the shaped arms only. `*_m001` is that same constant
margin with shaping **off**, so:

- `_sgft` vs `_m001` → one field (`sgft_coef`). This is the SGFT ablation.
- `_m001` vs `_dynaibrl` → one field (the schedule). This is the margin question.
- `_sgft` vs `_dynaibrl` → still two fields. Do not read it as either.

`SGFT_MARGIN=0.01` came from a Threading fixed-margin sweep at **one seed**
(policy-only, 75–100k steady): `0.01 → 0.873`, `0.05 → 0.775`, anneal
`0.05→0.01 → ~0.88`, anneal `0.05→0.00 → ~0.855`. The 0.873-vs-0.855 gap is
0.018 — well under the noise floor CLAUDE.md rule 1 sets, and never replicated on
a fresh seed. The `_m001` arms are the first honest measurement of it.

## The model-free arms (`sq_ibrl_m001` / `cof_ibrl_m001`)

`train_rl.py`, no world model anywhere, same 100k budget, same 20 target demos,
same constant margin as the `_m001` dyna arms. They answer "how much of the
margin's value needs a model at all?" — the Square/Coffee counterparts of
Threading's `ibrl_m001`.

⚠️ Against any dyna arm this is **two** differences, not one: no world model
*and* a different prior (target-domain BC here, the frozen stage-2 distilled
policy there). It bounds the model's contribution; it does not isolate it.

`*_ibrl_m001_qsrc` is the same arm with **one** field added: the SOURCE IBRL
expert's critic as Q's starting point (`--load_pretrained_agent <src expert>
--load_policy_only 0 --load_critic_only 1`). `load_critic_only` was added to
`train_rl.py` for this: it keeps critic + critic_target from the pretrained agent
and restores the freshly-initialised actor, so the RL actor still cold-starts and
only Q carries over. Verify in the log that

```
[warm-start] load_critic_only=1 -> critic + critic_target from the pretrained agent, actor re-initialised
```

appears **without** a `skipped N shape-mismatched keys` line — a shape mismatch is
skipped silently and the arm would then be a duplicate of `*_ibrl_m001`.

Prior evidence is mixed and none of it is this measurement: `docs/tasks.md` §41b
has the source expert critic *losing* to the distilled V̂ by 0.085 (z=2.8) as an
MPPI scorer, and the target-domain IQL-pretrained critic bought only speed, not
ceiling (~96%, matched). Both are different uses of a critic than IBRL's argmax.
`threading_ibrl_simdist_srccritic_mppiactor_seed1` is **not** a clean precedent —
it moved the actor, the margin schedule and the critic at once.

**Prerequisites** (neither exists on the cluster until you ship it):

| asset | how |
|---|---|
| `exps/mimicgen/{square_il_ft_target_rotw1,coffee_il_ft_target_g5x}_seed*` | `bash scripts/train_target_bc_priors.sh` on the workstation |
| the 3 source IBRL experts (`*_qsrc` arms only) | already on the workstation; `cluster/pack_ibrl_priors.sh` ships them |
| `release/cfgs/mimicgen/{square_rotw1,coffee_g5x}_ibrl_m001.yaml` | `python3 scripts/make_ibrl_m001_cfg.py` (already generated, committed) |

```bash
bash scripts/train_target_bc_priors.sh          # 3 tasks x 10 seeds, round-robin over GPUs
bash cluster/pack_ibrl_priors.sh                # -> ibrl_m001_priors.tar.gz (~260 MB)
# scp, then AT THE REPO ROOT on the cluster: tar xzf ibrl_m001_priors.tar.gz
JOBS=cluster/jobs_sqcof_ibrl_m001.txt \
  sbatch --account=def-<pi> --array=1-40 --export=ALL,JOBS cluster/job_sqcof_dynaibrl.sh
```

The priors ship **with their `cfg.yaml`** — `train_rl.py` rebuilds the BC prior
from `dirname(bc_policy)/cfg.yaml`, so a tarball of bare `.pt` files fails at
load time, after the queue wait.

### Coffee: two stage-1 families, only one is G5x

`scripts/make_ibrl_m001_cfg.py` copies the env from the **stage-1 cfg**, because
that is where the dyna arms get their eval env (`train_simdist_adapt.py:398`) and
two numbers are only comparable if the env matches (rule 19).

⚠️ On Coffee the right stage-1 is **`coffee_s1g5x_dynens` / `coffee_s1g5x_plain`**
(`friction 0.2 / lid_damping 5 / mass 8 / pod_holder_friction 5`), which is what
all 31 finished runs in `exps/coffee_g5x_3arm` inherit and what
`coffee_d0_state_baseline_target_g5x.yaml` and `coffee_ibrl_target_G5x_seed1` use.
`coffee_simdist_{diversev1,plain}_h5_seed1` are the **C50-era** stage-1s on a
different target (`friction 0.5 / lid 50 / mass 3 / pod 2`); this job pointed at
them until 2026-09-06. The directory names do not say which family is which --
confirm by reading `stage1_ckpt` out of an existing run's cfg.yaml. Full table:
`docs/infra.md` §13.


## What the score means

Dyna-IBRL has **no planner**. `score/score` is the IBRL hybrid: `argmax` over
{RL actor, FROZEN stage-2 prior} using the critic — one forward pass per step, no
CEM, no dynamics rollout at deploy time. The world model is used only during
training (to generate data and to warm-start Q).

`--eval_policy_only 1` logs the frozen prior as `score/policy_only`. That is the
floor: it is constant within a run because stage 3 never touches the prior, and
an arm that does not clear it has learned nothing. On Threading the gap reached
+0.29 by 100k.

⚠️ This is **not** comparable to the `score/score` of the SimDist adapt arms in
the same directories (`sq_dynens_anch_*`, `cof_dynens_anch_*`) — those are MPPI
scores, i.e. what a planner achieves. Comparing them is meaningful ("same budget,
does the policy match the planner?") but they are different quantities.

## The anchor: off by default, but Coffee probably wants it

`ANCHOR_COEF` defaults to **0**, matching the Threading batch.

```bash
ANCHOR_COEF=1.0 sbatch --account=def-<pi> --export=ALL,ANCHOR_COEF \
    cluster/job_sqcof_dynaibrl.sh
```

`docs/tasks.md` §40e measured the stage-2 anchor as **the** lever on Coffee G5x
(+0.40, against +0.03 for stage-3 adapt), and Dyna-IBRL starts from exactly that
stage-2 policy — so a Coffee run with the anchor off starts from a much weaker
prior and may be uninformative. Square's own adapt arm uses the far smaller 0.1.
Consider `ANCHOR_COEF=1.0` for Coffee and `0.1` for Square if you run them
separately (the value is per-task tuned; CLAUDE.md rule 9 — do not port it).

### Why the anchor data is small here

`_build_bc_anchor` reads only `obs/state` from `bc_anchor_episodes` (2000)
episodes and keeps 40000 states — nothing else in the file is touched. The
originals are 5.5 GB (Square) and 7.7 GB (Coffee). `scripts/extract_anchor_pool.py`
writes a same-layout file holding 4000 episodes of states: **72 MB** and
**222 MB**. Keeping 4000 > 2000 means each seed still draws its own subset
(`_build_bc_anchor` seeds its RNG with `cfg.seed`).

Verified to load identically: `(40000, 23)` for Square and `(40000, 66)` for
Coffee, matching the source `state_dim`.

## Run it

```bash
# 1. ship (~470 MB)
CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
  bash cluster/sync_sqcof_dynaibrl.sh --dry-run     # check, then drop it

# 2. env: cluster/README.md step 3. The job now defaults to the container when
#    cluster/ibrl.sif exists, and fails fast if neither env is present.

# 3. submit -- round 1 (dynens arms) if they are not already done
sbatch --account=def-<pi> --array=1-40 cluster/job_sqcof_dynaibrl.sh

#    round 2 (plain arms + the constant-margin arms)
JOBS=cluster/jobs_sqcof_dynaibrl_round2.txt \
  sbatch --account=def-<pi> --array=1-80 --export=ALL,JOBS cluster/job_sqcof_dynaibrl.sh
```

Per task: 1 GPU, 12 CPUs, 24 GB, **14 h**. Longer than the Threading batch
because Coffee's episodes are longer (~251 steps vs ~168) and `q_updates=5000`
makes Dyna-IBRL ~5× heavier than the old Dyna arm. Threading took 8–10 h on the
workstation at 4-to-a-box.

⚠️ **Coffee needs the mimicgen asset bind**, which the job already sets up:
mimicgen installs non-editable in the image and ships no XML, so `Coffee_D0`
dies on `coffee_pod.xml` without it. Square escapes this because NutAssembly's
assets come from robosuite. Confirm `deps/mimicgen/mimicgen/models/robosuite/assets/*.xml`
is 19 files on the cluster (`git submodule update --init --recursive`).

## Before the big submit

1. **One task interactively.** `salloc` a GPU and run task 1 by hand with
   `--max_transitions 3000 --eval_every_rounds 1`.
2. **Check these four lines appear**, in this order:
   ```
   [dyna-ibrl] prior frozen (N params); stage 3 trains actor + Q only
   [dyna-ibrl] actor initialised from prior: mse 0.4x -> 0.00x
   [dyna-ibrl] Q initialised from source value: mse 0.xx -> 0.000x
   [dyna-ibrl] preloaded 20 target demos into the TD replay: ~N transitions, 20 terminal (r=1)
   ```
   The last one is the quiet failure mode: `20 terminal (r=1)` means the demos'
   reward field made it into the replay. A **0** there means the critic has no
   positive examples at all — that is exactly what made the plain-Dyna arm sit at
   0.000 for all of 100k.
3. **Check the resolved env line** says the right task and σ=0.1. Two numbers are
   only comparable if it matches (CLAUDE.md rule 19), and rule 17: the eval env
   follows the checkpoint, so a wrong `stage1_ckpt` silently changes the task.
