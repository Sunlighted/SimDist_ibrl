# Threading_D0 — margin batch

Everything lands in `exps/simdist_threading_3arm`, i.e. the same wandb project as
the existing groups, so these overlay on one chart.

## The question

With the IBRL prior margin annealed 0.05 → 0, how does an actor that learns to
**act** compare against the planner it is deferring to — when both are built on
the **same single-head dynamics**?

## Arms

| arm | script | model | what it is |
|---|---|---|---|
| `plain_h5` | `train_simdist_adapt.py` | single head, h=5 | plain SimDist MPPI. The planning reference. |
| `dynaibrl_plain` | `train_simdist_dyna_ibrl.py` | single head, h=5 | Dyna + IBRL margin 0.05 → 0 over 50k |
| `dynaibrl_plain_m001` | `train_simdist_dyna_ibrl.py` | single head, h=5 | the same with margin **constant** at `SGFT_MARGIN` (0.01) |
| `dynaibrl_plain_sgft` | `train_simdist_dyna_ibrl.py` | single head, h=5 | constant margin + `--dyna.sgft_coef 1.0` |
| `ibrl_m005anneal0` | `train_rl.py` | none | model-free IBRL, same margin schedule |
| `dynaibrl_dynens` | `train_simdist_dyna_ibrl.py` | 5 heads, h=5 | as `dynaibrl_plain`, on the ensemble stage-1 |
| `dynaibrl_dynens_m001` | `train_simdist_dyna_ibrl.py` | 5 heads, h=5 | as `dynaibrl_plain_m001`, on the ensemble stage-1 |
| `dynaibrl_dynens_sgft` | `train_simdist_dyna_ibrl.py` | 5 heads, h=5 | as `dynaibrl_plain_sgft`, on the ensemble stage-1 |

⚠️ `dynaibrl_plain_sgft` was described here as "one field apart" from
`dynaibrl_plain`. It is not: it differs in `sgft_coef` **and** the margin schedule
(constant vs annealed), because the annealed shaped arm was measured to do badly
and the schedule was dropped for the shaped arms only. The `_m001` arms were
added to close that: read `_sgft` against `_m001` (one field, `sgft_coef`) and
`_m001` against the annealed arm (one field, the schedule).

## ⚠️ `--array=1-40` silently dropped rows 41-60

`jobs_threading_margin.txt` holds **60** lines while the `#SBATCH` header said
`1-40`, so a plain `sbatch` never ran either `dynaibrl_dynens` arm. The header is
now `1-60` and the job aborts if `--array` runs past the end of `$JOBS`.

## Job lists

`$JOBS` selects the list (default `cluster/jobs_threading_margin.txt`, 60 lines,
arm-major). The constant-margin arms live in a separate seed-major list:

```bash
JOBS=cluster/jobs_threading_margin_m001.txt \
  sbatch --account=def-<pi> --array=1-20 --export=ALL,JOBS \
  cluster/job_threading_margin.sh      # 2 arms x 10 seeds; --array=1-2 smokes both
```

`--export=ALL,JOBS` is not optional — without it the job silently uses the
default list.

Both `_sgft` arms have already run, and the 0.05 → 0 schedule is not being
re-spent, so `jobs_threading_margin_m001.txt` **is** the remaining work: the two
constant-margin, shaping-off arms, 10 seeds each. Build any other subset by name
with `cluster/make_joblist.py` rather than by `--array` arithmetic.

⚠️ The `_sgft` numbers they will be read against were produced by an earlier
submission. Before putting them on one chart, check `[EVAL ENV]` / `[setup]` in
both `train.log`s agree on task, σ, `strict`, and `ep_len` (CLAUDE.md rule 19),
and that the `_sgft` runs used the same `stage1_ckpt` (rule 17: the eval
environment follows the checkpoint).

`plain_h5`'s deliverable **is** the planner, so its `score/score` is the MPPI
score and `--eval_policy_only 1` logs the bare policy alongside as
`score/policy_only`. The Dyna arms have no planner at eval; their `score/score`
is the policy acting alone. The model-free arm has no model at all.

## Two things that were deliberately held fixed

**The dynamics head.** Every world-model arm uses
`threading_simdist_diversev2_h5_seed1` — one head, horizon 5, `diverse_v2`
source. The existing workstation `dynaibrl_*` runs all sit on the 5-head
`dynens` stage-1, so "Dyna+margin vs planning" was confounded with "ensemble vs
single head" there. Ensemble is now a separate axis, not a term inside this
comparison.

**The demo standard.** All four arms use `sat_160000` (25 episodes, 20 used),
the standing standard since `docs/demo-collection.md` 38 measured **+0.139** from
this switch alone; `dynens_h5 → dynens_h5_slow`, identical but for this path,
moved round-0 from 0.274 to 0.417.

## The margin schedule

`margin(t)` anneals `prior_margin_init → prior_margin_final` over
`prior_margin_steps`, then stays at `final`. Evidence for 0.05 as the init, from
the fixed-margin sweep documented in `rl/simdist_dyna_ibrl.py`:

| margin | score |
|---|---|
| 0 | 0.000 — argmax decided by critic noise; ~5.8% action deviation kills the episode |
| **0.05** | **0.775** — best; the actor clears the bar occasionally |
| 0.1 | 0.632 — bar too high, the actor essentially never acts |

High early (survive an uncalibrated critic), zero late (let a good actor
through) is what the schedule buys.

⚠️ `MARGIN_STEPS=50000` on **both** the Dyna and the model-free arm so the two
see the same schedule. `ibrl_m005_seed1` did something different: `steps=100000`
with `init=final=0.05`, i.e. a **constant** margin, no anneal. Override with
`MARGIN_STEPS=100000 sbatch --export=ALL,MARGIN_STEPS ...` if you want that
schedule length — but change it for all arms or you reintroduce the confound.

## Prerequisite: the BC priors

`ibrl_m005anneal0` needs a target-domain BC policy as its action prior. The six
existing `exps/mimicgen/il_ft_target_seed*` are unusable here: they cover 6 of
the 10 seeds, and they were fine-tuned on `sat_400000`. Pairing a `sat_400000`
prior with `sat_160000` preload data would confound this arm with a
demo-standard difference larger than most effects being measured.

`cluster/job_threading_bc_s160.sh` re-runs that fine-tune on `sat_160000` for
all 10 seeds into `il_ft_target_s160_seed*`. Everything except the demo path
matches `exps/mimicgen/il_ft_target_seed1/cfg.yaml`.

Arms 1–30 (`plain_h5`, `dynaibrl_plain`, `dynaibrl_plain_sgft`) do not need it
and can start immediately.

## Running it

```bash
# from the workstation
CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
  bash cluster/sync_threading_margin.sh --dry-run   # inspect, then drop --dry-run

# on the cluster
sbatch --account=def-<pi> cluster/job_threading_bc_s160.sh     # 10 tasks, ~2h
sbatch --account=def-<pi> --array=1-30 cluster/job_threading_margin.sh
# once the 10 il_ft_target_s160_seed*/model0.pt exist:
sbatch --account=def-<pi> --array=31-40 cluster/job_threading_margin.sh
```

wandb is offline by default (compute nodes have no internet, and `online`
deadlocks long runs — a logging thread parks in `__skb_wait_for_more_packets`).
Sync from a login node afterwards:

```bash
wandb sync wandb/offline-run-*
```

## Not comparable to

- the workstation `dynaibrl_*` and `ibrl_m005_seed1` — `sat_400000` demos, and
  `dynaibrl_*` additionally sits on the `dynens` stage-1
- any pre-2026-08-31 `dyna_*` run — see the four pre-fix defects documented at
  the top of `cluster/job_threading_dyna_plain.sh`
