# Square_D0 + Coffee_D0: Dyna-IBRL

20 array tasks = 2 tasks × 2 arms × 5 seeds, 100k target transitions each,
mirroring `cluster/job_threading_margin.sh` so the three tasks read the same way.

| arm | stage-1 | margin |
|---|---|---|
| `sq_dynaibrl` / `cof_dynaibrl` | 5-head dynens, h=5 | annealed 0.05 → 0 over 50k |
| `sq_dynaibrl_sgft` / `cof_dynaibrl_sgft` | same | **constant** `SGFT_MARGIN` (0.01) + `sgft_coef 1.0` |

Two more arms are defined but not in the default job list — `*_dynaibrl_plain`
and `*_dynaibrl_plain_sgft`, on the single-head stage-1. Add lines to
`jobs_sqcof_dynaibrl.txt` and widen `--array` to run them.

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

## The SGFT arms are not a controlled ablation

They differ from their plain counterpart in **two** fields: `sgft_coef` **and**
the margin schedule (constant vs annealed). The annealed shaped arm was measured
to do badly, so the schedule was dropped for the shaped arms. Read them on
absolute score. To restore the single-variable pair, run the SGFT arms with
`SGFT_MARGIN` matching the annealed schedule instead.

`SGFT_MARGIN=0.01` is the best constant measured on Threading (single seed,
75–100k steady, policy-only): `0.01 → 0.873`, `0.05 → 0.775`, anneal
`0.05→0.01 → ~0.88`, anneal `0.05→0.00 → ~0.855`.

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

# 3. submit
sbatch --account=def-<pi> cluster/job_sqcof_dynaibrl.sh
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
