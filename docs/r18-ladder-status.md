# r18@224 ladder — RUN LOG (2026-09-07 .. 2026-09-09) — **COMPLETE**

> ✅ **Conclusions live in `docs/simdist.md` §47**, and the decision-relevant parts
> are backfilled into `CLAUDE.md` (rewritten ★★★ PIXEL block + the iron-law-#14
> addendum) and `docs/infra.md` §7z (SLURM orchestration pitfalls).
> This file is the chronological run log: job ids, what broke, what was verified.
> Keep it only as provenance for §47's numbers; it is safe to delete afterwards.

## FINAL RESULTS (independent n=500, strict, sigma=0.1, shared demo draw)
| rung | window (n=13) | independent n=500 |
|---|---:|---:|
| A `plain_h3` | 0.365 +-0.099 | **0.418** (209/500) |
| B `dynens_h5` | 0.425 +-0.054 | **0.334** (167/500) |
| C `dynens_h5`+anchor | 0.875 +-0.030 | **0.766** (383/500) |

**VERDICT — ensemble does NOT reproduce (+0.060 window / -0.084 independent, signs
disagree, < rung A's own SD). Anchor DOES, and dominates: +0.432 / +0.450,
opposite in sign to vit96's -0.072 and ~5.3x state's +0.081.**
See §47a for the TopkSaver selection confound that explains why only the ensemble
delta is spoiled, and §47b for the encoder x anchor interaction (+0.504).


Working file for the in-flight experiment. When all three rungs have independent
n=500 numbers, the conclusions go into `docs/simdist.md` as a numbered section and
this file can be deleted.

## The question
Do the **dynamics ensemble** and the **source anchor** reproduce on pixel (r18@224)
the gains they gave on state? Two deltas, not a pixel-vs-state comparison.

```
state (published, docs/on-policy-adaptation.md, n=10 seeds, 75-100k window):
  plain_h3 0.564  --+0.249 ensemble-->  dynens_h5 0.813  --+0.081 anchor-->  0.979
vit96 (HYPOTHESIS ONLY, n=13 window, do not quote as a result):
  plain_h3 0.194  --+0.182 ensemble-->  dynens_h5 0.376  ---0.053 anchor-->  0.323
r18@224 (this run, independent n=500):
  plain_h3  [A]   --   ?  ensemble-->   dynens_h5  [B]   --   ?  anchor-->   0.766
```

## Rungs

| rung | what | stage-1 | adapt | n=500 headline |
|---|---|---|---|---|
| **A** `plain_h3` | 1 head, H=3 | `threading_pix224_plainh3_stage1_seed1` (NEW, 150k, 6-chunk chain) | `pix224_plainh3_adapt_seed1`, `ANCHOR_COEF=0` | pending |
| **B** `dynens_h5` | 5 heads, H=5, no anchor | `threading_pixel224_stage1_seed1` (**reused**, 150k done) | `pix224_dynens_noanchor_seed1`, `ANCHOR_COEF=0` | pending |
| **C** `dynens_h5`+anchor | 5 heads, H=5, λ=1 | same as B | `threading_pix224_adapt_full_seed1`, `ANCHOR_COEF=1` | **0.766** (383/500), job 5247283 |

Iron law #20: `num_dynamics_heads`/`horizon` are built in `WorldModel._build` and
`load_state_dict` is strict ⇒ **A needs its own stage-1**. The anchor is an
adapt-time knob ⇒ **B reuses C's stage-1 ckpt; it was NOT retrained.**

## Matched budget (all three rungs)
Stage-1 150k steps / 6.48M-transition cap / 5.9 epochs / 100k-traj JPEG replay.
Adapt `NUM_DEMO=20`, `MAX_TRANS=100000`, `INTERVAL=2000`, `EVAL_N=100`, `DYN_ANCHOR_COEF=0.1`.
Rung C verifiably reached the full 100k in 51 eval points — A and B use identical values.
Demo draw is **shared** across all rungs (`..._pix224_s400k_n200`, same episodes as
the 96 draw) ⇒ the comparison is PAIRED (~0.027), not carrying the 0.092 single-draw sigma.

## Jobs launched 2026-09-07
| job | name | what | depends on |
|---|---|---|---|
| 5268895 | `B_noanch_ad` | rung B adapt, 23:59, 96G | — |
| 5268896 | `B_noanch_fin` | rung B n=500 headline | afterany:5268895 |
| 5268897 | `A_plainh3_s1` | rung A stage-1 chunk 1/6, 11:59, 200G | — |

Rung A's adapt+eval are **not** in the queue yet and that is by design: stage-1
self-chunks, so there is no final job id to depend on at submit time. The chain
fires `scripts/slurm/launch_pix224_plainh3_followon.sh` from its new `THEN_SBATCH`
hook when `global_step` reaches 150k, and that script submits the adapt and chains
the eval behind it. Marker file `<save>/.then_submitted` makes it fire exactly once.

Pre-existing and unrelated to this ladder, left running:
`5256003 pix_stage1` = chunk 5/6 of `threading_pix224_hist_stage1_seed1`
(120k/150k) — the *history* arm, a different question (the velocity hypothesis).
`5268705 plain_fin` / `5268706 noanch_fin` = n=500 re-evals of the **vit96** plain
and no-anchor arms. Those upgrade the vit96 hypothesis ladder from its n=13 window
to n=500; they are context for the sign question, **not** results to quote.

## Changes made for this run
- **NEW** `release/cfgs/mimicgen/threading_d0_pixel_simdist_r18_224_plain_h3.yaml`
  — copy of `..._r18_224.yaml` with exactly three values changed: both `horizon:
  5` → 3 (planner AND world_model) and `num_dynamics_heads: 5` → 1. Edit anchored
  on `num_dynamics_heads`, never on `horizon` (§46 d7).
- `scripts/slurm/pixel_stage1.slrm` — added the `THEN_SBATCH` completion hook.
  **Default empty ⇒ inert**, so the in-flight hist chain that re-submits itself
  through this same file is unaffected. Propagated in `--export` so it survives
  down the chain.
- **NEW** `scripts/slurm/launch_pix224_plainh3_followon.sh`.

Gates run after the change (all pass): `test_configs_load.py` 17/17 (new config
reads `heads=1 H=3`), `test_world_model_branches.py`, `test_jpeg_replay.py`,
`test_history_layout.py`. The dedup regress job was not run — this change adds a
config and a slurm hook and touches no training code.

## Reading the results
- Headline = **independent n=500** re-eval of the saved ckpt (iron law #14). Never
  `model0.pt`'s in-training best. Note rung C's in-training points read 0.90-0.94
  at 98-100k against 0.766 independent — the optimism here is large.
- In-training curves: `scripts/cmp_adapt_runs.py` (does the 75-100k window and
  refuses to compare across budgets). Single n=100 points span 0.03-0.58.
- Iron law #1/#2: deltas under 0.05 are unreadable; the ensemble delta on state
  was +0.249 and on vit96 +0.182, so it should be readable if present. The anchor
  delta is the delicate one (+0.081 state, -0.053 vit96) — at n=500 per rung the
  95% CI half-width is ~0.04, so a ±0.08 effect is resolvable but a ±0.03 is not.

## Throughput: aip-rhinehar (added 2026-09-07, mid-run)
The association landed while the ladder was in flight. It is worth ~6x the
priority factor because it has NO usage history:

| account | NormShares | RawUsage | FairShare |
|---|---:|---:|---:|
| aip-gigor | 0.027 | 3.07e10 | 0.106 |
| **aip-rhinehar** | **0.125** | **0** | **0.657** |

Every PENDING ladder job was moved with `scontrol update job=<id>
Account=aip-rhinehar`, which **keeps the job id so dependency chains stay
intact** (do NOT cancel-and-resubmit -- that breaks `afterany` links and loses
accrued age priority). Measured effect: priority 530,690 -> 3,284,671 and the
scheduler reason went `(Priority)` -> `(None)`.

Moved: 5268895 (rung B adapt), 5268896 (rung B eval), 5269763 (rung A stage-1
chunk 2). NOT moved: 5268897 (rung A chunk 1, already RUNNING -- leave running
jobs alone) and 5269758 (the unrelated *history* arm's chunk 6, on gigor).

★ **Account propagation now has to hold in two places or the speedup silently
leaks back to gigor**, because `adapt.slrm`, `eval_simdist.slrm` and
`pixel_stage1.slrm` all hardcode `#SBATCH --account=aip-gigor`:
1. down the stage-1 chunk chain -- already handled, `ACCT="${SLURM_JOB_ACCOUNT}"`
   in `pixel_stage1.slrm`. Chunk 2 runs on rhinehar => chunk 3 inherits it.
2. from the final chunk into the follow-on adapt+eval -- this was MISSING and is
   now fixed in `launch_pix224_plainh3_followon.sh` (same `ACCT` idiom, passed as
   `--account="$ACCT"` to both submissions).
⚠️ `--export=ALL` carries ENV VARS ONLY, never sbatch resource/account flags --
the same trap that OOM'd job 5212432 via the `--mem` fallback.

## ✅ RESOLVED — THEN_SBATCH propagation verified (was an open checkpoint)
The risk: `THEN_SBATCH='bash scripts/slurm/launch_pix224_plainh3_followon.sh'`
contains SPACES and reaches the successor chunk inside a COMMA-separated
`--export` list. If SLURM split on whitespace, the hook would vanish and rung A's
adapt would never launch.

Two independent checks, both green:
1. **Inert branch, on a live chain.** The unrelated history chain's final chunk
   (5269758) hit `already at 150000 -- chain ends here`, which is the exact branch
   `fire_then` sits in. It produced no output, no `.then_submitted` marker and no
   spawned job => the hook is genuinely inert for chains that do not set it.
2. **Propagation, isolated probe** (job 5289598, 2 min, b1). Replicates the
   `--export` line's shape with a spaces-containing value plus vars on BOTH sides:
   ```
   CFG=[release/cfgs/x.yaml]
   THEN_SBATCH=[bash scripts/slurm/launch_pix224_plainh3_followon.sh]
   MAX_CHUNK=[6]
   PROPAGATION_OK
   ```
   The value survives whole AND `MAX_CHUNK` after it is intact, so the space did
   not truncate list parsing. ⇒ **SLURM splits `--export` on commas only.**

⚠️ Do NOT edit `scripts/slurm/pixel_stage1.slrm` while chunks are in flight.
Bash reads a script incrementally, so changing its byte layout under a running
interpreter can make it resume at a stale offset. Any fix waits until no chunk
is live.

Fallback if the hook ever does fail to fire (marker absent AND no `A_plainh3_ad`
job after stage-1 hits 150k): `bash scripts/slurm/launch_pix224_plainh3_followon.sh`.
Note `fire_then` writes the marker BEFORE submitting, so a marker with no job
means the SUBMIT failed -- running the launcher by hand still works.

## Results as they land (independent n=500, strict, sigma=0.1)
| arm | in-training window (n=13) | independent n=500 |
|---|---|---|
| vit96 `plain_h3` | 0.194 +-0.036 | **0.218** (109/500), job 5268705 |
| vit96 `dynens_h5` | 0.376 +-0.050 | **0.338** (169/500), job 5268706 |
| vit96 `+anchor` | 0.323 +-0.068 | **0.266** (133/500), job 5247287 |
| r18 `plain_h3` (A) | pending | pending |
| r18 `dynens_h5` (B) | pending | pending |
| r18 `+anchor` (C) | 0.875 +-0.030 | **0.766** (383/500), job 5247283 |

### vit96 ladder COMPLETE at independent n=500 (2026-09-07)
```
plain_h3 0.218  --+0.120 ensemble-->  dynens_h5 0.338  ---0.072 anchor-->  0.266
         (109/500)   z=+4.27                  (169/500)   z=-2.49            (133/500)
```
| delta | n=500 | 95% CI | n=13 window had said |
|---|---:|---|---:|
| ensemble | **+0.120** | [+0.065, +0.175] | +0.182 |
| anchor | **-0.072** | [-0.129, -0.015] | -0.053 |

★ **The vit96 anchor's NEGATIVE sign is now confirmed at n=500**, CI excluding
zero (z=-2.49), where before it was an n=13 hint inside +-0.050/+-0.068 windows.
The ensemble delta is also confirmed but SMALLER than the window suggested
(+0.120 vs +0.182). Both window estimates were biased away from zero, which is
the expected direction for a TopkSaver-influenced in-training measure.

⚠️ **What these CIs do and do not cover.** n=500 bounds the EVAL-SAMPLING noise
only. Each rung is ONE stage-1 seed and ONE demo draw, so stage-1 training-seed
variance is unmeasured and the shared draw only makes the rungs paired against
each other (~0.027, §38), it does not make any single rung's absolute value
robust. Treat the deltas as solid and the absolute levels as single-seed.

⇒ The r18 question is now sharper than "does the sign flip": on vit96 the anchor
genuinely HURTS by a resolvable margin. §42 predicts the anchor's gain is largest
on a WEAK base, and r18's base is much stronger, so a null or negative r18 anchor
delta would be consistent with vit96 while a positive one would contradict §42's
weak-base reading. Either way it is informative.

## Benign log noise — do NOT read these as failures
Every pixel run emits two teardown artifacts at EVERY in-training eval point
(~15x per chunk), both AFTER the work is done and both harmless:
1. `OSError: [Errno 16] Device or resource busy: '.nfsXXXXXXXX'` inside
   `multiprocessing/util.py _remove_temp_dir -> shutil.rmtree`. The mp eval
   workers' tempdir sits on NFS, which keeps a `.nfs*` silly-rename handle open
   until the last fd closes.
2. `EGLError(err = EGL_NOT_INITIALIZED, baseOperation = eglMakeCurrent)` raised
   in `MjRenderContext.__del__` / `EGLGLContext.__del__`, always prefixed
   `Exception ignored in:`. EGL is already torn down when the finalizer runs.

⇒ A crash filter must match the EXCEPTION line and subtract these, not match
`Traceback (most recent`. `/scratch/yifanr/temp/ladder/watch.sh` does this and was
validated in both directions (0 matches on a healthy rung-A log; still catches a
synthetic `CUDA out of memory`).

## Rung A stage-1 verified healthy (2026-09-07, chunk 1)
`1.43 it/s = 700 ms/step` (vs dynens_h5's 1.08 it/s -- H=3 costs fewer encode
calls), 10k/150k at 1:56. ETA ~29 h => ~3 chunks, MAX_CHUNK=6 has margin.
★ `[budget] transitions=6479983 batch=256 steps=150000 => EPOCHS = 5.9 cap=6500000`
-- **byte-identical to rungs B and C**, and `[seq-buffer]` shows 4500 x 8 shards
= 36,000 episodes, so all 8 generation seeds are represented (§46 ⑪).
⇒ the ladder's matched-budget claim is MEASURED on all three rungs, not assumed.

## ⚠️ MAINTENANCE OUTAGE gates the critical path (found 2026-09-07 13:20)
`scontrol show reservation` -> `maint_sept2026`, **all 178 nodes kn[001-178]**,
`StartTime=2026-09-08T09:00:00`. The `EndTime=2027-09-08` / `Duration=365-00:00:00`
is a PLACEHOLDER -- the real end is not knowable from SLURM. Window from
discovery to outage: **19.7 h**.

It hits the two arms completely differently, and the reason is §46i:

| arm | shape | fits the 19.7 h window? |
|---|---|---|
| rung A stage-1 | chunked, **has real resume** | yes -- chunk 1 ends 15:59 (~61.7k), chunk 2 at 11:59 ends 2026-09-08T03:58 => **~123k/150k banked** |
| rung B adapt | **21.5 h contiguous, CANNOT resume** | **no** -- 21.5 > 19.7 (rung C measured 21:32:36) |

⇒ Rung B waits for the nodes to come back. It is queued and correctly blocked
(`ReqNodeNotAvail, Reserved for maintenance`); nothing to fix.
⇒ **Do NOT shrink rung B's walltime to make it fit.** It cannot resume, so a
19 h run simply ENDS at ~90k transitions: a truncated window that
`cmp_adapt_runs.py` flags NON-STANDARD and that changes which ckpt TopkSaver
selects, i.e. not comparable to rung C's full 100k on either measure.
⇒ Both adapts (B, and A's after its stage-1) land post-maintenance regardless,
so **the delivery date is now a function of the outage length, not of compute.**

### Using the dead window: rung B plumbing smoke (job 5289149)
Rung B's exact configuration -- `ANCHOR_COEF=0` **on r18@224** -- had never been
run: rung C proved `ANCHOR_COEF=1` on r18, and the vit96 arm proved
`ANCHOR_COEF=0`, but not the combination. Discovering a fault in that after the
outage, on a 21.5 h critical-path job, would be expensive. So a 2-round smoke
(`MAX_TRANS=4000`, throwaway dir `pix224_noanchor_SMOKE`, 2:59 on rhinehar) runs
in the otherwise-idle window.
★ Per §46 e2 the smoke's SCORE is meaningless. The evidence is: stage-2 BC runs
with the anchor DISABLED, the stage-3 collect+dynamics loop runs, and `adapt rc=0`.

### Smoke verdict: rung B's config is PROVEN (job 5289149, 1:10:17, rc=0)
Plumbing evidence (§46 e2 -- the score is NOT the evidence):
- `adapt rc=0`
- `[adapt] using ORIGINAL SimDist MPPI recipe (base=100+rand=250, iters=3,
  elites=64, horizon=5 [kept])` -- same recipe rung C ran
- `stage-2 done: final policy BC loss = 0.01489` (non-trivial, anchor OFF)
- `[dyn-anchor] coef=0.1 | 2000 source episodes -> 40000 (z,a) pairs`

★ **Positive control in BOTH directions** that `ANCHOR_COEF=0` really disabled the
anchor rather than the log line merely being absent:

| run | ANCHOR_COEF | anchor-pool lines |
|---|---:|---:|
| rung C | 1 | `[anchor] encoding 40000 states ... 40000 x 512 (82 MB)` + `bc/anchor_loss` |
| **smoke** | **0** | **0** |
| vit96 no-anchor (known-good) | 0 | 0 |

The smoke matches the known-good no-anchor arm exactly and differs from rung C
exactly at the anchor.

### ★ A runtime check that rung A is really the plain_h3 rung
The adapt logs `horizon=N [kept]`, inherited from the stage-1 architecture
(iron law #20). Both pix224 runs so far read `horizon=5 [kept]`.
**Rung A's adapt MUST read `horizon=3 [kept]`.** If it reads 5, it loaded the
wrong stage-1 and the ladder's bottom rung is invalid -- check before trusting
any rung-A number.

## Chunk-boundary behaviour, measured (2026-09-07)
Chunk 1 ended `TIMEOUT` -- the DESIGNED boundary, not a failure. Verified:
`global_step = 50000` persisted, and chunk 2 logged
`progress before this chunk: 50000 / 150000` => real resume, not a restart.
Account propagated: chunk 2 ran on `aip-rhinehar` and printed
`propagating to successor: ... --account=aip-rhinehar`, so the `scontrol update`
on a PENDING chunk feeds correctly into the script's `SLURM_JOB_ACCOUNT` logic.

★ **USABLE steps per chunk != raw tqdm steps.** Chunk 1's tqdm read 50,249 but
`resume_state.pt` holds **50,000**: progress is checkpointed on the
`log_per_step: 10000` boundary, so a walltime kill discards up to 10k steps.
⇒ budget stage-1 at **~50k usable steps per 11:59 chunk** at 224+r18+H=3, i.e.
**3 chunks for 150k**, not the 2.6 a naive 1.43 it/s reading gives.
(Not worth "fixing" mid-run; noted so future sizing uses the usable rate.)

### Post-maintenance critical path (both arms run in parallel)
| arm | work remaining after nodes return | ETA |
|---|---|---|
| rung B | adapt 21.5 h + eval 0.5 h | **~22 h** |
| rung A | chunk 3 ~12 h -> hook -> adapt 21.5 h + eval 0.5 h | **~34 h** |
⇒ **~1.5 days after the outage ends**, no intervention required: rung A's chain is
self-launching (hook verified) and rung B is already queued behind the reservation.

## Claiming the pre-outage window (job 5298608, 2026-09-08 04:10)
Chunk 2 delivered exactly the predicted 50k usable steps (`global_step = 100000`),
which left a **4h57m window** that chunk 3 could not use: chunk 3 needs 11:59 and
was correctly held at `ReqNodeNotAvail, Reserved for maintenance`.

That window is worth ~20k usable steps (40% of the remaining 50k), so a ONE-OFF
bonus chunk was submitted against the same `SAVE` dir. Three properties keep it
from damaging the real chain -- all three verified in its log:
1. `CHUNK=6 == MAX_CHUNK=6` => hits the backstop and **queues no successor**, so
   its short 4:30 walltime is never propagated. (Its log still PRINTS a
   "propagating to successor" line before the backstop check -- that line is
   emitted unconditionally and is misleading; confirm the
   `no successor queued` line beneath it.)
2. `THEN_SBATCH` deliberately NOT passed => it cannot fire the follow-on hook.
   Chunk 3 (5290666) already carries it (probe-verified).
3. `--deadline=2026-09-08T08:55:00` => SLURM CANCELS it if it cannot finish in
   time, instead of letting it run past the outage and have TWO jobs writing the
   same `resume_state.pt` concurrently.
⇒ generalisable: this is the safe way to use a pre-reservation gap on a chunked
run -- a deadline-guarded, successor-less, hookless extra chunk.

### OUTCOME: banked 110,000, not the 120,000 I predicted (self-correction)
It ran 20,000 RAW steps in 3:57:31 (1.40 it/s) after a 32.5 min replay load, so raw
reached exactly 120,000 -- but the walltime kill landed ON that boundary and the
120k checkpoint never finished writing. `resume_state.pt` holds the prior save:
**110,000**. Yield was 10k usable, not 20k.

★ **SIZING LESSON: a chunk's usable yield rounds DOWN to the last COMPLETED save.**
Size a short/bonus chunk to land COMFORTABLY PAST a `log_per_step` boundary with
room for the ~169 MB `latest.pt` + `resume_state.pt` write -- never to arrive
exactly on one. Predicting "raw steps crosses the boundary" is not enough; the
save has to finish inside the walltime.

| | steps | post-outage rung A |
|---|---|---|
| before bonus chunk | 100k/150k | ~34 h |
| predicted | 120k/150k | ~28 h |
| **actual** | **110k/150k** | **~30 h** |

⇒ the bonus chunk was still worth ~4 h (not ~6 h). Chunk 3 completes stage-1 in
one pass: 40k steps needed = ~7.8 h train + ~0.5 h load = **~8.3 h**, well inside
its 11:59 walltime. Then the hook fires.
Revised post-outage path: **rung A ~30 h**, rung B ~22 h, in parallel.

## The outage began 2026-09-08 09:00 -- and it takes slurmctld down too
At 09:21 `squeue` returned `slurm_load_jobs error: Unable to contact slurm
controller (connect failure)`. During this maintenance the SCHEDULER itself is
down, not just the compute nodes, so nothing can be queried or submitted.

⚠️ **A stale monitor emitted a FALSE `[LEFT QUEUE] pix_stage1` event** -- its
filter read the failed `squeue` as "the job left the queue". Any queue watcher
whose logic is "job absent from squeue => job gone" will lie during an outage.
Verify with `sacct`, and treat an unreachable controller as "unknown", not "gone".

★ **Silence-is-not-success hazard.** A state-CHANGE monitor emits only on new
`(id|name|state)` triples, so if maintenance WIPES the pending queue it emits
NOTHING -- indistinguishable from "still held behind the reservation". A positive
survival check is therefore required, not optional:
`/scratch/yifanr/temp/ladder/await_slurm.sh` blocks until slurmctld answers, then
reports per-job SURVIVED / NOT-in-queue (with the `sacct` state) and prints an
explicit ACTION REQUIRED list. Watching the three critical ids:
5290666 (rung A chunk 3), 5268895 (rung B adapt), 5268896 (rung B eval).

If any were dropped, resubmit per the recipes above. Rung A's stage-1 resumes from
`global_step = 110000`, so a dropped chunk costs queue time only, not compute.

### Outage verified survivable (2026-09-08 09:25)
The controller blip was brief (09:21 unreachable -> 09:25 back). Positive check
returned **ALL THREE SURVIVED**; all still held at
`ReqNodeNotAvail, Reserved for maintenance`. Reservation now `State=ACTIVE`,
160/178 nodes reserved, `EndTime` still the 365-day placeholder.

★ **Correction to my own worry above:** the state-change monitor DOES cover a
queue wipe after all. Its `sacct` branch re-checks every id it has ever seen, and
a cancelled job reports `CANCELLED` there -- a NEW `(id|name|state)` triple, hence
an event. The uncovered case would be a job vanishing from squeue AND sacct, which
does not happen (sacct retains records). So no extra presence-poller is needed;
`await_slurm.sh` was still worth running, but as a one-shot confirmation rather
than an ongoing necessity.

## HANDOFF -- if this session ends before the outage lifts
Nothing needs doing. Both arms are self-driving:
- `5290666` resumes rung A stage-1 from `global_step=110000`, needs ~40k more
  steps (~8.3 h incl. load), then its verified `THEN_SBATCH` hook submits
  `A_plainh3_ad` (adapt) and `A_plainh3_fin` (n=500 eval) on aip-rhinehar.
- `5268895` -> `5268896` run rung B's adapt (~21.5 h) and headline eval.
To pick the thread back up: read this file top to bottom, then
`squeue -u $USER`, then `scripts/cmp_adapt_runs.py` on the two adapt dirs.
Verify before trusting rung A: its adapt log must read `horizon=3 [kept]` and
must contain NO `[anchor]` pool lines.

## ★ OUTAGE OVER -- both arms running (2026-09-08 10:01)
The maintenance lasted **~1 hour**, not days: reservation active 09:00, gone by
10:00, and both held jobs started at 10:00:57. (⇒ the 365-day `EndTime` really was
a pure placeholder; do not plan around it in either direction.)

**rung A chunk 3** (5290666, kn059): `progress before this chunk: 110000 / 150000`,
needs 40k steps ~= 8.3 h ⇒ this chunk COMPLETES stage-1 and fires the hook.
Chunk 4 (5302052) is queued as the backstop and will early-exit.

**rung B adapt** (5268895, kn001) verified on all three checks:
| check | rung B | rung C reference |
|---|---:|---:|
| anchor pool lines (`[anchor]`/`encoded once`/`bc/anchor_loss`) | **0** | 4 |
| MPPI recipe | ORIGINAL | ORIGINAL |
| `horizon=N [kept]` | **5** (dynens stage-1) | 5 |

### Projected completion
| deliverable | ETA |
|---|---|
| rung B headline n=500 | **2026-09-09 ~08:00** |
| rung A headline n=500 | **2026-09-09 ~16:20** |

## ★ Rung A self-launched correctly; PROVISIONAL early read on the anchor
### The self-launching chain worked end to end (2026-09-08 18:25)
- stage-1 `global_step = 150000` (complete, 3 real chunks + the bonus chunk)
- 5290666 hit `distillation COMPLETE at 150000` -> `[then] submitted OK`
- chunk 4 (5302052) early-exited and logged `[then] hook already fired (marker
  ...) -- not resubmitting` ⇒ **both completion paths and the fire-once guard
  exercised for real**
- adapt 5310474 + eval 5310475 both on **aip-rhinehar** (account propagation held
  through the hook)

### ★★ Rung A geometry CONFIRMED at runtime
`horizon=3 [kept]` and **0** anchor lines ⇒ it genuinely loaded the plain_h3
stage-1, not dynens geometry. This was the check flagged above; it passes.

### PROVISIONAL: matched mid-run comparison, rungs B vs C at 30-40k transitions
B and C differ ONLY in `ANCHOR_COEF` (same stage-1 ckpt, same demo draw, same
recipe), so this is a clean paired contrast -- but it is NOT steady state.

| rung | 30-40k mean (n=6) | own 75-100k window |
|---|---:|---:|
| B (no anchor) | **0.317** | pending |
| C (anchor) | **0.672** | 0.875 |

⚠️ **DO NOT QUOTE THIS.** Iron law #4: mid-run gaps in this project have flipped
sign at steady state (-0.025 mid-window -> +0.103 steady; a 2x2 interaction
+0.151 -> -0.136). Rung B has 60k transitions left and C's own curve climbed
0.672 -> 0.875 across exactly that stretch.

If the +0.355 direction DOES survive to steady state it is a real finding, not a
confirmation: §42 holds that the anchor's gain is largest on a **WEAK** base, so
r18's stronger base predicts a SMALLER anchor effect than vit96's. Observing a
much LARGER one -- and with the opposite sign to vit96's confirmed -0.072 --
would contradict §42's ordering in both directions and require an explanation
that is not about base strength.

## ★★ RUNG B COMPLETE -- the anchor delta on r18 is LARGE and POSITIVE (2026-09-09)
Rung B reached the full 100k in 20:22:02. Both arms steady-state, matched budget,
same stage-1 ckpt, same demo draw, differing ONLY in `bc_anchor_coef`:

| rung | 75-100k window (n=13) | policy_only | planner mult |
|---|---:|---:|---:|
| B (no anchor) | **0.425 +-0.054** | 0.203 | x2.09 |
| C (+anchor) | **0.875 +-0.030** | 0.249 | **x3.51** |

**anchor delta = +0.450** (in-training window measure).

⚠️ **Statistics done honestly.** `cmp_adapt_runs.py::stat` returns **SD, not SE**,
and the 13 points are SERIAL evals along one checkpoint trajectory ⇒
autocorrelated. A t-test on them would overstate precision. The defensible claim
is that **+0.450 is ~8x the larger SD**, decisive under any correlation
assumption. Single stage-1 seed and single (shared, hence paired) demo draw, so
the DELTA is well measured while seed variance is not.

★ The mid-run +0.355 GREW to +0.450 rather than flipping ⇒ iron law #4's failure
mode did not bite here, but checking was still correct: it has bitten before.

### Mechanism worth following up
The anchor barely moves the policy head (0.203 -> 0.249) but nearly DOUBLES the
planner multiplier (x2.09 -> x3.51). Since the stage-2 anchor pulls the policy
toward pi_src, and the policy IS MPPI's proposal distribution, this reads as
"keep the proposal inside the region the dynamics model actually models well"
rather than "make the policy better". That would tie to §36 (proposal quality is
what pays; +0.138 for a flow proposal) and §27 (only model error ON THE
ON-POLICY DISTRIBUTION matters).
⚠️ Provisional: the multiplier is computed from in-training points and inherits
the same optimism as the window (rung C: window 0.875 vs independent 0.766).

## ★★★ RUNG B HEADLINE + THE REAL FINDING: encoder x anchor INTERACTION (2026-09-09)
Rung B independent n=500: **0.334 (167/500)**. Window said 0.425, so optimism was
0.091 -- close to rung C's 0.109, i.e. roughly CONSTANT across arms, which is why
the delta survives the change of measure (window +0.450 vs independent +0.432).

### The 2x2 factorial (dynens_h5 fixed, n=500/cell, SAME demo draw, same harness)
| | no anchor | +anchor | anchor effect |
|---|---:|---:|---:|
| **vit96** | 0.338 (169/500) | 0.266 (133/500) | **-0.072** (z=-2.5) |
| **r18@224** | 0.334 (167/500) | **0.766** (383/500) | **+0.432** (z=+15.2) |
| **encoder effect** | **-0.004** (z=-0.1) | **+0.500** (z=+18.3) | **INTERACTION +0.504** (z=+12.4) |

★★ **Without the anchor, r18@224 buys NOTHING over vit96 (0.334 vs 0.338).**
The whole "resolution + ImageNet prior is worth ~0.5" effect is an INTERACTION
with the source anchor, not an encoder main effect.

⇒ **This revises the ★★★ PIXEL ENCODER DECISION block in `CLAUDE.md`**, which
attributes ~0.5 to the encoder alone ("比这个项目试过的任何其他杠杆都大得多").
The 0.5 is real but only in the `+anchor` row. Quoting it as an encoder effect
overstates what the encoder does on its own by the full amount.
⇒ CLAUDE.md's own warning to redo the vit96 ladder on r18 (because §42's
interaction structure might flip the anchor's sign) was CORRECT about the
mechanism -- but the flip runs OPPOSITE to §42's prediction. §42 says the anchor's
gain is largest on a WEAK base; here the STRONGER base shows the larger gain
(+0.432 vs -0.072). So "base strength" does not explain the pixel anchor.

⚠️ Limits: ONE stage-1 seed per cell. The interaction is ~12x its own SE from
eval sampling, but stage-1 seed variance is unmeasured; the shared demo draw makes
the cells paired (~0.027) rather than carrying the 0.092 single-draw sigma.

### Standing ladder (independent n=500 unless noted)
```
state (published, n=10 seeds, window):
  plain_h3 0.564  --+0.249-->  dynens_h5 0.813  --+0.081-->  0.979
vit96:
  plain_h3 0.218  --+0.120-->  dynens_h5 0.338  ---0.072-->  0.266
r18@224:
  plain_h3 [A]    --   ?   -->  dynens_h5 0.334  --+0.432-->  0.766
```
Only the r18 ENSEMBLE delta is still open; it needs rung A (5310474 -> 5310475).
