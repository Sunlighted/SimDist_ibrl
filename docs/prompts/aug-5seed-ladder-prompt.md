# TASK PROMPT — add the missing augmentations, then run the REAL 5-seed r18@224 ladder

You are picking up a live research codebase. **Read `CLAUDE.md` end to end first**
(all 24 iron laws), then `docs/simdist.md` **§44, §46, §47** and `docs/infra.md`
**§7z**. §47 is the single most important one: it is the previous, single-seed
version of the experiment you are about to run properly.

---

## THE DELIVERABLE (two parts, in order)

### Part 1 — implement the three missing augmentations
### Part 2 — run the 3-rung ladder at **5 seeds** and report the two deltas with error bars

The scientific question is unchanged from §47, but §47 was **n=1 seed** and its
ensemble delta is **under-powered by construction** (see "Why 5 seeds" below).
This run is the one that gets quoted.

```
           base SimDist+demos      +ensemble           +anchor
rung        A  plain_h3      -->   B  dynens_h5  -->   C  dynens_h5+anchor
heads/H     1 head,  H=3           5 heads, H=5        5 heads, H=5 (same stage-1 as B)
ANCHOR_COEF 0                      0                   1.0
```
Report **ensemble delta (B−A)** and **anchor delta (C−B)**, each as a
mean-over-5-seeds with a paired-within-seed SD, on BOTH measures (see "Analysis").

Reference points you are trying to place these against:
```
state (published, docs/on-policy-adaptation.md, n=10 ADAPT seeds, 75-100k window):
  plain_h3 0.564 ±0.171  --+0.249-->  dynens_h5 0.813 ±0.085  --+0.081-->  0.979 ±0.010
r18@224, n=1 seed, SHIFT-AUG ONLY (§47, independent n=500):
  plain_h3 0.418          ---0.084-->  dynens_h5 0.334          --+0.432-->  0.766
```

---

## PART 1 — AUGMENTATION

### What exists
`rl/simdist_agent.py::_augment_obs` (called from `update_distill` via
`:422`/`:424`) currently implements **random shift only**, driven by
`world_model.aug_shift_pad` (r18 configs use **8**, = the repo's conventional
`pad=4` at 96 rescaled to 224 — see §46⑫, getting this wrong is silent).
`world_model.aug_prop_noise` is **implemented but set to 0.0**.

`RandomShiftsAug` (replicate-pad + random crop back to size) IS the reference's
**random crop**, so that row of §44 is already satisfied. Do not add a second crop.

### What to add
| §44 reference aug | status | action |
|---|---|---|
| random crop | ✅ = `aug_shift_pad` | none |
| proprio Gaussian noise | implemented, **off** | expose + turn on |
| **colour jitter** | ✗ missing | **implement** |
| **Gaussian blur** | ✗ missing | **implement** |

### HARD CONSTRAINTS (each of these has already caused a real bug here)
1. **ONE augmentation draw per (sample, camera), SHARED across all H+1 frames.**
   This is the load-bearing invariant of `_augment_obs`. Independent per-frame
   parameters inject apparent camera jitter into the very transition the dynamics
   head is learning, AND desynchronise the network input from the detached
   consistency target built from the same frame. Applies to jitter and blur
   exactly as it does to the shift: draw the jitter factors / blur sigma once per
   (sample, camera), reuse for every timestep.
2. **Every new knob defaults to 0 / off**, so state arms and all existing pixel
   checkpoints stay byte-for-byte identical and old ckpts still `strict` load.
3. **float32 is FINE — do not contort to stay in uint8.** The uint8 path exists
   because whole-pixel shifts are exactly lossless there; jitter and blur are
   inherently float. Measured: the `[256,6,3,224,224]` block is 231 MB as uint8
   and **925 MB as float32 — a 694 MB delta, 1.4% of an L40S's 48 GB.** Convert
   once, do jitter+blur+shift in float, cast back at the end. Do NOT cite memory
   as a reason to skip these.
4. **Iron law #23 — `WorldModel.__init__` has THREE branches that each `return`**
   (`raw_state`, `pixel`, `latent`). Anything you add must exist on all three, or
   it fails at runtime on one arm only. Declare new attributes BEFORE the
   branches. Read `getattr(..., None)` at every use site.
5. Augmentation is **stage-1 only** by construction — anchor pools and eval must
   stay untouched (§46⑨). Do not let it leak into `train_simdist_adapt.py`.
6. Order matters and should match the reference's spirit: geometric (shift/crop)
   → photometric (jitter) → blur. Keep it deterministic given a seed.

### Suggested knobs (names are yours, but keep the 0=off contract)
`aug_color_jitter` (brightness/contrast/saturation/hue strength, 0=off),
`aug_blur_sigma_max` (0=off; sample σ∈(0,max] per sample-camera),
`aug_prop_noise` (already exists — set it non-zero in the new configs).

### GATES — all must pass before you queue anything (seconds, local, no GPU)
```
python scripts/test_world_model_branches.py   # all 3 arms build; new attrs on every path
python scripts/test_jpeg_replay.py            # JPEG round-trip + frame/state alignment
python scripts/test_history_layout.py
python scripts/test_configs_load.py           # pyrallis-loads EVERY config
sbatch scripts/slurm/dedup_regress.slrm       # must stay distill/loss 9.27062321
```
★ `dedup_regress` is **required this time** (unlike §47's config-only change): you
are editing the training path. With all aug knobs at 0 the loss must be
**bit-identical**. If it moved, your default-off contract is broken — stop.

### ADD A NEW GATE (you are expected to write this)
`scripts/test_augmentation.py`, login-node, seconds. Must assert:
- each aug is a **no-op at 0** (tensor-equal), so defaults are safe;
- with a knob on, the output **actually differs** from the input (catches a
  silently-inert aug — §44's whole problem was a hook nobody ever called);
- **the same (sample, camera) gets the SAME parameters across all H+1 frames**
  and **different samples get different ones** — this is constraint 1 and it is
  the one that will silently ruin the dynamics head if wrong;
- uint8→float32→uint8 round-trip stays in range and the shift stays lossless;
- runs on all three WorldModel branches.
⚠️ The dedup gate CANNOT catch a broken aug (it runs with aug off). This new gate
is your only protection.

---

## PART 2 — THE 5-SEED LADDER

### Configs
Copy the two §47 configs and add the new aug knobs (jitter, blur, prop noise):
- `release/cfgs/mimicgen/threading_d0_pixel_simdist_r18_224.yaml` (B/C: 5 heads, H=5)
- `release/cfgs/mimicgen/threading_d0_pixel_simdist_r18_224_plain_h3.yaml` (A: 1 head, H=3)
Name the new pair `..._r18_224_aug.yaml` / `..._r18_224_plain_h3_aug.yaml`.
⚠️ **Anchor every edit on a block-unique key such as `num_dynamics_heads`, NEVER
on `horizon`** — both `simdist.planner` and `simdist.world_model` have a
`horizon`, and anchoring there once put a knob under `planner` and pyrallis
rejected the run *after* the queue wait (§46 d7).
⚠️ **Iron law #20**: `num_dynamics_heads`/`horizon` are built in
`WorldModel._build` and `load_state_dict` is `strict` ⇒ **rung A needs its own
stage-1; it cannot be ablated at adapt time.** Rungs B and C **share** one
stage-1 per seed (the anchor is an adapt-time knob) — do NOT retrain for C.

### NO DATA RE-COLLECTION IS NEEDED
Augmentation is **train-time only**: `update_distill` calls
`states = self._augment_obs(states)` on the DECODED uint8 frames before the first
encode. The stored 340 GB JPEG replay
(`/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_*`) is reused unchanged.
§44's "the colour-jitter/blur/crop augmentations need raw pixels at load time" is
an argument that JPEG/video STORAGE suffices (you decode to raw at load), NOT that
`D_sim` must be regenerated. ⇒ 0 GPU-h of collection; the only new compute is the
stage-1 retrains, which this 5-seed run needs regardless.

### ★★ WHAT THE PUBLISHED STATE PROTOCOL ACTUALLY DID (read this before designing)
From `docs/on-policy-adaptation.md`:
- **ONE stage-1 world model per architecture**, and the 10 seeds are **ADAPT
  seeds** (it lists "one stage-1 world model" in its own Caveats).
- **Each seed uses a DIFFERENT 20-demo draw, shared across arms within that seed
  (paired)** — lines 138 and 151 together. So the published sds (`plain_h3`
  0.171, `dynens_h5` 0.085, `anchor` 0.010) **DO include §38's draw variance**.
  The blocked draw design below is therefore FAITHFUL, not a departure.
- ⚠️ **`dynens_h5*` scores are `score_best`, a max over 4 pessimism
  coefficients, carrying ~+0.03-0.05 of selection bias** that `plain_h3` cannot
  take. So the published `dynens_h5` 0.813 and the +0.249 ensemble delta are
  **hyperparameter-selected and inflated by roughly that much**. Our arms take no
  such max. ⇒ when you compare your ensemble delta to +0.249, say this out loud;
  the honest published comparator is closer to **+0.20-0.22**.
- ⚠️ The published **anchor** +0.081 was measured on `dynens_h5_slow`
  (unsaturated demos, 0.898 → 0.979), not on `dynens_h5`. Name the rung you compare to.

### ★ Seeds: **1 stage-1 per architecture + 5 ADAPT seeds** (matches published)

**PHASE 1 — this is what you run.** Train exactly TWO stage-1 models:
`A_stage1` (1 head, H=3) and `BC_stage1` (5 heads, H=5). Then run 5 adapt seeds
x 3 rungs on them. Rungs B and C **share `BC_stage1`** — the anchor is an
adapt-time knob, do NOT retrain for C.

**Why not 5 stage-1 seeds** (an earlier draft of this plan said to; it was wrong):
the two deltas are affected differently.
- **The anchor delta is IMMUNE to stage-1 seed count**, because B and C share one
  stage-1. At 5 adapt seeds it is a clean paired contrast — exactly the design
  that produced the published +0.081 at z=6.0.
- **Only the ensemble delta is confounded** with checkpoint identity, since it
  compares two architectures ⇒ two checkpoints. **But the published ladder has
  that same confound** and lists "one stage-1 world model" in its Caveats.
⇒ Using 5 stage-1 seeds would put a variance component in OUR error bars that is
absent from the published ones, making an equal true effect look weaker here.
That is an apples-to-oranges comparison and defeats the purpose of the rerun.
**Match the estimand first.**

**PHASE 2 — ONLY if Phase 1's ensemble delta lands ambiguous** (|delta| < ~0.10,
or the window and independent measures disagree in sign as they did in §47):
add **2 more stage-1 seeds per architecture** and rerun rungs **A and B only**
(C is unaffected — it shares B's stage-1 and its delta is already clean).
That de-confounds the ensemble delta from checkpoint identity. ~212 GPU-h.
Report it as a clearly-labelled extension that goes BEYOND the published
protocol, not as the headline comparison.

**Power, for reference.** Published sds are `plain_h3` 0.171, `dynens_h5` 0.085,
`anchor` 0.010, and they include draw variance. At n=1 the ensemble delta SE is
~0.19, so even a true +0.249 is ~1.3sigma — undetectable, which is why §47's n=1
ensemble result was worthless. At n=5 paired, SE ~0.085 ⇒ ~2.4-2.9sigma against
the honest (de-selected) comparator of +0.20-0.22. The anchor delta was already
~5sigma at n=1, which is why §47's anchor result is trustworthy.

### ★★ Demo draws — THERE IS A BLOCKER YOU MUST FIX FIRST
`rl/replay.py:206` prints `loading first {num_data} episodes` — the 20 demos are
the **first 20, unseeded**. If you do nothing, all 5 seeds train on the IDENTICAL
demos and your error bars EXCLUDE §38's single-draw σ = **0.092, the largest
noise source in this project**. That would make the whole 5-seed run misleading.

Fix: add a `target_demo_offset` (or seeded-subset) flag so seed *s* takes a
**disjoint** 20-episode slice. The pool
`release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5`
holds **200 episodes ⇒ 10 disjoint 20-demo draws**; use offsets 0,20,40,60,80.
Default the flag to 0 so existing runs are unchanged; add it to the config gate.

**Blocked design — this is the point:**
- **within a seed**, all 3 rungs share that seed's draw ⇒ the two deltas are
  PAIRED against ~0.027 instead of carrying 0.092;
- **across seeds**, draws differ ⇒ the error bar on the mean delta honestly
  includes draw variance.
Log the offset in each run so it is auditable.

### Matched budget — verify, do not assume
Every arm: stage-1 **150k steps**, `seq_replay_max_transitions: 6500000`,
`source_replay_num_data: 4500` per shard × 8 shards. Each run prints
`[budget] transitions=... => EPOCHS = ...`; §47's three arms all read
**`transitions=6479983`** and yours must too. `[seq-buffer]` must show 8 shards
(§46⑪: a capped buffer fed from several files evicts from the FRONT and can
silently keep only the last shards).
Adapt, all arms: `NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100`,
`DYN_ANCHOR_COEF=0.1`, ORIGINAL MPPI recipe.
⚠️ `INTERVAL=2000` is required to reach 100k inside walltime — **raise INTERVAL,
never lower EVAL_N** (EVAL_N drives TopkSaver selection ⇒ changing it is a
protocol change, §46i). `train_simdist_adapt.py` **CANNOT RESUME**: a timed-out
adapt is finished where it stopped.

### Commands (measured and working)
```bash
source set_env_drac.sh
REPLAY='/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_*/dataset.hdf5+/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_*/_shard_*.hdf5'
SRC=exps/mimicgen/threading_ibrl_source_rnnprior_seed1/model0.pt

# stage-1 (self-chunks; ~50k USABLE steps per 11:59 chunk -- see infra §7z ③)
CFG=<aug cfg> SAVE=exps/mimicgen/<name> SRC=$SRC REPLAY="$REPLAY" \
  STEPS=150000 CHUNK=1 MAX_CHUNK=6 \
  THEN_SBATCH='bash scripts/slurm/<your followon>.sh' \
  sbatch --time=11:59:00 --mem=200G --cpus-per-task=12 --account=aip-rhinehar \
  scripts/slurm/pixel_stage1.slrm

# adapt
ARM=pix224 STAGE1=<stage1>/latest.pt \
  DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5 \
  ANCHOR=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5 \
  SAVE=<adapt dir> NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100 \
  ANCHOR_COEF=<0 or 1.0> DEMO_OFFSET=<0/20/40/60/80> \
  sbatch --time=23:59:00 --mem=96G --account=aip-rhinehar scripts/slurm/adapt.slrm

# headline (iron law #14 -- NEVER quote model0.pt's in-training best)
ADAPT=<adapt dir> N=500 MP=10 TAG=<tag> \
  sbatch --time=2:59:00 --account=aip-rhinehar --dependency=afterany:<adapt> \
  scripts/slurm/eval_simdist.slrm
```

### Orchestration — read `docs/infra.md §7z`, it will save you a day
- A self-chunking stage-1 has **no final job id**, so you cannot hang the adapt on
  `--dependency=afterany:<stage1>`. Use the **`THEN_SBATCH` hook** in
  `pixel_stage1.slrm` (default empty ⇒ inert; marker `<save>/.then_submitted`
  makes it fire exactly once; already verified on two real chains). Values with
  spaces survive `--export` (probe-verified).
- `--export=ALL` carries **env vars only, never sbatch flags** ⇒ forward
  `--mem/--time/--cpus/--account` at every hop, including out of the hook (the
  slurm scripts hardcode `#SBATCH --account=aip-gigor`).
- Use **`--account=aip-rhinehar`**: fairshare 0.657 vs gigor's 0.106 (~6× the
  priority factor). To move an already-queued job use
  `scontrol update job=<id> Account=aip-rhinehar` — it keeps the job id so
  dependency chains survive; cancel-and-resubmit does not.
- **Do not edit a slurm script while chunks are executing it** — bash reads
  scripts incrementally and can resume at a stale offset.
- Ignore the two benign per-eval tracebacks (NFS `.nfsXXXX` "Device or resource
  busy"; `EGLError(EGL_NOT_INITIALIZED)` in `__del__` under "Exception ignored
  in"). A crash filter must match **exception lines** minus those, not
  `Traceback (most recent` — see §7z ⑥.

### Cost (measured; aug adds a few % to stage-1 step time)
| item | GPU-h |
|---|---:|
| data collection | **0** (train-time aug; 340 GB replay reused) |
| **Phase 1** stage-1: 2 runs (plain_h3 ~30.6 h + dynens ~40.5 h) | ~71 |
| **Phase 1** adapt: 5 seeds × 3 rungs (A 14.4 h, B 20.4 h, C 21.5 h) | ~282 |
| **Phase 1** 15 headline n=500 evals | ~8 |
| **Phase 1 total** | **~360** (~15 GPU-days; ~2–3 days wall clock on rhinehar) |
| Phase 2 (conditional): +2 stage-1 seeds/arch + rungs A,B | +~212 |

---

## ANALYSIS — how to report it (do not skip this section)

1. **Report BOTH measures for every rung**: the 75-100k window
   (`scripts/cmp_adapt_runs.py`) and the independent n=500
   (`scripts/eval_simdist.slrm`). The published state ladder reports **window
   means**, so for comparing DELTAS the window is the like-for-like estimand;
   independent n=500 is the right way to state an **absolute** number.
2. **★ §47a — the independent re-eval does NOT remove TopkSaver's selection
   bias.** `model0.pt` is the **max** over in-training eval points, and the
   premium scales with that arm's curve variance. Measured in §47:
   `plain_h3` SD 0.099 → best was **+0.245** above its window; `dynens` 0.054 →
   +0.095; `+anchor` 0.030 → +0.065. `plain_h3` was the ONLY arm whose
   independent eval landed ABOVE its window mean.
   ⇒ **Always print each arm's window SD next to its delta.** If two arms'
   SDs differ materially, their independent numbers are NOT exchangeable and the
   delta must be read on the window. This is precisely why §47's ensemble delta
   flipped sign between measures.
3. **Compute deltas PAIRED WITHIN SEED**, then average over the 5 seeds, and
   report the SD of the 5 paired deltas. Do not compare cross-seed means
   directly — that throws away the blocking and reinflates draw variance.
4. `cmp_adapt_runs.py` refuses to compare runs at different budgets and marks a
   truncated window NON-STANDARD. Do not work around it; fix the run.
5. **Iron laws #1/#2**: deltas < 0.05 are unreadable; ±0.1 claims need 300-500
   episodes. **#4**: read WINDOWS, never single points — single n=100 points span
   0.03-0.58 here, and mid-run gaps in this project have flipped sign at steady
   state. **#3**: anything you discover post-hoc needs a fresh pre-registered seed.

## WRITE-UP
New numbered section in `docs/simdist.md` (**§48**; §47 is taken). It must contain:
- the 3×5 table, both measures, with per-arm window SDs;
- the two deltas as paired means ± SD over 5 seeds;
- an explicit verdict: **does the ensemble reproduce? does the anchor?**
- a comparison to §47's n=1 numbers, saying plainly which §47 conclusions
  survive 5 seeds and which do not — §47's ensemble delta is expected to be the
  casualty, and if the anchor's +0.432 shrinks materially, SAY SO;
- whether adding jitter/blur/prop-noise moved anything, by comparing against
  §47's shift-only arms (confounded by seed count — label it as suggestive, and
  note the §44 rows still unmatched: 2 views not 3, 20 Hz not 5 Hz, 512-d SimNorm
  latent not 64-d MLP, no history).
Backfill `CLAUDE.md` **only** if a conclusion changes how the next experiment
should be run — that is its stated bar. §47's ★★★ PIXEL block and its iron-law-#14
addendum are the two places that would need editing.

## THINGS THAT WILL BITE YOU (all measured, all in the docs)
- Stage-1 progress checkpoints only at `log_per_step` boundaries ⇒ **usable steps
  ≠ tqdm steps** (measured: tqdm 50,249 but `resume_state.pt` held 50,000).
  Budget ~50k usable per 11:59 chunk.
- §46k: **never rank stage-1 checkpoints by `score/score`** — it runs with
  `use_planner=0`, pre-adaptation, and two datasets differing 11× downstream both
  read 0.02 there.
- §46j: 150k vs 800k stage-1 steps is worth ~+0.076. All arms here use 150k so it
  cancels within the ladder, but it means **absolute numbers are not comparable
  to the published state ladder** — only deltas are.
- Verify at runtime that each rung loaded the right geometry: the adapt logs
  `horizon=N [kept]`. **Rung A must read `horizon=3 [kept]`**; B and C must read 5.
  Rungs A and B must show **zero** `[anchor]` pool lines; C must show them.
- History is **dropped** by decision: measured **0.638 vs 0.766 (z=−4.47)**, i.e.
  K=8 history HURT, and it also dragged `policy_only` 0.249 → 0.166. Do not
  re-add it. Hz, view count, and the Eq.3 objective are likewise out of scope
  (SimDist has a 20-demo `+BC` variant, so our stage-2 is faithful to it).


## COST SPLIT, and the one place to economise
Per seed: **stage-1 71 h (56%)** + **adapt 56 h (44%)**. Stage-1 dominates.
★ But per §46i, **~95% of an adapt's wall-clock is the 50 x 100-game evals**, not
the dynamics training — you are paying for the measurement protocol, not the
learning. So if you ever need to cut adapt cost, the honest lever is eval DENSITY
(raise `INTERVAL`), never `EVAL_N` (which changes TopkSaver selection ⇒ protocol
change, iron law #15). Do NOT cut it for this run: `INTERVAL=2000` is already the
minimum that reaches 100k in walltime, and §47's rungs used exactly that.
If you must shrink scope, cut adapt SEEDS (5 → 4 → 3) and state the reduced power
in §48. Never cut the protocol itself, and never skip Phase 1's 5 adapt seeds down
to 1 — that is exactly the failure mode §47 already demonstrated.
