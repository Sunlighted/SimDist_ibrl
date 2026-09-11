# Agent prompt — r18@224 ablation ladder: does ensemble + anchor help on PIXEL? (2026-09-07)

Hand this to a fresh agent working in `/project/6101830/yifanr/SimDist_ibrl`.

## THE RESEARCH QUESTION — the only one
**Do the dynamics ensemble and the source anchor reproduce on PIXEL the gains they
gave on STATE?** This is NOT a pixel-vs-state comparison. You are rebuilding
*this* ladder on pixel and reporting the two deltas:

```
state (published, docs/on-policy-adaptation.md, n=10 seeds, 75-100k window):
  plain_h3 0.564  --+0.249 ensemble-->  dynens_h5 0.813  --+0.081 anchor-->  0.979
r18@224 (YOUR JOB):
  plain_h3  ???   --   ?  ensemble-->   dynens_h5  ???   --   ?  anchor-->   0.766*
```
`*` already measured: r18@224 `dynens_h5` + anchor = **0.766** (383/500, independent
n=500). You need the other two rungs on r18.

## ★ ENCODER: r18@224 ONLY. ViT@96 IS RETIRED.
`enc_type: resnet18_pretrained`, `rl_image_size: 224`. Do NOT run ViT@96 and do not
quote vit96 numbers as results. Measured, same demo draw / same replay / independent
n=500: **r18@224 0.766 vs vit96 0.266.** Resolution + the ImageNet prior is worth
~0.5 — more than data scale (40x), augmentation (null), or history.

⚠️ A vit96 ladder EXISTS (`plain_h3 0.194 -> dynens 0.376 -> +anchor 0.323`, i.e.
ensemble **+0.182** but anchor **−0.053**). Treat it as a HYPOTHESIS to test, not a
result to confirm: the anchor's sign may flip on r18 because §42 says the anchor's
gain is largest on a WEAK base, and r18's base is far stronger.

## WHAT TO RUN — 2 new arms (the third exists)

| rung | stage-1 config | adapt | status |
|---|---|---|---|
| A `plain_h3` | NEW: r18 + `num_dynamics_heads:1`, `horizon:3` (wm AND planner) | `ANCHOR_COEF=0` | **run it** |
| B `dynens_h5` | `threading_d0_pixel_simdist_r18_224.yaml` (exists) | `ANCHOR_COEF=0` | **run it** |
| C `dynens_h5`+anchor | same as B | `ANCHOR_COEF=1` | DONE = 0.766 |

⚠️ **Iron law #20**: `num_dynamics_heads`/`horizon` are built in
`WorldModel._build` and `load_state_dict` is strict ⇒ arm A needs its OWN stage-1.
The anchor is an adapt-time knob (`bc_anchor_coef`), so B reuses C's stage-1 ckpt
`exps/mimicgen/threading_pixel224_stage1_seed1/latest.pt` — **do not retrain for B.**

Make arm A's config by copying `threading_d0_pixel_simdist_r18_224.yaml` and
changing ONLY heads/horizon. Anchor the edit on `num_dynamics_heads`, **never on
`horizon: 5`** — both `simdist.planner` and `simdist.world_model` have a `horizon`,
and anchoring there once silently put a knob under `planner` and pyrallis rejected
the run after the queue wait (§46 d7).

## EXACT COMMANDS (these are the measured, working settings)
```bash
source set_env_drac.sh
REPLAY='/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_*/dataset.hdf5+/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_*/_shard_*.hdf5'
SRC=exps/mimicgen/threading_ibrl_source_rnnprior_seed1/model0.pt

# stage-1 (arm A only; ~926 ms/step => ~38.6 h => self-chunks, MAX_CHUNK=6)
CFG=<arm A cfg> SAVE=exps/mimicgen/<name> SRC=$SRC REPLAY="$REPLAY" \
  STEPS=150000 CHUNK=1 MAX_CHUNK=6 \
  sbatch --time=11:59:00 --mem=200G --cpus-per-task=12 scripts/slurm/pixel_stage1.slrm

# stage-2+3 (INTERVAL=2000 is REQUIRED to reach 100k inside 23:59)
ARM=pix224 STAGE1=<stage1>/latest.pt \
  DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5 \
  ANCHOR=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5 \
  SAVE=exps/mimicgen/<name>_adapt NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 \
  EVAL_N=100 ANCHOR_COEF=<0 or 1> \
  sbatch --time=23:59:00 --mem=96G --dependency=afterany:<stage1 job> scripts/slurm/adapt.slrm

# headline (iron law #14 — NEVER quote model0.pt's in-training best)
ADAPT=exps/mimicgen/<name>_adapt N=500 MP=10 TAG=<tag> \
  sbatch --time=2:59:00 --dependency=afterany:<adapt job> scripts/slurm/eval_simdist.slrm
```

## ASSETS (all built and verified — do NOT regenerate)
- **source replay**: 8 merged shards, **100k traj / 17.89M transitions / 340 GB**,
  224x224 JPEG q95 (15.9x, MAE 0.73/255). Both glob patterns in `$REPLAY` above;
  the loader de-dups and tolerates a truncated file.
- **target demos**: `..._pix224_s400k_n200` — n=200, 224x224, from `snapshot_400000`.
  The 96 and 224 draws are the SAME episodes, so all arms share one draw and the
  comparison is PAIRED (~0.027) instead of carrying the 0.092 single-draw sigma.
  **Do not re-collect.** Use `NUM_DEMO=20` (the standard rung).
- **buffer**: `seq_replay_max_transitions: 6500000` => 36,000 episodes / 6.48M /
  **5.9 epochs**, ~119 GB resident (measured 18,319 B/transition). 200G ask.

## GATES — run these after ANY code change (seconds, local, no GPU)
```
python scripts/test_world_model_branches.py   # all 3 arms build; aux_state on every path
python scripts/test_jpeg_replay.py            # JPEG round-trip + frame/state alignment
python scripts/test_history_layout.py         # train-vs-act history layout
python scripts/test_configs_load.py           # pyrallis-loads every config
sbatch scripts/slurm/dedup_regress.slrm       # must stay distill/loss 9.27062321
```

## READ FIRST
`CLAUDE.md` (all iron laws) and `docs/simdist.md` **§46** end to end. §46 records
everything measured in this push, including seven defects and four of my own wrong
conclusions. The ones that will bite you:
- **§46k — stage-1 `score/score` DOES NOT predict downstream value.** It is
  `use_planner=0` and pre-adaptation. Old and new data both read 0.02 there while
  differing ~11x after adaptation. Never rank stage-1 ckpts by it.
- **§46i — `train_simdist_adapt.py` CANNOT RESUME.** A timed-out adapt is finished
  where it stopped. Hence `INTERVAL=2000`.
- **§46j — 150k vs 800k stage-1 steps is worth ~+0.076** (state 0.430 -> 0.506).
  Real but modest. All ladder arms use 150k, so it cancels within the ladder.
- **Iron law #4** — read WINDOWS. Single n=100 points span 0.03-0.58 here.
  `scripts/cmp_adapt_runs.py` does the 75-100k window and refuses to compare runs
  at different budgets. `scripts/cmp_stage1_runs.py` does the same for stage-1 and
  will NOT call a narrowed source-target gap a win.
- **Iron law #1/#2** — deltas under 0.05 are unreadable; ±0.1 needs 300-500 episodes.
- The chunk chain propagates `--mem/--time/--cpus/--account`; before that fix a
  successor fell back to the 72G header and OOM'd (§46 d).

## DELIVERABLE
The two r18 deltas with independent n=500 numbers for all three rungs, written into
`docs/simdist.md` as a new numbered section, plus a one-line verdict:
**does the ensemble reproduce on pixel? does the anchor?** If the anchor's sign
differs from vit96's −0.053, say so explicitly and connect it to §42's
weak-base interaction.

Budget: 2 stage-1 (one is ~38.6 h, chunked) + 2 adapts (~15-22 h) + 2 re-evals.
~3 days wall clock on `aip-gigor`. `aip-rhinehar` exists on Killarney and would
roughly double throughput if the association is added (same filesystem, no data
movement; pass `--account=` and the chain propagates it).
