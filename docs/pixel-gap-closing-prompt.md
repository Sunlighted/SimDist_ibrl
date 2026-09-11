# Agent prompt — close the pixel/state gap (2026-09-02)

Hand this to a fresh agent working in `/project/6101830/yifanr/SimDist_ibrl`.
Governing principle: **match the reference recipe (100k trajectories) and our own STATE
arm's settings everywhere except the encoder.**

⚠️ **Both of our existing source replays are under-sized, for reasons that were never
deliberate.** See "DATA PROVENANCE" below before doing anything in Phase B.

---

You are working in `/project/6101830/yifanr/SimDist_ibrl`, a fork of `ibrl` implementing
SimDist (latent world model + MPPI) for sim2sim transfer on robosuite Threading_D0. Your
job: close the gaps between our PIXEL port and our own STATE arm, WITHOUT sacrificing the
two things this project deliberately adds — the dynamics ensemble and target-time policy
finetuning.

## FIRST: read these, in order. Do not skip.
1. `CLAUDE.md` — the "方法论铁律" iron laws (all 22). Non-negotiable; each was learned by
   breaking something.
2. `docs/simdist.md` §43, §44, §45. **§45 is the key finding.** §44 is the paper-vs-us
   divergence table. §43's *earlier* conclusions were RETRACTED — read the correction
   notices and do not build on the withdrawn claims.
3. `docs/pixel-progress.md` — current state and full results table.
4. `docs/infra.md` §7 — known pitfalls.

Environment: `source set_env_drac.sh`. SLURM (Killarney), account `aip-gigor`. Never train
on the login node.

## GOVERNING PRINCIPLE
**Match our own STATE arm's CONFIG everywhere except where genuinely inapplicable (the
encoder), and match the REFERENCE DATA SCALE (100k trajectories) rather than what our
state arm happens to have on disk.**

Config parity: our state arm is measured to work (dynamics-only target transfer
0.053 → 0.280; +BC 0.540 at independent n=500), so copy its settings rather than the
paper's where they differ.

Data scale: do NOT copy our state arm's dataset size — it is itself 5× under the reference
(see DATA PROVENANCE). The project owner confirms the reference recipe is **100k
trajectories**. ⇒ **the state arm's 0.540 is a PROVISIONAL baseline**, produced on a
5×-undersized replay; expect it to move when regenerated, and do not treat it as settled.

## ⚠️ DATA PROVENANCE — read before Phase B
Neither existing source replay is the size anyone intended:

| replay | on disk | intended | what actually happened |
|---|---:|---:|---|
| `threading_d0_source_diverse_v2` (state) | **20k traj / 4.08M** | 100k | job 5113397 was launched `NUM_TRAJ=20000` and finished in **1:59:42 of a 2:59:00 window**. 100k would have taken ~10 h and fit an 11:59 (b2) job. The dataset was sized to the fast-queueing 3 h tier — **5× the data traded for a shorter queue wait.** |
| `threading_d0_source_diverse_pix` (pixel) | **2400 traj / 0.44M** | 3000 (itself ~8× too small) | job 5113396 requested `NUM_TRAJ=3000`, ran 24 min, died **OUT_OF_MEMORY**, having written 2400. **No pixel generation job has ever completed.** |

`scripts/slurm/gen_source_replay.slrm` documents `NUM_TRAJ=100000` for the full run in its
own usage comment; the 400 default is a calibration size. So the reference number was in
the script the whole time.

⇒ Two consequences: **(a)** `docs/simdist.md` §45's framing of 2400 as "a storage decision"
is wrong — it was a crashed job whose partial output we then trained 461 epochs on;
correct that section. **(b)** The pixel generator OOMs at 3000 trajectories, so it CANNOT
produce 100k as written. Fixing it is a Phase B prerequisite, not an afterthought.

## THE CORE PROBLEM (docs/simdist.md §45)
Our pixel stage-1 does **461 epochs** over a **0.44M-transition** replay; the state arm
does **50 epochs** over **4.08M**; the reference recipe is **100k traj / ~20.4M** and the
paper does **2 epochs**. Result: pixel `source_score` 0.968–0.980 (memorised) but target
transfer ~0.01. See DATA PROVENANCE above for why both replays are under-sized — neither
reason was a modelling decision.

## INVARIANTS — do not change
- `num_dynamics_heads: 5`, `world_model.horizon: 5` (the "dynens" arm; +0.249, findings §31).
- Stage-2 policy BC with `bc_anchor_coef: 1.0` (this fork's contribution; §43 ⑧).
- `--original_mppi 1` on all adapt/eval runs — it already matches the paper's MPPI exactly.
  Confirm via the `[adapt] using ORIGINAL SimDist MPPI recipe ...` log line.
- `decoder_coef: 0.0` (no reconstruction).
- Leave `aux_state_coef: 0.0` — implemented but UNTESTED and deprioritised.

## ⚠️ CRITICAL HAZARD: live jobs share this checkout
`state_dynens` / `pixel_dynens` are training now, each with an `afterany` successor queued.
**A requeued chunk re-invokes python from THIS repo path**, so a code change can land
mid-run and either break a strict `load_state_dict` or silently change behaviour.

⇒ **Every code change must be inert by default**, gated behind a new config field
defaulting to off/zero (see `aux_state_coef` for the pattern). After ANY edit to
`rl/world_model.py`, `rl/simdist_agent.py`, `rl/sequence_replay.py`, `train_simdist.py`:

    sbatch scripts/slurm/dedup_regress.slrm

and confirm the `shared+detach` row still reads `distill/loss 9.27062321` /
`__w_encoder 126.76865` (and `no_detach` 126.83760). If those move, revert.
Check `squeue -u yifanr` before and after. Do not cancel anything you did not start.

## WORK, in order. Validate each phase before starting the next.

### Phase A — augmentation + epoch instrumentation (free, no new assets)
A1. Wire augmentation into stage-1 pixel distillation. The hook exists —
    `rl/pixel_obs_encoder.py:136 forward(obs, augment_fn=None)` and
    `rl/world_model.py:561 encode(state, augment_fn=None)` — but **no caller ever passes
    one**, so we train with ZERO augmentation while the paper uses four kinds (proprio
    Gaussian noise, colour jitter, Gaussian blur, random crop). `common_utils/py/data_aug.py`
    has `RandomShiftsAug`; `bc/bc_policy.py:64` and `rl/q_agent.py:137` use `pad=4`.
    - Gate behind `world_model.aug_*` fields, ALL defaulting to off.
    - ★ ONE random shift per (sample, camera), shared across all H+1 frames of that
      sample. Independent per-frame shifts inject apparent camera jitter into the
      transition the dynamics head is learning. `RandomShiftsAug` is channel-agnostic:
      reshape to `[B*ncam, (H+1)*3, H, W]`, augment, reshape back.
    - Training ONLY. Never at eval/act time, and never when building anchor pools (anchor
      targets are frozen π_src snapshots on a specific view; re-augmenting the input while
      holding the target fixed injects label noise).
A2. Add a startup log line printing transitions / batch / steps / **resulting EPOCH count**
    so this divergence can never hide again. Also add a TARGET-side checkpoint-selection
    / early-stop option: `TopkSaver` currently ranks on `score/source_score`, which IS the
    memorisation signal. (`train_simdist.py` already builds `eval_env` from the cfg's
    `eval_domain_shift`, which is the §12 target.)
    ⚠️ Do NOT pick an epoch count now. At the Phase B target (100k traj / 20.4M
    transitions) 800k steps IS ~10 epochs, which lands between our current 461 and the
    paper's 2 without any step change. Fix the DATA first; revisit steps only if 10 epochs
    still shows the memorisation signature (source high, target flat).
A3. Validation: short pixel stage-1 (~100k steps) on the EXISTING replay with augmentation
    ON, dynens config, plus a matched augmentation-OFF control at the same step count (one
    knob — iron law #9). Report BOTH `score/source_score` and `score/score` (target).
    Success = the source↔target gap narrows vs the aug-off baseline (source 0.98 / target ~0.01).

### Phase B — data scale to the REFERENCE recipe (the main fix)

**Target: 100k trajectories ≈ 20.4M transitions** (mean 204 steps/traj, measured from
`diverse_v2`), generated with the SAME recipe as
`release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5` (same
`scripts/generate_simdist_data.py` settings — snapshot ensemble, p_expert, hand_init
sigma), differing ONLY by `--rl_cameras agentview+robot0_eye_in_hand` (use `+`, NOT a
comma — `sbatch --export` splits on commas). The script's docstring documents the faithful
Algorithm-2 recipe and its CLI defaults already match it.

Sizing (mean 204 steps, 2 cams, 96×96 uint8 = 55,296 B/transition):

| traj | transitions | pixel raw | JPEG @~12× | 800k steps ⇒ |
|---:|---:|---:|---:|---:|
| 20k (what we have) | 4.1M | 0.23 TB | 19 GB | 50 epochs |
| **100k (target)** | **20.4M** | **1.13 TB** | **94 GB** | **10 epochs** |
| paper | ~25M | — | — | 2 epochs |

`/scratch` has ~1.7 TB free (`df -h /scratch`) — raw 96² *fits on disk* but see the RAM
blocker. Note 100k also brings the epoch count to **10**, far closer to the paper's 2 than
our current 461.

#### B0 — PREREQUISITE: fix the generator's OOM
`scripts/generate_simdist_data.py` **died OUT_OF_MEMORY at 3000 trajectories** (job
5113396, 24 min, `--num_proc 10`). It cannot produce 100k as written. Diagnose before
scaling: the likely cause is accumulating all episodes in memory before writing (the
function signature `collect(..., num_traj, ...)` returns `a list of (ep_dict, success)`),
which at 2 cams × 96² × 204 steps is ~11 MB/trajectory per worker. Make the writer
**streaming/incremental** (append each episode to the hdf5 and free it), and re-verify with
a mid-size run (e.g. 5k traj) before launching 100k. Measure `MaxRSS` via
`sacct -o JobID,MaxRSS` and size `--mem` from the measurement, not a guess.
Also: the state generator ran 20k in 1:59:42 at `--num_proc 10`, so 100k ≈ 10 h ⇒ request
**11:59:00** (b2), not the 2:59:00 that caused the original under-sizing.

#### B1 — JPEG storage (load-bearing, not an optimisation)
**RAM, not disk, is the binding constraint.** `SequenceReplayBuffer` is CPU-RAM resident
(`device="cpu"`). 20.4M transitions raw = **1.13 TB of RAM** — impossible. JPEG at ~12×
is **~94 GB**, which needs roughly a 128G `--mem` ask (nodes have 515 GB+, but large asks
queue worse — iron law #16). Implement in `scripts/generate_simdist_data.py` (writer) and
`rl/sequence_replay.py` (loader, `image_keys`, `_frame()`), **backward compatible** so
existing uncompressed datasets still load. **Benchmark ms/gradient-step with and without**
— decode must not starve the GPU. If it does: decode in worker processes, or cache decoded
batches, or fall back to a capped `seq_replay_max_transitions` and report the cap
explicitly (never silently).

#### B2 — remove the buffer cap
**`seq_replay_max_transitions: 450000`** (in `threading_d0_pixel_simdist_dynens.yaml`) is
the ACTUAL cap on what reaches training — it is why the pixel arm sees 445k transitions
regardless of dataset size. Raise it to match the dataset, or more data changes nothing
(the buffer just evicts). **Log the resulting epoch count at startup (Phase A2) so a cap
can never silently undo this work again.**

#### B3 — re-run stage-1
Pixel dynens stage-1 over the new replay, augmentation on, 800k steps (= 10 epochs at
100k traj). `scripts/slurm/pixel_stage1.slrm` self-chunks correctly via
`--dependency=afterany` (iron law #22).

#### B4 — regenerate the STATE replay too, and re-baseline
The state arm is also 5× under the reference. For an honest pixel-vs-state comparison both
arms need the same data scale, so generate a 100k **state** replay (cheap — 0.7 GB per 20k,
so ~3.5 GB, no images) and re-run state dynens stage-1 + adapt. Until then every
state-vs-pixel delta in `docs/pixel-progress.md` is provisional. Do this in parallel with
the pixel work — it is nearly free and it is the actual baseline the project reports.

### Phase C — resolution and encoder (only after B validates)
96×96 was NOT a cost decision; it was inherited (`ResNet96Encoder` is sized for 96,
`networks/min_vit.py` hardcodes 96-based patch counts). Owner's guidance: **224×224 with
ImageNet-pretrained ResNet-18, 2 views, resolution before view count.** Third view, if
ever used, is `sideview` (NOT `birdview` — framed too wide).
C1. Add `enc_type: resnet18_pretrained` to `rl/pixel_obs_encoder.py` (alongside `vit` /
    `resnet96`). ImageNet weights must be downloaded **ON THE LOGIN NODE** — compute nodes
    have no egress. Note `networks/encoder.py:33` hard-asserts raw 0–255 input and
    normalises internally, while a torchvision ResNet expects ImageNet mean/std — handle
    normalisation explicitly and document it.
C2. **Benchmark ms/gradient-step BEFORE requesting walltime** (iron law #16; patterns in
    `scripts/slurm/bench_encoders.slrm`, `bench_arms.slrm`). 224 is 5.44× the pixels of
    96. Measured at 96: state 33.6 ms/step, ViT-2cam 59.9 ms, resnet96-2cam 202.8 ms. Conv
    cost is ~linear in pixels ⇒ ResNet-18@224 plausibly 200–400 ms/step ⇒ at 800k steps
    ~45–90 h (3–4 chunks of 23:59:00). Storage at 224 for the 20.4M-transition target is
    ~6.1 TB raw / ~510 GB JPEG — **exceeds /scratch's 1.7 TB free unless compressed**, so
    JPEG is mandatory and you must re-check `df -h /scratch` before generating.
    **Do not launch a 224 run on an estimate.**

### Phase D — DO NOT IMPLEMENT; write up cost/risk in docs only
History encoder (paper: proprio history + actions + only the most recent image) — changes
the obs contract, the buffer, and MPPI's latent rollout. And 5 Hz control vs our 20 Hz
(H=5 gives them 1.0 s of lookahead, us 0.25 s) — switching invalidates EVERY existing
asset (source replay, demos, experts). Note that raising H at 20 Hz is the cheaper way to
test the lookahead hypothesis.

## CONFIG PARITY — already audited, do not redo
I diffed the **RESOLVED** (pyrallis-loaded, not YAML) configs of state-dynens vs
pixel-dynens. Exactly ONE unjustified difference: `use_wb` (1 vs 0), cosmetic since we pass
`--use_wb 0`. All others are intended and pixel-specific: `use_state`, `rl_camera`,
`encoder_type`, `decoder_coef`, `pi_grad_t0`, `seq_replay_max_transitions`, data paths.
`update_freq`, `simdist.planner.*` and BOTH domain-shift blocks resolve IDENTICALLY — a raw
YAML diff misleads, because the pixel file omits fields whose defaults already equal the
state file's explicit values. **Use resolved-config diffs, never YAML diffs.**

⚠️ One flagged deviation: **`pi_grad_t0=1` is on for pixel only and is not from the paper
or the state arm.** It was validated only as "6.4× larger conv gradient at identical loss",
never against a final score. Run a `pi_grad_t0=0` control for parity when convenient — as
a separate single-knob run, never bundled with the data change.

## RULES OF ENGAGEMENT (from CLAUDE.md; violations have cost this project days)
- **One knob at a time** (#9). Every claim needs a matched control.
- Effects <0.05 are unreadable; ±0.1 conclusions need 300–500 episodes (#1,#2). The
  single-20-demo-draw σ is 0.092.
- **Read windows, not points** (#4). n=50 in-training evals swing wildly (0.20→0.76→0.34 on
  adjacent points).
- `num_eval_proc` must divide `num_eval_episode` (#12) — it is an assert.
- **Headline numbers come from an independent re-eval**, never `model0.pt`'s in-training
  best (optimistic by 0.06–0.19, #14). Use `scripts/slurm/eval_simdist.slrm` — it
  auto-stages adapt checkpoints, which `eval.py` cannot read directly.
- Two numbers compare only if the printed `[EVAL ENV]` line matches (#15). Ours:
  `task=Threading_D0 shift=fixed nut_z_rot=FULL(0-360) sigma=0.1 strict=True ep_len=400`.
- **SLURM**: sit just UNDER a tier — 2:59:00 / 11:59:00 / 23:59:00. Keep CPU/mem small
  (8–12 cores, 32–48G). A 4 h ask for a 9 min job once waited 15 h23 (#16).
- **NEVER put a `#` comment on the line after a `\` continuation** (#19) — it terminates
  the logical line and silently drops every remaining flag.
- **Verify the ARTIFACT, not the exit code.** Four separate `rc=0` traps have bitten here
  (generator produced 0 trajectories; collector produced 0 demos; dynamics update silently
  skipped; flags silently dropped). Beware inverted exit codes too: a final grep that finds
  nothing exits 1 and makes SLURM report FAILED on a good run.
- **Check your own reporting tools** (#7). Multiple confidently-wrong conclusions here came
  from tooling, not experiments: a grep matching its own banner and reporting PASS on a
  failing job; `ls | tail -1` picking `snapshot_80000` over `snapshot_440000` by
  lexicographic sort; `sacct | grep -m1` returning a stale job row. When a number surprises
  you, suspect the tool first.

## DELIVERABLES
1. Code changes, inert by default, dedup regression gate passing.
2. Per phase: measured numbers with matched controls, plus the throughput/walltime
   measurements behind any large job launched.
3. New numbered section in `docs/simdist.md` (§46 onward; numbers are globally unique) and
   a refreshed `docs/pixel-progress.md`. Add a CLAUDE.md iron law only if you learn
   something that would otherwise be re-broken.
4. Final report: what closed the gap, what did not, what you did not attempt and why.

If a decision would cost more than ~12 h of GPU and the evidence is ambiguous, stop and say
so rather than guessing.
