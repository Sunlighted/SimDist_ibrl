# PIXEL arm — progress log

> ⚠️ **2026-09-02: every number below this line was measured on an OVER-FITTED
> stage-1** (pixel = 461 epochs over a 0.44M-transition replay that was the
> partial output of a crashed job; state = 50 epochs over a 4.08M replay sized to
> fit a 3 h queue tier). `docs/simdist.md` §45 explains why, and §46 records the
> rebuild. **Treat the whole table as PROVISIONAL** — in particular the state
> arm's 0.540 and every state-vs-pixel delta — until the 100k re-baseline lands.
>
> **In flight (2026-09-02):** 100k-trajectory replays for BOTH arms, generated as
> parallel shards; a 224 + ImageNet-ResNet-18 pixel arm; augmentation wired in for
> the first time. Both arms are now matched at **6.5M transitions / 150k steps =
> 5.9 epochs**, differing ONLY in the observation space. Runs:
> `threading_pixel224_stage1_seed1`, `threading_state100k_stage1_seed1`,
> plus the Phase A augmentation A/B (`pixel_phaseA_aug|off_seed1`) on the OLD data.
>
> Measured while building it (see §46 for the full table):
> ResNet-18@224 = **674 ms/step encoder-only** (16.3x vit@96; the planning doc
> guessed 200-400); end-to-end **847 ms/step**; JPEG **15.9x** compression at
> MAE 0.73/255 ⇒ 100k traj ≈ **340 GB**; generation **~148 transitions/s**
> sustained (the 295/s calibration was startup-dominated).


Goal: reproduce the state-based target-adaptation result with camera observations, keeping
this project's two additions (dynamics ensemble, target-time policy finetuning).
Mechanism analysis: `docs/simdist.md` §43 (retractions noted), §44 (paper vs us), **§45
(the key finding)**. Next steps: `docs/pixel-gap-closing-prompt.md`.

## Headline

**The pixel port works at every stage except target transfer.** Stage-1 matches or beats
the state arm on source at every checkpoint; adaptation on target does not move.
Root cause (§45): the pixel arm trains **461 epochs over 0.44M transitions** vs the state
arm's **50 over 4.08M** — 230× more passes over 8.5× less data ⇒ the source distribution
is memorised. The 2400-trajectory replay was a storage decision sized against the
*documented* 100k-trajectory reference, never recomputed against our own 20k-traj state arm.

## Results

### Stage-1 (source distillation), `source_score` 5-pt window
| arm | steps | source_score |
|---|---:|---:|
| state plain (h3, 1 head) | 800k ✅ | 0.956 |
| pixel plain (h3, 1 head, ViT) | 800k ✅ | **0.968** |
| state dynens (h5, 5 heads) | 700k/800k | 0.964 |
| pixel dynens (h5, 5 heads, ViT) | 400k/800k | **0.980** |

⇒ the visual encoder is **not** the weak link — 4 independent checkpoints, 2 architectures.

### Target adaptation — the 2×2 (all 20k transitions, `--original_mppi 1`)
| | with stage-2 BC (`SimDist+BC`) | dynamics-only (`SimDist`, Eq. 3) |
|---|---:|---:|
| **state** | **0.540** (indep. n=500) | 0.097 → **0.393** peak, 0.323 @20k |
| **pixel** | **0.028** (indep. n=500) | 0.000 → 0.023 @13k |

State reproduces its rung both ways (ref `plain_h3` 0.564 ± 0.171). Pixel does neither.
Both plain arms ran `horizon=3 [kept]` vs the paper's H=T=5; dynens arms are H=5.

### Stage-2 diagnostics (pixel), `policy_only` = policy head, planner off
| variant | pixel | state |
|---|---:|---:|
| stage-1 only, no target BC | 0.012 | 0.044 |
| N=20 demos, bc_steps 1000 | 0.070 | **0.360** |
| N=20, bc_steps 4000 / 8000 / 16000 | 0.10 / 0.08 / 0.08 | — |
| N=50 / N=100 / **N=200** | 0.13 / 0.19 / **0.24** | — |
| **N=200** | — | **0.75** |
| encoder UNFROZEN, no anchor | 0.16 (MPPI **0.00**) | — |
| frozen, no anchor | 0.02 | — |

- **Demo count is the dominant stage-2 axis** and 20 under-powers BOTH arms (state
  0.360 → 0.75). All of `demo-collection.md` is about *which* 20, never *how many*.
- **Not an optimisation problem**: 16× the BC steps moved nothing and drove BC loss *below*
  the state arm's.
- Unfreezing the encoder is a **probe, not a fix** — it invalidates the frozen value head
  MPPI ranks with (`policy_only` 0.16 while MPPI stayed 0.00).
- Anchor is worth +0.08 on the frozen pixel arm (§42 predicted: largest on a weak base).

### Latent probe (`scripts/probe_latent_pose.py`, closed-form ridge + MLP, held out BY EPISODE)
| arm | `prop` [0:9) CONTROL | `object` [9:37) pose |
|---|---:|---:|
| pixel | 0.218 / 0.384 | **0.033 / 0.016** |
| state | 0.655 / 0.800 | **0.528 / 0.650** |

⚠️ **Does not support the conclusion I first drew from it.** It probes *absolute* pose,
while control needs the *relative* needle-to-hole offset; and stage-1's own policy hits
0.968 on source using this very latent. Kept for one real finding: the CONTROL never
reaches 1.0 even solved exactly, though `prop` is a literal subset of the state arm's
encoder input ⇒ **SimNorm is genuinely lossy on both arms**.

### Supporting assets
Source expert 0.9767 src / 0.2767 tgt (n=300, all §12 knobs verified — the 2.1× vs the
reference 0.13 is a real property of our RNN-prior expert, ~6σ). Target expert stopped at
440k/1M, honest n=500 = **0.846** (in-training said 0.98 ⇒ optimism 0.134, iron law #14);
22 snapshots retained. Demo pools: 20 and 200 from `snapshot_400000`, verified (both cams,
length-aligned, no truncation vs the 400 cap). Pre-adapt state ladder: policy head 0.06 →
**+MPPI 0.223**.

## Port status
- **encode calls/step 17 → 7**, bit-identical via post-step weight checksums.
- **MPPI needed zero changes**; `--original_mppi 1` already matches the paper exactly.
- Source anchor caches **latents** (82 MB vs 2.2 GB) — exact only while the encoder is
  frozen *and* deterministic; `update_bc_policy` now raises if combined with unfreezing.
- **Headline eval path was broken for BOTH arms** — an adapt `cfg.yaml` is an `AdaptConfig`
  so `eval.py --mode simdist` could load no adapt ckpt. Fixed:
  `scripts/stage_adapt_ckpt_for_eval.py` (also copies the planner flags across).
- `aux_state_coef` implemented, inert at 0, **untested** — deprioritised (paper's
  reconstruction ablation is pixel-level, so privileged-state prediction is untested rather
  than contraindicated).

## Environment (rebuilt on Killarney/DRAC)
`docs/setup.md`'s machine is gone. New venv, py3.10.13, torch 2.6.0+cc; every fix inline in
`set_env_drac.sh`. Time sinks: mujoco has no CC wheel (unzip the PyPI one), `MUJOCO_GL=egl`
required, **`module load` silently no-ops** under a compiler layer.

Measured: **physics dominates, not rendering** (2 cams = +2.23 ms on an 11.82 ms step ⇒
falsifies `robosuite_wrapper.py:161-165`); mp eval scales 9.71× on 10 workers; **RNN vs
flow BC prior** = IBRL 6h23 vs ≥3 d (iron law #17); encoder bench at 96²: state 33.6 ms,
**ViT 59.9 (1.8×)**, resnet96 202.8 (6.0×) ⇒ ViT chosen, cost is FLOPs not params.

## 2026-09-02 — items 1-3 EXECUTED (see docs/simdist.md §46 for the full record)

Status of the "Open / next" list below: **1, 2 and 3 are done or in flight.** Numbers
here are MEASURED on Killarney, and several correct earlier estimates in this file.

| what | measured | supersedes |
|---|---|---|
| resnet18@224 encoder | **674.4 ms/step** (vit@96 = 41.5 ⇒ **16.3×**) | item 3's "plausibly 45-90 h" was for 800k steps; the real lever was cutting steps, not resolution |
| JPEG @224 q95 | **15.9×** compression, 18,935 B/transition | item 2's "226 GB RAM" ⇒ 100k traj at 224 is **340 GB**, not 6.1 TB raw |
| JPEG GPU decode | **110 ms/step** at 3072 frames/step; CPU is 919 ⇒ unusable | — |
| generation | **295 transitions/s**; 100k traj = 17 h ⇒ run as **8 parallel b1 shards** | — |
| stage-1 end-to-end | **1.04 s/step** measured in the 224 smoke ⇒ 150k steps ≈ 43 h | — |

**224 turned out to be free to produce**: `image_size: 224` / `rl_image_size: 96` means we
were already rendering at 224 and discarding it in the wrapper's resize.

Assets now on disk / in flight:
- **target demos at 224**: 200 demos, `(T,3,224,224)` uint8 ×2 cams, ep len 86/121/378,
  from `snapshot_400000` — **the same sha256 (`ca225fba…`) as the existing 96 draw**, so
  pixel-vs-resolution stays a PAIRED comparison (±0.027) instead of two independent
  20-demo draws (σ=0.092, §38). Verified through `load_demos_to_sequence_buffer`.
- **source replay at 224**: 8 shards × 12.5k traj, JPEG q95, never merged
  (`source_replay_path` now takes globs).
- **state replay at 100k**: 4 shards, for the §45 item-3 re-baseline.

Two new arms, deliberately paired (iron law #9 — the 224 arm changes data volume,
resolution, backbone AND augmentation at once, so a win would be unattributable):
- `..._r18_224.yaml` — the real arm.
- `..._vit96_data100k.yaml` — **data-scale control**: same replay, downsampled 224→96 at
  load via the new `pixel.resize_to`, so it needs no second dataset and costs ~1/16.

⚠️ **Item 5 (`pi_grad_t0`) is still unvalidated** and is now riding along in both new
configs. It remains owed a single-knob control.

⚠️ Everything measured in §43 was tuned on the over-fitted 2400-traj stage-1 and must be
re-measured once these land.

## Open / next
1. **Augmentation** — the `augment_fn` hook exists and *no caller passes one*; we train
   with zero augmentation while the paper uses four kinds. Free, and targets §45 directly.
2. **Data parity with the state arm** — 20k traj / 4.08M transitions. Blocked on two
   things: `seq_replay_max_transitions: 450000` is the real cap, and the buffer is
   CPU-RAM resident (4.08M ⇒ **226 GB RAM** vs our 72G ask) ⇒ **JPEG storage is
   load-bearing even at 96²**, not just a disk optimisation.
3. **224×224 + ImageNet ResNet-18, 2 views** (owner: resolution before view count). 96²
   was inherited, not chosen. Cost hinges on the epoch decision, not the resolution:
   at state parity (800k steps) ResNet-18@224 is plausibly 45–90 h. Benchmark first.
4. Deferred: history encoder; 5 Hz control (invalidates every existing asset — test
   lookahead by raising H at 20 Hz instead).
5. ⚠️ `pi_grad_t0=1` is on for pixel only, is my own invention, and was never validated
   against a final score. Needs a single-knob control.

## Open / next (updated 2026-09-02)
- **Lazy/disk-backed loader.** The buffer is still CPU-RAM resident, so the run is capped
  at `seq_replay_max_transitions: 8000000` (~151 GB, `--mem=200G`) = 4.8 epochs. The full
  18M transitions would need ~340 GB of RAM; the remaining trajectories sit on disk until
  a lazy loader exists. This is the one piece of §45's item 2 that is NOT done.
- **Stage 2/3 walltime is unmeasured at 224.** `adapt.slrm`'s 480 s/round was measured at
  vit@96 and does not transfer to a 16×-costlier encoder ⇒ `scripts/launch_pixel224_adapt.sh
  probe` measures s/round first (iron law #16: 时长从 sacct 量出来).
- **Dropout.** The reference implementation (`../simdist`, quadruped arm) carries
  `dropout: {mlp: 0.2, attention: 0.05}`; we have none. Free, and aimed at the same
  over-fitting §45 diagnoses. Not bundled with the current change.
- **History encoder — cheaper than this file assumed.** The reference makes history a
  cross-attention CONTEXT (`history_length: 25`, interleaved proprio/action tokens), not a
  framestack. Crucially `prop` is the first 9 dims of the stored 37-d state and actions are
  already stored ⇒ **history needs NO regeneration**, and in our TD-MPC2 port it does not
  touch the MPPI rollout (the encoder only ever sees the real current obs). The faithful
  version (transformer dynamics, T=25) IS a rewrite and would drop the dynamics ensemble.
