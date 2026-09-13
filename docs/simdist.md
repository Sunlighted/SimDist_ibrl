# SimDist 主线：pipeline 与各阶段

SimDist 本体（world model + MPPI）、两阶段 target adapt、在其上再加 residual/IBRL、省交互变体、stage-2 的 source 锚。改 pipeline 前读这个。

> 从 `CLAUDE.md` 拆出（2026-08-14）。速查表、方法论铁律、文件地图仍在 `CLAUDE.md`；已归档分支见 `CLAUDE_ARCHIVE.md`。

---

## 17. SimDist baseline (state-based latent world model + MPPI)

### What
Port of **SimDist** (Levy et al., *Simulation Distillation*, RSS 2026; original
JAX/quadruped, model-based latent world model + MPPI online planning; ref at
`/home/Chenyang/simdist`) into a **state-based, PyTorch** ibrl baseline on the
Threading_D0 source→target dynamics gap. The state-based latent world model is
essentially **TD-MPC2** (SimDist's own ref impl is built on TD-MPC2): an MLP
encoder maps the 37-d state to a latent z; an MLP latent-dynamics model predicts
z'; reward/value/policy heads decode from z. Execution = **MPPI** planning on the
world model.

Two user-directed changes vs the original:
1. **No online adaptation.** The original adapts at deploy time with real
   rollouts; here stage 2 is **offline** finetuning on the fixed TARGET demos
   (`release/data/mimicgen/threading_d0_target_dynamics`, 20 ep, dynamics-only
   target §12).
2. **Two stages back-to-back in one run** (`train_simdist.py`):
   - **Stage 1 (SOURCE, online)**: distill all heads. Data collection is
     **mixed** — the frozen SOURCE expert (`threading_ibrl_source_seed1`, +noise)
     for the first `expert_collect_steps` env steps (coverage + learnable under
     sparse reward), then the world model's own MPPI planner (on-policy,
     TD-MPC2-style).
   - **Stage 2 (offline)**: finetune **only dynamics + policy** heads
     (encoder/reward/value frozen) — latent-consistency on demo (s,a,s') +
     policy BC on demo (s,a).

TARGET strict-success is the deployment metric (`score/score`); SOURCE eval is a
diagnostic (`score/source_score`).

### Where
- `rl/world_model.py` (NEW) — `WorldModel(obs_dim, action_dim, WorldModelConfig)`:
  MLP `encoder/dynamics(residual z+Δ)/reward/value/policy(tanh)` + an EMA
  `value_target`. Helpers `encode/next/predict_reward/predict_value/pi`;
  `trainable_parameters()` (all but the EMA target) and
  `dynamics_policy_parameters()` (stage-2 subset). Generic `mlp()` builder.
- `rl/simdist_planner.py` (NEW) — `MppiPlanner` ported from
  `simdist/control/mppi.py`: encode→sample N action seqs (a `mixture_coef`
  fraction seeded from the policy head's rollout)→latent rollout→
  `return = Σγ^t r̂_t + γ^H V̂(z_H)`→top-k elites→softmax-reweight (mean,std), iterate.
  **Stateless by default** (`warm_start=0`, mean re-init each call) so `act()` is
  mp-eval-pickle-safe.
- `rl/simdist_agent.py` (NEW) — `SimDistAgent(nn.Module)` holds the world model +
  planner. `act(obs, eval_mode, stddev, cpu)` runs MPPI (handles **single AND
  batched** obs — mp_eval sends a `[B,obs]` stack and expects `[B,act]`).
  `update_distill` (stage 1, all heads: consistency + reward + value-TD +
  policy-BC, per-step `rho**t` weighted over the H-rollout, EMA value_target) and
  `update_finetune` (stage 2, dynamics+policy optimizer only). NOT a QAgent
  subclass but mirrors its public surface. `SimDistConfig` nests
  `WorldModelConfig` + `PlannerConfig` + finetune knobs.
- `rl/sequence_replay.py` (NEW) — `SequenceReplayBuffer` (+ `load_demos_to_
  sequence_buffer`). **Why needed**: ibrl's `rl/replay.py` wraps
  `rela.SingleStepTransitionReplay`, whose `sample` returns a SINGLE-step
  transition (`next_obs` is n-step later, `reward` discounted-summed) — the
  intermediate steps are unrecoverable, so it CANNOT supply the (H+1)-step
  contiguous segments the world model needs. This minimal PyTorch buffer stores
  whole episodes and `sample(B,H)` returns `states[B,H+1,obs] / actions[B,H,act]
  / rewards[B,H]`. The demo loader re-concats state from `STATE_KEYS[task]` (same
  as the env + `add_demos_to_replay`) so the world-model input dim matches
  `train_env.observation_shape` (37) and truncates each demo at first success.
- `train_simdist.py` (NEW, modeled on `train_gated.py`) — `MainConfig` +
  `Workspace` (`_setup_env` SOURCE/TARGET, `_setup_replay` seq + demo buffers,
  `warm_up`, `train()` = stage 1, `finetune()` = stage 2, `eval`, `log_and_save`,
  `load_model`). Reuses `train_rl.load_frozen_agent` for the stage-1 expert.
  `main()` = `train()` → `finetune()` → `os._exit(0)` (EGL teardown hangs).
- `evaluate/eval.py` — added `--mode simdist` → `train_simdist.load_model`.
- `release/cfgs/mimicgen/threading_d0_state_simdist.yaml` (NEW) — demo config.
- `scripts/train_simdist_threading.sh` (NEW) — `SEED/SOURCE_AGENT/
  EXPERT_COLLECT_STEPS/DISTILL_STEPS/FINETUNE_STEPS` wrapper + `"$@"` passthrough.
- `scripts/test_simdist.py` (NEW) — synthetic-batch smoke test.

### How
```bash
source set_env.sh
bash scripts/train_simdist_threading.sh                          # mixed collect (default)
EXPERT_COLLECT_STEPS=0 SOURCE_AGENT=none bash scripts/train_simdist_threading.sh  # pure on-policy
bash scripts/train_simdist_threading.sh --simdist.finetune_policy 0  # finetune dynamics only
# raw:
python train_simdist.py --config_path release/cfgs/mimicgen/threading_d0_state_simdist.yaml \
    --source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
    --save_dir exps/mimicgen/threading_simdist_seed1
# standalone eval (TARGET, strict-success):
python evaluate/eval.py --weight exps/.../latest.pt --mode simdist --num_game 100 --mp 10 --strict_success 1
```
`state_stack` MUST be 1 (matches `threading_ibrl_source_seed1`; `load_frozen_agent`
asserts the source/run obs space match).

### 43. ★ PIXEL stage-2: mostly a DEMO-BUDGET effect, not a latent defect (2026-09-02)

⚠️ **This section was rewritten after two corrections. The first version claimed the
frozen latent had discarded the geometry a policy needs. That claim does not survive the
evidence below — do not cite it.**

#### What is measured (all: target domain, strict, σ=0.1, s400k draw, frozen encoder)

`policy_only` = the policy head driving the robot with the planner OFF, i.e. proposal
quality. Stage-1 target scores before any BC: state 0.044, pixel 0.012.

| stage-2 variant | pixel `policy_only` | state `policy_only` |
|---|---:|---:|
| N=20 demos, bc_steps=1000 (the reference recipe) | 0.070 | **0.360** |
| N=20, bc_steps=4000 / 16000 | 0.10 / 0.08 | — |
| N=20, bc_steps=8000 | 0.08 | — |
| N=50, bc_steps=8000 | 0.13 | — |
| N=100, bc_steps=8000 | 0.19 | — |
| **N=200, bc_steps=8000** | **0.24** | **0.75** |

#### ① Demo COUNT is the dominant axis, and 20 under-powers BOTH arms
Pixel triples 0.08 → 0.24 across N=20→200, monotone, **not saturating** (~+0.055 per
doubling). The STATE arm doubles too: **0.360 → 0.75**. ⇒ the project-wide habit of fixing
N=20 (all of `demo-collection.md` is about *which* 20, never *how many*) is itself a
binding constraint. Any pixel-vs-state number taken at N=20 is measured in a
data-starved regime for both arms.

#### ② It is NOT an optimisation problem
bc_steps 1000→16000 at N=20 moved nothing (0.07→0.08) and drove BC loss *below* the state
arm's. 8× the steps bought nothing; 10× the data tripled the score. BC loss RISES with N
(0.017→0.028) — the healthy direction.

#### ③ The gap does not close at matched N, and that residual is the real question
At N=200: pixel 0.24 vs state 0.75. Extrapolating pixel's slope needs ~100k demos to
reach 0.75 ⇒ more target demos help a lot but cannot be the whole answer.

#### ④ Why the "latent lacks the geometry" story is WRONG
- **Stage-1's own policy scores 0.968 on SOURCE using exactly that frozen latent.** If the
  latent could not support precise control, that is impossible. The difference is that the
  stage-1 head was co-trained with the encoder on **445k transitions**; stage-2 re-fits it
  from **2951** (20 demos) — 150×.
- The latent probe (`scripts/probe_latent_pose.py`, closed-form ridge + MLP, held out BY
  EPISODE) measured **absolute** pose (the 28-d `object` block): pixel R² 0.03/0.02 vs
  state 0.53/0.65. But a controller needs the **relative** needle-to-hole offset, and a
  latent encoding only that difference scores ~0 on absolute pose while being fully
  sufficient. ⇒ the probe target was wrong; the numbers do not establish the claim.
- Probe caveat worth keeping anyway: the CONTROL (`prop`, dims [0:9), which is fed straight
  INTO the encoder) only reaches R² 0.655/0.800 on the state arm and 0.218/0.384 on pixel.
  Even solved exactly, a literal subset of the encoder input is not fully recoverable ⇒
  **SimNorm is genuinely lossy on both arms.** True, but it does not by itself explain
  stage-2.

#### ⑤ Unfreezing the encoder in stage 2: a probe, not a remedy
+0.14 (0.02 → 0.16, matched no-anchor) — so information IS recoverable from the images.
But reward/value are frozen from stage-1 and fitted on the stage-1 latent, so moving the
encoder invalidates the signal MPPI ranks with (`std_i(γ^H V)`=0.0068, 1.4% of range,
§33). Measured: `policy_only` 0.16 while MPPI stayed **0.00**. Also mutually exclusive
with the anchor, whose pixel pool caches latents (`update_bc_policy` now raises).

#### ⑥ The anchor is worth +0.08 on the frozen pixel arm
0.02 → 0.10 at N=20, consistent with §42's prediction that it helps most on a weak base.

#### ⑦ An auxiliary privileged-state loss is UNTESTED (not contraindicated) — corrected
`aux_state_coef` is implemented (`world_model.py` `_build_aux_state`, t=0 grad latent only,
z-scored, inert at coef=0 and bit-identity-verified). Clarified by the project owner:
the paper's *Raw Obs. Reconstruction* row (0.90 → 0.32 on peg insertion) is **PIXEL**
reconstruction, and §IV-A's mechanism is that it pressures the latent to encode randomized
texture/lighting/rendering artifacts. The row is heterogeneous — the quadruped has no
camera (proprio + 21×15 height map), so its 23.34 is LOW-DIM reconstruction and it
slightly HELPS. **Neither case tests predicting privileged state (object pose, contact),
which is a different signal from reconstructing the observation.** Also Fig. 9: an
auxiliary probe recovers IMAGES from the frozen latent ⇒ the information survives without
the loss. ⇒ treat as untested; lower priority than §45's data/resolution gaps.

#### ⑧ Stage-2 is this project's CONTRIBUTION, not a mistake — corrected
An earlier version of this section said we were "running a stage the method does not
have". That framing is wrong. Per the project owner (paper §V-B, Fig. 4 separate curves):
- **SimDist** = adapt ONLY the latent dynamics from on-policy rollouts, **no demo access**.
- **SimDist+BC** = additionally finetune π_θ on the teleop action labels.
Both are paper arms. **This fork's line of work is improving on SimDist by adapting the
policy too, and it already beats SimDist on the state-only case** ⇒ stage-2 is the point,
not an error.
⚠️ Caveat worth keeping: the base policy IS trained during PRETRAINING via Eq. 2's c₃ BC
term (our `policy_coef=10`), and MPPI warm-starts from it. So `bc_steps=0` means "no
real-world policy updates", NOT "no policy head".
Measured here, matched protocol (20k transitions, n=300 in-training, `--original_mppi 1`):

| | state | pixel |
|---|---:|---:|
| `policy_only` (frozen stage-1 policy on target) | 0.053 | 0.010 |
| MPPI @ round 0 | **0.097** | 0.000 |
| MPPI @ 3k | **0.280** | 0.007 |
| MPPI @ 9k | — | 0.017 |

⇒ plain SimDist (dynamics-only) reproduces on state and does NOT rescue pixel. Both plain
arms ran `horizon=3 [kept]`, vs the paper's H=T=5; the dynens arms are H=5.

### 44. ★ PAPER vs OUR PORT — input & objective divergences (from the authors' spec)

Supplied by the project owner from the paper; recorded because several of these are large
and were introduced by our port, not chosen deliberately.

| aspect | paper | ours | note |
|---|---|---|---|
| **deployment objective** | Eq. 3: consistency ONLY, `C,E,R,V,π` frozen, only `f_θ` updated | our stage-2 TRAINS the policy on 20 demos | ⚠️ we ran the `+BC` variant as the method (§43 ⑧) |
| views × resolution | **3 × 224×224** (wrist, overhead, side) | **2 × 96×96** (agentview, wrist) | 5.4× fewer px/view. Owner: **224 matters MORE than view count** (ResNet-18 is ImageNet-pretrained; the prior is lost at 96) ⇒ prefer **2 views @224 over 3 @96**. Third view = `sideview` (not `birdview`, framed too wide) |
| image encoder | **ImageNet-pretrained ResNet-18** → 3×512 | **from-scratch** ViT embed2/d1 | large missing prior |
| proprio | 6 joint observations | 9-d eef_pos+quat+gripper_qpos | different content |
| latent | **64-d**, MLP fusion | **512-d + SimNorm** | SimNorm is TD-MPC2's, not the paper's |
| history | **history encoder**: proprio history + actions + most recent image | none, `obs_stack=1` (hard-asserted in `rl/replay.py`) | partial observability |
| sim data | **100k trajectories**, ~36% optimal | **2400** traj / 445k transitions | ~40×; see storage note |
| augmentation | proprio Gaussian noise, colour jitter, Gaussian blur, random crop | **NONE** — `augment_fn` hook exists in `pixel_obs_encoder.forward` but no caller ever passes one | free to fix; repo's own pixel BC/RL use `RandomShiftsAug(pad=4)` |
| control / horizon | **5 Hz**, H=5 ⇒ **1.0 s** lookahead | 20 Hz, H=5 ⇒ **0.25 s** | 4× shorter in real time |
| pretraining | 2 epochs over D_sim, ~200k steps, batch 256, lr 2e-4 cosine→1e-4, 10k warmup | 800k steps | ours longer |
| reconstruction | none; adding *Raw Obs. Reconstruction* drops **0.90 → 0.32** | `decoder_coef=0` | ✅ matches; contraindicates §43 ⑦ |
| MPPI | 250 cand + 100 noised base, 3 iters, init_std 1.0, min_std 0.05, base_policy_std 0.1, 64 elites, temp 0.4, momentum 0.0, γ 0.99 | **`--original_mppi 1` resolves to EXACTLY this** (`apply_original_recipe`); the cfg *defaults* (512/6/0.5/0.8/0.5/0.1) are the ibrl-tuned recipe and are NOT what the adapt runs use | ✅ **already faithful** — verify via the `[adapt] using ORIGINAL SimDist MPPI recipe ...` log line |
| demos | 20 teleop, used by `SimDist+BC` | 20 (now also 50/100/200, §43 ①) | matches the variant |

⚠️ **Storage — corrected (my petabyte estimate was wrong).** ~100k trajectories × ~250
steps = 2.5e7 timesteps; ×3 views ≈ 7.5e7 images ≈ **8 TB raw uint8**, or **0.5–1 TB as
JPEG**. Our 23.9 GB for 2400 traj at 96²×2 scales to the same ~8 TB figure — the tell that
we store UNCOMPRESSED. Nothing in the paper implies on-the-fly rendering: pretraining is
fully offline over a fixed `D_sim` for exactly 2 epochs, and the colour-jitter/blur/crop
augmentations need raw pixels at load time. ⇒ **per-frame JPEG or per-episode video
encoding reaches their budget without touching the training loop.**

### 45. ★★ THE PIXEL ARM IS 230× OVER-TRAINED ON 57× TOO LITTLE DATA (2026-09-02)

The single largest divergence from the paper, found by computing effective epochs:

| | transitions | grad steps × batch | **epochs over D_sim** |
|---|---:|---:|---:|
| **paper** | ~2.5e7 (100k traj × ~250) | 200k × 256 | **2** |
| our **state** | 4.08M (20k traj) | 800k × 256 | **50** |
| our **pixel** | **0.44M** (2400 traj) | 800k × 256 | **461** |

The paper does **2 epochs**. We do 50 (state) and **461** (pixel).

⚠️ **Provenance corrected (2026-09-02) — neither replay size was a modelling decision:**
- **state, 20k traj**: job 5113397 was launched `NUM_TRAJ=20000` and finished in
  **1:59:42 of a 2:59:00 window**. `gen_source_replay.slrm`'s own usage comment documents
  `NUM_TRAJ=100000`, which at that rate is ~10 h and fits an 11:59 (b2) job. The dataset
  was sized to the fast-queueing 3 h tier ⇒ **5× the data traded for a shorter queue
  wait.** The project owner confirms **100k is the reference recipe**, so our state arm is
  5× under it too, and **the state arm's 0.540 is a PROVISIONAL baseline.**
- **pixel, 2400 traj**: job 5113396 requested `NUM_TRAJ=3000`, ran 24 min and died
  **OUT_OF_MEMORY** having written 2400. **No pixel generation job has ever completed**,
  and we then trained 461 epochs on that partial output. The generator therefore cannot
  produce 100k as written — fixing the OOM is a prerequisite, not an afterthought.

At the reference 100k traj (~20.4M transitions), 800k steps is **~10 epochs** — close to
the paper's 2 without changing the step count at all.

**This explains the whole pixel picture without any of §43's earlier stories:**
- `source_score` 0.968 (plain) / 0.980 (dynens) — excellent, i.e. the source distribution
  is MEMORISED after 461 passes.
- target transfer ~0.01, and dynamics-only adaptation cannot lift it (§43 ⑧) — no
  generalisation to lift.
- more TARGET demos help monotonically (§43 ①) — target data substitutes for the
  generalisation the source pretraining failed to produce.
- the latent does not linearly encode absolute pose (§43 ④) — a memorising encoder has no
  need to build a pose representation.
- **zero augmentation** (§44) — the standard remedy for exactly this regime, and part of
  the paper's recipe, never wired in our port (`augment_fn` hook exists, no caller).

⇒ **Fix order for the pixel arm, cheapest-first:**
1. **Augmentation** (colour jitter, blur, random crop, proprio noise). Free, in the paper,
   hook already present. Directly targets over-fitting.
2. **Scale D_sim to the reference 100k trajectories** (~20.4M transitions) — that alone
   takes 800k steps from 461 epochs to ~10, with NO step change. Prerequisites: fix the
   generator OOM, add JPEG storage (20.4M raw = 1.13 TB, i.e. impossible in RAM; ~94 GB
   compressed), and raise `seq_replay_max_transitions` (currently 450000 — the real cap).
   Also switch checkpoint selection off `source_score`, which IS the memorisation signal.
3. **Regenerate the STATE replay at 100k too** and re-baseline — it is also 5× under the
   reference, so every current state-vs-pixel delta is provisional.
4. **224×224 + ImageNet-pretrained ResNet-18**, 2 views (owner's guidance: resolution
   before view count).
5. History encoder; 5 Hz control so H=5 buys 1.0 s of lookahead, not 0.25 s.
Only then revisit stage-2 knobs. Everything measured in §43 was tuned on top of an
over-fitted stage-1 and should be re-measured after (1)–(3).

### Pre-adapt baseline ladder on OUR assets (2026-09-01, n=300, strict, σ=0.1)

Measured with `scripts/slurm/eval_simdist.slrm` on the fresh Killarney assets, so these
are the numbers our adapt runs must be read against — NOT the reference ladder, which was
produced on a different source expert (ours uses the RNN-GMM prior; see the tripwire note).

| rung | target score | note |
|---|---:|---|
| source IBRL expert, zero-shot | **0.2767** | tripwire, n=300 |
| state stage-1 policy head, `use_planner=0` | **0.06** | `score/score` at 800k |
| **state stage-1 + MPPI, NO adaptation** | **0.2233** | 67/300, 8m45s |
| reference post-adapt plain / dynens | 0.88 / 0.962 | `findings.md` §31, different expert |

Two readings worth keeping:
- **Planning alone is worth 3.7×** on the un-adapted world model (0.06 → 0.223). That is
  the planner doing real work before any target data exists.
- The distilled model + MPPI lands just *below* the source expert it was distilled from
  (0.223 vs 0.277) — expected distillation loss, and a sanity check that stage 1 did not
  somehow exceed its teacher.

⇒ adaptation has to carry **0.223 → ~0.88**. State stage-1 finished at 800k with
`source_score` 0.956 (5-pt window); pixel is tracking at or above it.

### Design choices
- **TD-MPC2-style state world model** (MLP, not the original Transformer — state
  obs needs no attention). Latent-consistency `‖ẑ' − sg(enc(s'))‖²` rolls the
  PREDICTED latent (multi-step), reward MSE, policy BC `‖π(z) − a‖²` on the
  collected/expert action. **The policy z is NOT detached** (fix 2026-06-05): the
  shared encoder is non-stationary (moved each step by consistency/value/reward),
  and a detached policy can't adapt to it → cloning plateaus at MSE 0.043
  (undershooting near-mean policy, closed-loop success 0). Letting the BC
  gradient also shape the encoder drops it to 0.005 (≈ a dedicated policy
  encoder / pure BC); `policy_coef` balances it against the other heads.
- **Value = DENSE distillation from the source critic** (`V̂(z)→V_src(s) =
  min-ensemble Q_src(s, π_src(s))`, reuses `QAgent.value`, §10). **This is the
  critical fix (2026-06-05)** — the first impl used a self-bootstrapped TD value
  `r+γ(1−done)V_target` "to reduce coupling", which **collapses to V≡0 under the
  sparse reward**: every non-terminal step is `r=0, done=0`, so `V(z)=γV(z')` has
  fixed point 0 and the lone `r=1` terminal is too rare to lift it. A flat value
  ⇒ MPPI returns `Σγ^t r̂+γ^H V̂` are ~constant ⇒ elite selection is random ⇒
  source AND target eval stuck at 0 (diagnosed: value std 0.002, while the policy
  head was already near-perfect, MSE 0.0009). SimDist itself distills the RL
  critic's value — the TD path was the deviation. With dense V_src the value head
  recovers spread (std≈0.10) and MPPI has a real gradient. TD path kept as a
  fallback when `source_agent` is absent (`value_targets=None`).
- **Stage-1 collection default = ALL-EXPERT** (`expert_collect_steps=0`): the
  source expert (+noise) collects the whole stage, off-policy — faithful to
  SimDist's offline expert data-gen and stable. `expert_collect_steps>0` switches
  to MPPI self-explore after N steps (on-policy); the first run showed this
  **destabilizes the world model on its own OOD rollouts** (consistency rebounds,
  grad_norm spikes to 2000+ right at the switch, train_score → 0). Keep 0 unless
  deliberately studying on-policy. `source_agent=none` ⇒ pure on-policy (TD value
  fallback, expect the collapse — diagnostic arm only).
- **Stage-2 = dynamics+policy only** (user choice): the task semantics
  (reward/value) are shared across source/target, so only the contact/friction
  dynamics + the action distribution need to adapt — SimDist's "dynamics-only
  finetune" spirit + the policy head (no online MPPI adaptation to lean on).
- **MPPI stateless** so `act()` carries no cross-step state and survives the
  multi-process eval pickle (`warm_start=1` available for single-process).
- **Own sequence buffer** (not rela): rela can't yield multi-step segments.
- **`simdist.use_planner`** (default 1): execution = MPPI. Set 0 to act with the
  policy head directly (ablation / diagnostic — the distilled proposal without
  planning). `act` handles single + batched obs in both modes.

### Verification status
Smoke-tested 2026-06-05:
- ✅ `scripts/test_simdist.py`: SequenceReplayBuffer add/sample shapes; world-model
  head shapes + value_target excluded from trainable; `update_distill` losses
  finite, encoder moves, value_target EMA-tracks; `update_finetune` moves ONLY
  dynamics+policy (encoder/reward/value frozen); planner+act single→`[act]` &
  batched→`[B,act]` in [-1,1]; state_dict round-trip.
- ✅ Demo loader: 20 demos / 1983 transitions, state dim 37 (matches env), act 7.
- ✅ End-to-end short run (distill 300 + finetune 50, mixed collect): two stages
  connect, distill losses decrease (consistency 0.025→0.0007, value→1e-4,
  reward→0), finetune runs, TARGET+SOURCE eval produce scores, ckpt saved,
  `os._exit(0)` clean.
- ✅ Standalone `evaluate/eval.py --mode simdist` (mp batched act) prints scores.

**2026-06-05 diagnosis + fix (after the first full run scored 0 throughout):**
- The first 200k run had `score`/`source_score` ≡ 0 the whole time, and
  `train_score` dropped to 0 right after the `expert_collect_steps` switch. Root
  cause = the TD-bootstrap **value collapse** above (value std 0.002) → dead MPPI,
  plus on-policy collection destabilizing the world model. Fixed by dense value
  distillation from the source critic + all-expert default.
- Post-fix component health (env-free, on a 20k-distill+5k-finetune ckpt): value
  std **0.10** (was 0.002), 5-step latent rollout MSE **3e-4** (dynamics
  accurate), policy MSE on target demos **6e-3**. All pieces individually sound.
**2026-06-05 deeper diagnosis — two MORE bugs + a stability caveat (after value
fix still gave 0):**
- **Policy wasn't cloning the expert.** On expert-visited source states the
  policy action differed from the source expert by mean|err| **0.24** with a
  systematic undershoot (|π|≈0.97 vs |a|≈1.4). Root cause: the policy BC loss
  used `pi(z.detach())`. Env-free controlled test (clone the 20 demos under the
  full multi-objective `update_distill`): detach **0.043** vs no-detach **0.005**
  vs dedicated-encoder **0.0045** vs pure-BC **0.008**. The shared encoder is
  non-stationary (moved each step by consistency/value/reward) and a DETACHED
  policy can't adapt to it → cloning plateaus near the mean action. **Fix:
  removed the detach** (`pi(z)`) so BC also shapes the encoder. → clone err
  0.24→**0.03**, no undershoot, SOURCE policy-only success **0→0.10→0.30→0.40**
  as distill trains, and after a 5k finetune SOURCE 0.40 / **TARGET 0.10**
  (end-to-end source→target adaptation finally works).
- **Collection noise must be ~0.** The executed action IS the BC target, so the
  `stddev` exploration noise both noises the target AND makes the noisy expert
  fail the precision task (worse demos). Measured: stddev 0.0 → source 0.40 /
  clone 0.03; 0.05 → 0.10 / 0.12; 0.3 → ~0. Default reverted to **0.0**.
- **MPPI must lean on the policy proposal.** On sparse Threading V_src is ~flat
  on the expert success-manifold, so value-ranking can't distinguish candidates;
  with the TD-MPC2 planner defaults (mixture 0.25, init_std 0.5) the random
  trajectories diluted the good proposals and MPPI scored BELOW the bare policy
  (0.0 vs 0.10). Tuned to **mixture_coef 0.8, init_std 0.15, temperature 2.0** →
  MPPI **0.20 > policy 0.10** (it refines the clone). Baked into defaults.
- **Stability — diagnosed AND root-fixed.** The no-detach policy BC and the
  dynamics-consistency loss FIGHT over the shared encoder: short runs are fine
  (30–40k) but a long distill diverges (consistency 0.006→0.017→0.055 @40/80/120k,
  SOURCE 0.35→0.25→0). **Root fix (implemented 2026-06-05): the policy gets its
  OWN encoder `enc_pi`** (`WorldModel.policy_action(state)=policy(enc_pi(state))`),
  fully decoupled from the dynamics encoder, plus a **`decoder` z→state** so the
  MPPI proposal can still roll the policy (`_policy_rollout`:
  z→decode→state→policy_action→a→next). Env-free validation (8000 steps on the
  demos): consistency stays FLAT (0.0005→0.0007, no blowup) while the policy
  clone improves monotonically to MSE **0.0009** and decode MSE ~0.006 — the two
  objectives no longer compete. `decoder_coef=1.0`; finetune trains
  dynamics+`enc_pi`+policy (encoder/reward/value/decoder frozen). NOTE: this
  changes the arch → pre-own_enc ckpts won't load.
- **Latent explosion → SimNorm (second stability fix).** own_enc fixed the
  POLICY, but the long run showed the DYNAMICS `consistency` still diverged
  (0.026→0.24 @40/80k, max 4.4), independent of the policy. Cause: the encoder
  output was unnormalized and grew unbounded (encode(s) std 7, norm **147**,
  absmax 26), so consistency MSE in that blown-up space exploded (policy-only
  eval was unaffected — source held 0.55-0.60 — but it kills MPPI). Fix: **SimNorm**
  (TD-MPC2) on the dynamics latent — `encode` and `next` softmax-normalize the
  latent in groups of `simnorm_dim=8`, and `next` is now DIRECT
  (`SimNorm(MLP([z,a]))`) not residual (a residual sum of simplexes isn't a
  simplex). Env-free (8000 steps): latent norm bounded **~6** (was 147),
  consistency **0.0002→0.0033** (flat, was →0.24), policy clone 0.0009. `enc_pi`
  is left unnormalized (it doesn't roll, doesn't grow).
NOT yet done: the long own_enc+SimNorm run's SOURCE-convergence + finetuned TARGET
strict-success (cf gated `force_c=1.0` ~13%). Running at
`exps/mimicgen/threading_simdist_ownenc_seed1`, wandb `cao1233/mimicgen`.

### 2026-06-16 RAW-STATE arm (no learned encoder/latent) — control for the latent
Added `WorldModelConfig.encoder_type: "latent" (default) | "identity"`. With
`"identity"` the world model has NO learned encoder and NO SimNorm: the "latent"
z IS the **z-scored raw state** (dim=obs_dim), `next` is a RESIDUAL next-state
predictor `z'=z+MLP([z,a])`, `decode` is the analytic denorm inverse (no params,
exact), and the policy runs directly on the normalized state. This drops the
ENTIRE §17 latent stability saga — SimNorm (latent explosion), `enc_pi` +
`decoder` bridge (policy-vs-consistency fight) all become unnecessary because the
policy (`s→a`) and dynamics (`(s,a)→s'`) no longer share an encoder. Rationale:
the 37-dim Threading state is already compact, so the latent buys little for
state-based control while costing the whole stability fight; this arm tests
whether the latent helps at all. z-score stats are fit on the SOURCE replay
states at the start of stage 1 (`SequenceReplayBuffer.state_stats()` →
`agent.set_normalization()`) and stored in ckpt buffers (identity until set), so
eval self-normalizes and old latent ckpts load byte-for-byte (default stays
`"latent"`; `pyrallis` fills the missing field with the default).
- Where: `rl/world_model.py` (`encoder_type` field + `raw_state` branch in
  `__init__`/`encode`/`next`/`decode`/`policy_action` + `set_normalization` +
  guarded `encoder_parameters`/`policy_parameters`); `rl/simdist_agent.py`
  (`set_normalization` passthrough, `enc_lr_scale` guard, skip the param-less
  decoder loss); `rl/sequence_replay.py` (`state_stats()`); `train_simdist.py`
  (`_maybe_set_state_normalization()` called after warm-up / before offline
  distill); `release/cfgs/mimicgen/threading_d0_state_simdist_rawstate.yaml`
  (identical to the latent YAML except `encoder_type: "identity"`);
  `scripts/test_simdist.py` (raw-state tests).
- How: `python train_simdist.py --config_path
  release/cfgs/mimicgen/threading_d0_state_simdist_rawstate.yaml --source_agent
  exps/mimicgen/threading_ibrl_source_seed1/model0.pt --save_dir
  exps/mimicgen/threading_simdist_rawstate_seed1`. Head-to-head vs the latent
  arm (`threading_d0_state_simdist.yaml`), same data + coefs.
- Smoke-tested 2026-06-16 (`scripts/test_simdist.py`, both arms pass): raw
  `encode`==z-score, `decode`==exact inverse, `next`==residual, no
  SimNorm/enc_pi/decoder modules, norm buffers in state_dict; distill losses
  finite + decoder loss==0 + heads move; finetune moves dynamics+policy only;
  act single/batched + state_dict round-trip preserves norm buffers; YAML parses.
- CAVEAT: `consistency_coef=20.0` was tuned for the SimNorm latent (small-valued
  simplex); in z-scored state space consistency MSE is ~unit-scale, a different
  balance vs policy/value — kept at 20.0 to isolate the encoder change, sweep if
  `distill/consistency` vs `distill/policy` looks off. NOT yet done: the full run
  + raw-vs-latent SOURCE/TARGET comparison.

### 2026-06-16 "dynamics-is-the-only-learned-component" — offline dynamics trainer + MPC eval
The plan validated in chat: train ONLY a dynamics model on SOURCE; borrow value
+ policy from the frozen source agent and use a HEURISTIC reward, then plan with
MPPI. Rationale (see chat): for a state-based setup the value can come directly
from the source critic `V_src(s)=min-ensemble Q_src(s,π_src(s))` (§10/§17 already
distilled exactly this — now just CALL it), the policy from the source RL actor
(as the MPPI candidate proposal), and the reward as an analytic function of the
PREDICTED state (MPPI can't `env.step` imagined rollouts). So the ONLY learned /
target-adaptable component is the dynamics; target transfer (later) = finetune
the dynamics on the few target demos. This collapses SimDist to "MPC in a
(future-target-finetuned) learned dynamics model, guided by the source agent" —
sidesteps the value/reward/policy collapse problems §17 fought.
- `train_dynamics.py` (NEW) — offline, no env/MPPI/target. Loads a
  `load_source_replay_to_sequence_buffer` dataset (obs/state precomputed),
  episode-level train/val split, fits z-score on TRAIN states, builds the
  raw-state `WorldModel(encoder_type="identity")` and trains ONLY
  `model.dynamics` with the multi-step (rho^t) consistency loss. Logs HELD-OUT
  1-step AND H-step rollout error (normalized MSE + raw-unit MAE) — the real
  dynamics metric, not an MPPI score. Saves the full WorldModel state_dict
  (incl. z-score buffers) → `best.pt`/`latest.pt` + `dyn_cfg.json`. Run:
  `python train_dynamics.py --data_path
  release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5
  --num_data 20000 --num_steps 200000 --save_dir
  exps/mimicgen/threading_dyn_diverse_v2_seed1`.
  (diverse_v2 = 100k mixed source trajs, 4.1 GB, mode=none, obs/state+expert_flags.)
- `rl/simdist_planner.py` — `MppiPlanner` gained EXTERNAL planning-source hooks
  `value_fn` / `reward_fn` (and the pre-existing `proposal_policy`), operating on
  RAW (decoded) state. `_estimate_returns` uses them when set, else falls back to
  the model's own head (vanilla SimDist unchanged). In raw-state mode `decode(z)`
  is the exact denormalized state, so `value_fn=source_agent.value` works directly.
- `rl/simdist_agent.py` — `SimDistAgent.set_planning_sources(value_fn, reward_fn,
  proposal_policy)` plugs them into the planner.
- `eval_simdist_mpc.py` (NEW) — loads the dynamics ckpt into a raw-state
  WorldModel + the frozen source agent (`train_rl.load_frozen_agent`), wires
  `value=source.value`, `proposal=source.act`, heuristic
  `reward=-‖needle−ring‖` (state[9:12] vs state[23:26] BODY positions — a
  planning proxy; the true success uses geom_xpos, off by default `--use_reward
  0` = rank by terminal V_src only), runs SINGLE-process MPPI on SOURCE and
  auto-prints the source-policy baseline as the bar. Run:
  `python eval_simdist_mpc.py --dyn_ckpt
  exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt --source_agent
  exps/mimicgen/threading_ibrl_source_seed1/model0.pt --num_game 50 --shift source`.
- Smoke-tested 2026-06-16: train_dynamics 400-step run on 300 trajs (val 1-step
  norm_mse 0.044 / 5-step 0.12, compounding as expected, ckpts saved); eval pipeline
  end-to-end (MPPI uses source value+proposal, prints baseline+result+diff); source
  state layout probed (needle=state[9:12], ring=state[23:26]); source policy alone
  strict 20-game = 0.65 (this ckpt; the bar is auto-printed each run).
- NEEDLE/RING are BODY positions in obs, NOT the geom_xpos the success check uses
  — heuristic reward is a planning proxy only; meaningful eval is SOURCE for now
  (dynamics is source-only, no target adaptation yet).

### 2026-07-06 target diverse v3 + 蒸馏内双 eval（policy-only vs MPPI）
- **数据**: `threading_d0_target_diverse_v3` — 150k 条 TARGET 域轨迹，成功率
  **26.9%**（旧 v2 成功率过高/覆盖窄的修正）。收集配方：snapshots =
  `threading_ibrl_target_snapshots_seed1` 的 16 个 `snapshot_*.pt`，expert =
  `threading_ibrl_target_snapshots_rlprior_seed1/model0.pt`，**HYBRID=0**（全部
  纯 RL 头 rollout，`load_frozen_agent` 强制 `act_method="rl"`，不挂 prior —
  这是成功率降下来的关键），p_expert=0.5，hand_init_sigma=0.1。同日删除了
  已被 v2 取代的 `threading_d0_source_diverse`（v1，2.8 GB）。
- **train_simdist.py `log_and_save`**: 当 `simdist.use_planner=1` 时每个 eval
  点额外跑一次 `use_planner=0` 的 eval，记 `score/policy_score`（MPPI 分照旧
  `score/score` + TopkSaver 排名依据），打印行含 `policy_score=`。
- **Run**: `threading_simdist_target_distill_v3_seed1` — 复用
  `threading_simdist_target_distill_sig01_seed1/cfg.yaml`，改
  `source_replay_path`=v3、`source_replay_num_data=-1`（全部 150k，旧 run 只用
  50k）、`seq_replay_max_episode=155000`、`distill_steps=800000`（旧 400k）。
  GPU1，log `exps_target_distill_v3.log`，wandb `cao1233/mimicgen`。目的：更广
  覆盖的数据能否让 MPPI 相对 policy-only 学出正增益。中期结果：MPPI 0.66–0.74
  @40–50k ≫ policy-only 0.18–0.22（弱 proposal 上 planner 增益巨大）。

### 2026-07-07 v4 数据（专家 hybrid + snapshots 裸头）+ 1M distill
- **v3 的教训**: HYBRID=0 把专家的 prior 也摘掉了 → 专家（rlprior model0 裸
  头）在 target 只有 30% → policy 头克隆天花板 30%。且"接近 100%"的专家分数
  只在 σ=0.02 窄初始化下成立：**同一 hybrid 专家在收集/部署用的 σ=0.1 下无
  动作噪声也只有 66%**（100 局 strict 实测）——66% 才是 σ=0.1 条件下现有 ckpt
  的真实天花板；v2 专家 65.8% 同因（区间动作噪声几乎无额外 episode 级伤害）。
- **generate_simdist_data.py 新增 `--snapshot_hybrid`**（shell: `SNAP_HYBRID`，
  默认 1=旧行为）：`HYBRID=1 SNAP_HYBRID=0` ⇒ 只有专家挂 prior 做 IBRL max-Q，
  snapshots 保持裸 RL 头。两个开关现在都写入 env_cfg.json 存档。
- **数据**: `threading_d0_target_diverse_v4` — 150k 条 TARGET 轨迹，成功率
  **45.3%**（专家 episodes 66% × 0.5 + snapshots 24% × 0.5）：专家动作干净可
  克隆（policy 头预期 ~66% 天花板），snapshot 部分保住 v3 的覆盖广度。
- **Run**: `threading_simdist_target_distill_v4_seed1` — 同 v3 run 配置，
  `source_replay_path`=v4、`distill_steps=1000000`、horizon=3（world_model +
  planner 显式传入，与默认一致）。GPU0，log `exps_target_distill_v4.log`。

### TODO
- [ ] **DYNAMICS MODEL: try MLP first on full 100k, then Flow if needed.** Train
      the deterministic residual MLP dynamics (`train_dynamics.py`) on the full
      diverse_v2 (100k trajs) and check HELD-OUT rollout error — esp. the H-step
      raw_mae in the CONTACT/insert phase. If it's poor (the MSE model averaging
      through multimodal `p(s'|s,a)` — obs≠full MuJoCo state, so the observed
      transition is stochastic), switch the dynamics to a generative head:
      FLOW-MATCHING (`bc/flow_policy.py` machinery: FourierTimeEmbedding +
      linear-OT + Euler — adapt to a conditional next-state flow `[s,a,x_t,t]→v`,
      net input grows by obs_dim+time), or the cheaper Gaussian ensemble (§23
      `DynamicsEnsemble`). CAVEAT for the planner: a flow dynamics needs an ODE
      solve per transition × 512 candidates × H × iterations (5-10× slower) and
      injects sampling variance into MPPI returns — use fixed-noise / few-step
      ODE / take-the-mode, or prefer the Gaussian ensemble for MPC. Dataset size
      is NOT the reason to switch (the MLP trains fine on 100k); multimodality is.
- [ ] After a good source dynamics: confirm MPPI(dyn+V_src+source proposal) ≥
      source policy on SOURCE (`eval_simdist_mpc.py`), THEN add the target stage
      (finetune dynamics on the 20 target demos) and eval on TARGET.
- [ ] Confirm the own_enc+SimNorm long run is stable + record TARGET strict-success;
      compare vs gated / H2O+ / DR.
- [ ] **RAW-STATE vs latent**: full run of `threading_d0_state_simdist_rawstate.yaml`
      vs the latent arm (same data/coefs) — does dropping the learned encoder match
      or beat the latent on SOURCE convergence + finetuned TARGET strict-success,
      with none of the §17 stability fights? Sweep `consistency_coef` for the
      z-scored-space balance.
- [ ] Sweep `expert_collect_steps` (mixed vs pure on-policy), `horizon`,
      `mixture_coef`, planner `num_samples/iterations`, `consistency_coef`.
- [ ] Ablate stage-2: dynamics+policy vs dynamics-only (`--simdist.finetune_policy 0`).
- [ ] Pixel-mode (needs an image encoder + re-rendered datasets).
- [ ] Optionally distill the value from the source IBRL critic (vs TD bootstrap).

---

## 24. SimDist TARGET adaptation (paper-style two-phase: policy BC → dynamics adapt)

### What
The SimDist (Levy et al., RSS 2026) deploy-time adaptation, run ON TOP of a
stage-1-distilled world model (§17), following the original paper's recipe (NOT
our offline `train_simdist.py::finetune()`). Loads a stage-1 SimDist ckpt and runs
two phases, both keeping the ENCODER (and everything not named) FROZEN:
1. **Stage 2 — TARGET policy BC**: behavior-clone ONLY the policy head on the 20
   TARGET demos (`release/data/mimicgen/threading_d0_target_dynamics`), flat MSE
   on (s,a). Encoder frozen ⇒ the policy re-maps the SAME (frozen) latent to the
   target action distribution. ~1000 grad steps.
2. **Stage 3 — TARGET dynamics adaptation**: roll the SimDist MPPI planner in the
   TARGET env, push transitions into a large sequence buffer; every 1000 collected
   transitions, train ONLY the dynamics head (multi-step latent-consistency, the
   frozen encoder produces the next-state targets) for `dyn_update_steps` (default
   1000, UTD≈1) grad steps, then EVAL 500 episodes on TARGET and log to wandb.
   Repeat to 20000 transitions (20 logged points, x-axis = #transitions). Round 0
   logs the post-BC baseline (transitions=0) before any dynamics adaptation.

Execution + eval use MPPI (`use_planner` is FORCED to 1 here, overriding the
stage-1 cfg which trained/evaluated with the policy head directly).

### Where
- `rl/simdist_agent.py` — two NEW lazily-built single-head optimizers + methods on
  `SimDistAgent`:
  - `update_bc_policy(states, actions, lr)` — `_bc_opt` over ONLY
    `model.policy.parameters()`; flat MSE BC over all (s_t,a_t). Encoder frozen by
    exclusion (with `shared_encoder=1`, `policy_action(s)=policy(encode(s))`, so
    only the policy MLP moves).
  - `update_dynamics_only(states, actions, lr)` — `_dyn_opt` over ONLY
    `model.dynamics.parameters()`; multi-step `||next(z,a)−sg(encode(s'))||²`
    rolled from the segment's first state (encoder under `no_grad` ⇒ frozen).
- `train_simdist_adapt.py` (NEW) — `AdaptConfig` + `AdaptWorkspace`. Rebuilds the
  stage-1 agent/envs from the stage-1 `cfg.yaml` (via `train_simdist.Workspace(
  from_main=False)`), loads the ckpt, forces `use_planner=1`, loads the TARGET
  demos, runs `bc_finetune()` then `adapt()`. `_collect` keeps the rollout episode
  state across rounds (no partial-episode loss). mp eval on TARGET.
- `scripts/train_simdist_adapt_threading.sh` (NEW) — `SEED/STAGE1_CKPT/GPU/
  BC_STEPS/ADAPT_INTERVAL/MAX_TRANSITIONS/DYN_UPDATE_STEPS/NUM_EVAL_EPISODE` env
  vars + `"$@"` passthrough.

### How
```bash
source set_env.sh
bash scripts/train_simdist_adapt_threading.sh                       # defaults
GPU=3 SEED=1 bash scripts/train_simdist_adapt_threading.sh
# raw:
python train_simdist_adapt.py \
    --stage1_ckpt exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
    --save_dir exps/mimicgen/threading_simdist_adapt_seed1 \
    --bc_steps 1000 --adapt_interval 1000 --max_transitions 20000 \
    --dyn_update_steps 1000 --num_eval_episode 500 --num_eval_proc 10
# ★ standalone eval of an adapted ckpt -- the ONLY valid source of a headline
#   number (iron law #14). An adapt save_dir's cfg.yaml is an AdaptConfig, so
#   pyrallis refuses it as train_simdist.MainConfig and eval.py dies with
#   "The fields {stage1_ckpt, bc_anchor_coef, ...} do not belong to the class".
#   VERIFIED FIX 2026-09-01 (works on state AND pixel):
python scripts/stage_adapt_ckpt_for_eval.py --adapt_dir <adapt dir> \
    --out_dir /scratch/yifanr/simdist/eval_staging/<name> --num_game 500 --mp 10
#   It copies the weights next to the STAGE-1 cfg and PRINTS the eval command
#   with --original_mppi / --planner_set taken FROM THE ADAPT CFG. Both matter:
#   the stage-1 cfg says use_planner=0 (stage 1 evals the policy head), so naive
#   staging silently measures no planning; and iron law #9 says planner knobs do
#   not transfer, so they must be copied, not defaulted.
#   --shift_from_cfg 1 IS correct on the staged dir: the adapt run takes its
#   TARGET env from the stage-1 cfg's eval_domain_shift (:277), and that block is
#   the §12 target (verified knob-by-knob).
#   Canonical line both arms must match (iron law #15):
#   [EVAL ENV] task=Threading_D0 shift=fixed nut_z_rot=FULL(0-360) sigma=0.1 strict=True ep_len=400
```

### Design choices
- **Single-head optimizers, not the stage-2 `finetune_opt`**: `finetune_opt` is
  built once at `__init__` from the cfg flags, so it can't switch between a
  policy-ONLY phase and a dynamics-ONLY phase in the same run. Two dedicated
  lazily-built optimizers scope each phase to exactly one head ⇒ every other
  parameter (incl. encoder) is frozen by exclusion.
- **Encoder frozen in BOTH phases** (the user's spec + the paper): the latent is
  fixed; stage 2 only re-maps it to the target action, stage 3 only re-fits the
  latent transition. With `shared_encoder=1` this is well-defined (the policy and
  dynamics each read the frozen `encode(s)`).
- **MPPI forced on**: adaptation is meaningless without planning over the adapted
  dynamics; the stage-1 run used the policy head directly (`use_planner=0`).
- **x-axis = transitions** (20 points to 20k); round 0 = post-BC baseline.

### Verification status
Smoke-tested 2026-06-16:
- ✅ synthetic-tensor freeze test: `update_bc_policy` moves ONLY `model.policy`
  (encoder/dynamics/reward/value unchanged); `update_dynamics_only` moves ONLY
  `model.dynamics` (encoder/policy unchanged); MPPI `act` single→`[A]` &
  batched→`[B,A]`.
- ✅ end-to-end short run (real stage-1 ckpt + 20 demos, bc_steps=5,
  adapt_interval=450, 1 round, eval 10): stage-2 BC loss drops, round-0 baseline +
  round-1 (collect→dynamics-only→eval→save) all run, wandb-off logging keyed by
  transitions.
NOT yet done: the full 20k-transition run's adaptation curve (in flight 2026-06-16,
`exps/mimicgen/threading_simdist_adapt_seed1`, GPU3, wandb; stage-1 ckpt =
`threading_simdist_tdmpc2_stage1_seed1/latest.pt`).

### TODO
- [ ] Record the TARGET strict-success adaptation curve vs #transitions; compare
      the post-BC baseline (round 0) vs the dynamics-adapted endpoint, and vs the
      other transfer baselines (gated `force_c=1.0` ~13% / H2O+ / weighted BC).
- [ ] Ablate: dynamics-adapt ON vs OFF (does adapting the dynamics on TARGET
      rollouts actually help MPPI over the frozen stage-1 dynamics?).
- [ ] Sweep `dyn_update_steps` (UTD), `adapt_interval`, and rollout exploration
      noise (currently deterministic MPPI, stddev=0).
- [ ] Consider also adapting the reward/value heads on TARGET (currently frozen) —
      under the sparse reward the value is the planning signal; a stale source
      value may bottleneck MPPI more than stale dynamics.

---

## 25. Residual off-policy RL on a frozen ADAPTED SimDist base (ResFiT-style)

### What
The adapted SimDist world model (§24, `threading_simdist_adapt_seed1/model0.pt`)
plateaued at TARGET strict-success ≈ **0.75** (MPPI) — dynamics adaptation can't
push it further. This trains a small **residual RL policy** on top of the frozen
adapted base to keep improving it, exactly like ResFiT (Ankile et al. 2025,
`~/residual-offpolicy-rl`): the frozen base proposes `a_base(s)` via MPPI, a
trainable residual actor adds a bounded `Δ`, executed `a = clip(a_base+Δ,-1,1)`,
and off-policy TD3 RL (critic on the FULL `a`, actor maximises Q) learns the
correction while the base stays frozen. It REUSES the repo's faithful ResFiT port
(§ `rl/residual_rl_agent.py`, `[[project-residual-rl-faithful]]`) + its RLPD
recipe (50/50 online+offline-demo mix, LayerNorm REDQ critic, base+noise warmup,
critic-only warmup, TD3 target smoothing, zero-init residual, `actor_lr=1e-6`,
`Q∈[0,1]` clamp), changing only what the SimDist base forces.

**The one hard constraint**: the adapted base's deployment behavior is **MPPI**
(`use_planner=1`, original recipe, 350 candidates × 3 iters × H3), far too
expensive to re-query 256×2 times per critic update. So — exactly as ResFiT
stores `observation.base_action` in the replay — the base action is computed ONCE
per collected env step and **STORED in the buffer's obs** (`base_action` key); the
update reads the stored value instead of re-planning. rela n-step gives
`next_obs["base_action"] = base(s_{t+n})` for free. MPPI is thus queried only at
collection (1 plan/step, reused across the act + the next step's stored value) and
eval (1 plan/step) — normal SimDist cost. UTD=4 (`num_critic_update=4`,
`update_freq=1`), n-step=5, ResFiT hyperparams. Train + eval in the TARGET domain
(the §24 adapt deploy domain: dynamics-only §12 shift + hand_init σ=0.1); the
buffer is initialised from the adapted-MPPI-base rollout, and the offline RLPD
anchor = the 20 TARGET demos PLUS successful MPPI-base rollouts harvested during
warmup.

### Where
- `rl/residual_simdist_agent.py` (NEW) — `ResidualSimDistAgent(ResidualRLAgent)`
  (+ `ResidualSimDistConfig = ResidualRLConfig`). Reuses `ResidualActor` /
  `EnsembleQ` / z-score / `_min_q` / `_clamp_q` / `_soft_update` /
  `set_normalization` / `attach_base_policy` verbatim; overrides only:
  - `_base_action(state_raw)` — the base is a `SimDistAgent` (no `__call__`);
    query `base.act({"state": s}, eval_mode=True, stddev=0, cpu=False)` (MPPI,
    single+batched). Used at act()/eval + collection only.
  - `act(...)` — reuses a precomputed `obs["base_action"]` if present (collection),
    else plans (eval path).
  - `update(batch, update_actor)` — consumes a rela `Batch` with base actions in
    obs/next_obs; TD3 target uses the n-step bootstrap `batch.bootstrap` (=γⁿ,
    already 0 at a success terminal), no separate γ·(1-done).
- `train_residual_simdist.py` (NEW, fork of `train_residual_rl.py`) — loads the
  adapted SimDist base (reads the adapt ckpt's cfg → `stage1_ckpt` → stage-1
  `cfg.yaml`, builds `SimDistAgent`, `apply_original_recipe` + `use_planner=1`,
  loads the adapt ckpt, attaches frozen). Two rela `ReplayBuffer`s (nstep=5):
  `online` (MPPI-base rollout) + `offline` (20 TARGET demos, base_action
  batch-planned per demo, + harvested warmup successes). Both store a minimal obs
  `{state, base_action}` so keys align for `Batch.merge_batches`. z-score fit on
  demo states. Warmup rolls MPPI base + uniform residual noise (stops only at a
  terminal so the rela episode ends flushed), then critic-only warmup, then the
  RLPD-mixed UTD-4 loop. `load_model` for standalone eval.
- `evaluate/eval.py` — `--mode residual_simdist`.
- `release/cfgs/mimicgen/threading_d0_state_residual_simdist.yaml` (NEW).
- `scripts/train_residual_simdist_threading.sh` (NEW) — `SEED/BASE_ADAPT_CKPT/GPU/
  STEPS/UTD` + `"$@"`.
- `scripts/test_residual_simdist.py` (NEW) — smoke test.

### How
```bash
source set_env.sh
bash scripts/train_residual_simdist_threading.sh                 # GPU 0, 100k steps
GPU=1 SEED=1 bash scripts/train_residual_simdist_threading.sh
# raw:
python train_residual_simdist.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_residual_simdist.yaml \
    --base_adapt_ckpt exps/mimicgen/threading_simdist_adapt_seed1/model0.pt \
    --save_dir exps/mimicgen/threading_residual_simdist_seed1
# eval every 1000 transitions on TARGET (100 episodes, mp). standalone re-eval:
python evaluate/eval.py --weight exps/.../latest.pt --mode residual_simdist \
    --num_game 100 --strict_success 1
```

### Faithfulness to ResFiT Algorithm 1 (audited 2026-07-07)
Verified `ResidualSimDistAgent.update` + `rl_train` line-by-line against ResFiT
Algorithm 1. Matches: warmup `at=εt+abt` storing `(st, abt, at, st+1, abt+1, rt,
dt)` (base action in obs → `abt`/`abt+1` via n-step rela); residual conditioned on
`abt`; `at=clip(abt+art)`; UTD critic updates; 50/50 offline/online batch; TD3
target smoothing on `art+1`; `y=rt+(1−dt)γⁿ·min_subset(Qφ′i)` + `[0,1]` clamp
(ResFiT `clip_q_target_to_reward_range`); all-heads critic MSE; soft targets.
**Two deviations were FIXED to match Algorithm 1** (the initial impl inherited them
from the §20 `ResidualRLAgent`): (1) **line 20 — the actor now maximizes the
ENSEMBLE MEAN `(1/N)Σᵢ Qφi`** (was `min` over 2 random heads = ResFiT's non-default
`min_random_pair`); (2) **actor updated once per env step** (`actor_updates_per_
iteration=1`, gated by the last UTD iteration in `rl_train`) and the **critic
target soft-updated after every critic update** (line 18), replacing the old
`policy_freq=2` gate. `_min_q` is still used for the critic TARGET (line 16), which
IS a min-subset. NOTE: the run started 2026-07-07
(`threading_residual_simdist_seed1`, ~0.90 by 90k) used the PRE-fix variant
(min-2 actor, actor 2×/step) — a valid RED-Q-ish variant that works well but is not
byte-faithful to Algorithm 1; restart for a strictly-faithful run.

### Design choices
- **Stored base action, not recomputed** — the whole reason the MPPI base is
  tractable; also strictly more faithful to ResFiT (`observation.base_action`).
- **n-step via rela** (not a flat buffer) — native n-step returns + the RLPD mix
  is just `Batch.merge_batches(online, offline)` (base_action rides in obs).
- **Warmup stops at a terminal** — a mid-episode break leaves a dangling rela
  episode and crashes the next `new_episode()` (`Episode.init` assert).
- **Offline anchor = demos + harvested MPPI-base successes** — richer target
  success coverage for the RLPD critic anchor (per the user's request).
- **Train+eval on TARGET, no gap weighting** — the base is already target-adapted
  and eval is target; this is straight ResFiT finetuning in the deploy env.

### Verification status
Smoke-tested 2026-07-06:
- ✅ `scripts/test_residual_simdist.py`: `_base_action` shape/range; `act`
  single+batched + `act==base` at zero-init + reuses stored `base_action`; rela
  n-step buffers + merged RLPD mix carry `base_action` in obs/next_obs; `update`
  moves critic+actor (finite losses); state_dict excludes the base + round-trips
  norm buffers.
- ✅ Tiny end-to-end run (real adapted MPPI base, 300 warmup / 30 critic-warmup /
  1200 steps): base loads (MPPI 350/3/H3), 20 demos load with per-state MPPI base
  actions + z-score, warmup rolls MPPI+noise & harvests successes into offline,
  RLPD-mixed n-step updates run (critic_loss≈0.004, q_mean≈0.15∈[0,1], delta_abs
  ≈0.011 ≪ action_scale, train episodes succeed), eval + ckpt save, run continues
  past the benign EGL teardown warnings.
- ✅ Standalone `evaluate/eval.py --mode residual_simdist` loads + re-attaches the
  MPPI base + strict-success TARGET eval (exit 0).
NOT yet done: the full 100k run's TARGET strict-success adaptation curve vs the
frozen adapted base's ~0.75 (early <~40k eval below base is normal — residual
starts at 0, critic uncalibrated, `[[project-residual-rl-faithful]]`); watch
`residual/q_mean` converging to the eval mean and `delta_abs` staying small.

### TODO
- [ ] Full 100k run: does the residual lift TARGET strict-success above the frozen
      adapted base's ~0.75? Record the curve (eval every 1000 transitions).
- [ ] Ablate the offline harvest (demos-only vs demos+harvested successes) and
      `offline_fraction`; sweep `action_scale` (0.1/0.2), `critic_warmup_steps`.
- [ ] MPPI-base eval is the wall-clock bottleneck (1 plan/step × 100 eps × 100
      evals) — consider fewer eval episodes or less frequent eval if too slow.

### IBRL-with-the-same-base comparison arm (train_ibrl_simdist.py)
The head-to-head for "residual on the base" vs "IBRL with the base as the prior".
SAME adapted SimDist base (MPPI, ~0.75) + SAME TARGET domain, but instead of a
bounded residual, standard **IBRL** (Hu et al.): a fresh RL actor competes with the
base in max-Q `a = argmax_{a in {pi_rl(s), pi_base(s)}} Q(s,a)`, RLPD recipe
(`mix_rl_rate=0.5` demo-mixed replay, RED-Q `num_q=5/num_k=2`, `bc_loss_coef=0.1`
actor reg, nstep=3). The base is the IBRL BC prior.
- `rl/train_ibrl_simdist.py` (NEW) — `StoredActionBCPolicy` (the base's MPPI action
  is computed once/step and STORED in the buffer's obs `base_action`; `.act(obs)`
  returns the stored value during training, falls back to LIVE MPPI at eval where
  the env obs has none — so IBRL's per-critic-update `pi_base(s')` on the 256-batch
  reads stored actions, never re-plans) + a QAgent(`act_method=ibrl`) forked from
  the residual trainer's data path (base load, demo seed with `base_action`, warmup
  rolling the MPPI base, collection storing `base_action`). One `rl/replay.py`
  ReplayBuffer (`use_bc=True`): 20 TARGET demos seeded (→ main + bc_replay) +
  online; `sample_rl_bc` for the RLPD mix + `sample_bc` for the actor BC loss;
  `agent.update(batch, stddev, update_actor, bc_batch, ref_agent=self.agent)`
  (`bc_loss_dynamic=0` ⇒ `ref_agent` unused). `load_model` re-attaches the prior.
- `release/cfgs/mimicgen/threading_d0_state_ibrl_simdist.yaml`,
  `scripts/train_ibrl_simdist_threading.sh`, `scripts/test_ibrl_simdist.py` (NEW);
  `evaluate/eval.py --mode ibrl_simdist`.
- How: `bash scripts/train_ibrl_simdist_threading.sh` (GPU=0). Eval every 1000
  transitions on TARGET.
- **Key contrast to expect**: the residual starts AT the base (~0.75, zero residual)
  and edges up; **IBRL from scratch cold-starts near 0** — the random RL actor is
  picked by max-Q until the RED-Q critic (fed by the prior/demo warmup + mix)
  calibrates to defer to the 0.75 prior, then the actor learns past it. Early eval
  ≈0 is EXPECTED (not a bug); watch `actor/bootstrap_bc` (fraction of max-Q that
  picks the prior) climb as the critic calibrates. This slow start vs the residual's
  warm start is exactly the comparison.
- Verification (2026-07-07): ✅ `scripts/test_ibrl_simdist.py` (StoredActionBCPolicy
  read/fallback/frozen; QAgent(ibrl) act with stored base_action; ibrl update on a
  stored-prior rela batch moves critic+actor; state_dict excludes the prior); ✅
  tiny e2e (base+prior load, 20 demos → replay+bc_replay, MPPI warmup, ibrl updates
  with `actor/bootstrap_bc≈0.26` + `rft/bc_loss` active, eval + ckpt, benign EGL
  teardown); ✅ standalone `evaluate/eval.py --mode ibrl_simdist` load path. NOT yet
  done: the full 100k IBRL-vs-residual TARGET-success comparison.

---

## 26. Interaction-thrifty 3-stage pipeline (30k adapt → reuse its buffer for IQL → residual / IBRL)

### What
Cut TARGET-domain interaction from ~620k transitions to **30k + stage-3**, keeping
the score. Two changes vs §25's pipeline:
1. **Stage 1 stops at 30k** SimDist adapt transitions (was 340k), taking the LAST
   ckpt (strict 30k, not best-by-eval).
2. **Stage 2 reuses stage-1's own replay buffer** for the offline IQL pretrain,
   instead of rolling a fresh 30k with the frozen base.

Then three stage-3 arms share those two stages: **residual** (§25),
**IBRL + MPPI-clone actor**, **IBRL + RWR absolute actor**.

### Why it works
`adapt` gains are heavily front-loaded. The 340k run's smoothed MPPI score only
went 0.81 (@30k) → 0.86 (@340k). The 30k rerun lands at **0.820** target strict
success (vs frozen094's 0.94, which is a best-by-eval pick off a noisy curve).
And the IQL critic trained on the (non-stationary) adapt buffer is **not degraded
at all** — see the gate table below.

### Where
- `train_simdist_adapt.py` — `eval_every_rounds` (dynamics still trains EVERY
  round; only the expensive rollout eval is thinned) + `dump_rollouts` (exports
  the adapt buffer as `rollouts.npz` in `scripts/verify_iql_critic.py`'s episode
  format) + `_dump_rollouts()`.
- `scripts/pretrain_iql_qagent.py` (NEW) — offline IQL into **QAgent's own nets**
  (`rl.critic.MultiFcQ` + `rl.actor.FcActor`, RAW state). Needed because IBRL
  cannot load the residual line's z-scored `EnsembleQ`. `--actor_mode
  rwr|awr|clone|none`; `--load_critic_from` skips the IQL phase and only re-trains
  the actor, so both IBRL arms share a bit-identical critic.
- `train_ibrl_simdist.py` — `offline_transition_path` (+ re-plans `base_action`),
  `init_critic_from`, `init_actor_from`, `"None"`→`""` normalization, and an
  assertion that `init_actor_from` ⊕ `actor_bc_pretrain_steps`.
- `train_residual_simdist.py` — `offline_replan_base_action` (default 0).
- `scripts/verify_iql_critic.py` — `--replan_base_action`.

### Stage-2 gate (2026-07-09, both PASS, zero degradation)
| | residual `EnsembleQ` (z-score, expectile .7) | IBRL `MultiFcQ` (raw, expectile .9) | historical |
|---|---:|---:|---:|
| per-demo Spearman(V, −steps) | **+1.000** | **+0.999** | +0.999 |
| Pearson(Q, γ^steps) | **+0.978** | +0.969 | +0.982 |
| MAE | 0.026 | 0.102 | 0.028 |

### ⚠️ AWR degenerates to BC here — a mathematical necessity, not a tuning miss
Each state in this data has exactly ONE action (deterministic MPPI base + demos),
so `Q(s,a)` is a function of `s` alone, `V(s)` fits it exactly, and `adv = Q − V`
collapses to fit residual. Measured: **adv_std = 0.011** vs **Q_std = 0.258**.
Raising expectile .7→.9 only SHIFTS the weights (`w_mean` 1.006→0.979), it does
not spread them (`w_std` 0.016). `beta` would need ~100, which amplifies critic
noise. **Use RWR instead** — weight by `Q` itself, batch-centred:
`w = exp(3·(Q − mean Q))` → `w_std 0.65-0.73`, `p10/p90 = 0.26/2.16`
(Q is 0.731 on success-episode transitions vs 0.358 on failures).

### Margin-collapse attribution (sharper than "floor=0 collapses")
The two collapsed runs (`offlinepretrain_collapsed` 11k→0,
`h5warm_criticpre_collapsed` 13k→0) both had `freeze_actor_during_margin=0`. The
two that survived (`permmargin` floor 0.1, steady 0.95@76k; `srccritic_mppiactor`
floor 0, peak 0.97@170k but oscillating) both had it ON. So **floor=0 is viable —
`freeze_actor_during_margin` is the necessary part.** The two IBRL arms here run
`prior_margin_init 1.0 → final 0.0` + `freeze_actor 1`, betting that a TARGET-domain
IQL critic damps `srccritic_mppiactor`'s oscillation (its critic was SOURCE-domain).

### How
```bash
# stage 1 (30k, eval every 5k, dump the buffer)
python train_simdist_adapt.py \
  --config_path exps/mimicgen/threading_simdist_adapt_long_h5_seed1/cfg.yaml \
  --save_dir exps/mimicgen/threading_simdist_adapt_30k_seed1 \
  --max_transitions 30000 --eval_every_rounds 5 --dump_rollouts 1

# stage 2a (IBRL-shaped: MultiFcQ critic + RWR absolute actor)
python scripts/pretrain_iql_qagent.py --rollouts <A>/rollouts.npz \
  --ibrl_cfg exps/mimicgen/threading_ibrl_permmargin_seed1/cfg.yaml \
  --save_dir <A>/iql_qagent --actor_mode rwr --expectile 0.9
# stage 2b (residual-shaped: EnsembleQ + AWR residual actor); rollouts.npz must be
# copied into <run>/iql_critic_verify/ so collection is skipped (cache hit)
python scripts/verify_iql_critic.py --base_run <R> --train_actor 1 --replan_base_action 1

# stage 3, three arms (300k each, eval every 1000 transitions)
# b1 residual (E config), b2 IBRL+clone actor, b3 IBRL+RWR actor
```

### Gotchas found while building this
- **`base_action` identity is broken.** The adapt buffer's actions were planned by
  a still-adapting dynamics model, so `action != base_action`. Both
  `train_residual_simdist.py:329` and `verify_iql_critic.py`'s AWR branch hard-coded
  the identity. Fixed behind `replan_base_action` flags (default OFF ⇒ existing
  runs byte-identical); when ON, base_action is re-planned in chunks with the final
  frozen base.
- **`actor_bc_pretrain_steps` silently overwrites `init_actor_from`** (it runs after
  `__init__`). Asserted mutually exclusive — this is exactly what separates the two
  IBRL arms.
- **pyrallis turns `--flag ""` into the string `'None'`** → `train_ibrl_simdist.py`
  tried to `open("None")`. Normalized in `__post_init__`.
- **NOT fixed (pre-existing):** `common_utils/rela/episode.cc:90` sets `bootstrap`
  purely by POSITION (0 for the last `nstep` steps of an episode), ignoring
  terminal/success — so a 400-step timeout failure is treated as an absorbing
  terminal (Q target 0). Every existing IBRL/residual run carries this bias, so it
  was left alone for comparability; loading the 30k rollouts (31% failures) widens
  it slightly (~0.8% of transitions). `verify_iql_critic.py::build_flat` does it
  correctly (`done = t==T-1 and success`), so IQL-pretrain and the IBRL replay have
  always disagreed on this point.

### TODO
- [ ] Record the three arms' TARGET strict-success curves vs the 0.820 base; is the
      30k base's 12-point deficit recovered? Compare to frozen094-based residual
      (~96% at 500 games).
- [ ] b2 vs b3: hard success-filtering (clone) vs soft Q-weighting (RWR) for the
      actor init.
- [ ] Confirm the `freeze_actor + floor=0` bet: does the TARGET IQL critic kill the
      `srccritic_mppiactor` oscillation, and does `bootstrap_bc` fall well below
      permmargin's 0.97 (i.e. the actor really takes over)?
- [ ] If the 30k base's deficit dominates, sweep the stage-1 stopping point
      (10k looked as good as 30k: 0.84 vs 0.82) — the cheapest lever of all.

---

## 42. ★ stage-2 的 policy 头受**两个竞争条件**约束 —— source 锚把弱 base 推 +0.164(2026-08-11/12)

### ★ 结论(用户框架):stage-2 出来的 policy 必须**同时**满足两件事
1. **靠近 π_src** —— value 头是 stage-1 在 `source_diverse_v2` 上蒸馏的 `V^{π_src}` 且**全程冻结**
   (`adapt_value=0`),所以策略应与它保持 on-policy 一致;
2. **在 target 上有足够成功率** —— 它是 MPPI 的 proposal + stage-3 采集的驱动者,要能热启动、
   让 dynamics 快速修上去。

两者**互相竞争**,而 `bc_anchor_coef=λ` 正是这个权衡的旋钮 —— 两个轴都随 λ 严格单调、方向相反:

| λ | **(1) D_SRC**(离 π_src,越小越满足) | **(2) bc_loss**(target demo 拟合,越小越满足) | policy_only | MPPI |
|---:|---:|---:|---:|---:|
| 0 | **0.2354** ✗ | **0.0083** ✓ | 0.407 | 0.793 |
| 0.1 | 0.0517 | 0.0088 | 0.570 | 0.926 |
| **1.0** | **0.0248** | **0.0103** | **0.608** | **0.957** |
| 3.0 | 0.0179 ✓ | 0.0120 | 0.477 | 0.912 |
| 10.0 | 更低 ✓ | **0.0160** ✗ | 0.317 | 0.817 |

⇒ **`policy_only` 与 MPPI 都在中间(λ=1)取峰,因为它们是两个条件共同作用的产物,不是其中一个轴。**
两端的失效机制各自清楚:λ 太小 ⇒ 违反 (1),流形外无约束(§35:`novel_frac`≈1.0,**100% 的 stage-3
采集在 demo 流形之外**,流形外跨 demo 行为分歧达动作自身散布的 80%)⇒ BC 在无数据处自由乱动;
λ 太大 ⇒ 违反 (2),锚压过 demo 项、学不到 target 动作 ⇒ 退化成接近 `nobc` 的行为(慢 7–18×,见下)。
⚠️ **别拿 `policy_only` 当条件 (2) 的度量**(我一开始就是这么错的):它是**结果**,λ=0 处它比 λ=1 更差
(0.407 vs 0.608),会让人误判"左侧不是权衡";真正代表条件 (2) 的是 **`bc_loss`**,它在 λ=0 处才是最好的。

### 起因
stage-2 是从 stage-1 权重**微调**而非从头 BC(实测:`nobc`=π_src 直接上 target 只有 0.15,
从头 MLP-BC 20 条 ≈0.25,微调 0.35–0.42),所以微调**必然**保留 π_src 的特征 ⇒ "用什么数据微调、
让它离原策略多远"是一个真正的设计自由度,而此前 stage-2 **零正则**(Adam 无 weight decay、policy 头
无 dropout),唯一的隐式旋钮只有"训多少步"。

### ★ 主结果:干预有效,是本项目测到的最大效应
新增 loss(`bc_anchor_coef=λ`,默认 0 ⇒ 逐位兼容):
```
L = MSE(π_θ(s_demo), a_demo)            # demo 状态: 照常学 target 动作
  + λ · MSE(π_θ(s_src),  π_src(s_src))  # SOURCE 状态: 钉回 π_src (目标在 stage-2 第一步前冻结)
```
**两项打在不同分布上是全部要点**:demo 覆盖处**不欠拟合**,只在 demo 从没到过、BC 本来自由乱动的
地方钉住。这是它区别于"少训 BC 步数"(各向同性地少动)的关键 —— §38 记的"按 policy_only 选更少
bc_steps ⇒ adapt −0.060 (z=−5.26)"因此不否定本方案。

**弱 base `sat_160000_s201`(λ=0 时 0.793),20–40k 窗口,3 seed/格(λ≥3 为 2 seed),demo 集固定:**

| λ | D_SRC | bc_loss | MPPI | sd | **配对 Δ vs λ=0** | t | policy_only | round-0 | 到 0.90 需 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2354 | 0.0083 | 0.793 | .014 | — | — | 0.407 | 0.650 | 未达到 |
| 0.1 | 0.0517 | 0.0088 | 0.926 | .020 | **+0.133** | 18.0 | 0.570 | 0.750 | 16k |
| **1.0** | 0.0248 | 0.0103 | **0.957** | .016 | **+0.164** | **112.6** | **0.608** | 0.737 | **9k** |
| 3.0 | 0.0179 | 0.0120 | 0.912 | .017 | +0.118 | 47.6 | 0.477 | 0.575 | 18k |
| **10.0** | (未测) | 0.0160 | **0.817** | .008 | **+0.023** | 2.7 | **0.317** | 0.535 | 未达到 |

(`bc_loss` = 各 seed stage-2 末步的均值;`D_SRC` 来自标定脚本,只扫到 λ=3,**λ=10 的 D_SRC 未实测**,
上面结论表里写的"更低"是按 λ=0.01→3 的单调趋势外插,不是测量值。)

**① 倒 U,峰在 λ=1,有效区间 ≈ [0.1, 3]。**λ=10 基本回落到 λ=0 的水平(Δ=+0.023;t=2.68 但
**n=2 ⇒ df=1,临界 t=12.7,p≈0.23,不显著**)。
`policy_only` 同步走倒 U(0.407→0.608→0.317)⇒ 右侧失效机制清楚:**锚压过 demo 项**,学不到
target 动作,退化成接近 nobc 的行为(nobc 慢 7–18×,见下)。
**② 效应极稳**:λ=1 三个 seed 的配对 Δ = +0.165/+0.161/+0.165(**t=112.6**),三个 λ 组零重叠,
9/9 配对同向。0.164 是 §38 采集抽签噪声 σ=0.092 的 **1.8 倍**、A/B 采集器差距(+0.068)的 2.4 倍。
**③ 把平庸 demo 集推到与最强 base 持平**:0.957 vs §31 的 `long_h5` 0.96(n=2000)。
**④ 效应分解(λ=1)**:round-0(dynamics 未动,只换 policy 头)已有 **+0.087**,20–40k 是 +0.164
⇒ **约一半在 adapt 阶段累积**,那一半只能经"更好的 policy → MPPI 采到更好的数据 → dynamics 修得
更好",与 value 无关。

### ⚠️ 轴 (1) 的**机制标签**未定:"value on-policy" vs "π_src 是好的流形外先验"
"靠近 π_src 有好处"由 λ 扫描直接证实(见上)。但**为什么**有好处,有两个候选,而它们**预测完全相同的
倒 U**,所以 λ 扫描本身分不开:

| 候选 | 现有证据 |
|---|---|
| **(1a) value 需要 on-policy**(原始假说) | ⚠️ oracle 反证(下面 (a));⚠️ 强 base null(下面 (b)) |
| **(1b) π_src 只是一个好的流形外先验/正则** | 与全部数据相容 |

**唯一给 (1a) 留缝的观测**:round-0(dynamics 完全相同)上 proposal ×1.5 只换来 MPPI **+0.087**,
而 §27 记的是固定 dynamics 下 proposal ×7.5(0.11→0.83)只让 MPPI 在 0.80–0.86 间动(+0.06)
⇒ **本轮每单位 proposal 变化的 MPPI 响应大得多**,超额那部分**可能**就是 (1a)。最省事的替代解释是
操作点不同(这里 0.65–0.74 离天花板远、§27 那里已接近)。**未判决。**
判决实验(便宜、不跑环境):在 λ=0 与 λ=1 各自 planner 的 on-policy 候选集上测 §33 的两个量 ——
`std_i(γ^H V)` 集内散布 + **top-64 elite 重合率**(§33 实测它与分数 Spearman **+0.943**,最灵)。
λ=1 若重合率明显更高 ⇒ (1a) 有独立贡献;相同 ⇒ 只剩 (1b)。

**(a) oracle 探针(真实模拟器 rollout + 同一个冻结 value 头,100 局):D_SRC 推 9.5×,oracle 不动。**
| λ | 0 | 0.01 | 0.1 | 1.0 | nobc |
|---|---:|---:|---:|---:|---:|
| D_SRC | 0.2354 | 0.1167 | 0.0517 | 0.0248 | **0** |
| oracle | 0.980 | 0.990 | 1.000 | 0.990 | 1.000 |

跨度 0.020,Fisher p=0.50。**完美 dynamics 下,策略离 π_src 多远对那个冻结 value 头的排序没有
可测影响** —— 而这是该机制唯一的作用渠道。
**(b) 强 base 上测不到增益(同一个 stage-1 ⇒ 同一个 value 头)**:若是 V 失配,把 `long_h5` 的
策略也拉近 π_src 应同样受益。实测 Δ=**−0.009**。

**⚠️ 但这两条反证只打死 (1a) 的纯粹形式,不打死条件 (1) 本身**:oracle 在**真实**终端读 value,
learned 在**想象**终端(偏 ~5.6mm,§27)读 ⇒ 它排除的是"V 对漂移策略到达的**真实**状态排错序",
**没排除交互形式**("策略漂移 ⇒ 候选落到模型出错处 ⇒ V 在**那里**排错序")。(b) 也可能只是天花板
(该窗口 λ=0 已 0.929、只剩 0.071 空间)。⇒ **条件 (1) 作为"要靠近 π_src"是实证成立的;只有"因为
value"这个理由待判。**

**旁证:锚的效应还带正则的签名** —— stage-2 此前零正则,而效应分解显示 +0.164 里 round-0 只有
+0.087、**另一半在 adapt 阶段累积**(必须经"proposal 更好 ⇒ 采集数据更好 ⇒ dynamics 更好"这条路,
与 value 无关)。⇒ 无论 (1) 的标签是什么,**至少有一半的收益走的是数据质量渠道**。
一个可证伪的强预测:若锚主要是正则,它应当**压小 §38 那个跨 demo 集的抽签方差 σ=0.092**
(目前只测了同 demo 集内换 stage-3 seed,λ=0 sd=0.014 vs λ=1 sd=0.016,**没变化,但那是另一种噪声**)
⇒ TODO 里"换第二个弱 demo 集"其实是这个预测的判决实验,不只是复现。
另一个便宜的判别:同样扫 policy 头的 `weight_decay`/dropout —— 若通用正则也给 +0.16 ⇒ 是容量控制、
π_src 不特殊;若只有 source 锚有效 ⇒ **π_src 的流形外行为本身有信息**,更靠近条件 (1) 的读法。

### ★ 效应大小 ∝ 该 base 到自己天花板的距离(与 §36 同型)
| base | λ=0 | λ=1 | Δ | 口径 |
|---|---:|---:|---:|---|
| **s201**(弱) | 0.793 | **0.957** | **+0.164** | 3 seed 配对, t=112.6 |
| **long_h5**(强) | 0.929 | 0.920 | **−0.009** | λ=0 仅 1 seed ⇒ SE=0.033 |

**交互 = +0.173, SE=0.033, t=5.2, CI [+0.108, +0.238]** ⇒ 结构结论稳。
⚠️ **强 base 那格只能说"测不到增益,上界约 +0.05"(CI [−0.076,+0.054]),不能说"无效"或"有害"**
(CI 覆盖 0;λ=0 只有 1 seed)。且天花板本身有约束:该窗口 λ=0 是 0.929、自身稳态 0.96 ⇒ 最多
只剩 0.071 空间,**"+0.164 没复现"部分是算术必然**。
与 §36 flow-proposal 那次结构**完全同型**(snapB 弱 WM +0.134 / long_h5 强 WM −0.026, 交互 0.160)
⇒ 两次独立干预、不同机制,同一个形状,再次印证「proposal 只在它 binding 时才值钱,值多少 ≈ 到该
系统天花板的那段距离」。

### 数据侧的对照(snap160 未触顶 vs snap400 已触顶 vs 蒸馏用的 source)
`source_diverse_v2` 参照:63.7% 成功,长度 212±146(全部)/ **105.7±46(成功轨迹)**;
`d_pi_src` 在 **source 数据自身**上是 **0.281**(是 B 的 1.8×、ORIG 的 2.7×)。

| | adapt | len | \|a\| | d_pi_src | D_SRC(真量) | **D_ORIG** | cover_lat% |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORIG(那个 0.93) | 0.965 | 99.2 | 0.322 | 0.104 | 0.2194 | 0.0905 | 67.4 |
| **A=snap160**(4 seed 均) | 0.819 | 93.0 | 0.342 | 0.126 | **0.2473**±.009 | **0.1231**±.008 | 49.1 |
| **B=snap400**(5 seed 均) | 0.751 | 71.4 | 0.404 | 0.153 | **0.2721**±.010 | **0.1551**±.007 | 30.8 |
| SOURCE | — | 105.7 | 0.406 | 0.281 | — | — | — |

**A/B 在同一批状态上零重叠、p=0.0003**(A: .1172/.1197/.1214/.1341 全部低于 B: .1456–.1615)⇒
未触顶采集器微调出的策略确实离 π_src 更近,方向与假说一致。**但三条把它的解释力卡死**:
① 是"大位移上的小调制"(两组都已移动 π_src 动作散布的 30%/61%/81%,A−B 之差只占 4.6–7.9%);
② **距离几乎是采集器的确定函数,adapt 不是** —— 组间差/组内 sd:`D_ORIG` **4.50**(零重叠) vs
`adapt` **1.03**(重叠 0.084)⇒ adapt 的方差绝大部分来自距离之外(§38 的 σ≈0.09);
③ **9 个 sat base 内部距离完全不预测 adapt**:`D_ORIG` ρ=−0.283 (p=0.46)、`D_SRC` ρ=**−0.033**
(p=0.93);B 组内部还直接倒挂(B_s401 的 D_SRC 更远却拿 oracle 1.000 / adapt 0.857)。
⇒ **"采集器 → 策略距离"链很紧;"策略距离 → adapt"链很松。** 这也是为什么"梯度版"的距离假说
(离得越近越好)不成立,而 **λ 这个主动干预**却有 +0.164 —— 两者渠道不同。

### `nobc` 的定位(修正一个旧印象)
`dynens_adapt_long_h5_nobc`(policy 头 = π_src,距离精确 0)**adapt tail-25 = 0.895,在所有
base 里排第 2**;真正差的是它的 **policy_only = 0.12–0.16**。所以"nobc 效果不好"来自 proposal,
不来自 adapt 水平。但它**慢得多**(首次达阈值所需 transitions):
| 阈值 | nobc | long_h5(20 demo) | 倍数 |
|---|---:|---:|---:|
| 0.8 | 35k | 5k | 7× |
| 0.85 | 92k | 5k | **18×** |
| 0.9 | 148k | 13k | 11× |
到 341k 仍在爬(0.888→0.902)。⇒ **倒 U 的两边各有机制**:左边(BC 太少)proposal 不会 target 动作
⇒ 采不到该采的接触相数据 ⇒ dynamics 修得慢;右边(BC 太远)见上。

### 可操作准则
**stage-2 要在"靠近 π_src"和"拟合 target demo"之间取中**,不是把任一个推到极端:
- **旋钮 = `bc_anchor_coef`,λ≈1**(区间 [0.1,3];λ=0 与 λ=10 都是端点失效,各自违反一个条件)。
- **监控两个轴,不要只看一个**:`D_SRC`(离 π_src)+ `bc/anchor_loss` 管条件 (1),
  `bc/policy_loss` 管条件 (2)。**`policy_only` 是结果不是轴**,只能事后验证(它应在最优 λ 处取峰)。
- **只对弱 base 值钱**(⚠️ **§44 已推翻这个推广**:真正的调节变量是 **dynamics 误差**,不是
  base 的分数 —— plain 单头 base 的 λ=0 分数与 dynens 几乎相同(0.767 vs 0.786),但 anchor
  在它上面只值 +0.055 而在 dynens 上值 +0.172,交互 t=4.2):该 base 的 MPPI 离它自己天花板越远,收益越大(s201 +0.164 / long_h5 −0.009,
  交互 t=5.2)。判据是"离天花板多远",不是"base 有多强"。
- **不要用减 `bc_steps` 代替它**:那是各向同性地少动(顺带欠拟合 demo ⇒ 违反条件 (2)),
  §38 实测 −0.060 (z=−5.26);锚是**有方向**地少动,两者不是同一个操作。

### Where(全部默认关、逐位兼容)
- `rl/simdist_agent.py::update_bc_policy(..., anchor_states, anchor_actions, anchor_coef)` ——
  加 `λ·MSE(π_θ(s_src), π_src(s_src))`,记 `bc/anchor_loss`。`anchor_coef=0` ⇒ `total is loss`、
  `torch.randint` 在 `if` 内不消耗 RNG ⇒ 现有 run 一字不改。
- `train_simdist_adapt.py`:`bc_anchor_coef/path/episodes/states/batch` 五个字段 +
  `_build_source_anchor()`(在 stage-2 第一步**之前**调用,那时 `model.policy` 还就是 π_src,
  把 `π_src(s_src)` 快照成常量 ⇒ 不需要保留第二份网络、目标不随 π_θ 漂移)。
- runs:`threading_lamadapt_{0.0,0.1,1.0,3.0,10.0}_s{1,2,3}`、`threading_lamgen_longh5_1.0_s{1,2}`、
  `threading_simdist_lam{0.01,0.1,1.0}_s201_seed1`(只跑 stage-2 + round-0 + oracle)。
- λ 标定脚本 + 真策略距离脚本在 scratchpad(未入库);标定顺带验证了重实现:λ=0 给
  `D_SRC=0.2354`/`D_ORIG=0.1162`,对上仓库实测的 0.2354/0.1172。

### ⚠️ 方法论(本轮踩的坑)
1. **`tail-25` 一直取在下降段**:s201 的曲线是 20–40k 0.809 / **40–60k 0.820(峰)** / 60–80k
   0.792 / 80–100k 0.772。我此前反复引用的"s201=0.780"是 75–100k 均值。不影响 A/B 相对结论
   (同口径),但绝对值偏低。**主读数改用 20–40k。**
2. **`100 % 8 ≠ 0`**:`num_eval_proc` 必须整除 `num_eval_episode`(§33 已记,又踩一次)。
3. **`nohup` 挡不住 Bash 工具 timeout(143)**:子进程同进程组会被一起带走。长任务必须
   `setsid ... < /dev/null & disown`,且**不要**在同一条命令里跟 sleep。
4. **round-0 不预测终点(§31)又验证一次**:long_h5 加锚 round-0 涨 0.190→0.38,终值 −0.009;
   且 round-0 带 demo-集特异成分(λ=0 的 s201 是 0.640,nobc 只有 0.400,与在 long_h5 上的方向相反)
   ⇒ **单点跨 base 比 round-0 不可靠**。
5. **oracle 在天花板附近没有分辨力**(0.98–1.00,Fisher p=0.50)⇒ 它只能定上界/证伪,不能用来
   排序下游效果;最终判据只能是 learned dynamics 下跑完 adapt。

### TODO
- [ ] **换第二个弱 demo 集复现 +0.164**(如 B 组最弱的 `sat_400000_s201` = 0.659),3 seed。
      **双重目的**:① 准则能否推广;② 若锚主要是正则,它应压小 §38 那个跨 demo 集的 σ=0.092
      ——这是正则解释的判决实验。**最该先做**。
- [ ] **判轴 (1) 的标签**(两个便宜实验,都不需要新的 adapt):
      ① §33 的 `std_i(γ^H V)` 集内散布 + **top-64 elite 重合率**,在 λ=0/λ=1 各自的 on-policy 候选集上
      测(不跑环境)⇒ 判 (1a) 的交互形式有没有独立贡献;
      ② 同样扫 policy 头的 `weight_decay`/dropout ⇒ 判"通用正则"还是"π_src 特有的先验信息"。
- [ ] 锚池目前固定用 `source_diverse_v2`(= stage-1 蒸馏所用的同一份数据),没有把"锚在哪个分布上"
      和"锚向哪个策略"分开。可换 target rollout 状态做对照。
- [ ] long_h5 的 λ=0 补到 3 seed(把上界从 +0.054 收到 ~+0.03);受天花板限制收益有限。
- [ ] λ ∈ (1,3) 加密找峰;λ 与 `bc_steps` 的二维交互(两者都控"走多远",但方向性不同)。
- [ ] 全部单任务(Threading_D0)、单 stage-1 seed。方法级结论需第二个任务 + ≥2 训练 seed。

### 40d. ⚠️ `base_policy_std=0` 会把 base 候选变成同一条序列的复制品（2026-08-13）

**现象**：Coffee stage-1 ckpt 在 target C50 上，只改 planner，成绩在 **22%–67%** 之间浮动。
单变量扫描（每点 100 局，其余固定为 SimDist "original" 配方 350 候选/3 轮/`init_std` 1.0/
`temp` 0.4/`mixture` 0.29/`base_policy_std` 0.1）：

| `base_policy_std` | 0.0 | **0.1** | 0.2 |
|---|---:|---:|---:|
| target | **22%** | **56%** | 46% |

| `init_std` | 0.5 | 0.7 | 1.0 | 1.5 |
|---|---:|---:|---:|---:|
| target | 56% | 48% | 56% | 52% |

| 预算 iters×samples | 3×350 | **6×350** | 3×700 | 10×350 | 3×1400 |
|---|---:|---:|---:|---:|---:|
| target | 56% | **67%** | 64% | 62% | 65% |

**机制**：`MppiPlanner` 的 base seeds 是 policy rollout 的**复制品**；`base_policy_std=0` 时这
`num_base` 条候选**逐位相同**，等价于只有 1 条 base + `num_rand` 条随机。所以它不是"少一点噪声"，
是**直接扔掉 `num_base` 份候选预算**。

**⇒ 这解释了一个我一度归因给 `init_std` 的错误结论。** 我自己配的 stage-1 planner
（`num_samples 512 / iterations 6 / init_std 0.5 / temp 0.5 / mixture 0.8 / base_policy_std 0.0`，
抄自 Square 的 `square_d0_state_simdist_diversev2.yaml`）在 target 只有 **33%**，我当时归因为
"`init_std 0.5` 太窄"。真相是 `mixture 0.8 + base_std 0` ⇒ **512 个候选里 410 个是同一条序列的
重复**，只剩 102 条真随机。验证：`base_std 0 + init_std 0.7` 单变量测得 **33.0%**，与那套配方
精确吻合；而 `init_std` 在 0.5–1.5 之间实测**没有显著差别**（46–56%，全在 ±1 SE 内）。

**⇒ 检查任何 SimDist/MPPI 配置时，先看 `mixture_coef × base_policy_std`**：
`base_policy_std==0` 且 `mixture_coef` 大 = 大部分预算在跑重复项。
⚠️ 影响范围（2026-08-14 更正，原文写得过宽）：cfg 里的 planner block **在默认流程里根本不
生效** —— stage-1 蒸馏是 `use_planner: 0`，stage-2/3 adapt 的 `original_mppi` 默认 **1** 会用
`apply_original_recipe()` 覆盖除 `horizon` 外的所有字段。**只有 `evaluate/eval.py --mode simdist`
读 cfg**（`train_simdist.load_model` → `pyrallis.load(MainConfig, cfg.yaml)`）。所以
`square_d0_state_simdist_diversev2.yaml` 的 `mixture 0.8 + base_std 0.0` 只污染 **standalone
eval**，不污染任何 adapt run —— 但那恰恰是本节那个 "33%" 的来源（我用 eval.py 评 stage-1 ckpt，
它读了这套 cfg），所以 Square 的 standalone eval 数字仍值得复核。

**本轮我犯的两个推断错误（都是多变量混淆）**：
① 把 `--original_mppi 1`（adapt 脚本默认）判为"跨任务照搬、可能削弱 planner"，据此停掉 adapt run
—— 实测它 target **56% vs 我的 A 配方 33%**、且便宜 3 倍，是**更好**的那个；
② 把 A 的 33% 归因给 `init_std`，实际是 `base_policy_std`。
**教训：一次只改一个参数；对照配方之间差 ≥2 个参数时，任何归因都是猜。**

**跨设置不可移植性再确认**：oracle dynamics 下最优 `init_std` 是 0.40（§40c），learned dynamics 下
0.5–1.5 无差别而 `base_policy_std` 才是主因 ⇒ **planner 超参不仅跨任务不可移植，跨"dynamics 是
oracle 还是 learned"也不可移植**，每换一个设置都要重扫。

---

## 43. ★ 约束的**形式**不重要，只有它造成的 demo 拟合退化重要 —— anchor 形式轴封闭(2026-08-20/21)

### ★ 结论
起点是一个具体问题:「把 stage-2 的 anchor 换成 KL divergence 会怎样」。答案分两层:

**(a) `bc_anchor_coef` 那行 MSE 已经就是 KL。** policy 头是确定性 tanh MLP,唯一能定义散度的
方式是提升成 `N(pi(s), sigma^2 I)`。等协方差高斯之间:

| 散度 | 闭式(令 `d = ||pi_theta(s) - pi_src(s)||_2`) |
|---|---|
| **KL**(两个方向相同) | `d^2 / (2 sigma^2)` |
| **Wasserstein-2** | `W2^2 = d^2` —— 与 KL **精确**差一个常数 |
| Hellinger^2 | `1 - exp(-d^2 / 8 sigma^2)` |
| Total Variation | `2 Phi(d / 2 sigma) - 1` |
| chi^2 / Renyi-alpha | `exp(d^2 / sigma^2) - 1` |

**整个 f-散度家族 + W2 全部只是同一个标量 `d` 的单调函数**,常数被 `bc_anchor_coef` 吸收。
⇒ 「改用 KL / W2 / JS / Hellinger / chi^2」在这个架构下是**空操作**,不必测。
二阶看:固定 sigma 的高斯策略 Fisher `F = G/sigma^2`(G = source 分布上的 Gauss-Newton),
所以 **F1 = TRPO 式 KL 信任域,且用的是正确的 Fisher 度量**。

**(b) 散度家族给不出新方向,真正的自由度只有四个**,`E_{s~D}[ g(||M (pi_theta(s)-pi_src(s))||) ]`
里的四个可动位置:① `g` 的形状 ② `D` 在哪些状态上量 ③ `M` 动作空间度量 ④ 比单步动作还是轨迹。
本节把 ① 和 ③ 测穿了。

**★ 主结果:形式只通过一个渠道起作用 —— 它让 target demo 拟合退化了多少。**
```
score ≈ 1.065 − 11.1 × bc_loss        R² = 0.93 (逐 run, n=12),  残差 sd = 0.010
```
⚠️ **§44 的限定**:这条定律成立的**前提是本节五个形式臂全被标定到了同一个 D_SRC**,即 D_SRC
这条渠道被固定住了。一旦造出"`bc_loss` 匹配而 D_SRC 差 6.3×"的点(§44 的 data-action 臂),
分数掉 **−0.081 (t=2.5)** ⇒ **D_SRC 是独立的第二条渠道**,真正的形式是
`score ≈ f(D_SRC) + g(bc_loss)`。本节只测了 g,§44 补上了 f。
残差 sd **小于 seed 噪声本身**(各臂 sd 0.014–0.019)。F1 未参与拟合却落在线上(预测 0.951 /
实测 0.957)。按臂均值:`bc_loss` R²=**0.97**,尾部形状 R²=0.47。

### 五个臂(全部默认 0,逐位兼容)
| 臂 | 加的 loss | 形式身份 | 旋钮 |
|---|---|---|---|
| **F1** | `λ·MSE(pi_theta(s_src), pi_src(s_src))` | KL **罚**,Fisher 度量(现役) | `bc_anchor_coef` |
| **F2** | Adam `weight_decay` on policy 头 | **不是散度**,吸引子是 theta=0 | `bc_weight_decay` |
| **F3** | `c·||theta − theta_src||^2` | KL 但 Fisher **换成单位阵** | `bc_l2init_coef` |
| **F4** | `λ·mean relu(d−eps)^2` | KL 的**硬信任域**(TRPO);`eps = sigma·sqrt(2 delta)` | `bc_anchor_eps` |
| **F5** | `λ·mean(1−exp(−d^2/tau))` | **Hellinger^2** = 饱和,g 轴上 hinge 的镜像 | `bc_anchor_sat` |

### Phase 0 — 53 个标定格(只跑 stage-2 + round-0,~8 min/格)
**为什么必须先标定**:直接扫形式会把「形式」和「强度」混在一起。全部标定到 **同一个 D_SRC**
(= `mean|pi_theta(s_src) − pi_src(s_src)|`,逐元素 MAE;§42 那张表用的就是这个,原文没写定义),
再比 `bc_loss`。

**D_SRC 定义反推 + 探针验证**(用已有 lamadapt ckpt,四个 λ 全部复现在 1.5% 内):
| λ | 0 | 0.1 | 1.0 | 3.0 |
|---|---:|---:|---:|---:|
| §42 记录 | 0.2354 | 0.0517 | 0.0248 | 0.0179 |
| 本轮实测(MAE) | 0.2390 | 0.0522 | **0.0246** | 0.0182 |

**F1 完整前沿(全部实测,seed 1,弱 base `sat_160000_s201`)**:
| λ | 0 | 0.3 | **1** | 2 | 3 | 6 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D_SRC | 0.2381 | 0.0354 | **0.0246** | 0.0203 | 0.0182 | 0.0147 | **0.0127** |
| bc_loss | .00795 | .00891 | **.00993** | .01093 | .0120 | .01373 | **.01575** |

⚠️ **补测了 §42 表里 λ=10 那行标注「未测」的 D_SRC = 0.0127**;该行 bc_loss 原记 0.0160,
本轮实测 0.01575(差 1.6%,交叉验证通过)。

**靶点 D_SRC=0.0246 处各形式的前沿代价**:
| 形式 | bc_loss 代价 vs F1 | 备注 |
|---|---:|---|
| **F2** weight decay | **结构性不可达** | D_SRC 下界 **0.2045**(= 靶点 8.3×),加大 wd 后**掉头上升**爬向 **0.3392** —— 因为 theta→0 ⇒ a→0,而零动作到 pi_src 的距离恰是 `mean|pi_src(s)|=0.3392`(即 `[anchor]` 打印的那个数)。⇒ **通用容量控制无法产生靠近 pi_src 的策略**,π_src 这个目标是必需的 |
| **F3** 参数空间 L2 | **+3.4×** | 实测(li=0.1 → D_SRC 0.0344 / bc 0.02980,对 F1 λ=0.3 的 0.0354 / 0.00891) |
| **F4** hinge | +10% | |
| **F5** tau=0.007 | +16% | |
| **F5** tau=0.03 | **打平**(−2%) | |

### F3 为什么这么贵:Delta-theta 几乎全落在 G 的近零空间
`||theta − theta_src||^2` 实测(F1 各 λ 的 ckpt 对 stage-1):
| λ | 0 | 0.1 | 1 | 3 |
|---|---:|---:|---:|---:|
| D_SRC | 0.2390 | 0.0522 | 0.0246 | 0.0182 |
| **`||Δθ||²`** | 238.6 | **304.7** | 250.6 | 186.7 |

**D_SRC 降 13× 而参数距离几乎不动(甚至非单调)** ⇒ anchor 的全部工作发生在一个很小的子空间里,
Delta-theta 的绝大部分落在 G 的近零方向(改了也不影响 source 状态上的动作)。F1 认得出这个子空间
(Fisher 度量),F3 认不出(单位阵),只能各向同性地压 ⇒ 同样的邻近度要付 3.4× 的拟合代价。
⇒ **「在哪个空间量距离」是实质设计选择,不是写法差异。**

### F5 的两个结构性事实(逃逸半径)
`1−exp(−d²/tau)` 的**小偏差刚度是 `λ/tau`**,不是 λ。逃逸半径 `sqrt(tau)`:
| tau | sqrt(tau) | vs F1 λ=1 的典型偏差 L2mean=0.0839 | 后果 |
|---:|---:|---|---|
| 0.002 | 0.0447 | 只有一半 | **完全无梯度**:状态越过 0.045 后 `exp(−312)≈0`,再也抓不回来。λ 跨 10× 毫无反应 |
| 0.007 | 0.0837 | 几乎相等 | 一半状态受约束 |
| 0.03 | 0.173 | 两倍 | 大部分在二次区 |

且 **F1 = F5 的 `tau→∞` 极限**(`1−exp(−x)→x`)。约束越紧 `d²/tau` 越小 ⇒ **所有 F5 变体在紧端
收敛到 F1 前沿**(实测:D_SRC≈0.0126 处 F1 λ=10 的 0.01575 vs F5 的 0.01512,差 −4%),
**只在松端(含最优工作点)才分开**。

### Phase 1 — 12 个 adapt run(4 臂 × 3 seed,40k,读 20–40k 窗口)
与 `threading_lamadapt_*` 逐字同 harness ⇒ F1 λ=1 的 3-seed 参照 **0.957** 直接可比(铁律 #15)。

| 臂 | 配置 | D_SRC | bc_loss | 尾 `p99/mean` | 三个 seed | **均值** | sd | Δ vs F1 | t |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| **F1** | `coef 1.0` | 0.0246 | .0103 | 3.97 | §42 参照 | **0.957** | .016 | — | — |
| **F5** | `sat .03 coef .013` | 0.0225 | .01024 | **6.88** | .947/.960/.932 | **0.946** | .014 | −0.011 | 0.9 |
| **F4a** | `eps .05 coef .5` | 0.0234 | .01122 | 3.23 | .942/.913/.925 | **0.927** | .015 | −0.030 | 2.4 |
| **F4b** | `eps .20 coef 100` | 0.0286 | .02472 | 2.06 | .810/.847/.831 | **0.829** | .019 | −0.128 | 9.0 |
| **F3** | `l2init .25` | 0.0245 | .03404 | 1.94 | .694/.687/.659 | **0.680** | .019 | −0.277 | 20 |

F5/F4a 在预注册阈值 0.04 内 ⇒ **打平**;F4b/F3 明显更差。

### ★★ 尾部形状:相关 0.90,因果 0 —— 只有干预能分开
F3/F4 三个臂上 `bc_loss` 与尾部厚度**完全共线**(Phase 0 测出这是结构性的:削尾巴必然花 bc_loss),
任何观察性分析都会把因果归给尾部。**F5 是专门为破这个共线造的** —— 它是唯一把尾巴推向 F1
**另一侧**的形式(饱和 ⇒ 放任大偏差):

| | D_SRC | bc_loss | p50 | p99 | **p99/mean** | 接触相% |
|---|---:|---:|---:|---:|---:|---:|
| F1 λ=1 | 0.0246 | .00993 | .0660 | .3332 | **3.97** | 1.0% |
| **F5** | 0.0225 | .01024 | .0494 | **.5374** | **6.88** | 0.2% |

**同 D_SRC、同 bc_loss、尾巴重 1.73× ⇒ 分数动 −0.011。**
尾部与分数的 Spearman **至今仍是 0.90**,但一个干预就把效应量打到零。
⇒ **教科书级的「相关不等于因果」。本项目此前多次栽在代理指标上(铁律 #10),这次的教训是:
共线的两个候选变量,必须造一个把它们分开的臂,否则拟合得再好也归错因。**

### 一并判死的假说(都有直接证据)
| 假说 | 怎么死的 |
|---|---|
| 通用正则也能给这份收益 | F2 结构性够不到工作点(下界 8.3×) |
| 度量空间无所谓 | F3 付 3.4× 拟合代价 ⇒ −0.277 |
| 罚函数 vs 硬约束有区别 | F4a 打平(−0.030,阈值内) |
| 尾部形状重要 | F5 单变量干预 ⇒ −0.011 |
| **偏差落在接触相是病因**(本轮提出) | **两次证否**:F3 接触相污染 10.2% 比 F4b 的 18.5% **轻**却更差;F5 污染 0.2% 但并不便宜 |

### ⚠️ 本轮踩的坑(都值得记)
1. **anchor 超参跨形式不可移植**(与铁律 #9 同型)。F5 首发按 F1 的习惯填 `λ=1, tau=0.005`,
   实际强了 **1400×**(饱和形式刚度是 `λ/tau` 不是 λ),D_SRC 直接冲到 0.0044。
   **这就是 8 分钟标定格存在的理由** —— 否则会把「形式不好」和「系数填错」混在一起,且要 2.3h 才发现。
2. **三次拿外推值下反预测的结论,全部撤回**。§42 的 λ=10 行 D_SRC 标「未测」,我拿它外推出
   「F5 更便宜」两次、「τ=0.03 更贵 39%」一次,补实测后全部反转。**铁律 #6 的又一次现场版。**
3. **进程管理三连自伤**:`pgrep -f <pattern>` 会匹配 ① mp eval fork 出的 **10 个 worker**
   (继承父进程完整命令行,1 个 run 数成 11 个)② **我自己诊断命令的 wrapper shell**
   (每查一次进度闸门计数 +1,观测行为改变被观测系统)③ 接 `kill` 时**把自己杀掉**(退出码 144)。
   ⇒ **不要用命令行文本识别进程**。最终架构 = 一臂一条串行链、各自 `wait` 自己的 job,
   并发天然等于链数,一个进程都不用数。需要计数时用 `/proc` 且要求 `comm == python`。
   另:`setsid` 起的脚本 `comm` 是**脚本名**不是 `bash`,`pgrep -x bash` 找不到它。
4. **铁律 #4 现场例子**:F4a_s1 部分窗口(n=2)读 0.920,完整窗口(n=11)是 **0.942**,偏 0.022。
5. `nohup`/重定向的 stdout 是**块缓冲**,`echo` 进度会滞留;真实输出看 `<save_dir>/train.log`
   (workspace 自己的 logger)。给 python 加 `-u`。

### ★ 可操作准则
- **形式轴已封闭。** 现役配置(KL 罚 + Fisher 度量 + source 状态分布 + λ≈1)在整个形式邻域里最优。
  任何新形式只要在同 D_SRC 下不增加 `bc_loss` 就一样好,而 Phase 0 显示 F1 已在前沿上。
- **别再试**:换散度名字(W2/JS/Hellinger/chi^2/TV/Renyi,恒等或落在已测的 g 轴上)、forward vs
  reverse KL(等协方差时对称)、动作维 Mahalanobis(逐维 std 只有 2.5× 跨度:`[.51 .56 .58 .32 .34 .45 .82]`,二阶)。
- ~~**唯一还没测的方向 = 轴 ②(`D`,在哪些状态上锚)**~~ ⇒ **已测,封闭,见 §44b**
  (11 个标定格 0/11 外移前沿 + 3 seed 闭环 ns)。当时的猜想是"按离 demo 支撑集的距离显式加权
  能外移前沿",实测**只会内移**:pool 本来就 99.6% 在 demo 支撑集之外(所以 §42 的原文理由早已
  自动满足),而 NTK 干扰几乎均匀(top10% 只携带 13%,LayerNorm 所致)⇒ 没有低成本子集。
  锚向**什么动作**(轴②的近邻)也已在 **§44** 封闭:目标必须是 π_src。
- **推论:anchor 的活已经干完了。** 下一个瓶颈不在这里,回 `docs/findings.md` §27/§31
  (dynamics 精度 / 任务容错窗口)那条线。

### Where(全部默认 0 ⇒ 逐位兼容,已验证)
- `rl/simdist_agent.py::update_bc_policy(..., anchor_eps, anchor_sat, l2init_coef, weight_decay)`
  —— `anchor_sat > 0` 走饱和;否则 `anchor_eps > 0` 走 hinge(`sqrt(d2+1e-8)`,**别用 `torch.norm`**,
  它的 backward 在 0 处是 NaN 而 anchor 残差后期确实收敛到 0);否则原来那行 `F.mse_loss` 一字不动。
  `l2init_coef > 0` 时在**第一次 `.step()` 之前**快照 `theta_src`(那时 `model.policy` 还就是 pi_src)。
  `weight_decay` 传给 Adam 构造(优化器只建一次)。新指标 `bc/l2init_loss`。
- `train_simdist_adapt.py`:`bc_anchor_eps` / `bc_anchor_sat` / `bc_l2init_coef` /
  `bc_weight_decay` / `bc_measure_dsrc` 五个字段。`bc_measure_dsrc=1` 时即使 anchor 关掉也建
  source pool,stage-2 末尾打印 `[D_SRC] <MAE> (L2mean ...) bc_loss ...`,并记 `bc/D_SRC`。
  ⚠️ 循环内 `torch.randint` 仍只在 `bc_anchor_coef > 0` 时执行 ⇒ **RNG 消耗不变,老 run 一字不改**
  (已验证:λ=1 重跑的 D_SRC/L2mean 与老 ckpt 复现到小数点后四位)。
- 分析脚本(在本轮 scratchpad,未入库):`dsrc.py`(逐 ckpt 的 D_SRC/MAE/RMSE 对照)、
  `tail.py`(p50/p90/p99/max + `p99/mean`)、`where.py`(top-1% 偏差状态在轨迹中的 t/T 分布)。

### 42b. Coffee 上的 λ 复现 —— λ≈1 是安全默认值，但**响应形状 target-dependent**（2026-08-22）

`bc_anchor_coef` 在 Coffee 的三个 target 上都是**最大的单一杠杆**（把 stage-2 后的 round-0
抬 **+0.35 ~ +0.44**），但最优 λ 与响应形状因 target 而异。全部 100 局 in-training eval：

| λ | C75kp25 (2 seed) | G5x (2 seed) | C50 (1 seed) |
|---:|---:|---:|---:|
| 0 | **0.19** | 0.653 | 0.58 |
| 0.3 | 0.68 | 0.859 | — |
| **1.0** | **0.71** | **0.907** | 0.62 |
| 3.0 | **0.41** | 0.868 | — |
| ∞ (noBC) | 0.45 | 0.786 | 0.78 |

**可信 / 不可信的部分**：
- ✅ **λ>0 必须**：λ=0 时 stage-2 摧毁 proposal（C75kp25 round-0 **0.020**，两 seed 逐位相同；
  `policy_only` 0.10–0.13、BC loss 0.013–0.020 ⇒ **demo 拟合得很好，作为 proposal 却无用**）。
- ✅ **λ≈1 是安全默认**：C75kp25 与 G5x 上都是峰或并列峰；G5x 上两个 seed **一致**胜出
  （0.896>0.820、0.918>0.898）。
- ❌ **"λ=3 一定不行" 不成立**：C75kp25 上 0.41（崩），G5x 上 0.868（不崩）。
- ❌ **λ=0.3 vs 1.0 在 C75kp25 上不可区分，且排序会随 seed 翻转**（seed1 λ=1 高 17 点、
  seed2 λ=0.3 高 11 点）⇒ 我一度写下"λ=1.0 是明确的峰"，那是单 seed 噪声，**已作废**。
  seed 间 σ ≈ 0.09–0.13。

**⇒ 操作准则**：换 target 后 λ 用 **1.0** 起步，并**必测 λ=0** 作为"anchor 值多少"的标尺
（那个差在 Coffee 三个 target 上是 0.35–0.44，不是可忽略量）。想找峰要 ≥2 seed，且别指望
分辨 0.3 vs 1.0。

**⚠️ 更重要的比例关系**（tasks §40e ①）：G5x 上 anchor 把 round-0 抬 **+0.40**，而在此之上
100k 的 stage-3 adapt 只再加 **+0.03**。**有限算力应优先投 stage-2，不是 stage-3。**

---

## 44. ★★ anchor 的收益由 **dynamics 精度**定；「锚向谁」是独立于 `bc_loss` 的第二条渠道(2026-09-04)

### ★ 两条结论
1. **同一个 `bc_anchor_coef=1`，在 plain(单头) base 上只值 +0.055，在 dynens(5 头) base 上值
   +0.172 —— 交互 +0.116 (t=4.2)。** §42 记的"弱 base 增益大、强 base 测不到"是**错的推广**:
   两个 base 的 λ=0 分数几乎相同(0.767 vs 0.786),真正的调节变量是 **dynamics 误差**,不是
   base 的分数。
2. **anchor 的目标必须是 π_src 本身。** 换成 source 数据集里**记录的 action**(同一批 states,
   只换回归目标)在 `bc_loss` 匹配到 4.2% 内时仍掉 **−0.081 (t=2.5)** ⇒ §43 那条
   `score ≈ 1.065 − 11.1·bc_loss` 不是全貌。**但注意:本节初稿据此写的"D_SRC 是第二条渠道"
   已被轴②那批 run 修正 —— 真正的充分统计量是 `policy_only`(proposal 质量),`D_SRC` 与
   `bc_loss` 都只是代理,详见下面「唯一的充分统计量」小节。**

### harness
plain stage-1 = `threading_simdist_diversev2_h5_seed1`(单头),dynens = `..._dynens_...`(5 头)。
adapt 参数与 `threading_lamadapt_*` **逐字相同**(target demo `threading_d0_sat_160000_s201`
n=20、40k transitions、`eval_every_rounds 2`、100 局 in-training eval、σ=0.1、`original_mppi 1`),
读 20–40k 窗口均值(11 点/run)。**两个 stage-1 的 cfg env 段已逐字 diff 过一致**
(`eval_domain_shift`、`strict_success.enabled=True`)—— 铁律 #17 的必做核对。
⚠️ caveat: 两个 stage-1 训于不同日期(06-17 vs 07-19),差异不只是 head 数;这延续的是项目里
既有的 plain/dynens 对照(CLAUDE.md 的 0.879 vs 0.962),没有新引入混淆。
runs: `threading_plain_{lam0.0,lam1.0}_seed1..5` + `threading_plain_datalam{0.3,1.0}_seed1..3`
(16 个 40k run, wandb project `mimicgen`, group 按臂)。

### 表 1 — ablation 1: 2×2(anchor × dynamics ensemble)
| stage-1 | λ=0 | λ=1 (π_src) | Δ | t | 剩余差距缩小 | Δlogit |
|---|---:|---:|---:|---:|---:|---:|
| **plain**(1 头) | 0.767 (sd .051, n=5) | 0.822 (sd .027, n=5) | **+0.055** | 2.15 | 23.7% | +0.34 |
| **dynens**(5 头) | 0.786 (sd .008, n=2) | 0.957 (sd .017, n=3) | **+0.172** | 15.6 | 80.0% | +1.81 |
| Δ | +0.019 | +0.135 | **交互 +0.116** | **4.16** | | |

铁律 #18 三个尺度一致 ⇒ **这不是天花板压缩**: plain 还剩 0.18 余量、缩小比例 24% vs 80%、
log-odds 差 5.3×。+0.055 恰好踩在铁律 #1 的读数线上,λ=0 从 3 seed 补到 5 就是为了把
SE 从 0.033 压到 0.023。

**机制读法**(与 §41b/§27 同源): anchor 的渠道是**保住那个 frozen 的蒸馏 V^{π_src}**,而
"value 与延续策略自洽"这个约束只有在 **dynamics 误差不再是主导瓶颈**时才 binding。plain 的
天花板由模型误差定(§27),修 value 一致性换不来分。与 §33 的限定语("dynamics 越准,换 value
的代价越小")是同一枚硬币的两面。

**附带结论(重要)**: **λ=0 时 dynamics ensemble 几乎没有优势(+0.019)。** §31 那个
0.88 → 0.962 的增益**与 anchor 强耦合、不可加** —— 两者是互补(交互项本身就是 +0.116)。

### 表 2 — ablation 2: anchor 锚向"数据集 action"
`bc_anchor_data_actions=1`: 同一批 source states(该数据集每条 episode 的 state/action 等长,
索引对齐 ⇒ **states pool 与 π_src 臂逐点相同**),只把回归目标从 `π_src(s)` 换成数据集记录的
`a_data`。`D_SRC` 始终对 π_src 量 ⇒ 两臂落在同一根轴上。

| 臂 | D_SRC | bc_loss | score (n=3~5) | vs π_src λ=1 | t |
|---|---:|---:|---:|---:|---:|
| λ=0 | 0.196 | .00835 | 0.767 | −0.055 | 2.15 |
| **π_src λ=1** | **0.025** | .01037 | **0.822** | — | — |
| data λ=0.3 | 0.158 | .01081 | 0.741 | **−0.081** | 2.47 |
| data λ=1 | 0.156 | .01336 | 0.649 (sd .122) | −0.173 | 2.42 |

**Phase 0 标定(53 格纪律的复用,~6 min/格)**: data-action 的 (D_SRC, bc_loss) 前沿
| λ | 0.3 | 1 | 3 | 10 | 30 |
|---|---:|---:|---:|---:|---:|
| D_SRC | 0.1589 | 0.1560 | 0.1546 | 0.1524 | **0.1514** |
| bc_loss | .01059 | .01337 | .01693 | .02226 | .02771 |

**λ 跨 100× 而 D_SRC 只从 0.159 挪到 0.151(几乎平坦),bc_loss 却涨 2.6×** ⇒ **结构性不可达**,
与 §43 的 F2(weight decay) 同型(那次下界是靶点 8.3×,这次 6.3×)。机制: MSE 收敛到
`E[a_data|s]`,噪声被平均掉,剩下的 0.15 是 replay 的**行为均值**与 π_src 之间**不可约**的偏差
—— `bc_anchor_path`(source diverse_v2)是 IBRL replay: `expert_flags=1` 只占 20%、
episode 成功率 **61.5%**、`|a|` mean 0.409 vs π_src 的 0.341、`mean|a_data − π_src(s)| = 0.283`。

### ★★ 唯一的充分统计量是 **proposal 质量**(`policy_only`) —— `D_SRC` 与 `bc_loss` 都只是代理
⚠️ **本小节推翻了本节初稿(以及 §43)的渠道说法。** 初稿写的是
`score ≈ f(D_SRC) + g(bc_loss)` 两条渠道,依据是两个"匹配一个、动另一个"的配对。
轴②那批 run(下面 44b)加进来后,`bc_loss` 那条渠道**方向翻转**:

| 配对(两边 D_SRC 已匹配) | Δbc_loss | Δpolicy_only | **Δscore** |
|---|---:|---:|---:|
| data λ=0.3 → data λ=1 (D_SRC .158/.156) | **+23.6%** | **−0.127** | **−0.092** |
| uniform λ=1 → 轴② st_f50 λ=10 (D_SRC .0251/.0242) | **+23.1%** | **+0.022** | **+0.021** |

同样的「D_SRC 匹配 + bc_loss 高 23%」给出 −0.092 和 +0.021 ⇒ **`bc_loss` 不是充分统计量。**
把 5 个**机制互不相同**的臂(uniform λ=0/λ=1、data λ=0.3/λ=1、轴② st_f50 λ=10)放在一起
逐臂回归(臂均值,n=5):

| 预测变量 | 斜率 | R² | 残差 sd |
|---|---:|---:|---:|
| **`policy_only`** | **+0.526** | **0.982** | **0.0104** |
| `D_SRC` | −0.673 | 0.513 | 0.0534 |
| `bc_loss` | −10.66 | 0.078 | 0.0734 |

`policy_only` 的残差(全部 ≤0.014)**小于任何一臂自己的 seed sd(0.027–0.122)** —— 这正是
§43 用来判"充分统计量"的标准。`bc_loss` 几乎无预测力(R²=0.078)。
⇒ **anchor 的一切效应都只通过一个量传导:stage-2 结束时 proposal(policy 头)有多好。**
`D_SRC` 是它的**不完美代理**(R²=0.51 —— 因为"离 π_src 多远"只在锚向 π_src 时才等价于
"policy 多好":data 臂锚向一个 61.5% 成功率的行为均值,同样的 D_SRC 对应差得多的 policy);
`bc_loss` 是**共线伴随量**,§43 之所以看到 R²=0.93,是因为它的五个形式臂全被标定到同一
D_SRC、`bc_loss` 在那个切片里与 `policy_only` 共线。**又一次"共线代理"事故(铁律 #10 / §43
尾部那格),这次栽的是我自己两小时前刚写下的结论。**

⚠️ **R²=0.98 是这 5 个臂上的数字;加进 44c 的三个"状态子集"臂后退化到 R²=0.752、残差 sd
0.03355**(仍远好于 `bc_loss` 的 0.078,但已不能当判据)。**跨机制会漂 ±0.05,详见 44c。**
⚠️ **`policy_only` 是解释变量,不是选择器。** 臂均值层面 R²=0.98,但 **seed 层面它反向**
(data λ=1 与轴②臂内部 Spearman 都是 **−0.50**),与铁律 #10 和
`feedback-simdist-stage2-bc-fit` 的历史结论一致。**别拿它调参,只拿它解释臂间差异。**

### ★★ 交互项的机制分解:planner 把 proposal 转成成绩的**转换率**由 dynamics 精度定
同一根 `policy_only` 轴在两个 base 上斜率不同(dynens 的 policy_only 从 lamadapt 日志读出):

| base | policy_only λ=0 → λ=1 | score λ=0 → λ=1 | **转换率 Δscore/Δp** |
|---|---|---|---:|
| plain | 0.419 → 0.551 | 0.767 → 0.822 | **0.42** |
| dynens | 0.414 → 0.603 | 0.786 → 0.957 | **0.90** |

两个 base 的 proposal **起点几乎相同**(0.419 vs 0.414),但转换率差 **2.1×**。表 1 那个
交互 +0.116 因此可分解为:**+0.024** 来自 anchor 在 dynens 上把 proposal 抬得更多
(Δp 0.189 vs 0.132)+ **+0.092(79%)** 来自转换率更高。
⇒ **这是 findings §36「proposal 只在 binding 时值钱」的定量版本**:dynamics 越准,planner
越能把 proposal 的改进兑现成成绩。⚠️ 每个 base 的斜率只由 2 个点估出(dynens λ=0 仅 2 seed),
量级可信、精度不可信。

### 机制证据 — policy 头裸跑(`policy_only`,21 次 eval 均值)
| 臂 | policy_only | MPPI |
|---|---:|---:|
| λ=0 | 0.418 (.370/.471/.414) | 0.767 |
| **π_src λ=1** | **0.560** (.574/.523/.583) | 0.822 |
| data λ=1 | **0.225** (.139/.256/.280) | 0.649 |

π_src anchor 把 policy 头**本身**从 0.42 抬到 0.56;data anchor 把它砸到 0.23。
⇒ data-action anchor 不是"约束方向不对",它是在**把一个好策略主动往 `E[a_data|s]`
(一个 61.5% 成功率的 replay 行为均值)上回归**。

⚠️ **又一次铁律 #10 现场版**: data 臂里 `policy_only` 最高的 seed3(0.280)恰好 MPPI **最低**
(0.508),排序完全反向;三个 seed 的 D_SRC/bc_loss 一致到小数点后三位,所以 seed3 的离群
**不能**归因给 stage-2 的 policy 质量。(该臂 sd 0.122 主要来自它。)

## 44b. 轴 ②(在哪些**状态**上锚)也封闭 —— 按离 demo 支撑集的距离/NTK 干扰加权(2026-09-04)

§43 把轴②列为"唯一还没测的方向",理由是 §42 说 anchor 该「只在 demo 从没到过的地方钉住」
而代码是**均匀**抽 40k 个 source 状态。本小节把它测穿:**没有余量,前沿只会内移。**

### 前置诊断(几分钟,在跑任何 40k 之前)
| 归一化 L2 | p0 | p50 | 落在 2× demo 自身间距内的比例 |
|---|---:|---:|---:|
| source pool → demo 最近邻 | **0.323** | 1.778 | **0.00%** |
| demo 内部最近邻(尺度) | — | 0.144 | — |

**pool 本来就 99.6% 落在 demo 支撑集之外**(3× 内 0.04%,最近的一点也在 2.2× 外)⇒
§42 那句原文理由**早已自动满足**,没有"近 demo 的 anchor 点"可以砍。
⚠️ 排除了"这只是几个域偏移维撑大了距离":距离贡献在 37 维上几乎均摊(最大单维 4.4%,
obj 组 75.5% ≈ 其维度占比 75.7%),边际域偏移总 L2 仅 1.058 vs NN 距离 1.778;砍掉偏移最大的
10 维后 p50 只从 1.778 → 1.208,2× 内仍只有 0.36%。

⇒ 那么 `bc_loss` 的退化**不可能**来自状态分布重叠,只能来自**参数耦合**。正确的加权变量是
NTK 干扰 `I(s) = mean_d |<∇_θ π(s), ∇_θ π(s_d)>|`(最后一层:`<h(s),h(s_d)>·<tanh',tanh'>`):

| | p10 | p50 | p90 | mean |
|---|---:|---:|---:|---:|
| source pool | 20.9 | 31.0 | 37.5 | **29.8** |
| demo 对 demo(参照) | 25.7 | 34.9 | 40.3 | **33.7** |

**比值 0.884** —— 随机一个 source anchor 状态对 demo 拟合的干扰已达 demo 状态彼此之间的 88%;
**几乎无偏斜**(干扰最大的 10% 只携带 **13.0%** 总干扰,均匀是 10%)。
⇒ 不存在"低成本高收益"的子集。机制 = **LayerNorm**:它把所有状态的特征范数钉在同一尺度
(`|h|` src 3.637 vs demo 3.673),任意两点的 NTK 内积都同量级,参数耦合几乎与"钉在哪里"无关
—— 这正好解释 §43 F3 的"D_SRC 降 13× 而 ‖Δθ‖ 不动"。
⚠️ Spearman(状态空间距离, NTK 干扰) = **−0.583**:方向对但只解释 34% 秩方差 ⇒ 状态距离是
个糟糕的代理,所以两种度量必须各跑一组(否则分不开"轴②封闭"和"度量选错")。

### Phase 0 — 11 个标定格,**0/11 外移前沿**(预注册阈值:同 D_SRC 下 bc_loss 低 >5%)
| 臂 | D_SRC | bc_loss | 前沿内插 bc | 相对代价 |
|---|---:|---:|---:|---:|
| st_f50 λ=1 | **0.0361** | 0.00943 | 0.00895 (uniform λ=0.3 实测 **0.0362**/0.00894) | **+5.4%** |
| st_f25 λ=1 | 0.0552 | 0.00913 | 0.00852 | +7.1% |
| st_pow2 λ=1 | 0.0455 | 0.00924 | 0.00860 | +7.4% |
| ntk_f50 λ=1 | 0.0351 | 0.00970 | 0.00902 | +7.6% |
| ntk_f25 λ=1 | 0.0486 | 0.00943 | 0.00857 | +10.0% |
| st_f50 λ=3 | 0.0298 | 0.01050 | 0.00945 | +11.1% |
| ntk_f50 λ=3 | 0.0280 | 0.01110 | 0.00962 | +15.4% |
| st_f25 λ=3 | 0.0481 | 0.00991 | 0.00858 | +15.5% |
| st_f50 λ=6 | 0.0262 | 0.01151 | 0.00980 | +17.5% |
| ntk_f25 λ=3 | 0.0421 | 0.01050 | 0.00870 | +20.7% |
| st_f50 λ=10 | 0.0240 | 0.01247 | 0.01016 | **+22.8%** |

为此把 uniform 前沿补到 6 个点(λ=0/0.15/0.2/0.3/1/3 → D_SRC .1971/.0456/.0414/.0362/.0250/.0184,
bc .00802/.00860/.00872/.00894/.00993/.01178),**11 个臂全部内插、无一外推**(§43 被外推害过三次)。
`st_f50 λ=1` 与 uniform λ=0.3 的 D_SRC 差 **0.3%** ⇒ 无需插值的干净配对,那里 bc_loss 高 5.4%。
剂量-响应单调:越偏离均匀(门限越紧 / λ 越大),代价越大。
**机制**:`D_SRC` 是在**全** pool 上量的,只在子集上钉住无法高效控制全 pool 的距离 ⇒ 要达到同
样的邻近度就得在保留子集上钉更紧 ⇒ 付更多 `bc_loss`。

### Phase 1 — 闭环确认(3 seed × 40k):**预注册预测反向失败**
`st_f50 λ=10` 的 D_SRC 与 uniform λ=1 匹配到 3.6%(.0242 vs .0251),`bc_loss` 高 **23.1%**。
按当时刚拟合的 g 渠道斜率 −36,**预注册预测 score ≈ 0.73(区间 0.70–0.76)**。

| 臂 | D_SRC | bc_loss | policy_only | **score** |
|---|---:|---:|---:|---:|
| uniform λ=1 | 0.0251 | 0.01037 | 0.551 | **0.822** (sd .027, n=5) |
| 轴② st_f50 λ=10 | 0.0242 | 0.01277 | **0.573** | **0.843** (sd .055, n=3) |

**实测 0.843,Δ=+0.021 (se .034, t=0.63) ⇒ 打平**;预测偏了 0.113。
⇒ ① **轴②不提升**(也不损害):前沿判定 11/11 内移、闭环 ns。
⇒ ② 预测之所以失败,是因为 `bc_loss` 不是充分统计量 —— 这一格正是上面那张"充分统计量"表的
第 5 个臂,也是**逼出 `policy_only` 这个结论的那个点**(D_SRC 与 policy_only 都匹配 ⇒ score 匹配,
`bc_loss` 高 23% 无影响)。**负结果买到了一个正结论。**
⚠️ 三个 seed 的 stage-2 与标定格逐位复现(seed1 的 D_SRC=0.0240/bc=0.01247 完全一致)。

### 44c. 轴②的最后一格:按 **provenance** 筛 anchor pool(expert / success 步)(2026-09-05)
用户提议:既然 `bc_anchor_path` 的动作太脏,就**只取用来训练 source policy 的那部分数据**。
在本仓库这有精确对应物 —— stage-1 的 policy BC 跑 **`bc_expert_only=1`**,即
**π_src 就是 `expert_flags==1` 那些步的 BC**。逐子集实测(2000 episodes,433k transitions):

| 子集 | 占比 | MAE(a, π_src) | **sysdev**(不可约偏移) |
|---|---:|---:|---:|
| ALL(现役 anchor pool) | 100% | 0.282 | **0.170** |
| expert_flags==0 | 79.1% | 0.324 | 0.203 |
| expert_flags==1 | 20.9% | 0.124 | 0.042 |
| **expert ∧ success** | 14.6% | 0.072 | **0.0247** |

⇒ **§44 那个"硬下界 0.151"被这个筛法消掉了**:expert∧success 的偏移 0.0247 ≈ π_src anchor
自己的工作点,外部动作源第一次能达到同样的邻近度。(顺带另一个对照:换成干净数据集
`source_bcpolicy_sigma0.1_n5000`,gap 0.095 ⇒ 下界 0.072,改善 2.1× 但仍够不到 0.025。)

### 三臂 × 3 seed × 40k(控制器钉在 held-out D_SRC=0.05,同一工作点)
| 臂 | anchor pool | target | policy_only | **score** |
|---|---|---|---:|---:|
| `uni_src` | 全 pool | π_src(s) | 0.631 | **0.833** (sd .014) |
| `es_src` | expert∧success | π_src(s) | 0.461 | **0.710** (sd .053) |
| `es_dat` | expert∧success | **原始 expert 动作** | 0.263 | **0.738** (sd .033) |

**分解(这就是 `es_src` 这个控制臂存在的理由 —— 筛子集同时动了状态和动作两件事)**:
- **动作质量的净效应 = `es_dat` − `es_src` = +0.028 (ns)** ⇒ 锚原始 expert 动作与锚 π_src
  **打平**。动作源不再是瓶颈,但也**没有收益**。
- **状态覆盖的代价 = `es_src` − `uni_src` = −0.123** ⇒ **全部损失来自这里**,与 44b 同机制
  (削减覆盖 ⇒ 得钉更紧:`es_src` 的 pool D_SRC 被压到 0.0144 才能让全 pool 达标,bc_loss 1.55×)。
- 合计 **−0.095**。

### ★ 为什么"用更好的动作源"注定没有收益(架构层面的自洽性)
π_src **就是** expert 动作的 BC。所以锚原始 expert 动作相对锚 π_src 能提供的全部新信息,
就是 stage-1 policy 头的**蒸馏残差**(在该子集上 = 0.0247)—— 而实测它值 +0.028 (ns)。
反过来,想在**全**状态空间上用 expert 动作当 target,就需要"expert 动作在全空间上的回归",
而那**恰恰就是 π_src 本身**。⇒ **想要更好的动作标签就得放弃状态覆盖,二者不可兼得。**
⇒ §44 的准则"目标必须是 π_src 本身"应精确为:**目标必须是一个与 π_src 系统性偏移足够小的
动作源(sysdev ≲ 工作点 D_SRC);满足这个条件后换谁都一样,不满足就崩(sysdev 0.170 ⇒ 0.649)。**

### ⚠️ `policy_only` 那条定律也退化了(8 臂重算)
| 拟合 | 斜率 | R² | 残差 sd |
|---|---:|---:|---:|
| 原 5 臂(§44) | +0.526 | **0.982** | 0.0103 |
| **全 8 臂(加 44c 三臂)** | +0.396 | **0.752** | **0.0335** |

退化但没崩(对比 `bc_loss` 的 R²=0.078)。残差方向有信息:**`es_src` −0.063 / `es_dat` +0.043**,
两臂唯一差别就是 anchor target ⇒ 同样的 `policy_only` 下,锚**带噪声的原始动作**表现**超出**预期、
锚确定性 π_src 于**窄状态子集**表现**不及**预期。假说:MPPI 拿 policy 当采样中心,**proposal 的价值
不等于它裸跑的分数**,还包含它的分布宽度/候选覆盖(可直接量 policy 在 rollout 分布上的散度,未测)。
⇒ **操作准则:`policy_only` 仍是最好的臂间解释变量,但跨机制会漂 ±0.05,只能当解释、不能当判据。**

### ⚠️ 两次预注册预测连续失败 —— 判据只有 stage-3
| 预测 | 依据 | 预测值 | 实测 | 误差 |
|---|---|---:|---:|---:|
| 轴② st_f50 λ=10 (44b) | 前沿 + `bc_loss` 斜率 −36 | 0.73 | **0.843** | **+0.113** |
| 44c 三臂 | `S ≈ .532+.526·policy_only` | .87/.79/**.69** | .833/.710/**.738** | `es_src`/`es_dat` **排序反了** |

标定格(stage-2 + round-0,~7 min)的 `policy_only` 本身**测得很准**(单次 100 局 vs 40k 的 21 次
均值,偏差 ≤0.03),失效的不是测量而是**外推关系**。⇒ **标定格只配当筛子筛大方向(>0.1),
±0.1 以内一律跑满 stage-3 × 3 seed**(铁律 #6 的又两个现场版)。

### ★ 轴②的判决
**封闭。** 三种形式(硬门 50%/25%、软权 `d²`)× 两种度量(状态距离 / NTK 干扰)× 四个 λ,
离线前沿 0/11 外移,闭环 ns。**别再试基于"离 demo 远近"的 anchor 状态加权**。
44c 用完全不同的筛法(按数据 provenance 而非距离)独立复现了同一机制:**任何削减 anchor 状态
覆盖的做法都要付代价(−0.123),而覆盖是 anchor 起作用的必要条件。**
若将来还想动轴②,先跑那两个诊断:pool 与 demo 支撑集的重叠比例、`I(s)` 的偏斜度 —— 两个数
都接近"无余量"时不必再往下做(本轮它们正确预测了 11/11 的方向,只是低估了闭环的宽容度)。

---

### ★ 可操作准则(§44/44b 合并)
- **换 base 必须重测 anchor 的价值**,不能从"base 分数弱"推"anchor 会有用"。判据是
  **dynamics 误差**(有没有 ensemble / oracle),不是分数。
- **anchor 的目标必须与 π_src 的系统性偏移足够小**(`sysdev` ≲ 工作点 D_SRC)。现成数据集的
  action 通常不满足:diverse_v2 全 pool 的 sysdev 0.170 ⇒ 掉 0.08–0.17。筛到 expert∧success
  (sysdev 0.0247)就满足了 —— 但那时它只是**打平**(+0.028 ns),而筛选本身的覆盖代价是
  **−0.123**(44c)。⇒ 实践上仍然是:**直接用 π_src 的前向输出**。
- **别为了"更好的动作标签"牺牲 anchor 的状态覆盖**(44b/44c 两种筛法独立证出同一机制)。
- **报 anchor 变体时给 `policy_only`**(臂均值),它是唯一 R²=0.98 的解释变量;`D_SRC` 次之
  (0.51),**`bc_loss` 不要单独用**(0.08)。⚠️ 但只用它**解释臂间差异**,不要用它**选参数**
  (seed 层面反向,ρ=−0.50)。
- **四个自由度里三个已封闭**:① `g` 的形状(§43)② `D` 状态分布(**44b** 距离/NTK 加权 +
  **44c** provenance 筛选,两条独立路径同一结论)③ `M` 动作空间度量(§43)+ 锚向什么动作
  (§44/44c)。**只剩 ④「比单步动作还是比轨迹」没测过。**
- **±0.1 以内的 anchor 结论一律跑满 stage-3 × 3 seed。** 标定格(stage-2 + round-0)只配筛
  >0.1 的大方向:44b/44c 两次预注册预测分别错了 0.113 和把两臂**排反**,而两次的离线量本身
  都测得很准 —— 失效的是外推关系,不是测量。
- 报效应量要在**有余量的 base** 上测(铁律 #18)。

### Where
- `train_simdist_adapt.py`: `bc_anchor_data_actions: int = 0`(默认 0 ⇒ 逐位兼容,已验证:
  重启后 λ=1 的 D_SRC 复现到 0.0250)。`_build_source_anchor()` 现在返回
  **(pool, tgt, ref)**: `ref` **永远**是 π_src 的动作快照 ⇒ `D_SRC` 保持 §42/§43 的语义,
  与 anchor 回归到什么无关。开关打开时额外打印
  `|a| mean / pi_src |a| mean / mean|a_data - pi_src(s)|`。
  ⚠️ RNG 消耗不变(`rng.choice` 调用顺序未动),老 run 一字不改。
- **轴②(44b)**: `bc_anchor_far_frac`(硬门:只保留最"远"的这个比例)/ `bc_anchor_far_power`
  (软权 `w ~ far^p`)/ `bc_anchor_far_metric`(`"state"` = 逐维标准化 L2 最近邻距离;
  `"ntk"` = `−I(s)`,最后一层 NTK 干扰)。三个默认 0/`"state"` ⇒ 关闭 ⇒ 逐位兼容。
  新函数 `_anchor_far_weights(pool)`,返回训练抽样权重(`None` = 均匀);
  `_build_source_anchor` 返回 **(pool, tgt, ref, w)**,BC 循环在 `w is not None` 时走
  `torch.multinomial` 而非 `randint`。
  ⚠️ **权重只改训练抽样分布;`D_SRC` 仍在均匀全 pool 上量**,否则各臂无法与 λ 前沿对标。
  打印 `[anchor-axis2] metric=.. frac=.. power=.. | far score p0/p50/p100 | kept/N | ESS=..`
  (`power=2` 的 ESS 从 40000 掉到 3856,是有意的极端点)。
- 分析脚本在本轮 scratchpad(未入库): 从 `train.log` 抽 `[adapt @ N] target_score=` 曲线
  → 20–40k 窗口均值 + `D_SRC`/`bc_loss`/`policy_only` 配对;`frontier.py`(前沿内插 + 预注册
  阈值判定);`diag_axis2.py` / `diag_ntk.py`(44b 的两个前置诊断:状态空间重叠 + NTK 偏斜度)。

---

### 46. ★★ PIXEL 224 + ImageNet ResNet-18 + JPEG: infrastructure & MEASURED costs (2026-09-02)

Executing §45's fix order. Everything below is **measured on this cluster**, not
extrapolated — §45/§44's own estimates were off by up to 3× in both directions.

#### ① Measured, and they revise the plan
`scripts/slurm/bench_pixel224.slrm` (job 5195192), batch 512 = 256×2 cameras,
step = 6 no_grad + 1 fwd/bwd encode calls:

| encoder | params | repr | fwd ms | f+b ms | **enc ms/step** |
|---|---:|---:|---:|---:|---:|
| vit embed2/d1 @96 (current) | 386k | 15488 | 4.6 | 13.6 | **41.5** |
| resnet18 imagenet @96 | 11.2M | 4608 | 12.1 | 41.4 | **113.9** |
| **resnet18 imagenet @224** | 11.2M | 25088 | 73.8 | 231.5 | **674.4** |
| resnet18 @224, freeze 2 stages | 11.2M | 25088 | 73.9 | 104.3 | 547.6 |

⇒ **ResNet-18@224 is 16.3× vit@96**, not the 3–6× the gap-closing prompt assumed
(it guessed "200–400 ms/step"). Encoder time alone is 30 h for 160k steps.
`freeze_stages=2` buys 19% and is the cheapest lever if this becomes binding.

JPEG decode at the real per-step load (256×6×2 = 3072 frames):

| res | q | B/frame | ratio | **GPU ms/step** | CPU ms/step |
|---|---:|---:|---:|---:|---:|
| 96 | 95 | 2348 | 11.8× | 105.1 | 231.1 |
| 224 | 95 | 9907 | 15.2× | 110.3 | 918.7 |
| 224 | 85 | 5917 | 25.4× | 107.8 | 776.6 |

⇒ GPU (nvJPEG) decode is ~**resolution-independent** (per-image overhead
dominates, not pixels) and adds ~16% to a 224 step. **CPU decode is not viable**
at 224 (919 ms/step would dominate the whole step) — the batched-GPU path is
load-bearing, not an optimisation.

Generation (`calib224`, job 5195174, 400 traj, `--rl_image_size 224 --jpeg_quality 95`):
**295 transitions/s**, **18,935 B/transition** (15.9× compression; agentview
14.4×, wrist 20.9×), mean episode 179.9 steps.
⇒ 100k trajectories ≈ **18M transitions ≈ 340 GB** (§45's "1.13 TB / 94 GB" was
computed at 96 and for a raw store; at 224 the raw figure would be 6.1 TB).

#### ② 224 costs nothing to *produce* — it was already being rendered and discarded
`image_size: 224` / `rl_image_size: 96`: the wrapper renders at 224 and throws it
away in `resize_transform`. Storing the native render is a one-field change with
**zero extra render cost**. But `image_size`/`rl_image_size` come from the
EXPERT's `cfg.yaml`, not the CLI, so `generate_simdist_data.py` needed explicit
`--image_size` / `--rl_image_size` overrides (default 0 = inherit = legacy).

#### ③ Two silent-corruption bugs found on the way
- **`merge_shards` dropped dataset attrs.** It walked `create_dataset(data=item[sub][()])`,
  which does not copy attrs — and the JPEG `codec` / `frame_shape` flags live
  there. Merged shards would have loaded as unreadable object arrays. Now uses
  h5py's native `Group.copy`, which preserves attrs and streams.
- **Flush-by-trajectory-count is the wrong unit for pixels.** `--flush_every 500`
  is ~31 GB/worker at 224 (×10 workers = OOM — this is what killed job 5113396).
  Added `--flush_mb` (default 2000), which state runs never reach.

#### ④ `aux_state` broke EVERY pixel run — a 3-branch `__init__` with early returns
`WorldModel.__init__` has three arms (`raw_state` returns at :335, `pixel` at
:369, `latent` falls through). The §43 aux-state work appended
`self._build_aux_state(...)` to the **fall-through path only**, so `self.aux_state`
never existed on a pixel model, while `train_simdist.py` read it unconditionally
⇒ `AttributeError` at startup for every pixel run. It went unnoticed because the
in-flight pixel job had started *before* that (uncommitted) code landed. Fixed by
calling `_build_aux_state` in the pixel branch (sets `None` when coef=0, so **no
state_dict key is added** and strict loads are unaffected) plus a defensive
`getattr` at the read site.
⚠️ **Generalisable:** in this file, a feature added after the `latent` branch is
invisible to the other two arms and fails only at *runtime, on that arm*.

#### ⑤ Correction to the gap-closing prompt
It claims "`TopkSaver` currently ranks on `score/source_score`, which IS the
memorisation signal". **It does not** — `train_simdist.py` saves on `eval_score`,
which is `self.eval(seed=...)` on the TARGET env. No change was needed.

#### ⑥ What is stored and how it is read
Frames stay **encoded in RAM** and decode per batch, so JPEG is a RAM saving, not
just a disk one (`JpegFrames` + `decode_jpeg_batch` + `_stack_frames`, all in
`rl/sequence_replay.py`). Raw datasets keep the legacy path byte-for-byte.
`scripts/test_jpeg_replay.py` gates it on 5 checks including
**frame[t]↔state[t] alignment through `sample()`** and attr survival through a
shard copy; `read_frames_array` serves the stage-2 anchor pools, which need a
materialised array rather than lazy frames.

#### ⑦ Sharding replaces the 17 h job AND the merge
100k trajectories is 17 h in one job (b2, slow queue) plus a 340 GB merge. As **8
independent 12.5k shards** it is 8 × ~2.9 h at b1, and `source_replay_path` now
accepts `+`-separated and/or globbed shard sets, so **the merge never happens**.
Seeds must differ per shard (`seed*100000 + rank` keeps the ranges disjoint).

#### ⑧ The data-scale CONTROL (iron law #9)
The 224 arm changes **four things at once**: data volume (2400→100k traj),
resolution (96→224), backbone (from-scratch ViT→ImageNet ResNet-18) and
augmentation (none→shift). A win would be unattributable. So
`pixel.resize_to` downsamples the 224 store to 96 at load, letting a **vit@96 arm
train on the identical replay** with no second dataset — isolating data scale
alone, at ~1/16 the compute.

#### ⑨ Augmentation, finally wired
`aug_shift_pad` / `aug_prop_noise` (both default 0 ⇒ inert). Applied ONCE to the
whole `[B,H+1,C,H,W]` block at the top of `update_distill`, with **one shift per
(sample, camera) shared across all H+1 frames** — independent per-frame shifts
would inject camera jitter into the transition the dynamics head is learning, and
desynchronise the input from the detached consistency target built from the same
frame. Kept in uint8 (whole-pixel shifts ⇒ bilinear `grid_sample` is lossless
here), which holds a 256×6×224² block at 231 MB instead of 925 MB.
Stage-1 only, so anchor pools and eval are untouched by construction.

#### ⑩ Run sizing actually used
`seq_replay_max_transitions` 450000 → **8,000,000** (§45 B2: this cap, not the
dataset, is what bounded the old arm at 445k). 8M ≈ 151 GB resident ⇒ `--mem=200G`.
150k steps × 256 = **4.8 epochs**, against the reference's 2 and the old pixel
arm's 461. Going past ~8M needs a lazy/disk-backed loader, not a bigger `--mem`;
the remaining trajectories stay on disk for that.

### 46. ★ PIXEL 224 + ImageNet ResNet-18: measured costs and the infrastructure (2026-09-02)

Everything here is MEASURED on Killarney L40S, not extrapolated. §45's fix order
(augment → scale D_sim → 224 + ResNet-18) is being executed; this section records
the numbers that decide the sizing, plus four defects found on the way.

#### a. Encoder + decode cost — the doc's estimate was 2-3x low
`scripts/slurm/bench_pixel224.slrm` (job 5195192), batch 512 = 256 x 2 cameras,
7 encode calls/step (6 no_grad + 1 fwd/bwd):

| encoder | params | repr | fwd | fwd+bwd | enc/step | 160k steps |
|---|---:|---:|---:|---:|---:|---:|
| vit embed2/d1 @96 (current) | 386k | 15488 | 4.6 ms | 13.6 ms | **41.5 ms** | 1.8 h |
| resnet18 imagenet @96 | 11.2M | 4608 | 12.1 | 41.4 | 113.9 | 5.1 h |
| **resnet18 imagenet @224** | 11.2M | 25088 | 73.8 | 231.5 | **674.4 ms** | **30.0 h** |
| resnet18 @224, freeze 2 stages | 11.2M | 25088 | 73.9 | 104.3 | 547.6 | 24.3 h |

⚠️ `docs/pixel-gap-closing-prompt.md` guessed "200-400 ms/step" for ResNet-18@224.
It is **674 ms**, i.e. **16.3x vit@96**, and it is now the dominant cost — NOT
storage, which is what that document treats as the blocker.
Freezing 2 residual stages buys 19% and is the obvious lever if this bites.

**End-to-end measured in the real training loop** (smoke job 5195389, 224 +
ResNet-18 + JPEG on the calibration replay): **1.18 it/s = 847 ms/step**, which
matches 674 (encoder) + 110 (decode) + heads/optimizer. ⇒ 150k steps ≈ **35 h**
= 3 chunks of 11:59, not the 9-18 h a naive "2 epochs" reading suggests.

#### b. JPEG frame store — RAM, not disk, was always the binding constraint
Raw 224x224x3 x2 cameras = 301,056 B/transition ⇒ 6.1 TB for 100k trajectories:
over the /scratch **quota** (2 TB, 318 GB used) and far over any node's RAM.
Measured compression on our renders (q95): **agentview 14.4x, wrist 20.9x,
15.9x combined ⇒ 18,935 B/transition**, so 100k traj ≈ **340 GB**. Fidelity
MAE 0.73/255.

Implementation: frames stay ENCODED in RAM and are decoded per batch, so the
saving is a RAM saving rather than merely a disk one.
- writer: `generate_simdist_data.py --jpeg_quality` (0 = raw, legacy).
- store: `rl/sequence_replay.py::JpegFrames` — ONE concatenated blob per
  (episode, camera) plus `starts`/`ends`. NOT a list of per-frame arrays: at 8M
  transitions x 2 cameras that is 16M live numpy objects of interpreter overhead
  and GC pressure. Explicit start/end (rather than a monotonic offset array) lets
  a frame be repeated (terminal padding) or dropped (truncation) with no copy.
- decode: `decode_jpeg_batch` uses torchvision's batched nvJPEG. **GPU decode is
  ~110 ms/step at 3072 frames/step and nearly resolution-independent (105 ms at
  96, 110 at 224) ⇒ it is per-image overhead, not pixels. CPU decode at 224 is
  919 ms/step and would starve the GPU**, so the GPU path is mandatory.
- **MEASURED resident cost** (`scripts/slurm/pixel224_loadtest.slrm`, job 5196175,
  2 real shards = 25,000 episodes / 4.45M transitions): **81.6 GB peak RSS =
  18,319 B/transition**, i.e. RAM ≈ the on-disk compressed size, which is the
  whole point of decoding per batch rather than at load. Load took **19.5 min**.
  Extrapolated: 4M -> 73 GB, **6.5M -> ~119 GB / ~28 min**, 8M -> 147 GB,
  18M -> 330 GB. ⇒ the 6.5M cap leaves ~81 GB of headroom under a 200G ask for
  torch/CUDA host memory and the 10 mp eval workers; 8M would have left ~53 GB and
  the full 17.9M is simply impossible in RAM (it needs the lazy loader).
  Load time is ~28 min/chunk and stage-1 self-chunks 3x ⇒ ~4% overhead on a 35 h run.
  `frame store: JpegFrames, blob objects=1` confirms the blob layout (1 object per
  episode-camera, not one per frame).
- gate: `scripts/test_jpeg_replay.py` — raw path byte-for-byte, round-trip
  fidelity, merge-preserves-attrs, and frame[t]↔state[t] alignment through
  `sample()`. Runs on a login node in seconds.

#### c. Generation: shard it (FINAL measured numbers)
Run as **8 independent 12.5k-trajectory shards** with DIFFERENT `--seed` (the
worker seed is `seed*100000 + rank`, so integer seeds give disjoint ranges).
Every shard COMPLETED inside its 2:59 (b1) window:

| | per shard | x8 |
|---|---:|---:|
| trajectories | 12,500 (all requested) | **100,000** |
| transitions | ~2.23M | **~17.9M** |
| size | ~42.8 GB | **~342 GB** |
| rate | **348-371 transitions/s** | — |
| wall | ~1:35 | (parallel) |
| success rate | 66.2-66.9% | (reference 63.7%) |
| B/transition | ~19,150 | (agentview 14.4x, wrist ~21x) |

⚠️ **Two of my own rate estimates were wrong in opposite directions, and both came
from measuring the wrong thing:**
- the 400-trajectory calibration reported 295 t/s, but it was startup-dominated;
- watching `du` grow during the first ~35 min gave **148 t/s** and a confident
  prediction that every shard would hit its walltime at ~64% — also startup-
  dominated, and wrong.
The truth is ~360 t/s sustained. ⇒ **Size a long job from a job that ran to
completion, never from a short calibration and never from a partial `du`.**
One job would have been ~14 h (b2); the shards were ~1.6 h each on b1.

`train_simdist.py::_expand_replay_paths` accepts a "+"-separated and/or globbed
`source_replay_path` and de-dups, so shards are consumed directly. In the end
every shard DID merge, so the unmerged-`_shard_N.hdf5` fallback was insurance
rather than load-bearing -- but it is kept, and the loader also tolerates (and
loudly counts) a truncated file, because a walltime kill during merge is a real
failure mode for any future larger run.

#### d. Four defects found (three would have corrupted or wasted the run)
1. **`merge_shards` dropped dataset attrs.** It walked `item[sub][()]` into a new
   `create_dataset`, which does not carry attributes — and the JPEG `codec` /
   `frame_shape` flags live there. Merged shards would have loaded as
   unreadable object arrays. Fixed to h5py's native `copy` (also streams instead
   of materialising every frame). This is now check [3] of the gate.
2. **Flush-by-trajectory-count is the wrong unit for pixels.** `--flush_every
   500` at 224 is ~31 GB per worker x 10 workers — precisely how job 5113396
   died OUT_OF_MEMORY. Added `--flush_mb` (default 2000), which state runs never
   reach, so it is inert for the legacy arm.
3. ★ **`WorldModel._build` never set `self.aux_state` on the pixel or raw-state
   arms.** Both branches `return` before `_build_aux_state()` at the end of the
   latent path, while `train_simdist.py` reads `model.aux_state`
   unconditionally ⇒ `AttributeError` before the first gradient step, on EVERY
   pixel run. The reader is uncommitted §43 work that had only ever been
   exercised on the state arm, so this was latent in the working tree: the
   running `pixel_dynens` held the pre-edit code in memory and its REQUEUED
   chunks would have crashed. Fixed by declaring `self.aux_state = None` before
   the branches (registers nothing ⇒ state_dict unchanged, strict loads
   unaffected) and calling `_build_aux_state` in the pixel branch so
   `aux_state_coef` actually works there.
4. **The stage-2 anchor pools read frames with a bare `np.asarray(ds)`**, which
   yields an OBJECT array on a JPEG dataset. Routed through
   `read_frames_array()`.
5. ★ **Both anchor builders materialised EVERY frame of every sampled episode
   before subsampling** to `*_anchor_states` (40k). At 96 that is ~2 GB and the
   waste is invisible; at 224 with `*_anchor_episodes: 2000` it is **~108 GB of
   host RAM** and OOMs the node before the encoder is reached. Now two-pass: read
   states, choose the subsample, then read ONLY those frames
   (`read_frames_array(ds, idx=...)`).
6. **The anchor encode chunk was a constant 2048**, tuned at 96. At 224 that is
   ~2.5 GB of float input plus ResNet activations ⇒ CUDA OOM (this is what the
   stage-2 smoke actually died of). Now scales as `2048 * 96² / (H*W)`.
8. ★ **Stage-2 BC built an autograd graph through a FROZEN encoder.**
   `update_bc_policy` called `policy_action(s_flat)` on `bc_batch_size=256 x H=5`
   flattened timesteps = 2560 images x 2 cameras; at 224 the first conv's
   activations alone are ~10 GB ⇒ CUDA OOM before step 1. The encoder is excluded
   from `self._bc_opt`, so encoding under `no_grad` in resolution-sized chunks is
   EXACT -- the same argument the cached latent anchor pool already relied on.
   ⚠️ **The first version of this fix was WRONG**: `policy_action` routes through
   the DEDICATED `enc_pi` when `shared_encoder=0`, so substituting
   `policy(encode(s))` would have silently used the DYNAMICS encoder -- a
   different network, i.e. a wrong policy that still trains and still reports a
   falling loss. The chunked path is therefore restricted to dict (pixel) obs,
   where `shared_encoder=1` is asserted; state arms keep the exact legacy call.
   **The dedup gate cannot catch this class** -- it only exercises
   `update_distill`.
7. **A config knob landed in the wrong block.** `aug_shift_pad` was inserted by
   anchoring on `horizon: 5` -- which exists in BOTH `simdist.planner` and
   `simdist.world_model` -- so it went under `planner` and pyrallis rejected the
   run at startup, after the queue wait. Gate added:
   `scripts/test_configs_load.py` pyrallis-loads every config locally in seconds.
   Anchor on a block-unique key (`num_dynamics_heads`), never on `horizon`.

#### e. Corrections to `docs/pixel-gap-closing-prompt.md`
- **Its B0 is already done.** `ShardWriter` + `--flush_every` bounds worker RAM;
  its own docstring records the old accumulate-then-write path OOM'ing at 7-12k
  trajectories. What remained was the byte-unit bug (d2), not the writer.
- **Its A2 checkpoint-selection concern is wrong.** `TopkSaver` already ranks on
  `eval_score` = the TARGET score (`train_simdist.py:673,689`), not
  `source_score`. No change was needed.
- **Its "/scratch 1.7 TB free" is a per-user QUOTA, not `df`** (`df` shows a 2.3 PB
  shared filesystem). The quota also caps FILES at 10M (1.02M used) ⇒ JPEG bytes
  must live INSIDE the hdf5, never as loose per-frame files.
- Storage at 224 is **340 GB**, not the 510 GB it estimates, because it assumed
  12x compression and 204-step episodes; measured are 15.9x and ~180 steps.

#### e2. Validation ladder actually run (all on the 400-step smoke ckpt)
The smoke config has `simdist.use_planner: 0` (cheap policy-head eval during
training), so a passing stage-1 smoke does **not** exercise MPPI. The plan path is
what stages 2/3 ride on, so it was validated separately:

| check | result |
|---|---|
| stage-1 train loop at 224 (JPEG + aug + dynens) | 1.18 it/s = **847 ms/step** |
| `[budget]` epoch instrumentation | `transitions=71950 ... EPOCHS = 1.4  cap=200000` |
| MPPI at 224, standalone eval | `[eval override] ORIGINAL SimDist MPPI recipe (base=100+rand=250, iters=3, elites=64, horizon=5)` — recipe confirmed, `[EVAL ENV]` matches the canonical line |
| real live-shard 224 JPEG through `sample()` | `(8,6,3,224,224) uint8`, range [0,255] |
| all three WorldModel arms + aux on/off | `scripts/test_world_model_branches.py` |
| frozen-BN ResNet-18 per-sample determinism | max diff 1.2e-07 (dedup-compatible) |
| **stage 2 (policy BC + source anchor) at 224** | `stage-2 done: final policy BC loss = 0.04610`; anchor pool `40000 x 512 (82 MB) encoded once` in `chunks of 376` |
| **stage 3 (collect + dynamics adapt) at 224** | `[dyn-anchor] 400 source episodes -> 40000 (z,a) pairs`; ran 3 eval points to `adapt rc=0` |
| buffer load at scale | 4.45M transitions, **81.6 GB RSS = 18,319 B/transition**, 19.5 min |

⚠️ EVERY score in this ladder is **0.000**, which is expected on a 400-step stage-1
and is NOT evidence about pixels. These are PLUMBING checks: the evidence is the
MPPI recipe line, the non-trivial BC loss, the anchor pool sizes and `rc=0` -- never
the score. ⇒ **stage 1 -> 2 -> 3 all execute at 224 with the ImageNet backbone.**

It took FOUR smoke iterations, each exposing a distinct 224-only blowup that is
invisible at 96 (d3, d5, d6, and the frozen-encoder BC graph in d8 below). None
would have surfaced from unit tests or from the dedup gate; all of them needed the
arm run end-to-end on real data.

#### g. THE THREE MATCHED ARMS (this is the experimental design)
The 224 arm changes FOUR things at once vs the old pixel baseline -- data volume,
resolution, backbone, and augmentation -- so on its own a win is unattributable
(iron law #9). Three arms are therefore run at an IDENTICAL data budget
(**6.5M-transition cap / 150k steps / batch 256 = 5.9 epochs**), on 100k-trajectory
replays, differing only in what is named:

| arm | config | encoder | what it isolates | ms/step | cost |
|---|---|---|---|---:|---:|
| **state** | `..._state_simdist_dynens_100k.yaml` | latent MLP | the observation space, vs pixel | ~60 | ~3 h |
| **pixel vit@96** | `..._pixel_simdist_vit96_data100k.yaml` | vit embed2/d1 | **DATA SCALE alone**, vs the old 2400-traj vit@96 | ~240 | ~10 h |
| **pixel r18@224** | `..._pixel_simdist_r18_224.yaml` | ImageNet ResNet-18 | + resolution + backbone | ~847 | ~35 h |

The vit@96 arm needs **no second dataset**: `pixel.resize_to` (new, default 0 =
off) downsamples the stored 224 frames to 96 with antialiasing before the
backbone, and the backbone is built for the resized shape, so `repr_dim` matches
a natively-96 encoder exactly (verified: `out_dim` 521 both ways).
⚠️ Its `aug_shift_pad` is **9, not 4**: augmentation runs in `update_distill` on
the STORED frames (224 here) and the encoder resizes afterwards, so a pad of 4
would be ~1.7 px once resized -- a weaker augmentation than the 224 arm gets, i.e.
a confound. 4/96 = 4.17%, and 4.17% of 224 = 9.
Note the vit@96 arm is only ~3.5x cheaper end-to-end, not 16x, because **JPEG
decode (~110 ms/step) is paid at the STORED resolution regardless of the encoder**.

⚠️ The state arm's published 0.540 and every state-vs-pixel delta in
`docs/pixel-progress.md` were measured on the 20k replay and are PROVISIONAL
until the state arm here replaces them.

#### h. FIRST RESULTS (2026-09-03)

**Phase A control (aug OFF, OLD 2400-traj replay, 100k steps = 57 epochs) --
the memorisation signature reproduced cleanly on a fresh run:**

| step | 10k | 30k | 50k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|
| source | 0.14 | 0.42 | 0.54 | 0.38 | 0.96 | 0.96 | 0.94 |
| target | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Target is **exactly 0.00 at all ten eval points** -- flat zero, not noisy-near-zero.
★ This SHARPENS §45: memorisation is already complete at **57 epochs**, and the old
`pixel_dynens` reaches essentially the same place (source 0.956) at 461. So 461 was
never a threshold that got crossed -- **0.44M transitions is too little data at any
epoch count in this range**, which is why augmentation alone was never likely to
rescue that arm and why the data-scale arms are the main line.

**★ PHASE A VERDICT: augmentation does NOT rescue the pixel arm on the 0.44M
replay.** Matched A/B, same config except `aug_shift_pad` (4 vs 0), both 100k steps:

| arm | target | source | gap |
|---|---|---|---|
| aug OFF | **0.000 +-0.000** | 0.736 +-0.298 | 0.736 |
| aug ON | **0.012 +-0.018** | 0.628 +-0.148 | 0.616 |

Target delta **+0.012 -- UNREADABLE** (floor 0.05, iron law #1). The aug arm's full
target trend is `0.00 0.00 0.00 0.02 0.00 0.04 0.02 0.00 0.00 0.00`: noise around
zero, ending at zero.

⚠️ **The gap narrowed 0.736 -> 0.616, and that is NOT a win.** The decomposition is
`source -0.108, target +0.012` -- the gap shrank because SOURCE FELL, i.e.
augmentation made the training distribution harder to memorise without buying any
transfer. Augmentation lowers source almost by construction, so "the gap narrowed"
is a trap here. `scripts/cmp_stage1_runs.py` originally printed "narrowing is the
win condition" and would have scored this null result as a success; it now
decomposes the gap and refuses to call source degradation a win.
⇒ **The win condition is TARGET RISING, never the gap alone.**

⇒ Augmentation stays ON in the data-scale arms (it is in the reference recipe, it
costs nothing, and it demonstrably suppresses memorisation) but it is NOT the fix.
The fix under test is data scale.

**State B4 re-baseline (150k steps, 5.9 epochs, NEW 100k replay):**
`target 0.060 +-0.063 (n=5 window) | source 0.828 +-0.149`. The published state
stage-1 policy-head baseline is **0.06** ⇒ reproduced exactly, on 5x more data at
5.3x fewer steps. ⇒ scaling data did NOT move the state arm, which is consistent
with it never having had the pathology (50 epochs over 4.08M, not 461 over 0.44M).
B4's value is that the state baseline is now VERIFIED on matched data rather than
inherited from a queue-tier-limited replay.

**★ The three arms are verifiably matched** (each printed its own budget):

| arm | episodes | transitions | epochs |
|---|---:|---:|---:|
| state | 9000 x 4 shards = 36,000 | 6,482,841 | 5.9 |
| pixel vit96 | 4500 x 8 shards = 36,000 | 6,479,983 | 5.9 |
| pixel r18@224 | 4500 x 8 shards = 36,000 | 6,479,983 | 5.9 |

**The demo draw is SHARED.** The 96 and 224 target-demo sets are the SAME episodes
(identical lengths and identical `actions[0]`; same snapshot_400000, same seed,
deterministic collection) -- only the stored resolution differs. So all three arms
adapt on one draw and the comparison is PAIRED (~0.027) instead of carrying the
single-draw sigma of 0.092 (§38). Do not re-collect per arm.

Measured stage-1 throughput: state 7.92 it/s (150k in 5:15), vit96 2.97 it/s
(~14 h), r18@224 1.08 it/s = 926 ms/step (~38.6 h) -- the r18 number is 9% above
the 847 ms/step bench prediction, and the chunk budgets hold.

#### l. ★★★ HEADLINE RESULTS -- state vs pixel, matched everything (2026-09-05)

Both arms: `dynens_h5` (5 heads, H=5) + `bc_anchor_coef=1.0` + `dyn_anchor 0.1` +
the ORIGINAL SimDist MPPI recipe, 100k-trajectory replay, 6.48M transitions,
**the SAME 20-demo draw** (the 96 and 224 demo sets are the same episodes -- same
snapshot_400000, same seed, deterministic collection; only stored resolution
differs), same `[EVAL ENV]`.

**Independent re-eval of the saved ckpt, n=500, strict, sigma=0.1** (iron law #14):

| arm | success | 95% CI |
|---|---:|---|
| **state** (B4 100k re-baseline) | **0.430** (215/500) | [0.387, 0.473] |
| **pixel vit96** (100k data, 96² via resize_to) | **0.254** (127/500) | [0.216, 0.292] |

CIs do not overlap ⇒ the 0.176 gap is real. It agrees with the in-training
matched-window gap of 0.173, which is a useful cross-check between two
independent measurements.

**★ The gap is ENTIRELY the planner.** Matched 54,750-73,000 window, 19 points each:

| arm | target | policy_only | planner |
|---|---:|---:|---:|
| state | 0.466 +-0.056 | **0.136** +-0.029 | **x3.43** |
| pixel vit96 | 0.293 +-0.054 | **0.136** +-0.037 | **x2.16** |

`policy_only` is **identical to three decimals (0.136 both)**. The pixel
representation supports a policy exactly as well as the privileged 37-d state
does. Same code, same data scale, same demos, same anchor, same MPPI recipe, same
window, matched policy -- the ONLY difference is what the world model's dynamics
are worth to a planner.

**Stage-3 is doing repair work that state never needs.** Pixel's planner
multiplier over the run: **x0.65 (harmful) -> x1.10 -> x1.47 -> x1.96 -> x2.16**,
monotone and NOT plateaued. State starts at x3.4+ with ZERO target transitions.

⇒ The pixel deficit is localised to **stage-1 dynamics quality**, and the standing
mechanistic hypothesis is that a SINGLE FRAME cannot encode velocity while the
37-d state can -- a policy can map appearance to action, a dynamics model must
predict motion. This is what the history arms test.

⚠️ **Both numbers sit far below the published ladder** (`dynens_h5` 0.813,
`dynens_h5_slow_anchor` 0.979). Suspected cause is §46j: 150k-step stage-1 vs the
published 800k. Under test. Until that resolves, these two numbers are comparable
to EACH OTHER and to nothing else. State's window is also non-standard (it never
reached 75k).

#### k. ★★ STAGE-1 `score/score` DOES NOT PREDICT DOWNSTREAM VALUE (2026-09-04)

Three claims I made this session were wrong, all from the same root: I read the
stage-1 in-training target score as if it ranked checkpoints by what they are
worth AFTER adaptation. It does not. It is evaluated with `use_planner=0` and
before any target data, so it sees neither the planner nor stage-2/3.

| claim | reality |
|---|---|
| "data scale bought efficiency, not transfer" | **~11x better at matched adapt budget**: old-data pixel MPPI @9k = 0.017 (§43) vs new-data @10k = **0.190** |
| "stage-2 does not lift the pixel arm" | 20 demos lift the pixel policy head **0.02 -> 0.170**, matching state's own 0.12-0.15 |
| "the entire gap is stage-1 dynamics" | stage-3 repairs it: planner **x0.65 -> x1.95** over 40k transitions |

Both old and new data read **0.02** at stage-1 while differing ~11x after
adaptation. ⇒ **never rank stage-1 checkpoints by `score/score`.** (Same family as
§46 e2: a metric that structurally cannot see the planner.)

**The ladder, stage by stage** (both arms = `dynens_h5` + anchor λ=1 +
`dyn_anchor` 0.1 + the ORIGINAL MPPI recipe -- NOT SimDist base):

| stage | state | pixel vit96 |
|---|---|---|
| stage-1 policy head (raw pretrain) | 0.060 | 0.020 |
| stage-1 + MPPI, no target data | 0.223 | 0.020 |
| **after 20 demos** (`adapt @ 0`) | **0.510** / pol 0.120 | **0.110** / pol **0.170** |
| + 30-40k transitions of stage-3 | 0.379 / pol 0.123 | 0.274 / pol 0.140 |

★ The decisive row is `adapt @ 0`: 20 demos fix the POLICY on both arms, but only
state's stage-1 dynamics are good enough to plan with. Pixel's planner is
*actively harmful* there (0.170 -> 0.110, **x0.65**) -- a planner can only hurt if
its model is confidently wrong. Stage-3 then repairs pixel to x1.95.

⚠️ **That 30-40k row is NOT a steady-state comparison and must not be quoted as
one** (iron law #4). State's own curve dips and recovers:
`0-10k 0.397 | 10-25k 0.348 | 25-45k 0.374 | 45-73k 0.462` -- so 30-40k sits in
state's DIP while pixel is still on its initial RISE. Single n=100 points span
0.22-0.58. Both arms must reach steady state before the gap means anything.

#### j. ★★ MY EPOCH-MATCHING UNDER-TRAINED EVERY ARM (2026-09-03, self-correction)

The state B4 control, adapted and re-evaluated INDEPENDENTLY at n=500:
**0.430 (215/500)**. In-training window was 0.466 +-0.056 and the best point 0.56,
so TopkSaver optimism was ~0.13 -- squarely in iron law #14's 0.06-0.19 band, which
is a nice confirmation of that law and a warning against ever quoting `model0.pt`.

But 0.430 is FAR below the published ladder for the same arm
(docs/on-policy-adaptation.md, n=10 seeds, 75-100k window):

| rung | published | ours |
|---|---:|---:|
| `plain_h3` | 0.564 | — |
| `dynens_h5` | **0.813** | — |
| `dynens_h5_slow` | 0.898 | — |
| `dynens_h5_slow_anchor` | 0.979 | — |
| ours: dynens_h5 + anchor(λ=1), 150k-step stage-1 | — | **0.430** |

Ours HAS the anchor and should therefore sit ABOVE 0.813. It is 0.38 short.

⇒ **The dominant suspect is my own design choice: 150k stage-1 steps against the
published 800k (5.3x less).** I set 150k across all three arms to land near §45's
"the reference does 2 epochs". That over-applied the argument: §45 showed 461
epochs over 0.44M is over-training, which does NOT establish that 5.9 epochs is
right -- and this project's own published state number came from **800k steps /
50 epochs**. Truncating adapt at 73k contributes little (the curve plateaued from
40k: 0.54, 0.47, 0.50, 0.49).

⚠️ **How this confounds the pixel arms -- narrowed after checking.** It is NOT
that the arms failed to learn: `source_score` saturates on both (state 0.828,
pixel **0.96**). In fact the new pixel arm reaches the SAME source competence as
the old one **7x faster** -- 0.96 at ~110k steps vs the old 0.44M-data arm's ~800k
-- while target stays 0.02 either way:

| | steps to source ~0.96 | final source | target |
|---|---:|---:|---:|
| old pixel (0.44M) | ~800k | 0.92-0.96 | 0.00-0.02 |
| new vit96 (6.48M) | **~110k** | 0.96 | 0.02 |

⇒ at 96², 40x the data bought TRAINING EFFICIENCY, not transfer.

But `source_score` is evaluated with `use_planner=0`, so it measures the POLICY
HEAD, not the dynamics -- and every quantity that depends on DYNAMICS is weak on
both arms (state's adapt 0.43 vs published 0.813; pixel's planner lift 1.0x vs
state's 3.4x). ⇒ the live hypothesis is that **the dynamics head keeps improving
long after the policy head saturates**, and nothing we log measures dynamics
quality directly. The 800k state run is the clean test: same data, same saturated
source, 5.3x the steps.

⇒ State stage-1 EXTENDED 150k -> 800k (stage-1 has real resume, so it continues
rather than restarts). The pixel arms should follow once the state arm confirms
the step count is the cause.
⇒ **Lesson: match the step budget that produced the number you are comparing to,
and only then vary it deliberately.** Epoch count is a diagnostic for
over-training, not a target to optimise.

#### i. ⚠️ `train_simdist_adapt.py` CANNOT RESUME -- plan walltime accordingly
It persists NOTHING that would let a run continue: no `global_step`, no optimizer
state, no replay. `TopkSaver` writes weights only. So a stage-2/3 run that hits its
walltime is FINISHED at whatever transition count it reached, and the 75-100k
protocol window may simply be unreachable.

MEASURED here: the state adapt ran **589 s/round**, 2.1x the 275 s/round the
adapt.slrm header assumes, so 100k transitions needs ~16.4 h against an 11:59
walltime ⇒ it stops near **73k**. Pixel rounds are dearer still.

Consequences and the way round them:
- **Headline numbers are NOT affected**: they must come from an INDEPENDENT
  re-eval of the saved ckpt anyway (iron law #14), and `TopkSaver` still wrote it.
  What truncates is the CURVE, not the reportable number.
- **A truncated window is not comparable.** A 60-73k mean must never be set beside
  a published 75-100k mean; `scripts/cmp_adapt_runs.py` marks it NON-STANDARD and
  says so rather than printing a number that looks equivalent.
- **To actually reach 100k inside the walltime, raise `INTERVAL` (2000 => 50 eval
  points), NOT lower `EVAL_N`.** Eval is ~95% of a round's cost, so halving the
  NUMBER of eval points halves the overhead while keeping n=100 per point; cutting
  `EVAL_N` instead would change the per-point noise AND the TopkSaver selection,
  which is a protocol change (iron law #15).
- Implementing real resume (global_step + optimizer + replay, as stage-1 already
  has) is the proper fix and is NOT done.

#### f. Instrumentation
`train_simdist.py::_log_epoch_budget` prints transitions / batch / steps /
**resulting epoch count** and flags an ACTIVE `seq_replay_max_transitions` cap at
startup, so §45's 461-epochs-on-0.44M divergence can never hide again.

#### ⑪ ★ A CAPPED buffer + a MULTI-SHARD load silently throws away most of your seeds
`seq_replay_max_transitions` evicts from the **front** (`_evict_for_transitions`
pops `episodes[0]`), and the shard loop loads shards in order. So with
`source_replay_num_data: -1` the 6.5M cap over 17.8M loaded transitions keeps only
the **last** shards — we would have generated 8 independent seeds and trained on
about 3. Reproduced on a miniature (8 shards × 10 eps, cap 300): the buffer ends
up holding **only shards 6 and 7**, i.e. 6 of 8 seeds evicted entirely.

Fix: set `source_replay_num_data` to a fixed **per-shard** slice
(4500 traj × 8 shards × ~178.8 steps ≈ 6.44M, just under the cap) so every seed is
represented. It also reads ~125 GB instead of 340 GB.
⚠️ Generalises to ANY capped buffer fed from several files. The STATE arm is
safe as it stands — its config sets no `seq_replay_max_transitions`, so it takes
the default 0 = unlimited, and its 100k replay is only ~3.5 GB (195 B/transition)
— but it becomes unsafe the moment anyone adds a cap there.

#### ⑫ Augmentation pad is in STORED-frame units, not encoder-input units
`_augment_obs` runs on the stored `[B,H+1,C,224,224]` block **before**
`PixelObsEncoder` applies `resize_to`. So for the vit@96 control the pad must be
expressed at 224: `pad=9` at 224 ≈ the repo's conventional `pad=4` at 96 after
downsampling. Setting `pad=4` there would have been a ~2.3× weaker shift than
intended — an easy and invisible mistake, since nothing about it fails loudly.

### 47. ★★★ THE r18@224 LADDER: the anchor reproduces (hugely), the ensemble does NOT (2026-09-09)

The question was narrow: **do the dynamics ensemble and the source anchor
reproduce on PIXEL the gains they gave on STATE?** Two deltas, rebuilt on r18@224.
All three rungs ran the full 100k target transitions at a **verifiably identical
budget** (`transitions=6479983` printed by each stage-1; `NUM_DEMO=20`,
`MAX_TRANS=100000`, `INTERVAL=2000`, `EVAL_N=100`, `dyn_anchor 0.1`, the ORIGINAL
SimDist MPPI recipe) on **one shared demo draw**, so the rungs are PAIRED (~0.027)
rather than carrying §38's 0.092 single-draw sigma.

| rung | stage-1 | 75-100k window (n=13) | independent n=500 |
|---|---|---:|---:|
| **A** `plain_h3` (1 head, H=3) | `threading_pix224_plainh3_stage1_seed1` | 0.365 ±0.099 | **0.418** (209/500) |
| **B** `dynens_h5` (5 heads, H=5) | `threading_pixel224_stage1_seed1` | 0.425 ±0.054 | **0.334** (167/500) |
| **C** `dynens_h5` + anchor λ=1 | same as B | 0.875 ±0.030 | **0.766** (383/500) |

```
state (published, n=10 seeds, WINDOW):
  plain_h3 0.564  --+0.249 ens-->  dynens_h5 0.813  --+0.081 anchor-->  0.979
vit96 (independent n=500):
  plain_h3 0.218  --+0.120 ens-->  dynens_h5 0.338  ---0.072 anchor-->  0.266
r18@224 (independent n=500):
  plain_h3 0.418  ---0.084 ens-->  dynens_h5 0.334  --+0.432 anchor-->  0.766
r18@224 (window, the estimand the published state ladder actually reports):
  plain_h3 0.365  --+0.060 ens-->  dynens_h5 0.425  --+0.450 anchor-->  0.875
```

#### ★ VERDICT
- **The ensemble does NOT reproduce.** State +0.249 and vit96 +0.120 are not
  recovered: r18 gives **+0.060 (window) / −0.084 (independent)** — the two
  measures disagree in SIGN, and the window effect is smaller than rung A's own
  SD (0.099). Null within noise on one seed; certainly not +0.249.
- **The anchor DOES — and it is now the dominant lever on pixel.**
  **+0.432 (independent) / +0.450 (window)**, both measures agreeing, ~5.3x
  state's +0.081 and OPPOSITE IN SIGN to vit96's −0.072.
⇒ The two arms swapped roles: on vit96 the ensemble transferred and the anchor
backfired; on r18@224 the anchor is everything and the ensemble is noise.

#### a. Why one delta is trustworthy and the other is not — a SELECTION confound
`model0.pt` is TopkSaver's **max** over the in-training eval points, so an
independent re-eval of it removes the in-training optimism but **preserves the
selection bias**. And the premium scales with the arm's curve variance:

| arm | window SD | best in-training pt | premium (best − window) | indep − window |
|---|---:|---:|---:|---:|
| A `plain_h3` | **0.099** | 0.61 | **+0.245** | **+0.053** |
| B `dynens_h5` | 0.054 | 0.52 | +0.095 | −0.091 |
| C `+anchor` | 0.030 | 0.94 | +0.065 | −0.109 |

A's noisier curve earned a **2.5x larger** selection premium, and A is the ONLY
arm whose independent eval landed ABOVE its window mean.
⇒ `independent n=500 of model0.pt` is **not a common estimand across arms of
differing curve variance.**
- The **anchor** delta compares two LOW-variance arms (0.054 vs 0.030; premiums
  +0.095 vs +0.065) ⇒ the confound largely cancels ⇒ both measures agree ⇒ robust.
- The **ensemble** delta compares a HIGH-variance arm to a low-variance one
  (0.099 vs 0.054; premiums +0.245 vs +0.095) ⇒ it does not cancel ⇒ the measures
  disagree in sign ⇒ unresolvable on one seed.

★★ **This REFINES iron law #14.** "Never quote `model0.pt`; use an independent
re-eval" is right, but incomplete: the independent re-eval inherits TopkSaver's
max-selection. When comparing arms, also compare their **window SDs**; if they
differ materially, the independent numbers are not exchangeable and the delta must
be read on the window (or with per-arm reruns from matched checkpoints).
★ Note the published state ladder reports **75-100k window means over 10 seeds**,
so for LADDER DELTAS the window is the like-for-like estimand; independent n=500
is the right way to state an ABSOLUTE headline for one checkpoint. Report both.
(For the vit96 rungs both measures agree in sign on BOTH deltas — its three arms
have comparable SDs 0.036/0.050/0.068 — so only r18's ensemble delta is spoiled.)

#### b. ★★★ The encoder's famous ~0.5 is an INTERACTION, not a main effect
Rung B closes a full 2x2 (encoder × anchor, `dynens_h5` fixed, independent n=500
per cell, one shared demo draw):

| | no anchor | +anchor | anchor effect |
|---|---:|---:|---:|
| **vit96** | 0.338 (169/500) | 0.266 (133/500) | **−0.072** (z=−2.5) |
| **r18@224** | 0.334 (167/500) | **0.766** (383/500) | **+0.432** (z=+15.2) |
| **encoder effect** | **−0.004** (z=−0.1) | **+0.500** (z=+18.3) | **INTERACTION +0.504** (z=+12.4) |

★★ **With the anchor OFF, r18@224 buys NOTHING over vit96: 0.334 vs 0.338.**
⇒ **The ★★★ PIXEL ENCODER DECISION block in `CLAUDE.md` overstates the encoder.**
It reads "分辨率 + ImageNet 先验值 ~0.5，比这个项目试过的任何其他杠杆都大得多" as an
encoder main effect. The 0.5 is real but lives ENTIRELY in the `+anchor` row; the
encoder's main effect at `anchor=0` is **zero**. Never quote the encoder gain
without naming the anchor condition.
⇒ That block's own warning — redo the vit96 ladder on r18 because §42's
interaction structure could flip the anchor's sign — was CORRECT about the
mechanism. But the flip runs **opposite to §42's prediction**: §42 has the anchor
paying most on a WEAK base, while here the STRONGER base shows the far larger
gain. **Base strength does not explain the pixel anchor.**

#### c. Mechanism worth chasing: the anchor guards MPPI's PROPOSAL
On r18 the anchor barely moves the policy head (`policy_only` 0.203 → 0.249) but
nearly DOUBLES the planner multiplier (**x2.09 → x3.51**; rung A is x1.62).
The stage-2 anchor pulls π toward π_src, and **π is MPPI's proposal
distribution** — so this reads as *keeping the proposal inside the region the
dynamics model actually models well*, not as making the policy better. That ties
to §36 (proposal quality is what pays: +0.138 for a flow proposal, while changing
the SEARCH was ns) and §27 (only model error ON THE ON-POLICY DISTRIBUTION
matters). It also explains the interaction in b: a sharper encoder's dynamics
model is only worth something if the proposal stays where that model is accurate.
⚠️ Multipliers are from in-training points and carry the same optimism as the
window; treat as a hypothesis, not a measurement.

#### d. Limits
ONE stage-1 seed per cell. The anchor delta (+0.432, z=+15.2) and the interaction
(+0.504, z=+12.4) are enormous against EVAL sampling noise, but **stage-1 seed
variance is unmeasured** — §38's noise floor and iron law #3 (pre-register a fresh
seed) both apply before either number is treated as final. The shared demo draw
makes cells paired, not absolutely robust. The ensemble delta needs several seeds
to settle sign, and per a it should be judged on windows from matched checkpoints.

### 48. ★★★ THE 5-SEED r18@224 LADDER, WITH §44'S FULL AUGMENTATION (2026-09-11, RUNNING)

**Status: PRE-REGISTERED, runs in flight.** Design, code and gates are below and
are frozen; the results table is filled in when the 15 adapt runs land. Recorded
before any number exists on purpose (iron law #3: a rule discovered post-hoc has
to be re-run on a fresh seed, and winner's curse was measured at 2.4x here).

#### The question
§47 asked whether the **dynamics ensemble** and the **source anchor** reproduce on
pixels the gains they gave on state, and answered on **n=1 adapt seed**. The
anchor answer (+0.432, z=+15.2) is trustworthy; the ensemble answer is not, for a
reason §47a nailed down: it compares a high-curve-variance arm (`plain_h3`, window
SD 0.099) to a low-variance one (0.054), so TopkSaver's max-selection premium does
not cancel and the two measures **disagree in sign** (+0.060 window / −0.084
independent). At n=1 the ensemble delta's SE is ~0.19, so even a true +0.249 is
~1.3 sigma. That result was under-powered **by construction**, not by bad luck.

This run re-does the ladder at **5 adapt seeds** and adds the three §44
augmentation rows §47's arms never had.

```
           base SimDist+demos      +ensemble           +anchor
rung        A  plain_h3      -->   B  dynens_h5  -->   C  dynens_h5+anchor
heads/H     1 head,  H=3           5 heads, H=5        5 heads, H=5 (B's stage-1)
ANCHOR_COEF 0                      0                   1.0
```

#### a. PART 1 — the three missing augmentations (implemented, gated)
`rl/simdist_agent.py::_augment_obs` had **random shift only**
(`aug_shift_pad`, 8 at 224 per §46⑫) and an `aug_prop_noise` that existed but was
pinned at 0.0. `RandomShiftsAug` IS the reference's random crop, so that row was
already satisfied and no second crop was added. Now added:

| §44 row | before | now | knob |
|---|---|---|---|
| random crop | ✅ live | unchanged | `aug_shift_pad: 8` |
| proprio Gaussian noise | implemented, **off** | **on** | `aug_prop_noise: 0.002` |
| colour jitter | ✗ missing | **implemented** | `aug_color_jitter: 0.3` |
| Gaussian blur | ✗ missing | **implemented** | `aug_blur_sigma_max: 2.0` |

- `aug_color_jitter` is a single strength `s`: brightness/contrast/saturation
  ~ U(1−s, 1+s), hue ~ U(−h, h) turns with h = min(0.5, s/3) — so **s=0.3 is
  exactly the conventional `ColorJitter(0.3, 0.3, 0.3, 0.1)`**. Hue is a YIQ
  chroma-plane rotation. Application order is FIXED (b → c → sat → hue);
  torchvision shuffles it, which would make a seeded run unrepeatable.
- `aug_blur_sigma_max` samples sigma ~ U(0, max] per (sample, camera) and applies
  a separable Gaussian as one grouped conv. Kernel width is a constant set by
  `sigma_max`, so a small sigma just collapses the kernel toward a delta.
- ⚠️ **`aug_prop_noise` is in METRES.** `prop` is RAW and unnormalised — the eef
  pose is concatenated as-is at `rl/pixel_obs_encoder.py:189` — so sigma acts
  directly on position channels. **0.002 = 2 mm**, just above the task's ~1.8 mm
  tolerance window. The obvious-looking 0.01 would be **1 cm** and would swamp
  the only signal the task turns on. Nothing warns you about this.

★ **THE LOAD-BEARING INVARIANT: ONE draw per (sample, camera), SHARED across all
H+1 frames.** Every parameter — shift grid, four jitter factors, blur sigma — is
drawn once OUTSIDE the `t` loop. Per-frame parameters would inject apparent
camera jitter into the very transition the dynamics head is learning AND
desynchronise the network input from the detached consistency target built from
the same frame. This is the one mistake that trains fine and reports nothing.

Implementation notes worth keeping:
- Order is geometric → photometric → blur.
- Everything runs in **float32**, cast back at the end. Do NOT contort to stay in
  uint8: the `[256,6,3,224,224]` block is 231 MB as uint8 and 925 MB as float32,
  a 694 MB delta = **1.4% of an L40S's 48 GB**. The loop is per-timestep, so the
  transient is one `[B,C,H,W]` slice (154 MB at 224), not the whole block.
- The shift is still applied with `round()` after `grid_sample`, so whole-pixel
  offsets stay **exactly lossless** in uint8 and the shift-only path is
  bit-identical to §47's — including the RNG stream, because a knob at 0 draws
  NOTHING.
- Both new knobs are **config fields on `WorldModelConfig`, not modules**, so
  unlike iron law #23's head/buffer case they exist on all three `__init__`
  branches by construction. They are still read via `getattr(..., None)` because
  an OLD ckpt's serialised cfg has no such attribute.

#### b. PART 1b — the demo-draw blocker that would have made the whole run a lie
`rl/sequence_replay.py::load_demos_to_sequence_buffer` (and `rl/replay.py:206`)
took the **first N episodes, unseeded**. Left alone, all 5 seeds would have
trained on the **IDENTICAL 20 demos**, and the error bars would have EXCLUDED
§38's single-draw sigma = **0.092, the largest noise source in this project** —
producing a 5-seed table that looks far tighter than the truth.

Fixed with `demo_offset` (default 0 ⇒ byte-for-byte legacy), threaded through
`train_simdist.py` (`target_demo_offset`), `train_simdist_adapt.py` (override +
unconditional log line) and `scripts/slurm/adapt.slrm` (`DEMO_OFFSET`). The
200-episode pool gives 10 disjoint 20-demo draws; this run uses **0/20/40/60/80**,
verified disjoint (100 distinct episodes across the 5 draws, 0 overlap) and
range-checked with an assert.

**Blocked design, and this is the whole point:**
- **within a seed** all 3 rungs share that seed's draw ⇒ both deltas are PAIRED
  against ~0.027 instead of carrying 0.092;
- **across seeds** the draws differ ⇒ the error bar on the mean delta honestly
  includes draw variance.

This matches the published state protocol rather than departing from it: its 10
seeds are ADAPT seeds over **one** stage-1 per architecture, each with a different
20-demo draw shared across arms within the seed. So the published SDs
(`plain_h3` 0.171 / `dynens_h5` 0.085 / `anchor` 0.010) already contain draw
variance, and using 5 stage-1 seeds here would inject a variance component the
published bars do NOT have — making an equal true effect look weaker in our
table. Match the estimand first.

#### c. Gates (all green before anything was queued)
| gate | result |
|---|---|
| `scripts/test_world_model_branches.py` | pass, all 3 arms, 0 missing attrs |
| `scripts/test_jpeg_replay.py` | pass (5/5) |
| `scripts/test_history_layout.py` | pass |
| `scripts/test_configs_load.py` | **19/19** load; the two new `_aug` configs read `heads=5 H=5` / `heads=1 H=3`, `shift8/jit0.3/blur2.0/prop0.002` |
| `scripts/slurm/dedup_regress.slrm` (job 5363904) | **`distill/loss = 9.270623207092285`** — bit-identical ⇒ the default-off contract holds |
| ★ `scripts/test_augmentation.py` (**NEW**) | pass |
| ★ CPU smoke of the real `update_distill` on the r18 pixel path | off 20.177292 → shift 20.117105 → full 19.738861 |

★ **`scripts/test_augmentation.py` is the only thing that can catch a broken
aug.** `dedup_regress` runs with augmentation OFF, so it is structurally blind:
it proves the default-off contract and nothing else. The new gate asserts every
knob is a tensor-equal no-op at 0; that a knob ON actually changes the pixels
(§44's whole failure was a hook nobody called — an INERT aug costs a multi-day
run and reports nothing); that identical input frames of a sample come out
identical while different samples AND different cameras get different draws (the
invariant above); that the uint8 round trip clamps instead of wrapping and the
shift introduces no new pixel values; and that it runs on all three branches.
The `update_distill` smoke is separate on purpose — per §46 d3/d8, a
`WorldModel`-level gate does not exercise the training path, and both of those
defects were pixel-only and runtime-only.

#### d. What is running
Two stage-1 models only — one per architecture, 150k steps, self-chunking
(`STEPS=150000 CHUNK=1 MAX_CHUNK=6`, `--mem=200G`, `aip-rhinehar`):

| arch | config | save_dir | rungs |
|---|---|---|---|
| `plain_h3` | `threading_d0_pixel_simdist_r18_224_plain_h3_aug.yaml` | `exps/mimicgen/thr_pix224_aug_plainh3_stage1` | A |
| `dynens_h5` | `threading_d0_pixel_simdist_r18_224_aug.yaml` | `exps/mimicgen/thr_pix224_aug_dynens_stage1` | **B and C** |

Rung A needs its own stage-1 (iron law #20: heads/horizon are built in
`WorldModel._build` and `load_state_dict` is strict). B and C **share** one —
the anchor is an adapt-time knob, so retraining for C would only add variance.

**Verified at startup, both arms printing IDENTICALLY** (jobs 5364043 / 5364044):
```
[seq-buffer] 8 source replay shards:                       <- §46⑪ satisfied
[budget] transitions=6479983  batch=256  steps=150000
         samples_seen=38400000  => EPOCHS over D_sim = 5.9   cap=6500000
[aug] shift_pad=8 prop_noise=0.002 color_jitter=0.3 blur_sigma_max=2.0
      | encoder=resnet18_pretrained | rl_image_size=224
```
`transitions=6479983` is **exactly** §47's figure ⇒ the budget is matched both to
§47 and between the two rungs. The `[aug]` line is the one §45 added precisely so
that a silent zero-augmentation run can never happen again; it reads the full
stack rather than the `<== NO AUGMENTATION` fallback. ⚠️ 5.9 epochs is still ~3x
the reference's **2**, though far from the 461 §45 measured — the epoch axis is
NOT fixed by this run and remains §45's open item.

Chaining uses the `THEN_SBATCH` hook (infra §7z ①), because a self-chunking
stage-1 has **no final job id** to hang `--dependency=afterany` on:
`scripts/slurm/then_aug48_{plain_h3,dynens}.sh` → `launch_aug_ladder_followon.sh`,
which submits that architecture's adapt seeds and chains one headline n=500 eval
behind each. The wrappers exist so the `THEN_SBATCH` value carries no `=` and no
spaces through the per-chunk `--export` re-forwarding, and the launcher has a
`DRYRUN=1` switch (exercised — all 15 sbatch lines verified before arming).
Every hop forwards `--account=aip-rhinehar` explicitly (§7z ②: `--export=ALL`
carries env vars, never sbatch flags).

Adapt budget, identical on all 15 runs: `NUM_DEMO=20 MAX_TRANS=100000
INTERVAL=2000 EVAL_N=100 DYN_ANCHOR_COEF=0.1`, ORIGINAL MPPI recipe,
`--time=23:59:00`. `INTERVAL=2000` is the MINIMUM that reaches 100k in walltime
and is what §47's rungs used; if this ever has to be economised the honest lever
is eval DENSITY (raise `INTERVAL`), never `EVAL_N` — that drives TopkSaver
selection, so changing it is a protocol change (§46i / iron law #15). And
`train_simdist_adapt.py` **cannot resume**: a timed-out adapt is finished where
it stopped.

#### e. How this will be read (fixed in advance)
1. **Both measures for every rung**: the 75-100k window
   (`scripts/cmp_adapt_runs.py`) and the independent n=500. The published state
   ladder reports **window means**, so the window is the like-for-like estimand
   for DELTAS; independent n=500 states an ABSOLUTE number for one checkpoint.
2. **Print each arm's window SD next to its delta** (§47a). The independent
   re-eval removes in-training optimism but NOT TopkSaver's max-selection, and
   the premium scales with the arm's curve variance (measured: SD 0.099 →
   +0.245; 0.054 → +0.095; 0.030 → +0.065). If two arms' SDs differ materially
   their independent numbers are **not exchangeable** and the delta must be read
   on the window.
3. **Deltas PAIRED WITHIN SEED**, then averaged over the 5 seeds, reporting the
   SD of the 5 paired deltas. Comparing cross-seed means directly throws away
   the blocking and reinflates draw variance.
4. ⚠️ **The honest published comparator for the ensemble is NOT +0.249.** The
   published `dynens_h5*` scores are `score_best`, a max over 4 pessimism
   coefficients, carrying ~+0.03-0.05 of selection bias that `plain_h3` cannot
   take. Our arms take no such max ⇒ compare against roughly **+0.20-0.22**.
   And the published **anchor** +0.081 was measured on `dynens_h5_slow`
   (0.898 → 0.979), not on `dynens_h5` — name the rung.
5. At n=5 paired the ensemble delta's SE is ~0.085 ⇒ ~2.4-2.9 sigma against that
   de-selected comparator. The anchor delta was already ~5 sigma at n=1.
6. Absolute levels are **not** comparable to the published state ladder (§46j:
   150k vs 800k stage-1 steps is worth ~+0.076; it cancels within this ladder but
   not against them). Only deltas travel.
7. Runtime checks: rung A's adapt log must read `horizon=3 [kept]`, B and C `5`;
   A and B must show **zero** `[anchor]` pool lines and C must show them; all
   three stage-1 logs must print `[budget] transitions=6479983` and a
   `[seq-buffer]` with **8 shards** (§46⑪: a capped buffer fed from several files
   evicts from the FRONT and can silently keep only the last shards).

#### f. Launch log (filled in as each arch lands)
**Rung A — stage-1 `thr_pix224_aug_plainh3_stage1` COMPLETE at exactly 150000
steps** (chunks 5364043 -> 5364063 -> 5390706, each ending in the designed
TIMEOUT, resume points exact at 50000/100000, `--mem=204800M` and
`--account=aip-rhinehar` propagated at every hop). Hook fired ONCE
(`[then] submitted OK` on chunk 3; chunk 4 read `hook already fired ... not
resubmitting` off the marker). 5 adapt + 5 chained evals submitted.

All five rung-A runs self-report the design correctly:

| seed | demo draw | horizon | `[anchor]` lines |
|---|---|---|---:|
| s1 | `[0, 20)` | `horizon=3 [kept]` | 0 |
| s2 | `[20, 40)` | `horizon=3 [kept]` | 0 |
| s3 | `[40, 60)` | `horizon=3 [kept]` | 0 |
| s4 | `[60, 80)` | `horizon=3 [kept]` | 0 |
| s5 | `[80, 100)` | `horizon=3 [kept]` | 0 |

plus `[adapt] using ORIGINAL SimDist MPPI recipe (base=100+rand=250,
init_std=1.0, min_std=0.05, iters=3, temp=0.4, elites=64, base_policy_std=0.1)`
on every one. ⇒ rung A's geometry, anchor condition and **five disjoint demo
draws** are all confirmed from the runs themselves, not assumed from the submit
command.

**Measured chunk rates** (11:59 walltime, ~30 min of which is the 8-shard load):
`plain_h3` **50k usable steps/chunk** (3 chunks; matches §7z ③'s estimate
exactly), `dynens_h5` **30k/chunk** (5 chunks, MAX_CHUNK=6 ⇒ one spare).

**Conditional Phase 2, pre-registered:** only if the ensemble delta lands
ambiguous (|delta| < ~0.10, or the two measures disagree in sign as in §47), add
**2 more stage-1 seeds per architecture** and rerun rungs **A and B only** (C is
unaffected — it shares B's stage-1 and its delta is already clean). ~212 GPU-h,
and it must be labelled as going BEYOND the published protocol, not as the
headline.
