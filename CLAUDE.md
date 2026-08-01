# CLAUDE.md — Project working notes

Living doc capturing every modification made on top of upstream ibrl, plus a
running TODO/roadmap. Update this when adding new features or discovering
gotchas. Sections are intentionally orthogonal so you can extend any one
without rewriting the others.

Last updated: 2026-07-28 (§33 为什么 value adapt 会让 SimDist 变差 —— 根因 =
MPPI 的候选区分信号只有 `std_i(γ^H V)=0.0068`（value 量程的 1.4%，判别信息在第 4
位有效数字上、且是同一状态内邻近点之间的差），而任何离线重拟合 value 头的残差都是
0.005-0.033 = 同量级或大 3-5 倍 ⇒ top-64 elite 集合被大面积改写 ⇒ 崩。
`val_mse=3e-4` 看着完美，实为 RMS 0.0173 = 信号的 2.6 倍 —— 所有离线指标结构性失明。
使能前提两条：CEM softmax 是死的（temp×std=0.003-0.006 ⇒ participation 64.00/64
⇒ 提交动作=top-64 首动作均匀平均）+ V 占 return 散布 98% ⇒ **value 唯一渠道是 topk
成员选择**。E0 剂量-响应（同配方只改精度）：残差/信号 0/0.33/0.43/0.59/0.86/0.93 →
0.775/0.749/0.725/0.660/0.450/0.345，Spearman −1.000。换 teacher/换评估点/换 source
数据/数据×13/lr/EMA/带新 value 重 adapt dynamics 30k 全部只值 ±0.01~0.13；唯一有效
的是"把现役函数自身当目标"(+0.25)。⇒ **value 头不是可独立替换的零件**，它是 stage-1
联合优化的产物且 dynamics 的 adapt 是带着它做的；"不同"本身就是伤害（重拟合的头对专家
value 的排序其实更好：+0.176 vs 现役的 −0.010，闭环却输 0.26-0.40）。必须 adapt 时唯一
没被否证的形态 = 以现役函数为锚的**门控残差** + 锚定 backup + 信任域（勿用 MC 回报：噪声
需每状态 ~5000 rollout）。附读数纪律：收敛要分数与 VAL_mse 同时平掉、`best.pt` 乐观
0.06-0.19 故用多点均值、默认开 `--value_ema 0.999`（跨度压 5.4×）。
scripts/value_head_recipe_ablate.py + diag_value_residual_vs_score.py + 7 个否证
假设的诊断脚本。Earlier: §15 Flow BC normalization + low-data config:
StateFlowPolicy gains state z-score + action min/max→[-1,1] normalization as
registered buffers [default ON, identity until set_normalization()], fit on
the train split by RobomimicDataset/CoTrainDataset and injected at construction
in train_bc. Buffers persist in the ckpt so eval auto-normalizes; identity
defaults + tolerant load_state_dict ⇒ pre-normalization ckpts load + behave
byte-for-byte. Motivation: generative heads are scale-sensitive, hurting the
20-demo regime where plain MLP beat flow. New
threading_d0_state_flow_bc_lowdata.yaml [smaller UNet, weight_decay 1e-3,
ema 0.999, num_data 20]. bc/flow_policy.py + bc/dataset.py + train_bc.py +
scripts/test_flow_norm.py. Earlier: §13 CFG on co-training flow BC:
StateFlowPolicy gains a `cfg_enabled` flag [default OFF] that concats a
one-hot source/target domain label to obs encoder output, dropouts to ∅
during training, and interpolates `s̃ = (1+λ)·s(c) − λ·s(∅)` at inference
(paper default λ=-0.5). CoTrainDataset tags each batch with domain_id;
single-dataset path unaffected. ADDA intentionally skipped — for state-based
obs (no visual nuisance) it has no signal. bc/flow_policy.py +
bc/dataset.py + scripts/test_flow_cfg.py + cotrain YAML knobs. Earlier:
§12 active target switched to "dynamics-only":
across all Threading_D0 baselines (ibrl_shifted / naive / dr / dr_oracle /
sgft / flow_bc / diffusion_bc / diffusion_transformer_bc / bc_rnn / cotrain),
the target_domain_shift was migrated from the old offset-bearing recipe
(friction 0.3, mass 3.0, solref 0.8, gripper_kp 0.5, hole_offset
[0.003,0.003,0], mask=true) to the validated "newtarget" recipe (friction
0.5, mass 3.0, solref/solimp 1.0, gripper_kp 0.12, hole_offset [0,0,0],
ring_size 0.85, ring_friction 2.0, mask=false). dr_oracle's DR ranges
re-bracketed; hole_offset DR removed. cotrain's target demo set still
matches the OLD shift — re-collect before evaluating cotrain on the new
target. Earlier: §11 IL co-training: CoTrainDataset mixes a source + target
dataset at a fixed per-batch ratio [default 95% src / 5% tgt = 380+20
episodes] and feeds the flow BC; bc/dataset.py learns the
collect_target_demos env_cfg.json-sidecar format + clamps ctrl_delta
actions; threading_d0_state_cotrain_flow_bc.yaml +
scripts/train_cotrain_flow_bc_threading.sh. Earlier: §10 SGFT-on-IBRL
value transfer: frozen source QAgent serves as both the IBRL BC prior and
the V_src for potential-based reward shaping r'=r+γV_src(s')-V_src(s) in
update_critic; target-domain only; q_agent.sgft_shaping/sgft_coef +
MainConfig.sgft_source_agent + threading_d0_state_sgft.yaml. Same day: §5c
strict success gating [dwell K=3 + ring-disp 0.04]; ring_size_scale/
ring_friction_scale precision shifts [0.85/3.0]; solref 0.8/solimp 1.0
anti-clip-through)

---

## 0. Project overview

Forked from `https://github.com/Chenyang Cao/ibrl` (state-based RL + BC for
robosuite/MetaWorld). Modifications cluster around four axes:

1. **MimicGen task integration** — 10 D0 tasks registered in the wrapper, BC
   data conversion tools patched.
2. **Domain shift framework** — sgft-style contact-phase shifts for
   `Threading_D0` with `fixed` and `dr` modes, plumbed through `train_rl.py`.
3. **Flow-matching BC policy** — robomimic-aligned `StateFlowPolicy` with
   action chunking, EMA, and pred>action horizon.
4. **Chunked-eval infrastructure** — `evaluate/` rewired to support per-worker
   action queues with `is_first_step` episode-boundary signaling.

Each section below: **What / Why / Where / How / TODO**.

---

## 1. Environment setup

### What
A conda env called `ibrl` (Blackwell-compatible, deliberately diverges from
the README's torch 2.1 / cu121 pin).

### Where
- Conda env actually lives at `/home/Chenyang/.conda/envs/ibrl`
  (NOT `/home/Chenyang/anaconda3/envs/ibrl` despite where anaconda is).
- Source repos vendored as **git submodules** under `deps/`:
  - `deps/robosuite` — pinned commit `b9d8d3d` (== v1.4.1, MimicGen requirement)
  - `deps/robomimic` — at upstream HEAD `e10526b` (**v0.5.0**, 2026-02). The
    MimicGen-recommended pin `d0b37cf` is preserved as the git tag
    `mimicgen-pinned-rollback` on that submodule if v0.5 ever causes issues.
  - `deps/mimicgen` — at HEAD (`72bd767`).
  - Legacy clones at `/home/Chenyang/ibrl_deps/` may still exist for backward
    compat (the editable install was originally there); safe to delete once
    the new `deps/` install is verified.
- C++ extension `common_utils/rela` is compiled (`cd common_utils && make`).
  Rebuild if torch is upgraded.

### Setup from a fresh clone
```bash
# 1) clone with submodules
git clone --recursive https://github.com/Sunlighted/ibrl.git
cd ibrl

# 2) create env + install torch (Blackwell)
conda create -n ibrl python=3.10 -y
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ibrl
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3) install pinned mujoco + deps in submodules
python -m pip install "numpy<2" "mujoco==2.3.2"
python -m pip install -e deps/robosuite -e deps/robomimic -e deps/mimicgen

# 4) install other ibrl deps
python -m pip install Cython==0.29.36 einops==0.7.0 fasteners==0.19 pyrallis==0.3.1 \
  tabulate==0.9.0 typing-inspect==0.9.0 wandb==0.16.4 cloudpickle "gym==0.26.2"

# 5) build the C++ extension
cd common_utils && make -j8 && cd ..

# 6) activate / set paths each session
source set_env.sh
```

### How
```bash
# activate
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh && conda activate ibrl
# or
cd /home/Chenyang/ibrl && source set_env.sh   # also sets PYTHONPATH + MUJOCO_GL
```

### Why diverge from README
- README pins **torch 2.1.0 + cu121** → silently fails on RTX Pro 6000 Blackwell
  (sm_120). Replaced with **torch 2.7.0 + cu128**.
- README pins **mujoco 3.1.3** → MimicGen needs 2.3.2, used that instead.
- Skipped **mujoco-py 2.1 + metaworld** entirely (old, painful on new system,
  not needed for MimicGen work). `env/metaworld_wrapper.py` will fail to
  import if anything tries to use it — accepted.
- `numpy<2` pinned (wandb 0.16.4 uses removed `np.float_`).
- `huggingface_hub==0.23.4` pinned (robomimic v0.5 hard pins).
- `robosuite_task_zoo` **NOT** installed → `HammerCleanup_D0` / `Kitchen_D0`
  unavailable, other 10 MimicGen tasks fine.

### Gotchas
- **`pip` has a stale shebang** pointing at `/home/Chenyang/anaconda3/envs/ibrl/bin/python3.10`
  (which doesn't exist). Always use **`python -m pip install ...`** in this env.
- 3× RTX Pro 6000 Blackwell available (sm_120, 97 GB VRAM each).

### TODO / known issues
- [ ] None known. If torch is bumped, remember to `cd common_utils && make` again.

---

## 2. MimicGen integration

### What
10 MimicGen D0 tasks registered in the robosuite wrapper, dataset conversion
tools patched to also register MimicGen envs.

### Where
- `env/robosuite_wrapper.py`
  - L12-22: `try: import mimicgen` triggers task registration on robosuite import
  - L25-67: `STATE_KEYS` dict extended for all 10 D0 tasks (all use `DEFAULT_STATE_KEYS`)
  - L68-85: `STATE_SHAPE` dict — precomputed (9 prop + object-state dim)
- `tools/dataset_states_to_obs.py`
  - L61-64: `import mimicgen` added (needed for `EnvUtils.create_env_for_data_processing` to find `Threading_D0` etc.)
  - L144-153: empty-cameras bug fix — `np.concatenate([])` crash when no `--camera_names` passed
- `tools/robomimic_dataset_conversion.py`
  - L20-23: `import mimicgen`

### Tasks available
Threading_D0, Stack_D0, StackThree_D0, Square_D0, Coffee_D0,
CoffeePreparation_D0, ThreePieceAssembly_D0, PickPlace_D0, NutAssembly_D0,
MugCleanup_D0. To add another, probe `obs["object-state"].shape[0]` once
and add the entry — see `STATE_SHAPE` block.

### How (state-only dataset prep)
```bash
cd /home/Chenyang/ibrl && source set_env.sh

# 1) download MimicGen-generated demos
python deps/mimicgen/mimicgen/scripts/download_datasets.py \
  --dataset_type core --tasks threading_d0 \
  --download_dir release/data/mimicgen

# 2) convert to ibrl format (state mode — no --camera_names)
python tools/dataset_states_to_obs.py \
  --dataset release/data/mimicgen/core/threading_d0.hdf5 \
  --output_folder release/data/mimicgen/core/threading_d0 \
  --done_mode 2
# Output: release/data/mimicgen/core/threading_d0/processed_data96.hdf5
```

For pixel mode add `--camera_names agentview robot0_eye_in_hand` (much slower:
re-renders every demo step at 96×96).

### Dataset facts (Threading_D0)
- MimicGen "core" release: **1000 demos**, **99.6% success**, ~224 mean
  steps/demo, **state shape = 37** (eef_pos[3] + eef_quat[4] +
  gripper_qpos[2] + object-state[28]), action_dim=7 (OSC_POSE + gripper)
- File downloaded by the user was at `/home/Chenyang/ibrl/dataset/processed_data96.hdf5`
  as of 2026-05-24 but **was removed** by 2026-05-25 — YAMLs hard-code this
  path and will fail until file reappears.

### Demo configs (BC + RL state)
- `release/cfgs/mimicgen/stack_d0_state_bc.yaml`
- `release/cfgs/mimicgen/stack_d0_state_rlpd.yaml`
- (Threading-specific configs documented in §3 and §4)

### TODO
- [ ] Re-download Threading dataset or update YAML paths once file reappears.
- [ ] Pixel-mode configs (currently only state mode is wired).
- [ ] Generalize `STATE_KEYS` to other MimicGen variants (D1, D2 — broader
      initial state distributions).

---

## 3. Domain-shift framework (Threading_D0 only)

### What
Contact-phase sim-to-real / sim-to-sim domain shifts mirroring sgft's
`peg-insert-side` setup. Two modes: `fixed` (single target shift) and `dr`
(per-reset random sampling).

### Where
- `env/domain_shift.py` (NEW, ~310 lines)
  - `DomainShiftConfig` — 9 fixed-shift params + 8 DR ranges + obs masking flag
  - `apply_contact_shift()` — mutates needle's `body_mass`, `body_inertia`,
    `body_ipos`, geom `friction`/`solref[:,0]`/`solimp[:,0:2]`, and Panda
    gripper actuator `gainprm[0]` / `biasprm[1]`
  - `apply_hole_offset()` — mutates **tripod freejoint qpos** (NOT body_pos —
    Threading's `_reset_internal` sets via qpos so body_pos writes get
    clobbered)
  - `apply_ring_shift()` (NEW 2026-05-27) — makes insertion need finer
    alignment WITHOUT moving the 12 mm success sphere. `ring_size_scale<1`
    grows every **collision** ring geom inward (y,z half-size +=
    `0.008·(1−scale)`) to shrink the 16 mm hole to `scale·16 mm`; geom centers
    are untouched so the success-check's averaged `ring_pos` is unchanged
    (and `radius=tripod.ring_size[1]` is a fixed object attr, not re-read from
    the model). `ring_friction_scale>1` raises the ring rim's (deliberately
    low, 0.3) sliding friction; MuJoCo combines contact friction as the
    element-wise max of the two geoms, so contact μ ≈ `0.3·ring_friction_scale`
    in the relevant regime. Visual `_vis` twins are intentionally NOT touched
    (friction is meaningless on non-colliding geoms; resizing them would shrink
    the *rendered* hole — a stray pixel-mode cue). Needs `geom_size` in the
    nominal snapshot so the scale composes from nominal each reset.
  - `mask_hole_offset_from_obs()` — subtracts offset from `obs["tripod_pos"]`,
    `obs["tripod_to_robot0_eef_pos"]`, and matching slices of
    `obs["object-state"]` so policy sees nominal hole location
  - `DRSampler` — per-reset random sampling + nuisance dims (arm mass, joint
    damping); does NOT randomize hole_offset / gripper_kp / com_offset (sgft
    design — those are the "blind" gap dims)
  - `resolve_threading_bindings()` — Threading-only env binding lookup; raises
    `NotImplementedError` for other env names
- `env/robosuite_wrapper.py`
  - L12-22: imports from `env/domain_shift`
  - L117: `PixelRobosuite.__init__` accepts optional `domain_shift` kwarg
  - L163-190: at init, snapshot nominal + resolve bindings + create DRSampler
  - L174-181: `_extract_images` calls `mask_hole_offset_from_obs` in fixed
    mode with `mask_hole_offset_in_obs=True`
  - L259-280: `reset()` re-applies shift after every `env.reset()` (because
    robosuite `hard_reset=True` rebuilds the sim and wipes mutations)
- `train_rl.py`
  - L14: imports `DomainShiftConfig`
  - L73-75: `train_domain_shift` + `eval_domain_shift` nested config fields
    in `MainConfig`
  - L200, 216: threaded into both `train_env` and `eval_env`

### How
Source/target mapping (in this framework):
- **`mode="none"`** → source (nominal env)
- **`mode="fixed"`** → target (perturbed by fixed amount)
- **`mode="dr"`** → source's "DR sub-distribution" (narrow random shifts
  centered on nominal; intentionally does NOT cover target values)

| Use case | `train_domain_shift` | `eval_domain_shift` | YAML |
|---|---|---|---|
| Pure source (no shift anywhere) | `none` | `none` | (omit fields) |
| Naive sim2real baseline | `none` | `fixed` | `threading_d0_state_naive.yaml` |
| DR baseline | `dr` | `fixed` | `threading_d0_state_dr.yaml` |
| IBRL on target | `fixed` | `fixed` | `threading_d0_state_ibrl_shifted.yaml` |

### Default values (tuned for Threading)
> **Historical / archived (2026-05-25 → 2026-05-27).** The values discussed
> below describe the OLD offset-bearing target recipe and the iterative
> tuning that led to it. As of **2026-05-28** the active target across all
> Threading_D0 YAMLs is the dynamics-only target — see **§12** for the
> current numbers + rationale. The discussion here is retained because the
> per-parameter justifications (clip-through risk, friction cone, ring
> geometry floor) still apply to anyone re-tuning the new recipe.

After empirical correction from sgft's MetaWorld values:
- `friction_scale: 0.3` ← **still sgft default; suspect — see TODO**
- `mass_scale: 3.0`
- `solref_scale: 0.8`, `solimp_scale: 1.0` ← **revised 2026-05-26**. The
  original `3.0` / `0.7` *softened* contacts so much that the needle clipped
  *through* the ring mesh: the success check is a pure position test
  (`‖needle - ring‖ < 12 mm`, `threading.py:_check_success`) with no
  penetration/threading topology check, so clip-through registered as a
  false-positive success. Clip-through only happens in the **soft** direction
  (timeconst ↑). `solimp < 1.0` is the most dangerous knob (it directly widens
  MuJoCo's allowed interpenetration), so `solimp_scale` is pinned to **1.0**.
  `solref_scale` is set to **0.8** (0.02→0.016 s timeconst) — slightly
  *stiffer* than nominal, which makes the contact phase mildly **harder**
  (rigid contact is less forgiving of misalignment) while being completely
  clip-through-safe. Stability floor: timeconst ≥ ~2×dt = 0.004 s
  (dt=0.002 s), i.e. `solref_scale ≳ 0.2`; below that the solver jitters and
  penetration returns. Safe "harder" range ≈ [0.3, 0.8].
- `gripper_kp_scale: 0.5`
- **`hole_offset: [0.003, 0.003, 0.0]`** — sgft's `[0.015, 0.015, 0]` was
  **2.1 cm offset** which exceeds Threading's 12 mm success radius
  (impossible). First reduced to `[0.005, 0.005, 0]` = 7.1 mm ≈ 60% of
  success radius, but source-only BC empirically only had ~4-5 mm headroom
  left and scored close to zero. Re-tuned to `[0.003, 0.003, 0]` = **4.2 mm
  ≈ 35% of success radius**, leaving room for BC trajectory noise so
  source-only baselines are non-trivial and the DR/IBRL gap is visible.
- **`ring_size_scale: 1.0`** (default = no shift) — **suggested target 0.85**
  (NEW 2026-05-27). Shrinks the 16 mm hole to `scale·16 mm`. The needle
  cross-section is **10×10 mm** (needle.py), so the opening must stay > 10 mm
  to be physically feasible:
  `1.00→16.0 mm (3.0 mm/side)`, `0.85→13.6 mm (1.8 mm/side)`,
  `0.80→12.8 mm (1.4 mm/side)`, `0.75→12.0 mm (1.0 mm/side, very tight)`,
  `≤0.625→≤10 mm (needle no longer fits → impossible)`. Safe "harder" range
  ≈ **[0.78, 0.95]**. Reference 0.85 keeps a comfortable 1.8 mm/side margin
  while clearly demanding cleaner alignment than nominal. Combine with
  `hand_init_perturb_joint_sigma` to stress alignment from both ends.
- **`ring_friction_scale: 1.0`** (default = no shift) — **suggested target 3.0**
  (NEW 2026-05-27). Scales the ring rim's nominal **0.3** sliding friction
  (set low in `ring_tripod.py` "to ensure easy insertion"). Contact μ ≈
  `0.3·ring_friction_scale` (MuJoCo max-combines the two geoms and the needle
  side is already ≤0.3). Reference 3.0 → rim μ 0.9, so the needle catches on
  the rim and a sloppy approach jams instead of sliding in. Useful range ≈
  **[1.0, 5.0]**; much above ~5 the needle can stick/refuse to slide.

#### Nominal vs shifted (Threading_D0 needle + Panda gripper)
**Active target = the dynamics-only target (2026-05-28; see §12).** The
table below reflects the *new* shift values; for the old offset-bearing
recipe see git history of this file or the bottom rows below (annotated).
Empirically measured nominal values (from `env.sim.model` after env.reset()):

| Parameter | Where applied | Nominal | Scale | **Shifted** | Semantic effect |
|---|---|---:|---:|---:|---|
| `body_mass` | needle root body | **0.0076 kg** (7.6 g) | ×3.0 | **0.0228 kg** | 3× heavier needle — harder to swing, more momentum into hole |
| `body_inertia` | needle root body | `[9.62e-6, 9.62e-6, 1.73e-6]` kg·m² | ×3.0 | `[2.89e-5, 2.89e-5, 5.18e-6]` | scales with mass; angular dynamics slower |
| `geom_friction[:,0]` (sliding μ) | `needle_obj_needle` (tip) | **0.3** (already low) | ×0.5 | **0.15** | tip μ halved; slippery on rim but still graspable |
| `geom_friction[:,0]` | `needle`/`handle` _vis_ geoms + `handle` | **1.0** | ×0.5 | **0.5** | handle μ halved; harder to keep in gripper but not impossible |
| `geom_solref[:,0]` (contact timeconst) | needle geoms | **0.02 s** (20 ms) | ×1.0 | **0.02 s** | unchanged — pinned to avoid clip-through false success |
| `geom_solimp[:,0]` (dmin) | needle geoms | **0.9** | ×1.0 | **0.9** | unchanged — pinned to avoid mesh interpenetration |
| `geom_solimp[:,1]` (dmax) | needle geoms | **0.95** | ×1.0 | **0.95** | unchanged — see above |
| `actuator_gainprm[0]` (gripper kp) | both finger joints | **1000.0** | ×0.12 | **120.0** | ~12% gripper stiffness — soft grasp; needle slips under load |
| `actuator_biasprm[1]` (gripper kp bias) | both finger joints | **−1000.0** | ×0.12 | **−120.0** | same as above |
| `tripod_joint qpos[:3]` (hole pos) | tripod freejoint | `(x₀, y₀, z₀)` | **+0 (none)** | `(x₀, y₀, z₀)` | **no offset** — gap is pure dynamics, not latent registration |
| `geom_size[:,1:3]` (ring y,z half) | 20 collision `tripod_obj_ring_*` | **0.002 m** | **+0.0012** | **0.0032 m** | hole opening 16.0→**13.6 mm** (`ring_size_scale=0.85`); centers fixed so success radius unchanged |
| `geom_friction[:,0]` (sliding μ) | 20 collision `tripod_obj_ring_*` | **0.3** (low for easy insert) | ×2.0 | **0.6** | grippier rim; contact μ=max(needle,ring)≈0.6, sloppy approach jams |

**Legacy / old-target values (no longer active; kept for cross-referencing
pre-2026-05-28 logs):**

| Parameter | Old scale | Old shifted | Why retired |
|---|---:|---:|---|
| `geom_friction[:,0]` (needle tip) | ×0.3 | 0.09 | too slippery — broke grasp; new target uses ×0.5 |
| `geom_friction[:,0]` (handle) | ×0.3 | 0.3 | same — handle as slippery as nominal tip; new target ×0.5 |
| `geom_solref[:,0]` | ×0.8 | 0.016 s | stiffer contact removed; new target pins to 1.0 |
| `actuator_gainprm[0]` (gripper kp) | ×0.5 | 500.0 | not soft enough; new target ×0.12 → 120 |
| `tripod_joint qpos[:3]` | +[0.003,0.003,0] | offset 4.2 mm | removed entirely — gap is dynamics-only now |
| `geom_friction[:,0]` (ring rim) | ×3.0 | 0.9 | too aggressive (needle could stick); new target ×2.0 → 0.6 |

Plus nuisance dims (only randomized under `mode="dr"`):

| Parameter | DR range | Notes |
|---|---|---|
| `body_mass` of arm/hand links | ×[0.9, 1.1] | small mass variation, doesn't affect grasp |
| `dof_damping` (global) | ×[0.7, 1.3] | joint damping; object freejoints have damping=0 so no effect there |

(`hole_offset` and `gripper_kp_scale` are not randomized by the default DR
config — they are the sgft "unknown gap" dims. The **oracle** DR config
`threading_d0_state_dr_oracle.yaml` does randomize them.)

### Geometry context (Threading_D0)
- Ring inner opening: 16×16 mm rectangular slot (frame of 20 collision box
  geoms `tripod_obj_ring_*`, each 4 mm thick); shrinkable via `ring_size_scale`
- Needle cross-section: **10×10 mm** (`needle.py` `needle_size[0,2]=0.005`
  half-size) → hard physical floor on `ring_size_scale` (~0.625)
- Ring rim nominal sliding friction: **0.3** (intentionally low); raise via
  `ring_friction_scale`
- Success criterion: `||needle_pos - ring_center|| < 12 mm`. **Note:** this
  radius is `tripod.ring_size[1]`, a Python attr fixed at object construction,
  NOT re-read from the model — so resizing ring geoms at runtime does not move
  the success threshold (that's why shrinking the hole adds a real physical
  obstacle while keeping the generous 12 mm tolerance)
- Reference: `env/domain_shift.py` + comments in
  `release/cfgs/mimicgen/threading_d0_state_ibrl_shifted.yaml` header

### What kind of gap is `hole_offset`? (config vs. dynamics; state vs. vision)
Mechanistically `hole_offset` is **NOT** an inertial/contact dynamics gap. It
only relocates the tripod (an object's pose) — the true transition
`P(s'|s,a)` over the *full* state (which includes the real hole pose) is
unchanged. So in the full-state MDP it is a **configuration / geometry gap**,
categorically different from `mass/friction/solref/solimp/gripper_kp` (those
genuinely change `P(s'|s,a)` = true dynamics gaps).

**But it CAN be legitimately framed as a dynamics gap — in the agent's
observation-space (POMDP belief) MDP — *iff* the offset is unobserved.**
Model-free RL operates over observations: it sees `o`, acts `a`, gets `o'`.
With `mask_hole_offset_in_obs=True` the needle pose stays observed but the
hole pose is masked to nominal, so:
- source: obs says "hole at nominal", hole *is* there → needle threads → `o'`.
- target: the **same** observed `o` ("hole at nominal"), but the hole is really
  at +offset → the same `a` drives the needle into solid ring → a **different**
  `o'` (the observed needle pose/velocity transitions differently — not just a
  reward change).

Same `(o, a)` → different `o'` ⇒ the effective transition kernel
`T_eff(o'|o,a)` shifts ⇒ a **dynamics gap in observation space**. Formally: the
true hole pose is a latent variable θ that affects `T(s'|s,a,θ)`; marginalizing
over the *unobserved* θ yields a shifted effective transition. Under partial
observability the "observation gap" vs "dynamics gap" distinction collapses.

Two conditions make this framing honest (don't claim it otherwise):
1. **The offset must be unobserved** — masked in state mode (done); in *vision*
   mode it cannot be masked out of pixels (the camera renders the true hole),
   so to keep it a hidden gap it must be relocated to **camera extrinsics /
   hand-eye calibration error** (not yet implemented — see §3 TODO / camera
   randomization). With an observed offset it degrades to a config/
   generalization gap, and the "dynamics gap" claim is invalid.
2. **It must change the observed transition, not only the reward** — satisfied
   here (the needle's contact outcome differs), so it is not a mere goal shift.

**Wording to use:** call it an *unobserved kinematic offset that induces a
transition (dynamics) gap in the observation-space MDP*, or a *latent
task-parameter shift*; the real-world analogue is **hand-eye / registration
calibration error** (the controller commands motion for the geometry it
*perceives* but the system responds with the *true* geometry — an unmodeled
disturbance feedback must reject). Be explicit that this is a
**kinematic/registration** dynamics gap, complementary to (not the same as) the
inertial/contact dynamics gaps above; never say the physics `P(s'|s,a)` itself
changed.

### TODO
- [ ] **Empirically validate `friction_scale=0.3` doesn't break grasping**.
      Needle's base friction is already low (0.3) and handle is 1.0;
      ×0.3 makes handle 0.3 which may make BC unable to grasp. Run a BC
      checkpoint under nominal vs shift; if shift breaks grasp at nominal
      eval, dial down to 0.5 or 0.7. See chat 2026-05-25 for analysis.
- [ ] Extend `resolve_*_bindings` to other MimicGen tasks (Stack, Square,
      Coffee, ...) — currently Threading-only.
- [ ] Optional: visual domain randomization (textures, lighting, camera) —
      orthogonal to current contact-phase shift; sgft also skips this.
- [ ] BC eval on target — see §4 TODO.

---

## 4. Flow-matching BC policy

### What
`StateFlowPolicy` — rectified-flow / linear-OT BC head for state-based
control. Aligned with robomimic's `algo/diffusion_policy.py` on the three
highest-impact features (chunking, pred>exec horizons, EMA) while staying
simpler in mechanism (MLP not UNet, continuous time not DDPM, Euler not
scheduler).

### Where
- `bc/flow_policy.py` (NEW, ~250 lines)
  - `StateFlowPolicyConfig` — dataclass with all knobs
  - `StateFlowPolicy` — module
    - Training: linear OT `x_t = (1-t)z + t·a`, target `v = a-z`, MSE loss
    - Inference: Euler ODE 10 steps from `z~N(0,I)` to `x_1`, reshape to (Tp, D)
    - `loss(batch)`, `act(obs)` [single action, back-compat],
      `act_chunk(obs)` [(B, Ta, D) for chunked eval]
    - `update_ema()` / `apply_ema()` / `restore_from_ema()` — shadow weights
  - `FourierTimeEmbedding` — fixed-frequency sin/cos features for continuous t
- `bc/dataset.py`
  - `DatasetConfig.action_horizon: int = 1` — chunk length (1 = legacy single-action)
  - L130-138: build `entry["action_chunk"]` with last-action padding at episode end
  - L211-219: `_convert_to_batch` routes `action_chunk` into `batch.action`
- `train_bc.py`
  - L14: import `StateFlowPolicy`
  - L23-26: `flow_policy` nested config + `use_flow` flag
  - L68-79: branch in policy instantiation; asserts `dataset.action_horizon ==
    flow_policy.pred_horizon` and `action_horizon <= pred_horizon`
  - L132-136: `policy.update_ema()` after every `optim.step()` (no-op for
    non-flow policies)
  - L185-198: `evaluate()` helper wraps `apply_ema` / `restore_from_ema` in
    try/finally — automatically applies for any policy with these methods
- `evaluate/multi_process_eval.py`
  - `EvalProc.start()`: workers send `(proc_id, obs, is_first)` tuple
  - Main loop: per-worker `action_queues: dict[int, deque]`, cleared on
    `is_first=True`; only calls policy for empty queues (cheap re-prediction
    amortization)
  - `_policy_act_chunk(agent, obs)` helper with non-chunked fallback
- `evaluate/eval.py`
  - Per-episode local `deque`, refilled from `_policy_act_chunk_single`
- `release/cfgs/mimicgen/threading_d0_state_flow_bc.yaml` — demo config

### How
```bash
cd /home/Chenyang/ibrl && source set_env.sh
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_flow_bc.yaml
# CLI overrides:
#   --flow_policy.pred_horizon 8 --dataset.action_horizon 8   # smaller chunk
#   --flow_policy.num_inference_steps 5                       # faster eval
#   --flow_policy.use_ema 0                                   # disable EMA
#   --flow_policy.hidden_dim 1024                             # bigger network
```

### Horizon convention (mirrors robomimic)
- `pred_horizon` (**Tp**): chunk length the network predicts/trains on. Must
  equal `dataset.action_horizon`.
- `action_horizon` (**Ta**): chunk length executed per re-prediction.
  Must be ≤ Tp. `act_chunk` returns `chunk_pred[:, 0:Ta]`.
- Default: Tp=16, Ta=8 (matches robomimic diffusion policy default).

### Design choices
- **Backbone is configurable via `flow_policy.backbone`** — default
  `"unet1d"` (robomimic's `ConditionalUnet1D`, FiLM-cond, ~5M params),
  fallback `"mlp"` (6-layer MLP at 1024-dim, ~6M params, concat conditioning,
  Fourier time embedding). The UNet1D default matches `bc/diffusion_policy.py`
  exactly so flow-vs-diffusion ablations isolate the training objective.
  Continuous flow time t∈[0,1] is rescaled by `time_scale=1000` before the
  UNet's `SinusoidalPosEmb` so the embedding spans its full frequency band
  (otherwise it collapses).
- **Linear OT path** `x_t = (1-t)z + t·a`, target velocity `v = a - z`,
  uniform time `t ~ U(0,1)`.
- **Loss reduction = mean** (`F.mse_loss(v_pred, v_target)`) — matches
  `bc/diffusion_policy.py`. Previously summed over (Tp*D) which scaled the
  loss by ~112× vs diffusion at the same lr; the same `lr=1e-4` is now
  apples-to-apples between heads.
- **EMA decay 0.9999** — standard diffusion/flow default. EMA shadow is
  stored as a plain dict (`self._ema_state`) but **is** persisted via
  overridden `state_dict()` / `load_state_dict()` under keys prefixed with
  `_ema_shadow__`. So saved ckpts include EMA; standalone eval auto-loads
  EMA weights (apply_ema()/restore_from_ema() work after a fresh load).
  Old checkpoints without EMA keys still load fine (back-compat: missing
  EMA → `apply_ema()` returns False, online weights are used).
- **Eval chunk queue lives in evaluate framework, not policy** — keeps the
  policy stateless / multi-process-safe. Workers signal episode boundaries
  via `is_first_step` bit so main can clear stale queue actions.
- **Back-compat**: `_policy_act_chunk` wraps non-chunked `act()` returns into
  length-1 chunks → existing `StateBcPolicy` / `BcPolicy` continue to work
  identically under the new eval code.

### Performance
- Params (default `backbone="unet1d"`, Tp=16, action_dim=7): **4.92M**
  — matches `bc/diffusion_policy.py`. The legacy `backbone="mlp"` variant at
  the new 1024×6 default is **5.94M**; the old 4×512 MLP was 1.1M.
- Eval chunking still amortizes the policy call over Ta=8 env steps.
- Old wall-clock numbers (1.86 ms / step at MLP 4×512) no longer apply; the
  UNet1D backbone is heavier per step but unlocks the architectural
  inductive bias along the chunk axis.

### TODO
- [x] **Wire BC eval onto target domain** — done 2026-05-25. See §5.
- [x] **Make EMA state part of `state_dict()`** — done 2026-05-26. Done via
      `state_dict` / `load_state_dict` overrides on `StateFlowPolicy`,
      `StateDiffusionPolicy`, `StateRnnGmmPolicy`, prefixing shadow tensors
      with `_ema_shadow__`. Standalone eval now sees the same EMA weights
      in-training eval used. Adds ~equal to params bytes to the ckpt size.
- [x] **Optional UNet1D + FiLM backbone** — done 2026-05-25. Default is now
      `backbone="unet1d"` (matches `bc/diffusion_policy.py`); MLP variant
      kept under `backbone="mlp"`. Old MLP-trained checkpoints will not
      load against new defaults; pin `backbone: "mlp"` in YAML to reuse them.
- [ ] Logit-normal time sampling (SD3 style) — vs current uniform. Cheap
      ablation when needed.
- [ ] Real end-to-end test of multi-process eval with chunked policy and a
      trained checkpoint (smoke tested in isolation only).

---

## 5. Source/target evaluation infrastructure

### What
- Two **mutually exclusive** robot init-noise knobs (analogs of sgft's
  `cfg.hand_init_noise`):
  - `hand_init_perturb` (3-tuple): EE-space uniform box + 3-DOF positional
    IK override. Only EE position is targeted; orientation drifts as a
    null-space side effect.
  - `hand_init_perturb_joint_sigma` (float): scale robosuite's built-in
    joint-space gaussian. Adds `N(0, σ)` to each of the 7 arm joint qpos;
    EE position **and** orientation are both perturbed via FK. `σ=0.02`
    reproduces robosuite/MimicGen's default demo-collection distribution.
- BC training-time eval can be configured to run on source or target.
- After training, **dual eval** automatically runs on BOTH source and target,
  reports source/target scores + their gap.
- CLI `evaluate/eval.py` accepts on-the-fly env overrides so you can re-eval
  any saved checkpoint under different conditions.

### Where
- `env/robosuite_wrapper.py` L125-174, L295-340: `PixelRobosuite(...
  hand_init_perturb=..., hand_init_perturb_joint_sigma=...)`. The two
  kwargs are **mutually exclusive** (asserted in `__init__`).
  - **Both `None`** (default): robosuite's `initialization_noise="default"`
    = **σ=0.02 gaussian on joint qpos** (≈22 mm EE std). Matches MimicGen
    demo collection — eval is then in-distribution with the training data,
    which is the right default.
  - `hand_init_perturb` 3-tuple regimes:
    - **`(0.0, 0.0, 0.0)`**: fully deterministic init. Robosuite's noise is
      disabled (`magnitude=0.0`) and we don't IK-perturb either. Use when
      you want eval variance to come *only* from the policy (clean for
      cross-checkpoint comparison).
    - **`(rx, ry, rz)` with any > 0**: robosuite's noise is disabled and
      we override via IK — at each reset, sample an EE offset from
      `uniform([-rx,+rx]×[-ry,+ry]×[-rz,+rz])` and damped-LS IK the arm
      there. Matches sgft's `hand_init_noise` convention (sgft uses
      MetaWorld's EE-direct `hand_init_pos`; we IK on top of robosuite).
      Gripper finger joints and object placements are NOT touched. IK is
      3-DOF positional; orientation drifts slightly for non-zero offsets
      (small for ≤ 2 cm). Baseline EE pos snapshotted lazily on first reset.
  - `hand_init_perturb_joint_sigma` float regime: sets
    `initialization_noise={"magnitude": σ, "type": "gaussian"}` directly,
    so robosuite adds `N(0, σ)` to each of the 7 arm joint qpos values at
    every reset. Same distribution shape as the default (σ=0.02), just
    rescaled — use when you want a "demo-style but wider/narrower" init
    distribution without leaving joint space. σ=0 ⇒ deterministic init
    (equivalent to `hand_init_perturb=(0,0,0)`, joint-space route).
- `bc/dataset.py` L52: `DatasetConfig.hand_init_perturb: Optional[tuple] =
  None`, threaded into `env_params` dict (BC env is only used for eval).
  Default `None` → robosuite's σ=0.02 init.
- `train_bc.py` L43-50: `MainConfig.eval_domain_shift: DomainShiftConfig`.
- `train_bc.py L160-187`: in-training eval uses `eval_domain_shift` if
  mode != "none"; otherwise source. Post-training final eval ALWAYS runs
  source AND (if mode != "none") target, writing `best_ckpt_source` /
  `best_ckpt_target` to the log + printing both + the gap.
- `train_bc.py L189-215`: `evaluate()` helper accepts `domain_shift` and
  `hand_init_perturb` keyword overrides.
- `evaluate/eval.py L113-156`: CLI flags `--shift_mode`,
  `--shift_hole_offset`, `--shift_friction_scale`, `--shift_mass_scale`,
  `--shift_solref_scale`, `--shift_solimp_scale`, `--shift_gripper_kp_scale`,
  `--shift_ring_size_scale`, `--shift_ring_friction_scale`,
  `--shift_mask_hole_in_obs`, `--hand_init_perturb`.

### Where checkpoints land
`TopkSaver` writes to `cfg.save_dir/`:
- **`latest.pt`** — overwritten every epoch (= the "last" checkpoint)
- **`model0.pt` / `model1.pt`** — top-2 by in-training eval score (kept
  rolling); `get_best_model()` returns the highest-scoring one
- **`cfg.yaml`** — pyrallis dump of the resolved config used for training

To load + eval any of them later, point `evaluate/eval.py --weight` at the
file. The CLI loads via `train_bc.load_model(weight, "cuda")` which reads
`<save_dir>/cfg.yaml` from the same dir.

### How (typical workflow)
```bash
# Train flow BC on Threading_D0 source data; track sim2real gap during training
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_flow_bc.yaml

# After training, dual eval auto-prints:
#   [final eval | source]  score = 0.78  (exps/bc/run1/model0.pt)
#   [final eval | target]  score = 0.45  (shift mode=fixed, hole_offset=(0.003,0.003,0))
#   [final eval | gap   ]  source - target = +0.33

# Re-eval the saved checkpoint under different conditions without re-training:
python evaluate/eval.py --weight exps/bc/run1/latest.pt --mode bc --num_game 100 \
    --shift_mode none                                     # eval on source
python evaluate/eval.py --weight exps/bc/run1/latest.pt --mode bc --num_game 100 \
    --shift_from_cfg 1                                    # reproduce the YAML's target
                                                          # (reads eval_domain_shift
                                                          # from <run>/cfg.yaml, applies
                                                          # all six shift fields)
python evaluate/eval.py --weight exps/bc/run1/latest.pt --mode bc --num_game 100 \
    --shift_from_cfg 1 --shift_hole_offset 0.008,0.008,0  # YAML target + override hole only
python evaluate/eval.py --weight exps/bc/run1/latest.pt --mode bc --num_game 100 \
    --shift_mode fixed --hand_init_perturb 0              # ad-hoc shift w/o cfg baseline
python evaluate/eval.py --weight exps/bc/run1/latest.pt --mode bc --num_game 100 \
    --shift_mode fixed --shift_friction_scale 0.3 --shift_mass_scale 3.0 --shift_solref_scale 0.8 --shift_solimp_scale 1.0 --shift_gripper_kp_scale 0.5 --shift_hole_offset 0.003,0.003,0 --hand_init_perturb_joint_sigma 0.1
```

WARNING: passing only `--shift_mode fixed` (without `--shift_from_cfg 1`
or explicit `--shift_*_scale` flags) constructs a *partial* shift where
all unset scales default to 1.0 (= no shift). That's almost never the
target you trained against — use `--shift_from_cfg 1` to reproduce the
YAML's exact target settings.

### Design notes
- **In-training eval is single-mode**: tracks `eval_domain_shift.mode`. If
  set to "fixed", the saver's topk ranking reflects target scores, not
  source. To get source-tracked saving + target dual eval at end, leave
  mode="none" in yaml and use the CLI for target eval afterwards.
- **EMA is applied during in-training eval AND persisted in `state_dict()`**
  (as of 2026-05-26). Saved `latest.pt` / `modelN.pt` now include the EMA
  shadow under `_ema_shadow__` keys, so standalone eval automatically uses
  EMA weights matching in-training scores. Old checkpoints (pre-fix, no
  EMA keys) still load fine but fall back to online weights.
- **Default (both knobs `None`)** = robosuite's built-in σ=0.02 gaussian
  on joint qpos (≈22 mm EE std) — same distribution MimicGen used to
  collect the demos, so eval starts in-distribution. To get fully
  deterministic eval (= old default), pin `hand_init_perturb: [0.0,0.0,0.0]`
  in YAML or pass `--hand_init_perturb 0,0,0` to evaluate/eval.py. To use
  EE-space IK perturb (sgft-style, e.g. `±15mm × ±15mm × ±10mm` box), set
  `[0.015, 0.015, 0.01]`. To widen/narrow the joint-space gaussian instead
  (keep distribution *shape*, just rescale), set
  `hand_init_perturb_joint_sigma: 0.05` (etc.) — mutually exclusive with
  `hand_init_perturb`.
- For BC: training never touches the env, so `dataset.hand_init_perturb` and
  `eval_hand_init_perturb` are both just "eval env construction" knobs. The
  split exists so the dataset cfg can carry a "demo-collection-style"
  default while you override with a different perturb at eval. RL is
  different — `train_rl.py` also has analogous fields (TODO: verify).

### TODO
- [ ] Smoke-test the post-training dual eval on a real trained checkpoint
      (currently only unit tests pass; haven't verified the full source/target
      gap reporting end-to-end).
- [ ] Maybe add `--dual_eval` flag to the CLI so a single invocation runs
      both source and target back-to-back (currently requires two calls).
- [ ] Same eval_domain_shift / hand_init_perturb plumbing for `train_rl.py`
      already covered by §3; double-check it routes through correctly.

---

## 5b. MimicGen-aligned BC baselines (RNN-GMM + Diffusion)

### What
Two new BC heads added alongside Flow:
- **`StateRnnGmmPolicy`** (bc/rnn_gmm_policy.py) — MimicGen's published
  baseline. Open-loop LSTM unroll over Tp=10 steps + per-step K=5 GMM head,
  tanh-squashed means, low_noise_eval. Open-loop = LSTM gets the same
  encoded obs at every of its Tp input steps; lets it emit a full action
  chunk from a single observation, which slots into existing chunked eval.
- **`StateDiffusionPolicy`** (bc/diffusion_policy.py) — DDIM noise-prediction
  policy with switchable backbone (`cfg.backbone`):
  - `"unet1d"` (default): `ConditionalUnet1D` from robomimic (FiLM-cond
    1D U-Net, sinusoidal step embed, GroupNorm+Mish).
  - `"transformer"`: ported `TransformerForDiffusion` from
    real-stanford/diffusion_policy (bc/diffusion_transformer.py). GPT-style
    encoder-decoder, causal self-attn on action axis, time+obs tokens as
    cross-attention memory. Matches the diffusion_policy paper's lowdim
    transformer config (n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3).
  Both backbones share `diffusers.DDIMScheduler`, the obs encoder MLP,
  EMA, and the loss/sampling code.

### Where
- `bc/rnn_gmm_policy.py` (NEW, ~210 lines) — `StateRnnGmmPolicyConfig` +
  `StateRnnGmmPolicy`. Same public surface as `StateFlowPolicy`
  (`loss / act / act_chunk / update_ema / apply_ema / restore_from_ema`).
  Open-loop unroll: LSTM input = encoded obs repeated Tp times.
- `bc/diffusion_policy.py` — `StateDiffusionPolicyConfig` +
  `StateDiffusionPolicy`. Obs encoder is a small MLP (default 256→128).
  Two backbones dispatched via `_predict_noise` in init:
    - UNet1D default `down_dims=[64,128,256]` (~5M params; robomimic image
      default [256,512,1024]≈65M is oversized for state-only).
    - Transformer default n_layer=8/n_head=4/n_emb=256 (~9M params at
      Tp=16, action_dim=7).
- `bc/diffusion_transformer.py` (NEW) — `TransformerForDiffusion` port of
  real-stanford/diffusion_policy. Slimmed vs source: only
  `obs_as_cond=True, time_as_cond=True` mode kept (the recommended
  default); inpainting / encoder-only-BERT branches removed.
  `SinusoidalPosEmb` is inlined. Includes `get_optim_groups(weight_decay)`
  helper for GPT-style decay/no-decay AdamW param split.
- `train_bc.py`
  - Imports both classes; nests `rnn_policy` / `diffusion_policy` configs.
  - Adds `use_rnn` / `use_diffusion` flags; assert exactly ≤1 is set.
  - `_load_model` knows to reconstruct both new classes from saved cfg.
- `release/cfgs/mimicgen/threading_d0_state_bc_rnn.yaml` (NEW) —
  Tp=10, Ta=8, rnn_hidden_dim=400, num_modes=5, lr=1e-3 (MimicGen low_dim).
- `release/cfgs/mimicgen/threading_d0_state_diffusion_bc.yaml` —
  Tp=16, Ta=8, 10 DDIM inference steps, EMA decay 0.9999, lr=1e-4.
  Implicit `backbone="unet1d"` (default).
- `release/cfgs/mimicgen/threading_d0_state_diffusion_transformer_bc.yaml`
  (NEW) — Same chunk shape + scheduler, `backbone: "transformer"`,
  weight_decay=1e-3 (paper default for transformer body).

### Why open-loop RNN
robomimic's BC-RNN-GMM at eval propagates hidden state across env steps and
resets every `rnn_horizon` steps — closed-loop, stateful inference. To
integrate with ibrl's stateless chunked eval (single `act_chunk` call → Ta
actions, no per-proc state in the policy), we use **`rnn.open_loop=True`**
semantics: train and eval both feed a single observation, LSTM unrolls Tp
steps emitting a chunk. This is one of robomimic's own supported modes (see
`algo_config.rnn.open_loop`) and the trade-off for short Tp (≤16) is small.

### How
```bash
# MimicGen BC-RNN+GMM baseline (matches MimicGen paper config)
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_bc_rnn.yaml

# Diffusion Policy w/ UNet1D backbone (chunked, EMA, DDIM)
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_diffusion_bc.yaml

# Diffusion Policy w/ Transformer backbone (lowdim default from diffusion_policy paper)
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_diffusion_transformer_bc.yaml
```

### Training-loop polish (applies to all heads)
- `MainConfig.eval_period` (epochs) — gate the rollout-eval cadence.
  YAML default 50 matches MimicGen's `experiment.rollout.rate` low_dim
  value. When skipped, `latest.pt` is still refreshed but `topk` is not
  touched (`saver.save(perf=None)`).
- `MainConfig.val_period` + `dataset.val_ratio` — log `val/loss` every N
  epochs over a held-out episode split. val_ratio=0 disables (legacy
  behavior). Episode-level split, fixed RNG seed (`val_split_seed=0` by
  default) so it's stable across runs.
- Default `dataset.eval_episode_len` bumped 300→400 to match MimicGen's
  Threading `rollout.horizon` (old 300 truncated long-tail demos).
- All three new YAMLs use `num_epoch=2000, epoch_len=100` (= 200K grad
  steps, matches MimicGen low_dim) for finer log granularity vs the
  previous 40×10000.

### TODO
- [ ] Closed-loop RNN-GMM variant (per-proc hidden state, requires eval
      framework extension). The open-loop version is the immediate
      apples-to-apples comparison to MimicGen for short Tp.
- [ ] Image-mode variants of RNN-GMM and Diffusion (obs encoder = ResNet18
      + SpatialSoftmax, matches MimicGen image config).

---

## 5c. Strict success gating (anti reward-hack, Threading_D0 only)

### What
Threading's raw success (`threading.py:_check_success`) is an **unguarded,
instantaneous** 3D position test — `‖geom_xpos["needle_obj_needle"] −
mean(ring geom centers)‖ < 12 mm` — with no check that the needle threaded
the hole cleanly. A learned policy reward-hacks it two ways:
1. **Ram / clip-through / fly-through**: a single control step landing the
   needle *center* inside the 12 mm ball counts (and with `end_on_success` the
   episode ends there). At 20 Hz a needle at 0.5 m/s moves 25 mm/step ≈ the
   ball diameter, so a wild swing crosses it between checks.
2. **Shove the ring**: the tripod is a **free body of only ~18 g**, and success
   is the *relative* distance, so ramming the lightweight ring toward the
   needle satisfies the test — this is the "硬捅 / 捅出屏幕 but success" failure.

`StrictSuccessConfig` layers validity gates on top, computed in the ibrl
wrapper (we do NOT edit the vendored mimicgen submodule). A frame counts only
when the raw test passes AND the gates pass; success is declared only after the
valid condition holds `dwell_steps` consecutive control steps. **When enabled,
the validated signal drives BOTH the returned `success` flag AND the per-step
reward**, so RL trains against the honest objective (not just measured honestly
at eval).

### Where
- `env/strict_success.py` (NEW) — `StrictSuccessConfig` + stateful
  `SuccessValidator` (holds the consecutive-valid streak + post-reset
  ring-center snapshot; `reset(env, bindings)` / `step(env, bindings,
  raw_success) -> bool`).
- `env/domain_shift.py` `resolve_threading_bindings()` — added `needle_joint`
  (`needle_obj_joint0`) for the optional speed gate; reuses `ring_geom_ids`.
- `env/robosuite_wrapper.py`
  - `__init__`: `strict_success` kwarg; builds `self._success_validator` when
    `enabled`; resolves bindings if shift **or** strict is active.
  - `reset()`: re-resolves bindings + snapshots ring center **after** any
    domain-shift `sim.forward()` (so a shifted `hole_offset` is part of the
    start pose, not counted as drift).
  - `step()`: `raw_success = step_reward==1`; if validator active,
    `frame_success = validator.step(...)` and `frame_reward = 1.0 if
    frame_success else 0.0`, replacing the raw sparse reward. Disabled →
    byte-for-byte upstream behavior.
- `train_rl.py`: `MainConfig.strict_success` (single field, applied to BOTH
  train and eval envs so reward + metric agree).
- `train_bc.py`: `MainConfig.strict_success` + `evaluate(..., strict_success=)`
  threaded into all 4 eval call sites (BC training never touches env, so this
  only affects eval scoring).
- `evaluate/eval.py`: CLI `--strict_success`, `--strict_dwell_steps`,
  `--strict_max_tripod_disp`, `--strict_max_needle_speed`.

### Gates (chosen design: dwell + tripod; speed optional/off)
| Gate | Config field | Reference | Kills |
|---|---|---|---|
| dwell | `dwell_steps` | **3** (@20 Hz = 0.15 s) | transient clip-through / fly-through |
| ring unmoved | `max_tripod_disp` | **0.04 m** | shove-the-ring + off-screen successes |
| needle speed | `max_needle_speed` | **0.0 = off** | high-momentum ramming (opt-in) |

`max_tripod_disp` is deliberately generous (4 cm, bumped up from 2 cm per
request) so small legitimate contact nudges aren't rejected. **WARNING:** too
high re-admits the shove hack (the cheat only needs to move the ring as far as
the needle is away), so don't go much past the workspace scale.

### How
```yaml
# in any threading_d0 YAML (RL or BC), enable + tune:
strict_success:
  enabled: true
  dwell_steps: 3
  max_tripod_disp: 0.04
  max_needle_speed: 0.0   # set e.g. 0.5 to also reject ramming
```
```bash
# re-score a saved checkpoint honestly without retraining:
python evaluate/eval.py --weight exps/.../latest.pt --mode bc --num_game 100 \
    --strict_success 1 --strict_dwell_steps 3 --strict_max_tripod_disp 0.04
# stack with the target shift:
python evaluate/eval.py --weight exps/.../latest.pt --mode rl --num_game 100 \
    --shift_from_cfg 1 --strict_success 1 --strict_max_tripod_disp 0.04
```

### Design choices
- **Default OFF** (`enabled=False`) → existing configs + trained-checkpoint
  evals are unchanged and remain comparable. Opt in per experiment.
- **Single field applied to train+eval** (unlike domain_shift's train/eval
  split) — the honest objective should be identical in both, else the reward
  and the metric disagree.
- **Reward becomes sparser when enabled** (dwell delays the +1 until the
  needle holds K steps). Intended: that's why the hack disappears. For RL this
  changes the MDP; for BC it only changes eval scoring.
- **Snapshot ring center post-shift** so the displacement gate measures drift
  from the true (possibly hole-offset-shifted) start, not from nominal.

### Verification status
Smoke-tested 2026-05-27: ✅ validator unit tests (dwell streak, reset-on-fail,
tripod-disp gate at 0.02/0.1 m, speed gate); ✅ wrapper integration (validator
shares shift bindings, ring snapshot reflects shifted hole, dwell gate fires on
forced raw-success at step K + terminates); ✅ default-off path unchanged;
✅ pyrallis parses the nested field in both train_rl / train_bc.
NOT yet verified: end-to-end effect on a *real trained policy*'s success rate
(need a checkpoint to confirm the hack-rate drop).

### TODO
- [ ] Run a known hacking checkpoint under `enabled=true` to quantify how much
      the "honest" success rate drops vs raw.
- [ ] Optional orientation/grasp gate (needle long-axis aligned with hole +
      still grasped) — deferred; dwell+tripod cover the observed failures.
- [ ] Generalize bindings (`ring_geom_ids` / `needle_joint`) beyond Threading.

---

## 6. Quick configuration reference

| YAML | Task | Algo | Train env | Eval env |
|---|---|---|---|---|
| `release/cfgs/mimicgen/stack_d0_state_bc.yaml` | Stack_D0 | BC | n/a | source |
| `release/cfgs/mimicgen/stack_d0_state_rlpd.yaml` | Stack_D0 | RLPD | source | source |
| `release/cfgs/mimicgen/threading_d0_state_flow_bc.yaml` | Threading_D0 | Flow BC | n/a | source + **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_bc_rnn.yaml` | Threading_D0 | BC-RNN+GMM (open-loop) | n/a | source + **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_diffusion_bc.yaml` | Threading_D0 | Diffusion (DDIM, UNet1D) | n/a | source + **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_diffusion_transformer_bc.yaml` | Threading_D0 | Diffusion (DDIM, Transformer) | n/a | source + **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_naive.yaml` | Threading_D0 | RLPD | source | **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_dr.yaml` | Threading_D0 | RLPD | DR(source) | **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_dr_oracle.yaml` | Threading_D0 | RLPD | DR_oracle(brackets new-target) | **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_ibrl_shifted.yaml` | Threading_D0 | IBRL | **new-target** | **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_ibrl_newtarget.yaml` | Threading_D0 | IBRL (warm-started from source ckpt) | **new-target** | **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_sgft.yaml` | Threading_D0 | SGFT-on-IBRL (needs source agent) | **new-target** | **new-target** |
| `release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml` | Threading_D0 | Co-train Flow BC (95% src + 5% tgt) | n/a | source+**new-target** ⚠️ demos still match OLD target |
| `release/cfgs/mimicgen/threading_d0_state_h2oplus.yaml` | Threading_D0 | H2O+ on IBRL (§14) | **source** (online "sim") | **new-target** (offline "real" demos + eval) |
| `release/cfgs/mimicgen/threading_d0_state_lwail.yaml` | Threading_D0 | LWAIL-cross (§18: ICVF latent + WGAN-GP state-occupancy) | **source** (online explore) | **new-target** (demos' expert states + eval) |
| `release/cfgs/mimicgen/threading_d0_state_rebrac.yaml` | Threading_D0 | ReBRAC offline TD3+BC (§19; MLP-BC actor) | n/a (offline) | **new-target** (20 target demos + eval) |
| `release/cfgs/mimicgen/threading_d0_state_rebrac_weighted.yaml` | Threading_D0 | Gap-weighted ReBRAC (§19; vs weighted BC) | n/a (offline) | **new-target** (gap-weighted source + 20 target + eval) |
| `release/cfgs/mimicgen/threading_d0_state_tdmpc2_source.yaml` | Threading_D0 | Online TD-MPC2 + IBRL prior (§28; source MLP-BC prior) | source (online) | source |
| `release/cfgs/mimicgen/threading_d0_state_tdmpc2_target.yaml` | Threading_D0 | Online TD-MPC2 + IBRL prior (§28; 20-demo target BC prior) | **new-target** (online) | **new-target** |

**new-target** = the dynamics-only target (see §12 — friction 0.5, mass 3.0,
solref/solimp 1.0, gripper_kp 0.12, hole_offset [0,0,0], ring_size 0.85,
ring_friction 2.0, mask=false). Migrated 2026-05-28.

All assume `dataset/processed_data96.hdf5` for Threading and Stack-specific
paths in `release/data/` for Stack. **All current demos are source-domain**
(MimicGen "core" datasets are generated under nominal env).

---

## 7. Verification status

Smoke-tested 2026-05-25:
- ✅ MimicGen env registration for all 10 D0 tasks
- ✅ `PixelRobosuite + domain_shift` Threading: mutation visible, persistent
       across `hard_reset`, hole offset effective, obs blinding works, DR
       reproducible across seeds
- ✅ Flow policy Tp/Ta split, EMA shadow/apply/restore, action chunking,
       end-to-end loss + backward
- ✅ Multi-proc eval helpers: chunked + non-chunked fallback paths
- ✅ Queue dynamics (is_first clears stale)

NOT yet end-to-end verified:
- ⚠️ Multi-process eval with real `mp.Process` workers and a trained
      checkpoint (only the helper functions and queue simulation pass)
- ⚠️ Full training run on real Threading data (file unavailable)
- ⚠️ Domain-shift defaults' effect on BC success rate (need a checkpoint to
      ablate; suspect `friction_scale=0.3` is too aggressive)

---

## 8. Open issues / known sharp edges

- **`dataset/processed_data96.hdf5` missing** (was 9.2 GB, gone as of
  2026-05-25). All Threading YAMLs reference this path — fix paths or
  re-download.
- **`pip` shebang bug** in `~/.conda/envs/ibrl/bin/pip` — always use
  `python -m pip install ...`. See §1.
- **`metaworld_wrapper.py` will fail to import** because we deliberately
  skipped `mujoco-py 2.1 / metaworld`. Only matters if you try to use
  MetaWorld envs.
- ~~**EMA not in `state_dict`**~~ — **fixed 2026-05-26**. EMA shadow is
  now persisted via state_dict overrides; old ckpts still loadable.
- **Domain shift Threading-only** — `resolve_threading_bindings` raises
  `NotImplementedError` for other env_names.
- **`set_env.sh` line 4** calls `conda activate ibrl` but doesn't source
  conda profile; needs `~/.bashrc` to have conda init, OR
  `source /home/Chenyang/anaconda3/etc/profile.d/conda.sh` first.

---

## 9. Roadmap / future work

Use this section as a parking lot for planned changes. When you start a new
feature, add a section above (§10, §11, ...) following the
What / Where / How / TODO template.

### Near-term (high signal-to-effort)
- [ ] Wire BC eval on target (see §4 TODO)
- [ ] Empirical sweep of `friction_scale` to find Threading-safe default
- [ ] Generalize domain shift to Stack_D0 / Square_D0 (simpler tasks =
      cleaner gap measurement)

### Medium-term
- [ ] Pixel-mode flow BC (add image encoder, re-render datasets at 96×96)
- [ ] UNet1D backbone option for flow policy (if Tp grows past 16)
- [ ] Resumable EMA (register as buffer)
- [ ] DR + obs/action noise (the third sgft mechanism, currently skipped)

### Long-term / exploratory
- [ ] Threading D1 / D2 variants (broader initial-state distribution)
- [ ] Target-domain demo generation (run MimicGen pipeline against shifted env)
- [ ] Cross-task generalization (multi-task flow policy)

### Reserved section template
When adding a new feature, copy this skeleton:

```
## N. <feature name>

### What
<one paragraph: what does this do and why>

### Where
- file:line  ...
- file:line  ...

### How
<command-line or code snippet to invoke>

### Design choices
- <bullet>
- <bullet>

### TODO
- [ ] <followup>
```

---

## 10. SGFT (value transfer) on top of IBRL — target-domain only

> ⚠️ **部分归档** —— 代码保留（`sgft_shaping`/`sgft_coef` 是 `rl/q_agent.py` 的开关、`train_rl.py` 的 `sgft_source_agent`），但 `threading_d0_state_sgft.yaml` 已归档到 `_archive/`（见 §34）。

### What
SGFT (Yin et al., ICLR 2025): pretrain a base agent on the source domain to get
a value function `V_src`, then on the target domain shape the reward with a
**potential-based** term so the source value guides exploration:
`r' = r + γ·V_src(s') − V_src(s)`. Potential-based ⇒ the optimal target policy
is unchanged; only exploration is guided. The sgft reference impl is TD-MPC2
(`/home/Chenyang/sgft`, `tdmpc2/trainer/online_trainer.py`), where `V_src` is a
learned value head. **Here it is realized on IBRL** (our actor-critic), where
the natural value is `V_src(s) = min-ensemble Q_src(s, π_src(s))`.

Per the task spec, only the **target-domain training** is implemented (the
source agent is just a normal `train_rl.py` run on the no-shift env). One frozen
source `QAgent` is reused for **both** roles:
1. **IBRL BC prior** — its deterministic RL policy becomes the prior the target
   actor competes with in IBRL max-Q (replaces a separately-trained BC policy).
2. **SGFT value source** — `V_src` for the critic's reward shaping.

### Where
- `rl/q_agent.py`
  - `QAgentConfig`: `sgft_shaping: int = 0`, `sgft_coef: float = 1.0`.
  - `QAgent.sgft_value_source: list` (list-holder, NOT a submodule → not in
    `state_dict`); `set_sgft_value_source(agent)`.
  - `QAgent.value(obs)` — batched `V(s)=min-ensemble Q(s, π(s))`; strips any
    caller `feat` and re-encodes pixel obs with its OWN encoder.
  - `update_critic`: when `sgft_shaping`, replaces the TD-target reward with
    `r + sgft_coef·(discount·V_src(s') − V_src(s))`. Telescopes correctly over
    the n-step transition because `discount` here is the batch bootstrap
    (γⁿ, or 0 at terminal so `V_src(s')` drops out). Logs `sgft/shaping`,
    `sgft/v_source`.
- `train_rl.py`
  - `MainConfig.sgft_source_agent: str = ""` (path to source `QAgent` ckpt);
    `__post_init__` normalizes "None"→"" and **asserts** `bc_policy` and
    `sgft_source_agent` are not both set (IBRL takes exactly one prior).
  - `FrozenAgentBCPolicy` — adapts a frozen `QAgent` to the BC-prior interface
    `_act_ibrl` expects (`.act(obs, …) → [B, action_dim]`, deterministic RL
    policy, strips `feat`, held in a list so not a submodule).
  - `load_frozen_agent(weight, env, cfg, device)` — rebuilds the source agent
    from its own `cfg.yaml` arch + weights, sets `act_method="rl"`, freezes.
  - `Workspace.__init__`: when `sgft_source_agent` set → load it, `add_bc_policy`
    (the prior) + `set_sgft_value_source` (the shaping) + warn if act_method≠ibrl.
  - `load_model`: re-adds the source agent as the BC prior for standalone eval
    (the value source is training-only, not needed at eval).
- `release/cfgs/mimicgen/threading_d0_state_sgft.yaml` — demo config.

### How
Convenience scripts (preferred): `scripts/train_rl_sgft_threading.sh` (training,
with the full 2-phase workflow + eval usage in its header) and
`scripts/eval_rl_threading.sh` (runnable target/source/strict eval helper).
```bash
# 0) source agent (no shift): produces exps/mimicgen/threading_rl_source_seed1/latest.pt
bash scripts/train_rl_source_threading.sh
# 1) SGFT-on-IBRL fine-tune on the target (shifted) env:
bash scripts/train_rl_sgft_threading.sh           # SOURCE_AGENT defaults to the Phase-0 path
SGFT_SHAPING=0 bash scripts/train_rl_sgft_threading.sh   # ablation: IBRL + source prior, no shaping
# 2) eval the trained ckpt (target + source, strict-success gated):
DOMAIN=both bash scripts/eval_rl_threading.sh exps/mimicgen/threading_sgft_seed1/latest.pt

# Equivalent raw commands:
python train_rl.py --config_path release/cfgs/mimicgen/threading_d0_state_sgft.yaml \
    --sgft_source_agent exps/mimicgen/threading_rl_source_seed1/latest.pt \
    --save_dir exps/mimicgen/threading_sgft_seed1
python evaluate/eval.py --weight exps/mimicgen/threading_sgft_seed1/latest.pt \
    --mode rl --num_game 100 --shift_from_cfg 1 --strict_success 1
```

### Design choices
- **`V_src(s) = min-ensemble Q_src(s, π_src(s))`** — the actor-critic analog of
  sgft's learned value head; uses the source's online critic + deterministic
  actor. With sparse reward + `end_on_success`, `V_src ∈ [0,1]` (≈ γ^steps-to-
  success), so the shaping term is naturally bounded.
- **Shaping in `update_critic`, not at buffer-add** — keeps the replay storing
  TRUE task rewards (eval/success logging unaffected; shaping toggleable without
  re-collecting). Computing `V_src` on the fly is one cheap forward.
- **Same agent for BC prior + value** — matches the task spec ("source RL agent
  as the BC base") and means one checkpoint configures the whole method.
- **Source frozen, held in a list** — never trained, never in the target's
  `state_dict` (so target ckpts stay small and the source is re-loaded by path
  at eval — exactly like the existing `bc_policy` prior).
- **Ablations from one config**: `q_agent.sgft_shaping=0` → plain IBRL with the
  source as prior (isolates the value-transfer term); `sgft_source_agent=""` +
  `bc_policy=…` → vanilla IBRL.
- **Obs space MUST match source** (gotcha): the frozen source consumes this
  run's obs, so `use_state / state_stack / obs_stack / prop_stack` must equal
  the source's — `load_frozen_agent` asserts this with a clear error (e.g.
  "set --state_stack N") instead of a cryptic size-mismatch. The existing
  source agents use `state_stack=1`, so `threading_d0_state_sgft.yaml` is set
  to 1 (NOT the 3 used by the other RL configs). `q_agent` *architecture*
  (hidden dims, num_q) may still differ between source and target — only the
  obs space and action dim must line up.

### Verification status
Real source ckpt (threading_ibrl_source_seed1, state_stack=1): ✅ loads + frozen
+ act_method=rl + value() works; ✅ obs-space mismatch (state_stack 1 vs 3)
raises the clear ValueError. Synthetic state agents: ✅ source wired
as BC prior (`len(bc_policies)==1`) + value source, NOT in target `state_dict`;
✅ `value(obs)` shape; ✅ `act` (ibrl) max-Q over {target, source} shape;
✅ `update` runs with shaping, logs `sgft/shaping` + `sgft/v_source`;
✅ `sgft_shaping=0` ablation drops the shaping; ✅ YAML parses; ✅ bc_policy ⊕
sgft_source_agent mutex assert. NOT verified: a real source→target run's success
curve (needs a trained source ckpt + the Threading dataset).

### TODO
- [ ] Real source→target run once a source ckpt exists; confirm SGFT > vanilla
      IBRL > naive on target success.
- [ ] Sweep `sgft_coef` (shaping strength) — 1.0 is a starting point.
- [ ] Optionally also warm-start the target actor/critic from the source
      (`load_pretrained_agent`) and compare vs. shaping-only.
- [ ] Pixel-mode: `value()` re-encodes from raw frames (no aug) — works but
      untested; the source encoder must match the obs the target collects.

---

## 11. IL co-training (source-heavy + a little target) for BC

> ⚠️ **部分归档** —— 代码保留（`bc/dataset.py::CoTrainDataset`），但 `*_cotrain_flow_bc*.yaml` 已归档到 `_archive/`（见 §34）。

### What
Train a BC policy (flow by default) on a LOT of source data + a LITTLE target
data, mixed at a fixed PER-BATCH ratio. Default 95% source / 5% target =
380 source + 20 target episodes. The fixed per-batch ratio (not pool-then-
uniform sampling) keeps the 95/5 split exact regardless of the two sets'
episode lengths/counts — the standard co-training knob.

### Where
- `bc/dataset.py`
  - `DatasetConfig`: `cotrain_path` (target hdf5; non-empty ⇒ co-train),
    `cotrain_num_data` (target episode cap), `cotrain_ratio` (per-batch source
    fraction, default 0.95). `path`/`num_data` are the source.
  - `env_config`: falls back to an `env_cfg.json` **sidecar** when the hdf5 has
    no `env_args` attr (the `collect_target_demos.py` format), and marks it
    `_sidecar` to skip the strict controller-cfg check.
  - `RobomimicDataset.__init__(cfg, build_env=True)`: `build_env=False` skips
    the eval-env build (co-train sub-datasets share one parent env); prefers a
    precomputed `obs/state` when present (sidecar format) else builds from
    `STATE_KEYS`; clamps `ctrl_delta` actions to [-1,1] (policy-collected demos
    can slightly exceed it; the OSC controller clips anyway, so the executed/BC
    target action IS the clipped one — no-op for in-range MimicGen demos).
  - `CoTrainDataset` — loads source + target as two `RobomimicDataset`s
    (`build_env=False`), builds one shared eval env, and `sample_bc` draws
    `round(B·ratio)` from source + the rest from target, merged via
    `_merge_batches`. Same surface train_bc needs (`sample_bc`/`sample_val`/
    `has_val`/`env_params`/`obs_shape`/`prop_shape`/`action_dim`/`cfg`).
- `train_bc.py` `run()`: `if cfg.dataset.cotrain_path: dataset = CoTrainDataset(...)`
  else `RobomimicDataset(...)`. Everything downstream (flow/diffusion/rnn,
  val, dual eval) is unchanged.
- `release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml` — demo config
  (source=threading_d0_source_sigma0.1, target=threading_d0_target_hole0.01,
  380/20, ratio 0.95; target eval shift matches the target demos' env_cfg.json:
  hole_offset=0.01, mask_hole_offset_in_obs=false).
- `scripts/train_cotrain_flow_bc_threading.sh` — runnable wrapper (RATIO/N_SRC/
  N_TGT env vars + eval usage in the header).

### How
```bash
bash scripts/train_cotrain_flow_bc_threading.sh                 # 95/5, 380+20
RATIO=0.90 N_TGT=40 bash scripts/train_cotrain_flow_bc_threading.sh   # 90/10
RATIO=1.00 METHOD=source_only bash scripts/train_cotrain_flow_bc_threading.sh  # source-only ablation
# raw:
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml
```
The two data dirs use the `collect_target_demos.py` format (env_cfg.json
sidecar + precomputed obs/state); the loader auto-detects it. Co-training is
head-agnostic — point the same `cotrain_*` fields at a diffusion/rnn config too.

### Design choices
- **Fixed per-batch ratio** (not pooled uniform): faithful to "95% / 5%
  mixing" even though source demos (~62 steps) and target (~66 steps) differ;
  matches the co-training-ratio knob used in the literature.
- **Two sub-datasets, one env**: reuses the full entry-building path
  (action_chunk padding, stacking, val split) for free; only one eval env is
  built (eval-time domain shift is injected by train_bc, not baked into data).
- **`val_ratio: 0.0` in the demo config** so all 380+20 episodes are used for
  training (the requested mix); set >0 for held-out val/loss tracking.
- **Action clamp**: the collected demos had ~1% of actions just outside
  [-1,1]; clamping makes the BC target equal the env-executed action.

### Verification status
Smoke-tested 2026-05-27: ✅ sidecar (env_cfg.json) load + precomputed obs/state;
✅ 380 source / 20 target episodes; ✅ `_split(256)=(243,13)` (95/5); ✅ merged
batch has flow `action_chunk` [256,16,7], actions clamped ≤1; ✅ 5 flow grad
steps on the mixed batch (loss finite + decreasing); ✅ YAML parses + branches
to CoTrainDataset. NOT verified: a full training run's source-vs-target success
(needs the eval rollouts).

### TODO
- [ ] Sweep the mix (`cotrain_ratio` / `cotrain_num_data`): source-only →
      95/5 → 90/10 → target-only, vs the SGFT (§10) and DR baselines.
- [ ] Co-train the other heads (diffusion/rnn) for an apples-to-apples table.
- [ ] **Re-collect target demos under the new dynamics-only target** (§12).
      The existing `threading_d0_target_hole0.01` demos were collected under
      the OLD shift (hole_offset=0.01, friction_scale=0.3, ...); the cotrain
      YAML's eval was migrated to the new target on 2026-05-28 but
      `cotrain_path` still points at the old-target demos, so train/eval are
      mismatched until new demos are collected via
      `scripts/collect_target_demos.py` with the §12 shift values.

---

## 12. Active target = "dynamics-only" (migration 2026-05-28)

### What
The Threading_D0 target_domain_shift was migrated from the original
offset-bearing recipe to a "dynamics-only" recipe (no `hole_offset`, with
ring-geometry shifts instead). Per §3's gap-taxonomy discussion, the original
recipe relied on a masked `hole_offset` to manufacture a "dynamics gap in
observation space"; the new recipe removes that latent registration term and
keeps the gap purely in contact/friction/geometry physics (true `P(s'|s,a)`
shifts even with full observability).

### Why
Validated 2026-05-27 against the frozen source IBRL policy: the dynamics-only
target produces **13% strict success** (down from 98% on source) — a sharp,
honest sim-to-sim gap. The failure split is ~25% grasp/transport, ~75%
align/insert, and the grasp phase is FEASIBLE (friction-cone margin ~10×, so
residual grasp failures are learnable approach-pose errors). See
`scripts/diag_failure_modes.py` and `scripts/diag_grasp_feasibility.py`.

Conceptual win: the gap no longer requires an "unobserved kinematic offset"
framing (the §3 hole_offset POMDP argument). With `hole_offset=0` the
transition `P(s'|s,a)` itself shifts (mass, friction, ring geometry), so the
gap is a clean dynamics gap even in the full-state MDP — easier to defend
and a stronger setting for value-transfer methods.

### Old vs new shift values

| Param | Old target | **New target** | Notes |
|---|---:|---:|---|
| `friction_scale` | 0.3 | **0.5** | needle handle μ 1.0→0.5 (was →0.3); tip 0.3→0.15 |
| `mass_scale` | 3.0 | **3.0** | unchanged — needle 7.6 g → 22.8 g |
| `solref_scale` | 0.8 | **1.0** | pinned (clip-through-safe; was mildly stiffer) |
| `solimp_scale` | 1.0 | **1.0** | pinned (avoid penetration) |
| `gripper_kp_scale` | 0.5 | **0.12** | finger servo kp 1000 → 120; much softer grasp |
| `hole_offset` | `[0.003, 0.003, 0]` (4.2 mm) | **`[0, 0, 0]`** | removed — gap is pure dynamics now |
| `mask_hole_offset_in_obs` | `true` | **`false`** | no offset → nothing to mask |
| `ring_size_scale` | (not set → 1.0) | **0.85** | hole 16 mm → 13.6 mm; needle is 10 mm |
| `ring_friction_scale` | (not set → 1.0) | **2.0** | ring rim μ 0.3 → 0.6; grippier insert |

Architecture impact: the `newtarget.yaml` config mirrors
`threading_ibrl_source_seed1` exactly (`state_stack=1`, q_agent dims) for
warm-start compatibility; the other migrated YAMLs keep their original arch
(some still have `state_stack=3`) — only the shift block changed.

### Where (configs touched on 2026-05-28)
All eval (and where applicable, train) shift blocks updated to the new
recipe:
- `threading_d0_state_ibrl_shifted.yaml` — train + eval
- `threading_d0_state_naive.yaml` — eval
- `threading_d0_state_dr.yaml` — eval (DR ranges unchanged — still blind)
- `threading_d0_state_dr_oracle.yaml` — eval + DR ranges re-bracketed:
  `dr_friction_range [0.4,1.2]`, `dr_gripper_kp_range [0.1,1.2]`,
  `dr_ring_size_range [0.8,1.0]`, `dr_ring_friction_range [1.0,2.5]`;
  `dr_hole_offset_range` removed (target has no offset)
- `threading_d0_state_sgft.yaml` — train + eval
- `threading_d0_state_flow_bc.yaml` — eval
- `threading_d0_state_diffusion_bc.yaml` — eval
- `threading_d0_state_diffusion_transformer_bc.yaml` — eval
- `threading_d0_state_bc_rnn.yaml` — eval
- `threading_d0_state_cotrain_flow_bc.yaml` — eval **only** (the
  `cotrain_path` demo set still matches the OLD shift — see §11 TODO)

NOT touched: `threading_d0_state_ibrl_source_handperturb.yaml` (source-only,
no shift), `threading_d0_state_ibrl_newtarget.yaml` /
`threading_d0_state_ibrl_newtarget2.yaml` (already the reference recipe),
`stack_d0_*` (different task).

### How
Same as before — pass the YAML path to `train_rl.py` / `train_bc.py` /
`evaluate/eval.py`. To reproduce the old offset-bearing target for an
ablation, set the shift block back manually or use
`--shift_hole_offset 0.003,0.003,0 --shift_friction_scale 0.3 ...` on
`evaluate/eval.py`. Old checkpoints trained under the old target still
load fine (only the env shift changed; obs/action spaces are identical).

### TODO
- [ ] Re-collect target demos via `scripts/collect_target_demos.py` under
      the new shift (for cotrain — §11 TODO).
- [ ] Sweep the new target's `ring_friction_scale` (1.5 vs 2.0 — the
      `newtarget2.yaml` uses 1.5; keep the more aggressive 2.0 unless
      training stalls).
- [ ] Update `release/cfgs/mimicgen/stack_d0_*` and other tasks once their
      target recipes are designed (Threading is the only one so far).

---

## 13. Classifier-Free Guidance (CFG) on co-training flow BC

### What
The CFG arm of CFG-ADDA (Lei et al. 2026, "Science of Co-Training") wired into
`StateFlowPolicy` + `CoTrainDataset`. ADDA itself is intentionally *not*
implemented — for state-based obs (no visual nuisance variation, source/target
share identical 37-dim obs space) the alignment side of CFG-ADDA has no signal
to operate on, and Table 1 of the paper shows adding the discriminator on top
of already-aligned representations *hurts*. CFG (the discernibility side)
remains directly applicable: explicitly condition the score on a one-hot
domain label, dropout to ∅ during training, interpolate at inference.

Default **OFF** (`flow_policy.cfg_enabled: 0`) so existing cotrain runs and
non-cotrain flow BC are byte-for-byte unchanged.

### Where
- `bc/flow_policy.py`
  - `StateFlowPolicyConfig`: `cfg_enabled` (default 0), `cfg_num_domains=2`,
    `cfg_dropout_p=0.2`, `cfg_guidance_lambda=-0.5`, `cfg_inference_domain_id=1`.
  - `_build_unet1d_backbone` / `_build_mlp_backbone` bump the conditioning
    input dim by `cfg_num_domains` when enabled (UNet1D global_cond, MLP
    concat at first layer).
  - `forward_velocity(x_t, t, state, domain_onehot=None)` — `None` ⇒ ∅
    (zeros) ⇒ the unconditional branch.
  - `loss(batch)` reads `batch.obs["domain_id"]`, one-hots it, masks rows to
    ∅ with prob `cfg_dropout_p`. Asserts if CFG is on but the field is
    missing (CFG needs CoTrainDataset).
  - `_sample_full_chunk(state)` — at each Euler step, two forwards
    (conditional on `cfg_inference_domain_id` and ∅), interpolate
    `v = (1+λ)·v_c − λ·v_∅`. λ<0 transfers "averaged across domains"
    knowledge (paper default −0.5); λ=0 reduces to standard conditioned
    inference; λ>0 amplifies domain-specific score (the classical CFG
    direction, paper says not useful here).
- `bc/dataset.py` `CoTrainDataset._tag_domain` — helper that injects a
  `domain_id` LongTensor into `batch.obs` for source (0) / target (1).
  `sample_bc` / `sample_val` use it on each sub-batch. Other policies index
  `batch.obs` by name, so the new key is a no-op for them.
- `train_bc.py` — when `use_flow=1` and `cfg_enabled=1`, asserts
  `dataset.cotrain_path` is set (CFG without two domains is meaningless).
- `release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml` —
  exposes the 5 CFG knobs at the end of the `flow_policy:` block; all set
  to OFF / paper-default values.
- `scripts/test_flow_cfg.py` (NEW) — smoke test: loss + backward + act_chunk
  on both backbones × {disabled, enabled}, the "missing domain_id ⇒
  assertion" path, param-count growth check, and CoTrainDataset tagging.

### How
```bash
# enable CFG on top of an existing cotrain run (CLI overrides)
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml \
    --flow_policy.cfg_enabled 1
# tweak guidance strength (negative = less per-domain specificity)
python train_bc.py ... --flow_policy.cfg_enabled 1 --flow_policy.cfg_guidance_lambda -0.5
# inference-time domain switch (eval as if in source — for ablation)
python train_bc.py ... --flow_policy.cfg_enabled 1 --flow_policy.cfg_inference_domain_id 0
```

### Design choices
- **One-hot concat at the encoder output**, not at the network input. Matches
  the paper ("concatenating a one-hot embedding c as environment labels to
  the observation features after the vision encoder"). Implementation: for
  UNet1D, append to `global_cond` (FiLM-conditioning input); for MLP,
  append to the first concat alongside `s_feat`, `a_feat`, `t_feat`.
- **Dropout to ∅ is per-sample, per-step.** A given sample is either fully
  conditional or fully unconditional in a given forward; we never partial-
  mask the one-hot. Mirrors standard CFG training.
- **Inference doubles ODE cost** (2 forwards per Euler step). With our
  default `num_inference_steps=10` that's 20 forwards/chunk vs 10 — still
  cheap.
- **No ADDA**. See §13 What — state-based obs lack the visual nuisance
  variation that ADDA needs to be useful, and the paper itself documents
  alignment-only methods underperforming when discernibility is critical.
  Skip until we move to pixel mode (§4 TODO).
- **Default OFF**: keeps `latest.pt` ckpts of existing runs loadable
  (different architectures otherwise — global_cond_dim differs by 2).

### Verification status
Smoke-tested 2026-05-28 (`scripts/test_flow_cfg.py`):
- ✅ `cfg_enabled=0` path unchanged (loss + act_chunk + grad flow, both
      UNet1D + MLP backbones).
- ✅ `cfg_enabled=1` loss reads domain_id, dropout fires at p=0.5, loss
      finite, gradients flow.
- ✅ act_chunk runs the two-forward CFG interpolation, returns
      `(B, Ta, action_dim)` finite chunks.
- ✅ CFG-enabled UNet1D has +7168 params over CFG-off baseline (= the
      extra global_cond → first-FiLM-block weights).
- ✅ Assertion fires when `cfg_enabled=1` but batch lacks `domain_id`.
- ✅ `CoTrainDataset._tag_domain` correctly tags source=0 / target=1.
- ✅ YAML parses with the new fields; defaults are off / paper values.
NOT yet verified: a real training run's source-vs-target gap with CFG on
vs off (needs the rollouts).

### TODO
- [ ] Run CFG-on vs CFG-off cotrain end-to-end and compare source/target
      success + the gap.
- [ ] Sweep `cfg_guidance_lambda` in (-2, +2) on a single mixing ratio,
      mirror paper Fig. 10 (expect U-shape with valley around λ≈-0.5).
- [ ] If we ever go to pixels: revisit ADDA — concat one-hot at encoder
      output and add a 3-layer MLP discriminator with gradient reversal
      (CFG-ADDA full method, paper Algorithm 1 lines 5–7). State-based has
      no signal for it.
- [ ] Optional: port CFG to `StateDiffusionPolicy` / `StateRnnGmmPolicy`
      for head-side ablations. Mechanically identical (concat one-hot to
      obs_feat, dropout, interpolate two score forwards at inference).

---

## 14. H2O+ on IBRL (hybrid offline-and-online RL with a dynamics gap)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
Port of H2O+ (Niu et al. 2025, *An Improved Framework for Hybrid
Offline-and-Online RL with Dynamics Gaps*; reference impl at
`/home/Chenyang/H2Oplus`) onto this repo's IBRL `QAgent`. H2O+ combines a LOT
of online data from a queryable simulator that has the WRONG dynamics with a
FEW offline demos from the deployment domain, and corrects the simulator's
dynamics gap by importance-weighting its Bellman error with a dynamics density
ratio. We keep the H2O+ scaffolding but swap its SAC backbone for IBRL.

**Mapping (sim2sim) — this is the crux:**
- H2O+ **"sim"** (online, queryable, biased dynamics) = our **SOURCE** domain
  (nominal, `train_domain_shift.mode=none`). The current policy is rolled out
  here to fill the online replay.
- H2O+ **"real"** (offline, fixed few demos, deployment/eval target) = our
  **TARGET** domain (dynamics-only: gripper_kp 0.12 / ring_size 0.85 /
  friction 0.5 / mass 3.0 / ring_friction 2.0, §12). Its data is the 20
  pre-collected demos at `release/data/mimicgen/threading_d0_target_dynamics`
  (state 37 / action 7 / sim_state 45); **no online interaction**, eval-only.
- the dynamics ratio `p_target(s'|s,a)/p_source(s'|s,a)` importance-weights the
  SOURCE/"sim" transitions' Bellman error.

This matches the sim2real framing in `[[project-dynamics-gap-detector]]`
(target = real world, fixed demos, not queryable; source = sim, queryable).

### Where
- `rl/h2oplus_agent.py` (NEW) — `H2OPlusAgent(QAgent)` + `H2OPlusConfig` + `FcV`.
  Inherits IBRL `_act_ibrl`/`_act_default`/critic/actor/BC-prior; adds:
  - `self.vf` (in-sample V, IQL quantile regression on TARGET demos only);
  - dynamics-ratio estimator chosen by `cfg.ratio_source`:
    - `"online"` (default): trains H2O `d_sa`/`d_sas`
      (`classifier_test/discriminator.py::DynamicsDiscriminator`, imported via
      importlib since `classifier_test/` has no `__init__`);
    - `"cf"`: loads the FROZEN counterfactual-paired discriminator
      `classifier_test/discriminator_cf.pt` (only queried);
    - `"none"`: ratio ≡ 1 (un-corrected mixed-buffer IBRL control);
  - `update_h2oplus(real_batch, sim_batch, stddev, update_actor)` — mirrors
    `SimpleSAC/h2oplus.py:train` order with the SAC actor replaced by IBRL:
    (1) train ratio (online); (2) blended explore(`r+γ·minQ_target(s',a')`,
    a' = IBRL max-Q over {target actor, source prior}) + exploit(`r+γ·V(s')`)
    backup with `exploit_coeff`; (3) critic loss = real (unweighted) + sim
    (weighted by `sqrt(clip(ratio))`); (4) in-sample V quantile loss on real;
    (5) IBRL DDPG actor loss on the mixed real+sim obs.
  Does NOT touch `QAgent.update`/`update_critic` (existing RL configs unaffected).
  STATE-ONLY (`use_state=1`).
- `train_h2oplus.py` (NEW) — adapted from `train_rl.py`; reuses its
  `load_frozen_agent` + `FrozenAgentBCPolicy` (SOURCE agent = IBRL prior),
  `rl/replay.py::ReplayBuffer`/`add_demos_to_replay`, `evaluate.run_eval(_mp)`.
  `train_env`=SOURCE (online rollout), `eval_env`=TARGET; `self.replay`=sim
  online, `self.real_replay`=target demos (frozen). `load_model()` for
  standalone eval.
- `evaluate/eval.py` — added `--mode h2oplus` branch (→ `train_h2oplus.load_model`).
- `release/cfgs/mimicgen/threading_d0_state_h2oplus.yaml` (NEW) — demo config.
- `scripts/train_h2oplus_threading.sh` (NEW) — `RATIO_SOURCE`/`SEED`/`SOURCE_AGENT`
  wrapper, 3-arm ablation usage in header.
- `scripts/test_h2oplus.py` (NEW) — synthetic-batch smoke test (3 ratio modes).

### How
```bash
source set_env.sh
bash scripts/train_h2oplus_threading.sh                      # online ratio (baseline)
RATIO_SOURCE=cf   bash scripts/train_h2oplus_threading.sh    # counterfactual ratio
RATIO_SOURCE=gt   bash scripts/train_h2oplus_threading.sh    # GROUND-TRUTH gap (auto nstep=1); vs weighted BC
RATIO_SOURCE=none bash scripts/train_h2oplus_threading.sh    # no correction (control)
# raw:
python train_h2oplus.py --config_path release/cfgs/mimicgen/threading_d0_state_h2oplus.yaml \
    --ibrl_source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
    --h2oplus.ratio_source online --save_dir exps/mimicgen/threading_h2oplus_online_seed1
# in-training eval runs on TARGET every log_per_step (TopkSaver keeps best-by-target).
# standalone re-eval:
python evaluate/eval.py --weight exps/.../latest.pt --mode h2oplus --num_game 100 --strict_success 1
# optional: BC warm-up the RL actor on TARGET demos first + also log SOURCE eval:
python train_h2oplus.py ... --pretrain_bc_epoch 3 --pretrain_bc_epoch_len 5000 --eval_source 1
# preferred warm start: copy the SOURCE agent's actor+critic into the trainable
# nets (~13% on target at step 0; the frozen source still serves as the prior):
python train_h2oplus.py ... --init_actor_critic_from_source 1
# via the script: INIT_FROM_SOURCE=1 RATIO_SOURCE=cf bash scripts/train_h2oplus_threading.sh
```
`state_stack` MUST be 1 (matches `threading_ibrl_source_seed1`; `load_frozen_agent`
asserts source/target obs-space equality).

**`score/score` is the TARGET eval** (the deployment metric); the agent only
*interacts* with SOURCE (online rollout), but is *evaluated* on the TARGET env
(`eval_env`, `eval_domain_shift`). `eval_source=1` additionally logs
`score/score_source` (nominal env) as a diagnostic of whether the actor learns
the source task at all.

**Optional BC warm-up (`pretrain_bc_epoch>0`)**: before online RL, the RL actor
is BC-trained on the TARGET demos (`real_replay`, via the inherited
`pretrain_actor_with_bc`), `actor_target` is hard-synced, then online H2O+
starts. Gives the actor a non-zero target success before it competes in IBRL
max-Q (otherwise it starts random and the hybrid leans on the weak source
prior). Off by default. Note: H2O+'s own AWR loss (`h2oplus.py:207-216`) lives
only in its `pretrain_steps` branch and is OFF by default (`warmup_steps=0`);
its main-phase actor is SAC max-Q over mixed obs — i.e. our IBRL DPG max-Q is
faithful to default H2O+, and the demos shape only Q/V, not the policy.

### Design choices
- **SAC→IBRL**: actor is DDPG-style (`-min Q(s,π(s))` on mixed obs), bootstrap
  uses IBRL max-Q over {target actor, frozen source prior}; H2O+'s
  `backup_policy_entropy` is OFF by default (IBRL has no α/log-π). The frozen
  SOURCE agent is BOTH the "sim expert" and the IBRL prior (one ckpt).
- **Two buffers, fixed per-batch sim ratio** (`batch_sim_ratio=0.5`): real
  (target demos, frozen) + sim (source online). The replay stores TRUE rewards;
  the ratio only reweights the sim TD error (toggle via `use_td_target_ratio`).
- **In-sample V** (quantile/expectile, `quantile` default 0.7) learned on the
  target demos only — the exploitation target `r+γ·V(s')` (vs the SAC/IBRL
  explore target) — exactly H2O+'s explore/exploit blend.
- **`ratio_source` default `online` is the requested faithful-H2O baseline, but
  it is DIAGNOSED to be ≈no-op here.** `classifier_test/` showed online
  `d_sa/d_sas` on state-based Threading learn the data-collection (s,a) behavior
  signature, not the dynamics → on-policy ratio ≈ 1 (the smoke run confirmed
  `h2o/is_weight≈0.98`). So this arm degrades to "two-buffer mixed IBRL +
  in-sample V". The validated `cf` discriminator (Spearman 0.33–0.50) is the
  arm expected to actually transfer the dynamics correction; `none` isolates
  whether any correction matters. Run all three and compare TARGET eval.
- **`ratio_source="gt"` (NEW 2026-06-09) — GROUND-TRUTH gap, the online-RL-vs-IL
  bridge.** Mirrors GatedGapAgent §16: train_h2oplus builds a dedicated TARGET
  `gap_env` and, per online SOURCE transition, counterfactually replays (s,a)
  there (`cf_utils.counterfactual_step` + `data_utils.state_gap`), maps the gap
  to `c = exp(-gap/gap_tau)` (default `gap_tau=0.024`) and carries it on the sim
  action dict as `dyn_c` (n-step-safe carrier; **requires `nstep=1`**, asserted +
  auto-set by the script). In `update_h2oplus` the sim IS-weight becomes
  `w = sqrt(c)`, so the squared sim Bellman error is weighted by EXACTLY `c` — the
  same per-transition gap weight `train_weighted_bc.py` applies to its BC loss
  (`[[project-weighted-bc]]`). This makes H2O+(gt) the **online-RL counterpart of
  weighted BC under one identical gap signal**: run both and compare TARGET
  strict-success to separate "online RL vs IL" from "which gap estimator". Costs
  ~2× rollout (one extra TARGET env step per online step). No discriminator is
  built for `gt` (treated like `none` in the agent ctor); `load_model` for
  standalone eval doesn't need `gap_env`.

### Verification status
Smoke-tested 2026-06-01 (+ `gt` arm 2026-06-09):
- ✅ `scripts/test_h2oplus.py`: all 4 `ratio_source` modes run one
  `update_h2oplus` step, losses finite, V/critic/actor all move, `none`→is_weight==1,
  `cf` loads `discriminator_cf.pt`, **`gt`→is_weight==sqrt(c).mean()** (sim batch
  carries `dyn_c`). YAML parses with `gap_tau`; `gt`+`nstep=3` raises, `gt`+`nstep=1` OK.
- ✅ Full training pipeline (online): target demos load (20 ep/1983 tr), source
  warm-up, 49 mixed updates with all H2O+ metrics (`h2o/disc_loss`,
  `is_weight≈0.98`, `real/sim_qf_loss`, `vf_loss`, blended TDs), IBRL hybrid
  active (`actor/bootstrap_bc≈0.54`), TARGET eval + checkpoint save.
- ✅ Standalone `evaluate/eval.py --mode h2oplus` loads agent + source prior,
  runs target eval (exit 0).
NOT yet done: a full-length run / the 3-arm target-success comparison (cf vs
online vs none); the BC prior is the SOURCE agent (no competent TARGET ckpt
exists), so IBRL leans on RL to close the gap.

### TODO
- [ ] Full runs + 4-arm ablation (`ratio_source` online/cf/gt/none); expect
      gt ≥ cf ≥ none ≈ online on TARGET strict-success. Record numbers here.
- [ ] **Headline online-RL-vs-IL**: H2O+(`ratio_source=gt`) vs weighted BC
      (`train_weighted_bc.py`, same GT-gap weighting) on TARGET strict-success —
      does online RL beat IL when both are guided by the identical perfect gap?
      Keep `gap_tau` matched between the two for a fair comparison.
- [ ] `cf` ratio is trained offline on demo counterfactual pairs and has a known
      OOD failure mode (flags novel states as high-gap, `[[project-dynamics-gap-detector]]`
      step 5) — watch whether the online source rollouts drift OOD and corrupt
      the weight; consider clamping/retraining the cf discriminator on rollout
      states.
- [ ] Optionally warm-start the target actor/critic from the source agent
      (`load_pretrained_agent`-style) instead of cold RL.
- [ ] Sweep `exploit_coeff`, `quantile`, `batch_sim_ratio`,
      `clip_dynamics_ratio_max`.
- [ ] Pixel-mode (needs an image dynamics discriminator + encoder; currently
      state-only).

---

## 15. Flow BC input/output normalization + low-data config

### What
`StateFlowPolicy` now normalizes its inputs and outputs:
- **state** → z-score `(s − mean)/std`
- **action** (chunk) → min/max mapped to `[-1, 1]`, inverted on output.

Stats are fit on the **train split** by the dataset and injected into the
policy at construction. Motivated by flow/diffusion losing to a plain MLP in
the ~20-demo regime: generative heads are far more scale-sensitive than
deterministic regression (the obs encoder otherwise wastes capacity aligning
the 37-dim state's heterogeneous scales, and the velocity target `a − z`
assumes unit-scale actions to match `z ~ N(0,I)`). Default **ON**, but with a
critical back-compat property: the buffers are **identity** (`mean=0, std=1,
min=−1, max=1`) until `set_normalization()` is called, and identity is a
mathematical no-op — so a checkpoint trained before this feature (no buffers,
no call) loads strictly and behaves byte-for-byte.

### Where
- `bc/flow_policy.py`
  - `StateFlowPolicyConfig`: `normalize_obs=1`, `normalize_action=1`.
  - `__init__`: registers buffers `norm_state_mean/std`, `norm_action_min/max`
    (identity init). **Buffers, not parameters** ⇒ saved in `state_dict`,
    moved by `.to()`, and NOT touched by EMA (EMA iterates `named_parameters`).
  - `set_normalization(state_mean, state_std, action_min, action_max)` —
    copies stats in (std floored at 1e-6); any `None` arg is skipped (image
    mode has no state stats).
  - `_normalize_state` / `_normalize_action` / `_denormalize_action` — applied
    in `loss()` (normalize both state + action_chunk), `act_chunk()` / `act()`
    (normalize state in, denormalize sampled action out). `forward_velocity` /
    `_sample_full_chunk` operate purely in normalized space.
  - `load_state_dict`: backfills any missing `norm_*` buffers from the current
    (identity) values so legacy ckpts load under `strict=True`.
- `bc/dataset.py`
  - `_compute_norm_stats(dataset_index_pairs, use_state)` — module helper;
    stacks the per-step entries (so it matches `obs_shape` even with
    `state_stack > 1`) → state mean/std + action min/max.
  - `RobomimicDataset`: computes `self.norm_stats` on `idx2entry_train`
    (no val leakage); `get_normalization_stats()` exposes it.
  - `CoTrainDataset`: stats fit over the COMBINED source+target train split;
    same getter.
- `train_bc.py`: after building the flow policy, `policy.set_normalization(
  **dataset.get_normalization_stats())` (guarded by `hasattr`).
- `release/cfgs/mimicgen/threading_d0_state_flow_bc_lowdata.yaml` (NEW) —
  20-demo config: backbone stays **unet1d** but smaller (`unet_down_dims
  [32,64,128]`, `obs_feat_dim 64`), `weight_decay 1e-3` (→ AdamW), `ema_decay
  0.999`, `num_data 20`, `val_ratio 0.0`, 100K steps.
- `scripts/test_flow_norm.py` (NEW) — smoke test.

### How
```bash
# low-data flow BC (normalization auto-on)
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_flow_bc_lowdata.yaml
# disable normalization for an ablation:
python train_bc.py ... --flow_policy.normalize_obs 0 --flow_policy.normalize_action 0
```
The default `threading_d0_state_flow_bc.yaml` also picks up normalization
automatically (the new fields default to 1).

### Design choices
- **Buffers, not config-time stats**: the checkpoint carries its own
  normalizer, so standalone `evaluate/eval.py` needs no dataset to denormalize.
- **Identity = no-op for back-compat**: default-ON is safe; old ckpts with no
  `norm_*` keys load via the tolerant `load_state_dict` and run unchanged.
- **Fit on train split only** (RobomimicDataset uses `idx2entry_train`) to
  avoid val leakage; CoTrain combines both domains (the policy sees a mix).
- **Flow-only for now.** Stats helpers are generic; diffusion/rnn/chunk_mlp
  can adopt `get_normalization_stats()` + the same buffer pattern later (their
  deterministic/GMM objectives are less scale-sensitive, lower priority).

### Verification status
Smoke-tested 2026-06-02 (`scripts/test_flow_norm.py`, both backbones):
✅ buffers in `state_dict`; ✅ loss+backward+EMA with normalization on;
✅ `act`/`act_chunk` return raw-scale (denormalized) actions; ✅ identity
buffers == `normalize_obs/action=0` path (bit-exact under fixed seed);
✅ legacy ckpt (norm buffers stripped) loads under strict=True;
✅ `_compute_norm_stats` shapes (state + image mode); ✅ lowdata YAML parses.
NOT yet verified: a real 20-demo training run's source/target success vs the
MLP baseline (needs the dataset + rollouts).

### TODO
- [ ] Run the 20-demo flow BC (normalized) vs `StateChunkMlpPolicy` vs
      `StateBcPolicy` and record whether normalization closes the gap.
- [ ] Port the same normalization to `StateDiffusionPolicy` (mechanically
      identical) for an apples-to-apples low-data table.
- [ ] Optional: logit-normal time sampling (§4 TODO) stacks on top of this.

---

## 16. Gap-gated sim fine-tuning with a frozen target-demo policy (GatedGapAgent)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
A NEW method (not from a paper) being validated. Keep fine-tuning a policy
`pi_phi` ONLINE in the SOURCE simulator, but do NOT trust the sim everywhere:
for each sim transition `(s,a,s')` a per-transition **dynamics-gap confidence**
`c(s,a,s') ∈ [0,1]` says how trustworthy the source dynamics is as a proxy for
the target. Where the gap is small (c→1) keep improving via normal RL; where it
is large (c→0) defer to a **frozen** policy `pi_demo` learned on the few TARGET
demos. Three gap-gated changes to a plain TD3/DDPG actor-critic:
1. **Gated Bellman backup**: `y = r + γ[ c·minQ_t(s',π_φ(s')) + (1−c)·minQ_t(s',π_demo(s')) ]`.
2. **Confidence-weighted critic loss**: `E[ c·(Q(s,a)−y)² ]` on sim transitions.
3. **Gated policy update**: `−Q(s,π_φ(s)) + λ(1−c)·‖π_φ(s)−π_demo(s)‖²`
   (KL between fixed-std Gaussians ⇒ MSE on the action means).

**This is NOT IBRL.** `pi_sim` = the source RL checkpoint
(`threading_ibrl_source_seed1`) is used ONLY to initialize `pi_phi` + critic;
it plays no role at update time (no max-Q-over-prior, no BC prior). `pi_demo`
is frozen for the whole fine-tune.

**Validation strategy**: replace the (future, learned) gap classifier with the
**GROUND-TRUTH gap** — a "perfect classifier". sim2sim lets us query both
domains, so for each online SOURCE transition we counterfactually replay the
SAME `(s,a)` in a dedicated TARGET sim (reset to the captured `sim_state`,
`qacc_warmstart=0`, step) and measure `‖s'_src − s'_tgt‖`. If even the perfect
gap can't beat the no-gating control arm, the idea is dead; if it helps, the
learned classifier (§ classifier_test) is worth pursuing.

### Where
- `rl/gated_agent.py` (NEW) — `GatedGapAgent(QAgent)` + `GatedGapConfig`
  + `gap_to_confidence()` + `build_iql_demo_policy()`.
  - `attach_demo_policy(module, kind)` — frozen `pi_demo` held in a LIST (not a
    submodule ⇒ kept out of `state_dict`, re-attached by path at eval).
  - `_demo_action(state)` — deterministic query; `kind="bc"` calls
    `module(obs)`→action (StateBcPolicy etc., handles chunk/tuple returns),
    `kind="iql"` uses `FcActor(obs,0).mean`. Clamped to [-1,1].
  - `update_gated(sim_batch, real_batch, stddev, update_actor)` — **hybrid
    offline-online** update. SIM (source online): the 3 gated mechanisms
    (gated backup + c-weighted critic + gated demo-imitation policy), c from
    `sim_batch.action["dyn_c"]`. REAL (`real_batch` = target demos): **full
    weight, ungated** — ground-truth target dynamics + reward, bootstrapped
    along pi_phi, anchoring Q to the true target return; also enters the actor
    loss with c=1 (no demo pull). `real_batch=None` ⇒ sim-only ablation.
    `force_c≥0` overrides the sim `c` (1.0 = no-gating control arm). Gate
    switches `gate_backup / gate_critic_weight / gate_policy` for ablations.
    **Why real_batch matters**: without it (the original implementation) Q was
    purely source-grounded, demo actions were OOD/pessimistically underestimated
    in the source critic, and Q negatively diverged (td_target drifted to ≈−1.8
    on a sparse 0/1 reward — see 2026-06-04 diagnosis); target success was
    capped at ≈pi_demo (0.25). The full-weight real term injects the true
    target value + a positive r=1 anchor so pi_phi can exceed pure imitation.
  - `clamp_q_target` (default ON, [0,1]): even WITH the demo buffer Q still drifted
    negative (2026-06-04: source critic starts healthy ~0.67 but td_target/q_pi/
    q_demo AND real_td_target all <0 by step 10k). Cause = sparse reward +
    pessimistic min-ensemble + OOD bootstrap negative-divergence loop; the demo
    buffer's positive anchor is only the ~1% terminal r=1 transitions (the other
    99% are bootstrapped and follow the critic down). reward∈{0,1}+end_on_success
    ⇒ true Q∈[0,1], so clamping the Bellman target to [0,1] is UNBIASED and severs
    the loop — the actual fix; the demo buffer alone is insufficient.
  - `build_iql_demo_policy(real_replay, ...)` — standard IQL (expectile V, TD Q,
    AWR policy extraction) on the TARGET demo replay → a frozen `FcActor`.
- `train_gated.py` (NEW, adapted from `train_h2oplus.py`) — `train_env`=SOURCE
  (online), `gap_env`=TARGET (dedicated counterfactual sim), `eval_env`=TARGET.
  Inits `pi_phi`+critic from `--source_agent` (reuses `train_rl.load_frozen_agent`
  just to read the weights). `pi_demo` built in `_setup_demo_policy` (load BC
  ckpt, or run IQL). **GT-gap carried on the ACTION dict**: `cf_utils.get_sim_state`
  before the env step, `cf_utils.counterfactual_step(gap_env, …)` +
  `data_utils.state_gap` after, mapped to `c` and stashed on the reply
  `{"action":…, "dyn_c": c}` before `replay.add` (rela passes arbitrary action
  keys through n-step + sampling). The action carrier (vs a next_obs key) aligns
  to the transition's own first action AND survives terminal steps (where the
  next obs is dropped by `replay.add`). The networks ignore the extra
  key (they index `obs["state"]`). `load_model` for `evaluate/eval.py`.
- `evaluate/eval.py` — added `--mode gated` → `train_gated.load_model`.
- `release/cfgs/mimicgen/threading_d0_state_gated.yaml` (NEW) — demo config.
  `q_agent` arch MIRRORS the source ckpt (num_q=5, hidden 1024, …) so its
  actor/critic load cleanly; `act_method="rl"` (NOT ibrl).
- `scripts/test_gated.py` (NEW) — synthetic-batch smoke test.

### How
```bash
source set_env.sh
# pi_demo = pure BC (the threading_target_mlp_bc_n20_seed1 ckpt)
python train_gated.py --config_path release/cfgs/mimicgen/threading_d0_state_gated.yaml \
    --source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
    --gated.demo_policy_kind bc \
    --gated.demo_policy_path exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt \
    --save_dir exps/mimicgen/threading_gated_bc_seed1
# pi_demo = IQL in-sample policy (extracted offline from the demos)
python train_gated.py --config_path release/cfgs/mimicgen/threading_d0_state_gated.yaml \
    --source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
    --gated.demo_policy_kind iql --save_dir exps/mimicgen/threading_gated_iql_seed1
# CONTROL ARM (no gating = pure sim RL from a demo-grounded init):
python train_gated.py ... --gated.force_c 1.0
# eval a saved ckpt on TARGET (strict-success gated):
python evaluate/eval.py --weight exps/.../latest.pt --mode gated --num_game 100 --strict_success 1
```

### Design choices
- **GT-gap via the action dict + `nstep=1`**: zero rela changes. The n-step
  transition keeps the FIRST action, so `batch.action["dyn_c"]` is exactly the
  sampled transition's own `c` — and unlike a next_obs carrier it survives
  terminal steps (verified: the obs carrier was off-by-one on terminals because
  `replay.add` drops the terminal next obs). `nstep=1` is kept so the
  single-step gap matches the single-step bootstrap. n-step
  would misalign it to the window's last step → `nstep=1` is asserted.
- **`c = exp(-gap/gap_tau)`** (soft, default) or `1[gap<gap_threshold]` (binary).
  Calibrated on `classifier_test/pairs_demo.npz` via
  `scripts/calibrate_gap_tau.py`: free-space (`needle_ring_dist≥0.15`) gap
  median≈0.0066 (p90≈0.010), contact-phase gap median≈0.0164 (p90≈0.028) —
  contact ≈2.5× free. Defaults: **`gap_tau=0.024`** (= contact-median/ln2 ⇒
  c=0.5 at the contact median; free mean c≈0.73, contact mean c≈0.55, c≈0.30
  at contact p90) and **`gap_threshold=0.013`** (= geo-mean of free-p90 &
  contact-median ⇒ 94% of free trusted c=1, 63% of contact gated c=0). NOTE:
  this is the DEMO (success-trajectory) gap distribution; online rollouts may
  be wider — watch the logged `gap/value` histogram and re-run the calibrator
  if needed.
- **`pi_demo` frozen, list-held** → not in the agent `state_dict`; re-attached
  by path at eval (BC) or re-extracted (IQL). Target ckpts stay small.
- **Cost**: one extra TARGET env step per online step (~2× rollout). Acceptable.
  Uses `cf_utils.counterfactual_step`'s `qacc_warmstart=0` cold-start so the GT
  gap is pure + reproducible.
- **Two ways to learn `pi_demo`** (`demo_policy_kind`): `bc` (pure behavior
  clone; loads any trained BC head via `train_bc.load_model`) or `iql`
  (advantage-weighted in-sample policy — stays in demo support, prefers
  high-advantage actions; per the original idea spec, NOT pure BC).

### Verification status
Smoke-tested 2026-06-03:
- ✅ `scripts/test_gated.py`: `update_gated` runs, 3 losses finite, critic+actor
  params move; `force_c=1.0` control arm (c≡1, imit_loss=0, pi_demo ignored);
  all-gates-off → plain TD3; `gap_to_confidence` soft+binary.
- ✅ Full `train_gated.py` pipeline (BC pi_demo): demos load, source warm-up,
  GT-gap counterfactual computed online (`gap/value avg≈0.0055`,
  `gap/confidence avg≈0.90`, `gate/c_min≈0.23`), gated updates with finite
  `gate/*` metrics, TARGET eval + checkpoint save.
- ✅ Standalone `evaluate/eval.py --mode gated` loads agent + re-attaches frozen
  pi_demo, runs TARGET eval (score 0 on the barely-trained smoke ckpt).
- (EGL errors at interpreter shutdown are benign MuJoCo context cleanup, after
  all work + saves complete.)
NOT yet done: a full-length run + the feasibility comparison (GT-gap gating vs
`force_c=1.0` control vs source-prior-only ~13% on TARGET strict success);
`gap_tau` calibration; the IQL pi_demo arm full run.

### TODO
- [ ] Full runs: **GT-gap gating** vs **`force_c=1.0`** (no-gating control) vs
      source-init-only — does the perfect gap beat the control on TARGET strict
      success? (the go/no-go feasibility signal).
- [x] Calibrate `gap_tau` / `gap_threshold` from `classifier_test/pairs_demo.npz`
      (done 2026-06-03, `scripts/calibrate_gap_tau.py`; soft τ=0.024, binary
      thr=0.013). Re-run on online-rollout gaps if they prove wider than demos.
- [ ] BC vs IQL `pi_demo` comparison.
- [ ] If feasible: swap the GT gap for the learned classifier (§ classifier_test
      `discriminator_cf.pt`) and measure the drop from "perfect" to "learned".
- [ ] Optional: c-weight the `−Q` term in the actor loss too (currently only the
      demo-imitation term is gated); sweep `kl_lambda`.

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

## 18. LWAIL-cross (cross-domain latent state-occupancy matching)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
A cross-domain (source→target) imitation method adapted from **LWAIL** (Yang et
al., *Latent Wasserstein Adversarial Imitation Learning*, ICLR 2026; ref at
`/home/Chenyang/LWAIL`). An agent **explores in the SOURCE sim** but is matched —
only in a frozen **ICVF latent φ** — to the **STATE distribution of a few TARGET
demos**. The target demo states say "which states to reach"; "how to get there"
is left to the downstream RL (a plain DDPG actor-critic = the ibrl `QAgent`). The
policy is finally **deployed/evaluated on TARGET**.

Two changes vs vanilla LWAIL (the rest — ICVF pretraining, WGAN-GP, sigmoid
reward, downstream RL — is carried over):
1. **State-level matching** (not state-pair): the discriminator `f` eats `φ(s)`
   only; pseudo-reward `r(s) = σ(−f(φ(s)))` (no `minus`, no `(s,s')`, no action).
2. **Data split**: online interaction + ICVF pretraining come from **SOURCE**;
   the expert distribution is the **TARGET demo states**.

This is NOT IBRL and NOT gap-gated (no max-Q-over-prior, no pi_demo, no dynamics
gap). The source RL ckpt only **warm-starts** pi_phi + critic (optional, default
on, with a from-scratch ablation). φ is frozen for the whole run.

### Three stages
```
Stage 1  ICVF pretrain (OFFLINE, no env): train a PyTorch ICVF on the existing
         SOURCE RL replay (release/data/mimicgen/threading_d0_source_replay/
         replay_buffer.hdf5; 1082 trajs, obs/state already 37-dim, ~45% success,
         a mixed/passive dataset). Freeze phi -> icvf_phi.pt (+ icvf_full.pt).
Stage 2  ★ sanity gate: phi-induced value V(s, g_success, g_success) must rank
         TARGET-demo states consistently with steps-to-success (Spearman>0). If
         ~0, the gap is too large in this phi -> STOP, revisit the representation.
Stage 3  occupancy matching: SOURCE online explore -> replay; TARGET demo states
         = expert pool; f WGAN-GP single-state; r=σ(−f(φ(s))) injected into the
         critic; DDPG update. Eval on TARGET (score/score), SOURCE diagnostic.
```

### Where
- `rl/icvf.py` (NEW) — PyTorch port of `icvf_release/src/` (JAX 0.3.25/TPU, can't
  run on Blackwell; LWAIL's `PhiNet` is itself a PyTorch port of this φ, so the
  port is validated). `PhiNet` (GELU MLP, optional LayerNorm, matches
  `LWAIL/network.py:PhiNet`), `MultilinearVF` (φ/ψ/T nets + matrix_a/b,
  `get_info`/`get_phi`; port of `icvf_networks.py:110-154`), `ICVFEnsemble`
  (2 members, min-Q), `expectile_loss`/`icvf_loss` (port `icvf_learner.py:15-91`),
  `ICVFLearner` (online + EMA target, `update`), `TrajGoalSampler` (port
  `gc_dataset.py:GCSDataset.sample`: hindsight goal/intent, `p_randomgoal=0.3/
  p_trajgoal=0.5/p_currgoal=0.2`, `p_samegoal=0.5`, `reward_shift=−1`),
  `load_source_replay_states` (reads `data/demo_N/obs/state`, already 37-dim).
- `train_icvf.py` (NEW) — stage 1, pure offline tensor training (no robosuite, no
  source agent). Saves `icvf_phi.pt` (φ state_dict + meta) and `icvf_full.pt`
  (full ensemble + cfg, for the gate). `os._exit(0)` at the end.
- `scripts/diag_icvf_sanity.py` (NEW) — stage 2 gate. Loads `icvf_full.pt` +
  TARGET demos, scores `V(s, g_success, g_success)`, reports Spearman vs
  −steps_to_go (global + per-demo), prints PASS/FAIL (threshold 0.3), optional
  scatter. Exit code 0 = pass.
- `rl/lwail_agent.py` (NEW) — `LwailAgent(QAgent)` (state-only). Inlines
  `FullyConnectedNet` + `gradient_penalty` (from LWAIL). `attach_phi` (frozen φ,
  **list-held** so NOT in state_dict; builds `f_net` over the latent + its Adam).
  `reward_from_f(s)=σ(−f(φ(s)))∈(0,1)`. `update_f(source_states, expert_states,
  n_iter)` (WGAN-GP: `tanh(f(φ(s_exp))).mean() − tanh(f(φ(s_src))).mean() + GP`,
  expert=target, sample=source). `update_lwail(sim_batch, stddev, update_actor)`
  (DDPG with reward REPLACED by `reward_from_f(sim_batch.obs["state"])` from the
  current f; **dense reward in (0,1) → NO clamp**, unlike the sparse-reward gated
  agent). f_net IS in state_dict; φ is not.
- `train_lwail.py` (NEW, trimmed from `train_gated.py`) — removes gap_env /
  counterfactual / cf_utils / pi_demo / dyn_c; keeps SOURCE online env + TARGET
  eval + warm-start + warm-up + dual eval/save. `_attach_phi` loads `icvf_phi.pt`;
  `_setup_expert_states` loads TARGET demo states (reuses
  `load_source_replay_states`); `pretrain_f` (LWAIL `pretrain`, warm-up source vs
  expert, `f_pretrain_steps`); train loop: f update every `update_timestep`, RL
  every `update_freq`. `load_model` re-attaches φ from `icvf_path` (strict=False).
- `release/cfgs/mimicgen/threading_d0_state_lwail.yaml` (NEW) — `state_stack=1`,
  q_agent MIRRORS `threading_ibrl_source_seed1` (num_q=5/hidden 1024/num_layer 3,
  for clean warm-start load), `act_method="rl"`, `eval_domain_shift` = §12 target,
  `lwail` (f_hidden_dim/lr_f/alpha/f_epoch), `icvf_path`/`source_agent`.
- `scripts/pretrain_icvf_threading.sh`, `scripts/train_lwail_threading.sh` (NEW) —
  `SEED/SOURCE_AGENT/ICVF_PATH/WARM_START/...` env vars + `"$@"` passthrough.
- `scripts/test_lwail.py` (NEW) — synthetic-batch smoke test.
- `evaluate/eval.py` — `--mode lwail` → `train_lwail.load_model`.

### How
```bash
source set_env.sh
# Stage 1: pretrain phi on the source replay (offline)
bash scripts/pretrain_icvf_threading.sh                 # -> exps/mimicgen/threading_icvf_seed1
# Stage 2: ★ go/no-go gate (must pass before stage 3)
python scripts/diag_icvf_sanity.py --icvf_full exps/mimicgen/threading_icvf_seed1/icvf_full.pt
# Stage 3: occupancy-matching training
bash scripts/train_lwail_threading.sh                   # warm-start (default)
WARM_START=0 bash scripts/train_lwail_threading.sh      # from-scratch ablation
# Standalone eval (TARGET, strict-success)
python evaluate/eval.py --weight exps/mimicgen/threading_lwail_seed1/latest.pt \
    --mode lwail --num_game 100 --shift_from_cfg 1 --strict_success 1
```

### Design choices
- **PyTorch ICVF** (not the JAX original): `icvf_release` pins `jax[tpu]==0.3.25`/
  `flax==0.6.2` which won't run on Blackwell sm_120; the algorithm is small and
  LWAIL's PhiNet proves the port. φ exported as a bare `PhiNet` state_dict so
  `lwail_agent` needs no ICVF machinery at train time.
- **Dense reward, no clamp**: `r=σ(−f)∈(0,1)` every step, so the true Q is in
  `[0, 1/(1−γ)]`, not `[0,1]` — the gated agent's `clamp_q_target` would be wrong
  here. The reward being always positive also avoids the negative-divergence loop
  that motivated the clamp there.
- **Reward computed at update time** with the CURRENT f_net (f changes every
  `update_timestep`; RL runs every `update_freq`) — equivalent to LWAIL's
  `get_pseudo_rewards` rebuild without rewriting the rela buffer.
- **Warm-start optional + ablation**: training is entirely in SOURCE dynamics
  (only eval/deploy is TARGET), so warm-start has no train-time conflict; the
  real transfer gap is measured by TARGET eval, and `init_actor_critic_from_
  source 0` is the from-scratch control.
- **No ADDA / no state-pair / no `minus`**: per the user's cross-domain spec, the
  matching is single-state in φ; that's the whole adaptation over LWAIL.

### Verification status
Smoke-tested 2026-06-06 (Blackwell, ibrl env):
- ✅ `scripts/test_lwail.py`: TrajGoalSampler shapes; ICVFLearner.update (finite
  loss, value moves, target EMA, phi export/reload); LwailAgent attach_phi builds
  f_net; f_net IN state_dict, φ NOT; update_f WGAN-GP finite; reward∈(0,1);
  update_lwail 3 losses finite + critic/actor move; state_dict round-trip.
- ✅ `train_icvf.py` 30-step run: value_loss finite + decreasing (0.24→0.18),
  phi_norm stable; saves icvf_phi.pt + icvf_full.pt.
- ✅ `diag_icvf_sanity.py`: loads ICVF + 20 TARGET demos, computes Spearman, prints
  PASS/FAIL + exit code (FAILed on the 30-step toy ICVF, as expected).
- ✅ `train_lwail.py` end-to-end short run: SOURCE env + warm-start (actor/critic
  load), φ attached (in=37 latent=256), expert pool (1983,37), f pretrain
  (**r_expert 0.530 > r_source 0.495** — correct direction even with toy φ), train
  loop, dual TARGET+SOURCE eval, TopkSaver save, clean `os._exit(0)`.
- ✅ `evaluate/eval.py --mode lwail`: load_model re-attaches φ, target shift loaded,
  rolls out + prints score.
NOT yet done: a real ICVF pretrain (500k) + the sanity gate verdict on a properly
trained φ; the full stage-3 run + TARGET strict-success vs baselines (gated
`force_c=1.0` ~13% / H2O+ / DR / SGFT); warm-start vs from-scratch comparison.
(EGL errors at interpreter shutdown are benign MuJoCo context cleanup.)

### 2026-06-06 first real run + diagnosis + fix
A real ICVF pretrain PASSED the sanity gate strongly (global Spearman **+0.948**,
per-demo **+0.990**, V range [-61.6, 2.07] — shared-V-on-phi transfers across the
gap). But the first stage-3 run (200k) gave `score/score`=`source_score`=0 with
`train_score`=0. `scripts/diag_lwail_init.py` (NEW) localized it:
- warm-start is GOOD: the bare warm-started RL actor scores **SOURCE 0.633** /
  TARGET 0.000 BEFORE any training — but the trained ckpt is **SOURCE 0.000** /
  TARGET 0.000. DDPG CATASTROPHICALLY forgot a working source policy.
- the discriminator is HEALTHY (no domain shortcut): reward(TARGET demo)=0.878,
  reward(SOURCE success)=**0.649**, reward(SOURCE fail)=0.254 — it rewards
  task-completing states, success ≈ target.
- root cause = **dense reward (0,1) ⊗ `end_on_success`**: success ends the
  episode and STOPS the reward stream, so the agent is rewarded for NEVER
  succeeding. `lwail/q≈10 ≈ 0.1·(1−0.99⁴⁰⁰)/0.01` = "stall 400 steps at 0.1/step",
  exactly the no-success optimum. (LWAIL's D4RL is fixed-horizon, so it never hit
  this.) Secondary: dense Q→~100 vs the source-warm critic's [0,1] scale.
- **fix**: `train_end_on_success=0` (train at fixed horizon; eval keeps
  success-termination) + `lwail.reward_scale=0.01` (=(1−γ), keeps Q ~O(1) to
  match the warm-started critic). Both wired in train_lwail.py / lwail_agent.py /
  the YAML. A 40k re-run is validating whether source holds ~0.63 and target lifts.

### TODO
- [ ] Real run: `bash scripts/pretrain_icvf_threading.sh` (500k) → check the
      sanity gate (Spearman) → stage 3; record TARGET strict-success.
- [ ] If the gate FAILs: try `--icvf.use_layer_norm 1`, more/other source data,
      or a different expectile; the gate is the early-stop for a bad φ.
- [ ] warm-start vs `--init_actor_critic_from_source 0` on TARGET success.
- [ ] Sweep `lwail.alpha` (GP), `lwail.f_epoch`, `update_timestep`, f batch size
      (LWAIL uses ~4000 source states per f step; we use `batch_size`).
- [ ] Pixel-mode: needs an image ICVF encoder + a pixel discriminator.

---

## 19. ReBRAC (offline TD3+BC) — head-to-head vs MLP-BC

> ⚠️ **部分归档** —— `train_rebrac.py` + `rl/rebrac_agent.py` **保留**（`rl/residual_rl_agent.py` 依赖其 `EnsembleQ`、`train_residual_rl.py` 依赖其 `_load_demos_flat`），但 ReBRAC 的 YAML 与 `scripts/` 已归档到 `_archive/`（见 §34）。

### What
Port of **ReBRAC** (Tarasov et al., *Revisiting the Minimalist Approach to
Offline RL*, NeurIPS 2023; JAX/Flax ref at `/home/Chenyang/ReBRAC`) onto ibrl as
a **pure offline RL** baseline. ReBRAC = TD3+BC with four "minimalist" knobs:
(1) **decoupled** actor / critic BC penalties (`actor_bc_coef` / `critic_bc_coef`
— the critic penalty rides the TD target: `next_q -= critic_bc_coef·‖a'_π−a'_data‖²`),
(2) deep **LayerNorm** twin-Q critic, (3) **Q-normalization** in the actor loss
(`λ=sg(1/|Q|.mean())`), (4) large batch + long training. Plus standard TD3:
target policy smoothing, twin-Q min, delayed actor+target updates (`policy_freq`).

**Deliberate deviation for the comparison**: the actor IS the MLP-BC actor
network — `bc.bc_policy.StateBcPolicy` (state→MLP→Tanh). So ReBRAC and
`threading_d0_state_mlp_bc_target` share the IDENTICAL actor arch
(num_layer 3 / hidden 1024 / dropout 0 / no LN); the ONLY difference is the
objective (TD3+BC vs pure MSE BC). Trains offline on the SAME 20 target demos
(`threading_d0_target_dynamics`, dynamics-only target §12); eval on TARGET
(strict-success gated) + nominal SOURCE diagnostic. **No source agent, no online
interaction.** Did NOT port the `calibration`-branch MC-return lower bound
(`maximum(…, mc_returns)`) — that's the separate Cal-QL-style addition, not core
ReBRAC; target = `r + (1−done)·γ·(min_q − critic_bc_coef·bc_penalty)`.

### Where
- `rl/rebrac_agent.py` (NEW) — `ReBRACAgent(nn.Module)` + `ReBRACConfig` +
  `EnsembleQ`. Actor = `StateBcPolicy` (+ a frozen `actor_target` deep-copy);
  `EnsembleQ` = `num_critics` independent concat-(s,a) MLPs with LayerNorm
  (+ `critic_target`). State z-score in registered buffers
  (`norm_state_mean/std`, identity until `set_normalization()`), applied in
  both `update` and `act` (so the ckpt self-normalizes at eval);
  `normalize_states=0` ⇒ identity = raw-state ablation matching MLP-BC's input.
  `update(batch)`: critic every call, actor + soft-target on `policy_freq` cadence.
  `act` handles single + batched obs (mp eval). NOT a QAgent subclass (TD3+BC is
  structurally unlike the IBRL max-Q family), but mirrors the eval surface.
- `train_rebrac.py` (NEW, trimmed from `train_simdist.py`) — offline only:
  `ReBRACOfflineDataset` flattens whole demo episodes (via
  `rl/sequence_replay.py::load_demos_to_sequence_buffer`) into flat GPU tensors
  `states/actions/rewards/next_states/next_actions/dones` (ReBRAC's
  `create_from_d4rl` layout; `next_action[t]=a_{t+1}`, terminal step dummy masked
  by `done=1`). Builds only the TARGET eval env, computes state stats →
  `agent.set_normalization`, runs `num_train_step` offline updates, periodic
  TARGET eval (+ SOURCE diagnostic), TopkSaver. `load_model` for standalone eval.
- `evaluate/eval.py` — added `--mode rebrac` → `train_rebrac.load_model`.
- `release/cfgs/mimicgen/threading_d0_state_rebrac.yaml` (NEW) — target-only;
  actor mirrors `threading_d0_state_mlp_bc_target.state_policy`; critic 3×256+LN
  twin-Q; batch 1024; 200K steps; §12 target eval + strict success + eval σ=0.1.
- `release/cfgs/mimicgen/threading_d0_state_rebrac_weighted.yaml` (NEW) — the
  gap-weighted source+target variant (see "Gap-weighted variant" below); data
  paths/cache/τ/ratio mirror `threading_d0_state_weighted_bc.yaml`.
- `scripts/train_rebrac_threading.sh` (NEW) — `SEED/N_DEMO/NORM_STATES/ACTOR_BC/
  CRITIC_BC/WEIGHTED/WEIGHT_TD` env vars + `"$@"` passthrough.
- `scripts/test_rebrac.py` (NEW) — synthetic-batch smoke test.

### Gap-weighted variant (offline-RL analog of weighted BC, §11/§16)
To compare head-to-head with `train_weighted_bc.py`, ReBRAC trains on the SAME
mixture: a LOT of gap-weighted SOURCE demos (`threading_d0_source_sigma0.1`,
2000) + a few TARGET demos (20, weight 1), drawn at a fixed per-batch ratio
(`target_batch_ratio=0.5`). Per-transition weight `c = exp(-gap/τ) ∈ [0,1]` from
the GT-gap cache (`gt_gaps_n2000.npz`, `scripts/compute_source_gaps.py`), keyed
`(ep_id, t_in_ep)` — the cache's `ep_id` is the same `demo_0,1,…` order the flat
loader uses (it tracks the ORIGINAL demo index even though it skips no-success
demos / truncates at first success, so the lookup aligns; coverage asserted).
**Where the weight enters** (mirrors weighted BC's `weighted_loss = Σw·e/Σw`):
- **actor BC penalty** `actor_bc_coef·‖π(s)−a‖²` → gap-weighted mean (THE direct
  analog of weighted BC's loss); the RL Q term `−λ·Q(s,π(s))` stays a plain mean.
- **critic BC penalty** `critic_bc_coef·‖a'_π−a'_data‖²` (in the TD target) →
  per-sample `c·(·)`.
- **critic TD/Bellman loss** `(Q−target)²` → weighted mean by `c` IFF
  `rebrac.weight_td=1` (DEFAULT ON; the §16 gated "trust the backup less where the
  source dynamics is unreliable"). `weight_td=0` weights only the BC terms.
Target-only mode (no `source_demo_path`) omits `weights` from the batch entirely
⇒ byte-for-byte the plain (unweighted) ReBRAC path.

### How
```bash
source set_env.sh
bash scripts/train_rebrac_threading.sh                       # 20 demos, z-score on
N_DEMO=10 bash scripts/train_rebrac_threading.sh             # 10-demo run
NORM_STATES=0 bash scripts/train_rebrac_threading.sh         # raw-state (MLP-BC input) ablation
ACTOR_BC=0.5 bash scripts/train_rebrac_threading.sh          # sweep actor BC strength
# gap-weighted source+target mix (vs weighted BC). Prereq: the GT-gap cache:
python scripts/compute_source_gaps.py --num_demos 2000        # once (slow)
WEIGHTED=1 bash scripts/train_rebrac_threading.sh             # weight BC terms + TD (default)
WEIGHTED=1 WEIGHT_TD=0 bash scripts/train_rebrac_threading.sh # weight only the BC terms
# raw:
python train_rebrac.py --config_path release/cfgs/mimicgen/threading_d0_state_rebrac.yaml \
    --save_dir exps/mimicgen/threading_rebrac_seed1
# standalone eval (TARGET, strict-success):
python evaluate/eval.py --weight exps/.../latest.pt --mode rebrac --num_game 100 \
    --shift_from_cfg 1 --strict_success 1
```
`state_stack` MUST be 1 (matches the MLP-BC target baseline's obs space).

### Design choices
- **Actor == MLP-BC network** (the whole point): isolates objective from arch.
  State z-score is an input transform, not an arch change — kept ON (ReBRAC
  recipe) but toggleable to 0 for a strictly-equal-input ablation.
- **Pure offline, no source agent**: ReBRAC is offline RL, not a hybrid /
  transfer method — the cleanest "can offline RL beat BC on the same 20 target
  demos" question. (Contrast §16 gated / §14 H2O+ which also use the source.)
- **Weight on the BC terms, not the data sampling** (weighted variant): unlike
  weighted BC's default `weight_mode=sample` (draw source ∝ c), ReBRAC uses the
  weighted-LOSS form (uniform draw, `c` scales the BC terms / TD loss). Cleaner
  for RL — every sampled transition keeps its r/s'/a' for the Bellman backup;
  the weight only changes how much each term contributes, matching weighted BC's
  `weight_mode="loss"` arm. The Q term is never weighted (it's the RL signal).
- **Standalone twin-Q critic** (not the QAgent `Critic`/`MultiFcQ`, which are
  encoder-coupled): a plain concat-(s,a) MLP matches ReBRAC's `EnsembleCritic`
  exactly and keeps the agent self-contained.
- **No MC-return calibration**: core ReBRAC only; the checkout's `calibration`
  branch adds it but that's a different method.

### Verification status
Smoke-tested 2026-06-08:
- ✅ `scripts/test_rebrac.py`: actor IS `StateBcPolicy`; act single→`[act]` &
  batched→`[B,act]` in [-1,1]; `update` moves critic every step, actor only on
  `policy_freq`, targets soft-update, losses finite; state_dict round-trip incl.
  norm buffers; `normalize_states=0` identity.
- ✅ Full pipeline (200 steps, real demos): 20 demos / 1983 transitions / obs 37
  / act 7 / 20 success-terminals loaded into the flat buffer; offline updates
  run; TARGET eval + checkpoint save; exit 0 (EGL teardown errors benign).
- ✅ Standalone `evaluate/eval.py --mode rebrac` loads + rolls out target (score
  0 on the 200-step toy ckpt, as expected).
- ✅ Weighted variant: `test_rebrac.py::test_weighted_update` (BC terms weighted,
  `weight_td` on/off, all-zero-weight Σw guard, finite + nets move). End-to-end
  short run loaded the mix — tgt 20 demos/1983 tr (mean w=1.000) + src 100
  demos/6283 tr (**mean w=0.513** from the cache, NOT all-1 ⇒ `(ep_id,t)`
  alignment confirmed), ratio 0.5/0.5, updates + TARGET eval + save.
NOT yet done: a full 200K run's TARGET strict-success vs the MLP-BC baseline
(`threading_d0_state_mlp_bc_target`) AND vs weighted BC
(`threading_d0_state_weighted_bc`) + the `normalize_states`/`actor_bc_coef`/
`weight_td` sweeps.

### TODO
- [ ] Full run + record TARGET strict-success vs MLP-BC AND vs weighted BC (the
      headline comparisons); also vs gated `force_c=1.0` ~13% / H2O+ / DR / SGFT.
- [ ] Weighted variant: `weight_td` on vs off (does weighting the Bellman backup
      help beyond weighting just the BC terms?); sweep `gap_tau` / `target_batch_ratio`.
- [ ] Sweep `actor_bc_coef` / `critic_bc_coef` (ReBRAC tunes these per dataset)
      and `normalize_states` on/off.
- [ ] 10-demo vs 20-demo (does offline RL's edge over BC grow in lower data?).
- [ ] Optional: port the MC-return calibration target for an antmaze-style
      lower-bound ablation.

---

## 19. Source-domain recovery-data collection (recovery-transfer feasibility)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
A go/no-go data-collection step for the hypothesis: **collecting V-gated
*recovery* data in the SOURCE sim can teach a downstream policy recovery
behaviors that help it learn from only ~20 TARGET demos.** Roll out the
TARGET-trained BC (`threading_target_mlp_bc_n20`) in the SOURCE domain; a source
RL agent (`threading_ibrl_source_strict`) estimates `V(s)` and TAKES OVER
(DAgger-style) when the BC trajectory is about to crash (V dips / stuck) —
exactly the §16 / `intervention_eval.py` gating. The **executed** action per
step is recorded (so a recovery segment stores the RL *recovery* actions to
clone). Episodes are typed by (took_over × success):
`A` = no-takeover+success (pure BC), `B` = no-takeover+fail, `C` =
takeover+success (recovery rescued), `D` = takeover+fail (recovery failed).

Four 500-traj datasets (state-only, NO images; same HDF5+`env_cfg.json` sidecar
layout as `collect_target_demos.py`, so `bc/dataset.py` reads them directly):
| Dataset | Composition | Regime |
|---|---|---|
| ds1 | `C + A` (recovery-success + clean-success) | recovery ON |
| ds2 | `C + D + A` (all-recovery + clean-success) | recovery ON |
| ds3 | `A + B` (no-recovery, success+fail; **fail-fraction matched to ds2**) | recovery OFF |
| ds4 | `A` only (no-recovery success) — **pre-existing** | — |

ds4 = `release/data/mimicgen/threading_d0_source_bcpolicy_sigma0.1` (2000 traj,
WITH images, **raw** success — collected by `collect_target_demos.py`, not
strict). The comparison logic: **#1 vs #4** = does successful-recovery data
help; **#1 vs #2** = does adding failed-recovery (D) hurt; **#2 vs #3** = at a
matched fail-ratio, do recovery-action segments differ from natural-BC-failure
data.

### Where
- `scripts/collect_recovery_data.py` (NEW) — loads BC + RL via
  `train_bc/train_rl.load_model`; builds a SOURCE env (`DomainShiftConfig(mode=
  "none")`, `hand_init_perturb_joint_sigma=0.1` to match ds4, `strict_success`
  ON by default, one rl-camera rendered but image keys dropped on save).
  `rollout(recovery_on)` snapshots obs BEFORE each `act()` (act mutates obs in
  place) and skips image keys to bound RAM over the 400-step D episodes;
  classifies A/B/C/D. Collects pool R (recovery ON) until `|A|+|C| ≥ N`, then
  pool P (recovery OFF) until it has `round(fail_frac2·N)` B + the rest A.
  `--probe K` rolls K episodes/regime, prints the A/B/C/D histogram, writes
  nothing. Per-demo HDF5 attrs carry `ep_type / took_over / n_takeover_steps`.

### How
```bash
source set_env.sh
python scripts/collect_recovery_data.py --probe 40          # distribution probe
python scripts/collect_recovery_data.py --n_per_dataset 500 # full ds1/ds2/ds3
# outputs: release/data/mimicgen/threading_d0_source_recovery_ds{1,2,3}/
```

### Results (collected 2026-06-08, seed 1)
Probe (40/regime): pure-BC SOURCE success **50%** (ample recovery headroom, not
saturated); recovery-ON took_over **70%**, success 57%. Full collection:
| Dataset | types | success | transitions |
|---|---|---|---|
| ds1 (C+A) | A=228, C=272 | 500/500 | 61,473 |
| ds2 (C+D+A) | A=119, C=148, D=233 | 267/500 | 126,940 |
| ds3 (A+B) | A=267, B=233 | 267/500 | 124,221 |

ds2 & ds3 match on BOTH traj count AND success/fail count (267/233,
`fail_frac=0.466`) AND ≈transition count (failures are the 400-step D/B
episodes) — a clean #2-vs-#3 control. ds1's transitions are ~½ of ds2 (the only
unmatched axis: D episodes are long + numerous).

### Design choices / gotchas
- **Executed-action relabel**: recovery segments store the RL takeover action,
  not the BC action — that IS the recovery behavior to clone (DAgger-style).
- **D episodes are 400-step, RL-driven-300+ and still fail**: once BC drives
  into a bad region V never recovers above `recover_threshold`, so RL drives to
  the horizon and fails. Legit failed-recovery trajectories, but they dominate
  ds2's transition budget. BC samples per-transition ⇒ ds2 training will be
  weighted toward failed-recovery RL actions.
- **strict_success ON** for ds1/2/3 (honest insertion, anti clip-through) but
  **ds4 was raw** success. For an apples-to-apples #1-vs-#4 the user should
  subsample ds4 → 500 and re-filter under strict (or re-collect ds4 strict).
- env (mode=none + sigma=0.1) intentionally mirrors ds4 so the four share an
  init distribution; the V-gate uses the validated target-tuned thresholds
  (thr 0.32 / recover 0.5 / rel_drop 0.08 / stuck_eps 0.02).

### Downstream results — weighted BC + weighted ReBRAC (2026-06-09)
The feasibility verdict. Each recovery set + 20 TARGET demos trains a
gap-weighted policy (§11 weighting, `gap_tau=0.01`), evaluated on the §12
dynamics-only TARGET (strict-success). To control for DATA VOLUME (BC/RL sample
per-transition), ds1/2/3 were **transition-matched** to ds4-first-500 = **50,554
transitions** via `scripts/make_tmatch_subset.py` (shuffle-then-pack to budget,
seed 0 — ds3 was written A-block-then-B-block so a leading prefix would collapse
its fail fraction; shuffling preserves the per-type transition mix). The matched
subsets: ds1 411 demos, ds2 198 (A50/C56/D92), ds3 193 (A96/B97); GT-gap caches
`<dir>_tmatch/gt_gaps.npz` built per subset. ds4 baseline retrained at 500 demos
(`threading_weighted_bc_bcpolicy_n500_seed1`, reuses `gt_gaps_n2000.npz`).

Final eval = **`model0.pt` (best ckpt) × 500 games** on TARGET, strict-success,
seed 1, `hand_init sigma 0.1` (`scripts/eval_recovery_final.sh`). The in-training
50-game `best_ckpt_target` numbers were **too noisy to trust and actively
misleading** (they ranked ds3 2nd and ds2 last; 500 games flips both) — always
re-eval at 500 before concluding. ±=binomial SE.

| set | composition | **weighted BC** (500g) | weighted ReBRAC (500g) |
|---|---|---:|---:|
| **ds1** | C+A (recovery-success) | **29.6% ±2.0** | 14.6% ±1.6 |
| ds2 | C+D+A (all-recovery) | 28.8% ±2.0 | **17.8% ±1.7** |
| ds4 | A only (baseline, pure success) | 24.6% ±1.9 | 9.0% ±1.3 |
| ds3 | A+B (no-recovery, fail-matched) | 21.8% ±1.8 | 12.2% ±1.5 |

(mode `weighted_bc` / `rebrac`; the BC cfg.yaml carries gap_* fields so `--mode
bc` can't load it — use `weighted_bc`. ds4 ReBRAC `model0` is from the run that
hung at 110k, but it's the best-by-target ckpt so the 500-game number is valid.)

**Verdict — recovery data helps BOTH heads over baseline; the heads disagree on
what to do with FAILURES:**
1. **Recovery helps (both heads beat the pure-success baseline ds4).** BC:
   ds1 29.6% / ds2 28.8% > ds4 24.6% (ds1−ds4 z=1.8, marginal; the larger,
   clearly-significant gaps are ds1/ds2 **vs ds3** z=2.8/2.6). ReBRAC: every
   recovery/failure set beats ds4 9.0% (ds2−ds4 z=4.1, ds1−ds4 z=2.8).
2. **Failed recovery does NOT hurt — earlier 50-game claim was noise.** BC
   ds1≈ds2 (29.6 vs 28.8, z=0.3, indistinguishable): adding the failed-recovery
   D segments neither helps nor hurts BC. ReBRAC ds2 is the BEST of all
   (17.8%) — offline RL extracts value from the failed-recovery transitions.
3. **Pure-failure coverage (ds3) is where the heads split.** BC: ds3 21.8% is
   the WORST, *below* baseline (ds4−ds3 z=1.05, ns but trending) — cloning
   no-recovery failure states/actions doesn't help BC; the useful signal is the
   recovery ACTIONS, not failure coverage. ReBRAC: ds3 12.2% > ds4 9.0% —
   offline RL still benefits from failures (the critic labels them low-value),
   so even un-recovered failures lift it above the success-only baseline.
4. **BC ≫ ReBRAC everywhere** (best BC 29.6% vs best ReBRAC 17.8%): this
   weighted-ReBRAC impl underperforms BC on Threading and collapses late in
   training (best ckpt ≫ final); BC is the method of record here. ReBRAC's
   monotonic data-richness ordering (ds2>ds1>ds3>ds4) is the cleaner *relative*
   story but at low absolute success.

Net: **collecting source-domain V-gated recovery data and gap-weighting it into
a 20-target-demo BC gives a small but real lift (~+4-5pt over the success-only
baseline); the win comes from the successful-recovery ACTIONS, and including the
failed-recovery attempts is harmless for BC and outright helpful for offline
RL.** Single-seed — multi-seed needed to tighten the ds1/ds2-vs-ds4 margin.

**Code touched for this run:**
- `scripts/make_tmatch_subset.py` (NEW) — shuffle-pack a dataset to a transition
  budget, re-indexed HDF5 + sidecar.
- `train_rebrac.py::_load_demos_flat` — added `include_failures` (source call
  passes True): no-success episodes are kept as full-length, `done=0` throughout
  (timeout, NOT absorbing terminal). Without it offline RL silently dropped every
  failure trajectory (ds2 fell to 106 demos / 14k trans, ds3 to 96 / 12k) —
  defeating ds2/ds3 AND breaking the transition match. Default False keeps the
  target-only path byte-identical.
- ds2/ds3 weighted-BC need `--dataset.eval_episode_len 401` (the D/B failures are
  exactly 400 steps; `bc/dataset.py:232` asserts `episode_len < eval_episode_len`).

### TODO
- [x] **Train weighted BC + weighted ReBRAC on ds1/2/3 (+ ds4 baseline), eval on
      TARGET** — done 2026-06-09 (table above). Recovery-SUCCESS data helps BC
      (+15pt); failed-recovery hurts BC but helps ReBRAC.
- [ ] Multi-seed (≥3) to de-noise — single-seed 50-game evals are noisy; the
      ds1>ds4 BC gap is the only robust-looking effect.
- [ ] ReBRAC: early-stop on best-target + lower actor_lr / stronger BC coef to
      kill the end-of-training collapse; rerun ds4 ReBRAC to full 200k with
      `mp_eval=0` (the mp_eval EGL deadlock on GPU0 is what hung it twice).
- [ ] Co-train (§11) ds1 source-recovery + 20 target demos vs plain co-train /
      the §16 GatedGapAgent — does the recovery signal stack with co-training?
- [ ] Truncate D episodes at takeover onset + a fixed tail (cheaper ds2 that
      isn't dominated by 400-step failed-recovery) and re-test the BC #1-vs-#2.

---

## 20. Residual off-policy RL on a frozen target-demo BC base (ResidualAgent)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
A "naive residual RL" baseline ported from **ResFiT** (Ankile et al. 2025,
*Residual Off-Policy RL for Finetuning Behavior Cloning Policies*; ref at
`~/residual-offpolicy-rl`), adapted to this repo's state-based IBRL `QAgent` +
the sim2sim dynamics-gap setup. A frozen **base** BC policy `a_base(s)` (trained
on the 20 TARGET demos, `exps/mimicgen/threading_target_mlp_bc_n20_seed1`)
proposes a nominal action; a trainable **residual** RL actor outputs a small
bounded correction `delta(s, a_base) in [-action_scale, +action_scale]` and the
executed action is `a = clip(a_base + delta, -1, 1)`. The critic learns `Q(s,a)`
on the FULL combined action; the residual actor maximises `-Q(s, clip(a_base+delta))`.
Only the residual + critic train; the base is frozen.

The TWIST vs vanilla ResFiT: the residual is trained ONLINE in the **SOURCE**
sim but deployed/evaluated on the **TARGET**. Source dynamics is only a
partially-trustworthy proxy, so each source transition `(s,a,s')` is weighted by
the **GROUND-TRUTH dynamics-gap confidence** `c = exp(-gap/gap_tau) in [0,1]`
(counterfactual replay in the target sim — the SAME signal weighted BC §11 /
GatedGapAgent §16 / H2O+(gt) §14 use), and the critic's Bellman error is
c-weighted. So the residual is fit where the source dynamics matches the target
and down-weighted where it does not. This makes it the **residual-RL counterpart
of weighted BC under one identical gap signal**.

Two ResFiT design choices kept because they stabilise residual RL:
1. the residual actor's LAST layer is **zero-initialised** → `delta=0` at step 0
   → the policy IS the base BC policy (RL departs smoothly from the prior);
2. tiny `actor_lr=1e-6` + small `action_scale=0.1` keep the correction local;
   plus a **critic-only warmup** (`critic_warmup_steps`) before the actor starts.

### Where
- `rl/residual_agent.py` (NEW) — `ResidualAgent(QAgent)` + `ResidualConfig`.
  Reuses QAgent's `MultiFcQ` critic (input state+action) + target nets; REPLACES
  the standard `FcActor` with a residual actor whose input is `[state, a_base]`
  (obs_dim+action_dim) and whose last layer is zero-init (`_zero_init_last_layer`).
  Frozen base policy held in a LIST (`attach_base_policy`, NOT in state_dict;
  re-attached by path at eval). `_base_action` (handles tuple/chunk returns),
  `_residual` (= `action_scale * tanh-sample`), `_combined` (= clip(base+delta)),
  `act` (single + batched; eval=deterministic mean, train=sampled), and
  `update_residual(sim_batch, stddev, update_actor)` — c-weighted critic Bellman
  error (`gate_critic_weight`/`force_c`) + `clamp_q_target` [0,1] (sparse-reward
  unbiased, cf. §16) + DDPG residual actor loss with optional `residual_l2_coef`.
- `train_residual.py` (NEW, forked from `train_gated.py`) — `train_env`=SOURCE
  online, `gap_env`=TARGET sim (counterfactual GT-gap via `cf_utils` +
  `data_utils.state_gap`, carried on the action dict as `dyn_c`), `eval_env`=TARGET.
  Base loaded via `train_bc.load_model`. No source RL agent, no target demo
  buffer — the residual is learned purely from SOURCE online interaction, GT-gap
  weighted. `rl_train` gates the actor update on `global_step >= critic_warmup_steps`.
  `load_model` for standalone eval. `os._exit(0)` at the end (EGL teardown hangs).
- `evaluate/eval.py` — added `--mode residual` → `train_residual.load_model`.
- `release/cfgs/mimicgen/threading_d0_state_residual.yaml` (NEW) — demo config.
- `scripts/train_residual_threading.sh` (NEW) — `SEED/BASE_POLICY/FORCE_C/
  ACTION_SCALE/WARMUP` env vars + `"$@"` passthrough.
- `scripts/test_residual.py` (NEW) — synthetic-batch smoke test.

### How
```bash
source set_env.sh
bash scripts/train_residual_threading.sh                       # GT-gap weighted (default)
FORCE_C=1.0 bash scripts/train_residual_threading.sh           # control: plain residual RL on source
ACTION_SCALE=0.2 bash scripts/train_residual_threading.sh      # wider residual
RESUME=1 bash scripts/train_residual_threading.sh              # resume an interrupted run (same SAVE_DIR)
# raw:
python train_residual.py --config_path release/cfgs/mimicgen/threading_d0_state_residual.yaml \
    --residual.base_policy_path exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt \
    --save_dir exps/mimicgen/threading_residual_seed1
# standalone eval (TARGET, strict-success):
python evaluate/eval.py --weight exps/.../latest.pt --mode residual --num_game 100 \
    --shift_from_cfg 1 --strict_success 1
```
`state_stack` MUST be 1 (matches the base BC ckpt; nstep=1 for dyn_c alignment).

**Resume** (`--resume 1`): unlike the other RL trainers (which only warm-start
weights), `train_residual.py` does an EXACT resume. Every log step it writes
`<save_dir>/train_state.pt` next to `latest.pt` with `global_step/train_step/
global_episode` + `critic_opt`/`actor_opt` state + torch/cuda/numpy RNG +
`wandb_run_id`. `--resume 1` reloads `latest.pt` into the agent, restores all of
that, re-warms the replay (NOT persisted — cheaper than storing ~9 GB), and
continues the SAME wandb run (via `WANDB_RESUME=allow` + `WANDB_RUN_ID`). Asserts
if `train_state.pt` is missing (e.g. a run that predates this feature — the 1M-step
runs started before 2026-06-10 have no train_state.pt and cannot be resumed
exactly). Loaded with `weights_only=False` (it holds optimizer + numpy RNG, not
plain tensors). Resume can also EXTEND the budget: `RESUME=1 STEPS=2000000 ...`.

### Design choices
- **Residual actor input = [state, base_action]** (ResFiT): the residual can
  condition on what the base proposes. Implemented by widening the FcActor's
  obs_shape by action_dim and concatenating `a_base` into `obs["state"]`.
- **Zero-init last layer** → starts as the base policy. NOTE: this gates the
  gradient to all earlier layers through the zero last-layer weight, so the FIRST
  actor update only moves the last layer (expected; the smoke test asserts on the
  last layer, not the first).
- **GT-gap weights the CRITIC** (not the actor) — the dynamics-gap correction
  belongs on the Bellman error (which transitions to trust as a target proxy);
  `force_c=1.0` is the unweighted control arm. The Q-maximisation stays a plain
  mean. (Mirrors weighted BC weighting its BC loss; contrast §16 which also gates
  the backup + the policy.)
- **Pure SOURCE online, no target buffer** — the "naive" framing the user asked
  for: can a small residual learned on source (gap-weighted) move the frozen
  target-BC base on the target? No hybrid offline-RL demo buffer (unlike §14/§16).
- **clamp_q_target [0,1]** (ResFiT's `clip_q_target_to_reward_range` + §16): sparse
  0/1 reward + end_on_success ⇒ true Q in [0,1] ⇒ clamping is unbiased and severs
  the negative-divergence loop.

### Verification status
Smoke-tested 2026-06-09:
- ✅ `scripts/test_residual.py`: zero-init ⇒ `act == base` at init (max|diff|=0);
  act single→`[A]` & batched→`[B,A]` in [-1,1]; base policy EXCLUDED from
  state_dict; `update_residual` moves critic every step, actor only when
  `update_actor=True` (last layer), losses finite; `force_c=1.0`→`c_mean==1`;
  state_dict round-trip.
- ✅ End-to-end short run (120 steps, real base ckpt): base loads, warm-up rolls
  out the base in source, online GT-gap counterfactual (`gap/value≈0.0065`,
  `gap/confidence≈0.77` — matches the demo gap distribution §16), c-weighted
  updates (`residual/c_mean≈0.82`, `delta_abs≈0.008` — tiny, as expected from
  zero-init + scale 0.1), critic warmup → actor starts, TARGET+SOURCE eval
  (`source_score=1.0` since delta≈0 ⇒ pure base, which is strong on the easier
  source; `target_score` is the metric to grow), checkpoint saved, `os._exit(0)`.

### TODO
- [ ] Full run + TARGET strict-success vs: weighted BC (the IL counterpart under
      the same GT-gap weighting), the frozen base BC alone (~25% target), gated
      `force_c=1.0` ~13%, H2O+(gt). Headline: does adding a gap-weighted residual
      RL on source beat the frozen target-BC base on the target?
- [ ] `force_c=1.0` (unweighted) vs GT-gap weighted: does the gap weighting help,
      or does plain residual RL on source already transfer?
- [ ] Sweep `action_scale` (0.1 vs 0.2 vs 0.3) and `critic_warmup_steps`; a too-
      large residual can leave the base support and lose the prior's benefit.
- [ ] Optional: add a small TARGET demo buffer (hybrid, à la §16's real_batch)
      to anchor Q on true target returns — the natural next step if source-only
      residual RL underperforms.

---

## 21. Oracle dynamics-map + state-occupancy reweighting (sim2sim upper bound)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
A NEW method validating the sim2sim ORACLE upper bound for transfer. The GT-gap
(§16) only scalar-DOWN-weights source transitions by `c=exp(-gap/τ)`, which
conflates two orthogonal gaps and never USES the target information. Here we
instead FIX both gaps with the sim2sim oracle, then train IBRL **from scratch**
on the corrected data:
1. **DYNAMICS gap** `p_tgt(s'|s,a)` vs `p_src(s'|s,a)` — fixed by the **ORACLE
   MAP**: for every `(s,a)` the SOURCE online rollout samples, counterfactually
   replay it in the TARGET sim and store the TARGET result
   `(s, a, r_tgt, s'_tgt, done_tgt)`. The replay then holds UNBIASED
   target-dynamics transitions, so NO `c`-weighting is needed (the transition
   itself is target). The oracle map is literally `cf_utils.counterfactual_step`'s
   output — the same call §16 used for the gap, but here we keep `s'_tgt/r_tgt`
   instead of collapsing them to a scalar.
2. **OCCUPANCY gap** `d_tgt(s)` vs `d_src(s)` — the oracle map corrects each
   `(s,a)`'s dynamics but the state distribution is still `d_src` (the rollout is
   source-driven). A `StateOccupancyDiscriminator` D(s) (TARGET-demo states =
   positive, SOURCE rollout states = negative) gives a density ratio
   `w(s)=d_tgt/d_src=D/(1−D)`; the IBRL critic + actor losses are
   self-normalized-weighted by `w` so capacity concentrates on target-reachable
   states and source-only (unreachable) states fade out.

The agent is IBRL **from scratch** (random actor/critic — NOT source-init, unlike
§16) with a frozen TARGET-demo BC prior (`act_method="ibrl"`). SOURCE = online
queryable sim; TARGET = eval env + oracle counterfactual sim. The whole point:
if even the PERFECT map + PERFECT occupancy ratio can lift TARGET success, the
learned approximations (§ the residual-map / learned-ratio TODO) are worth
pursuing; if not, the bottleneck is elsewhere.

### Where
- `rl/oraclemap_agent.py` (NEW):
  - `TransitionReplayBuffer` — flat GPU ring buffer storing
    `(state, action, reward, next_state, bootstrap)`. **Why needed**: rela's
    `ReplayBuffer` derives a transition's `next_obs` from the NEXT obs pushed
    into the episode, so it CANNOT store a relabeled next state. The oracle map
    needs an arbitrary `s'_tgt` ≠ the source rollout's next obs, so rela is out.
    `bootstrap = 0` on target success else `γ` (end_on_success semantics).
  - `StateOccupancyDiscriminator` — spectral-norm MLP + BCE (label-smoothed,
    optional gradient penalty) anti-collapse; `weight(s)=clip(exp(logit),floor,w_max)`.
    `occ_weight_enabled=0` ⇒ returns ones (oracle-map-only ablation).
  - `OracleMapAgent(QAgent)` — `update_oracle(batch, weights, stddev, update_actor)`:
    IBRL bootstrap (`_act_ibrl` max-Q over {actor_target, bc_prior}) + `_clamp_q`
    [0,1] (sparse 0/1 + end_on_success, cf. §16) + self-normalized
    occupancy-weighted critic `(w·Σ_k td²).sum()/w.sum()` + weighted actor
    `−(w·minQ(s,π(s))).sum()/w.sum()`.
- `train_oraclemap.py` (NEW, forked from `train_gated.py`) — `train_env`=SOURCE
  online, `gap_env`=TARGET oracle sim, `eval_env`=TARGET. `_collect_step` does one
  source step then `_oracle_relabel` (counterfactual TARGET replay) and pushes the
  relabeled transition to `TransitionReplayBuffer`. `rl_train` trains the
  occupancy discriminator (source rollout states vs target demo states) then runs
  the weighted IBRL update. BC prior loaded via `train_bc.load_model`; demo states
  via `cf_utils.load_demo_episodes`. `load_model` for standalone eval.
- `evaluate/eval.py` — added `--mode oraclemap`.
- `release/cfgs/mimicgen/threading_d0_state_oraclemap.yaml` (NEW).
- `scripts/test_oraclemap.py` (NEW) — synthetic-batch smoke test.

### How
```bash
source set_env.sh
python train_oraclemap.py --config_path release/cfgs/mimicgen/threading_d0_state_oraclemap.yaml \
    --save_dir exps/mimicgen/threading_oraclemap_seed1
# ablations:
python train_oraclemap.py ... --oracle.occ_weight_enabled 0   # oracle-map only (no occupancy reweight)
# standalone eval (TARGET, strict-success):
python evaluate/eval.py --weight exps/.../latest.pt --mode oraclemap --num_game 100 \
    --shift_from_cfg 1 --strict_success 1
```
`state_stack` MUST be 1 (matches the BC prior + the §16 cf machinery). `nstep=1`
(single-step relabeled transitions). Cost: ~2× rollout (one extra TARGET env step
per online step for the oracle map).

### Design choices
- **Explicit transition buffer, not rela** — the relabeled `s'_tgt` is not the
  source rollout's next obs, which rela's episode-implicit-next cannot represent.
  TD(0) doesn't need trajectory continuity, so a flat `(s,a,r,s',boot)` buffer is
  exactly right (and stores state-only, so it's tiny).
- **No `c`-weighting after the oracle map** — the gated agent's `c` exists because
  source transitions are biased; relabeling makes them unbiased target
  transitions, so the dynamics correction is EXACT, not down-weighted. `c` is
  replaced entirely by the occupancy weight `w(s)` (a DIFFERENT, orthogonal
  correction: state distribution, not transition).
- **Occupancy ratio = D/(1−D), self-normalized** — the discriminator's optimal
  output gives the density ratio directly; self-normalizing `w` within the batch
  cuts variance (standard SNIS) at a small bias, fine for support-filtering.
  Spectral norm + label smoothing guard against the `D→{0,1}` collapse that would
  send `w` to `{0,∞}`.
- **IBRL from scratch (not source-init)** — the question is whether oracle-correct
  target data ALONE (no source-policy warm start) suffices; the target-demo BC
  prior supplies the IBRL max-Q prior so early rollouts aren't fully random.
- **clamp_q_target [0,1]** — sparse 0/1 reward + end_on_success ⇒ true Q in [0,1];
  clamping is unbiased and severs the negative-divergence loop (§16).

### Verification status
Smoke-tested 2026-06-12:
- ✅ `scripts/test_oraclemap.py`: TransitionReplayBuffer add/sample/sample_states
  shapes; StateOccupancyDiscriminator separates +2/−2 gaussians (acc>0.8,
  `w_tgt>w_src`, clipped ≤w_max), disabled→ones; `update_oracle` moves
  critic+actor, losses finite, actor only on `update_actor`; state_dict round-trip
  EXCLUDES the BC prior; uniform-weight (occ off) path trains.
- ✅ End-to-end short run (300 warm-up + 200 steps): BC prior loads, occupancy
  pool (1983 target demo states), oracle relabel logs `oracle/target_reward`,
  occupancy discriminator trains (`occ/loss`, `occ/w_mean≈0.79`, `acc_source≈0.77`),
  IBRL bootstrap active (`actor/bootstrap_bc≈0.24`), TARGET+SOURCE eval + ckpt save.
- ✅ Standalone `evaluate/eval.py --mode oraclemap` loads agent + BC prior, applies
  target shift, rolls out (score 0 on the 200-step toy ckpt, as expected).
**2026-06-12 cold-start finding (IMPORTANT).** A first FROM-SCRATCH run
(random actor/critic, target-BC prior only) DEADLOCKED: at 30k steps
`oracle/target_success=0`, `data/batch_R=0.0006`, `train_source_score=0`,
`critic_qt=0.517` (self-bootstrapped, no reward grounding). Mechanism: random
critic ⇒ IBRL max-Q mostly picks the random actor (`bootstrap_bc≈0.11`) ⇒
source rollout never succeeds ⇒ EVERY oracle-relabeled reward is 0 ⇒ the critic
has no reward signal ⇒ rollout stays random. A sparse-reward hard task has no
reward to learn from until a rollout stumbles into a target-success `(s,a)`,
which a random policy ~never does. **Fix: warm-start actor+critic from the
source RL ckpt** (`source_agent` + `init_actor_critic_from_source`, reusing
`train_rl.load_frozen_agent`). The source agent solves source (~98%), so source
rollouts succeed immediately and the oracle map gets target reward to learn
from: smoke went from 0→**9 target-success transitions in 300 warm-up steps**,
`batch_R` 0.0006→0.027, `source_score` 0→0.60. The occupancy discriminator was
healthy throughout (acc 0.83/0.91) — the deadlock was pure RL cold-start, not
the occupancy/oracle machinery. (This means the run is NOT pure from-scratch;
the source ckpt is a warm start, NOT an IBRL prior — the IBRL prior stays the
target-demo BC.)

NOT yet done: the full 400k runs' TARGET strict-success. Two source-init runs
in flight (2026-06-12): `exps/mimicgen/threading_oraclemap_srcinit_seed1`
(oracle map + occupancy reweight, GPU0, wandb `threading_oraclemap_srcinit_seed1`)
and `exps/mimicgen/threading_oraclemap_noocc_seed1` (oracle-map-only ablation,
`occ_weight_enabled=0`, GPU1) — their difference isolates the occupancy-reweight
contribution.

### TODO
- [ ] Full run + record TARGET strict-success vs: gated `force_c=1.0` ~13% / GT-gap
      gated / H2O+(gt) / weighted BC / the frozen target-BC prior (~25%). Does
      oracle-correct data + occupancy reweight beat the scalar-gap methods?
- [ ] Ablate `occ_weight_enabled` 0 vs 1 — isolate the dynamics-fix (oracle map)
      from the state-distribution-fix (occupancy reweight). **Watch the `w`
      histogram**: if >90% of source states are pushed to w≈0 the source/target
      occupancy barely overlap (the LWAIL-saturation failure mode,
      [[project-lwail-threading-saturation]]) — reweighting can only RESHAPE the
      overlap, not fill the unreachable窟窿; that's the signal to add branch-rollout
      (generate target-reachable data in the target sim) instead.
- [ ] If oracle (perfect) works: swap the oracle map for a learned residual
      dynamics model (`s'_tgt = s'_src + g_θ(s,a)`, demo-trained, ensemble +
      pessimism) and the oracle ratio for a learned discriminator — measure the
      drop from perfect to learned (the actually-deployable sim2real version).
- [ ] Sweep `occ_w_max` / `occ_w_floor` / `occ_train_per_rl`; try `occ_grad_penalty>0`.
- [ ] Optional branch-rollout (§ the occupancy-window fix): from each `s'_tgt`,
      roll a few steps in the TARGET sim to actively GENERATE target-reachable
      states, shifting the visitation distribution toward `d_tgt` rather than only
      reweighting `d_src`.

### 2026-06-12 update: source-init oracle map ALSO collapses (actor drift)
The source-init runs (§21 fix) cured the cold-start deadlock (warm-up gets 88
target-success transitions) but then **collapsed a different way**: `source_score`
fell 0.98→0.10 by 10k steps and to 0 by 50k; `target_score`=0 throughout;
`train_source_score`=0, `batch_R≈0.001`. Mechanism = **actor drift under sparse
target reward**: target-success transitions are only ~3% (60/1983 in the demos,
and source rollout must pass a "target-single-step-success" state), so the critic
gets almost no reward grounding; the free actor maximizes `-Q(target critic)`,
`clamp_q[0,1]` caps Q's magnitude but not the actor chasing Q's OOD over-estimates,
so it drifts off the good source-init policy → source rollout degrades → new
target-success transitions dry up → positive-feedback collapse to 0.
**The relabel itself is CORRECT** (`scripts/verify_oracle_relabel.py`, golden
test on demo (s,a): demo success steps reproduce reward=1 100%, set_state
fidelity med 3mm, next-state reproduce med 9mm — the max-9cm drift is the
intended `qacc_warmstart=0` cold-start divergence in contact). So oracle map
fixes dynamics but NOT the orthogonal "sparse target reward → critic ungrounded
→ actor diverges" problem. The plain-IBRL oracle map is now just a **baseline**
(`threading_oraclemap_srcinit_seed1`); the fix is §22.

---

## 22. Residual RL on a frozen target-demo BC base + oracle map (anti actor-drift)

> ⛔ **代码已归档，未随本仓库发布** —— 本地 `_archive/`（见 §34）。下文提到的文件路径在本仓库中不存在。

### What
The fix for §21's actor-drift collapse: combine §20 residual RL (frozen base BC
policy `pi_base` = the target-demo BC + a small, bounded, zero-init residual
actor) with §21's oracle dynamics-map relabel. The residual structure ANCHORS
the policy at `pi_base` — the residual can only make a bounded
`delta∈[-action_scale,+action_scale]` correction, so the policy CANNOT drift off
into the ungrounded-Q regions that collapsed plain IBRL (§21). The transition is
still oracle-relabeled to the TARGET (unbiased target dynamics), and a state-
occupancy discriminator reweights the residual critic+actor toward target-
reachable states. reward / next-state / bootstrap in the update are all TARGET.

### Where
- `rl/residual_agent.py::ResidualAgent.update_residual_oracle(batch, weights,
  stddev, update_actor)` (NEW method) — occupancy-weighted version of
  `update_residual`: `batch` holds ORACLE-relabeled TARGET transitions (from the
  §21 `TransitionReplayBuffer`), `weights` is the occupancy ratio `w(s)` (self-
  normalized). Same residual actor (zero-init, bounded, input `[state, a_base]`)
  + `_clamp_q[0,1]` as §20, but no c-weighting (dynamics already target).
- `train_oracle_residual.py` (NEW, fuses train_oraclemap §21 data path + §20
  residual agent) — SOURCE online rollout → oracle counterfactual relabel →
  `TransitionReplayBuffer`; trains the `StateOccupancyDiscriminator`; `rl_train`
  calls `update_residual_oracle` with `critic_warmup_steps` gating the actor.
  Frozen base BC via `train_bc.load_model`. **Full exact-resume** (`--resume 1`,
  `train_state.pt`: step + optimizers + occ discriminator + RNG + wandb run id;
  buffer re-warmed). `load_model` for standalone eval.
- `release/cfgs/mimicgen/threading_d0_state_oracle_residual.yaml` (NEW).
- `evaluate/eval.py` — `--mode oracle_residual`.
- **Resume was also added to `train_oraclemap.py`** (§21) — same `train_state.pt`
  pattern, `--resume 1`.

### How
```bash
python train_oracle_residual.py --config_path release/cfgs/mimicgen/threading_d0_state_oracle_residual.yaml \
    --save_dir exps/mimicgen/threading_oracle_residual_seed1
python train_oracle_residual.py ... --resume 1                 # exact resume
python train_oracle_residual.py ... --oracle.occ_weight_enabled 0   # oracle-map-only
python evaluate/eval.py --weight exps/.../latest.pt --mode oracle_residual --num_game 100 \
    --shift_from_cfg 1 --strict_success 1
```

### Design choices
- **Residual base = anti-drift.** §21 plain IBRL drifted because the free actor
  had no anchor; the residual is bounded around the frozen target-demo BC, so the
  policy stays in BC support — directly removes the collapse mechanism.
- **No source warm-start needed.** Unlike §21, the base BC already gives a
  competent policy from step 0 (zero-init residual ⇒ policy==BC); the critic
  cold-starts but `critic_warmup_steps` lets it calibrate on the base before the
  residual moves.
- **Occupancy reweight kept** (the §21 state-distribution fix), `occ_weight_enabled=0`
  isolates oracle-map-only.

### Verification status
Smoke-tested 2026-06-12:
- ✅ `update_residual_oracle`: occupancy-weighted critic+actor, finite losses,
  residual moves (`delta_abs`), clamp_q.
- ✅ End-to-end short run: base BC loads, oracle relabel (warm-up target-success),
  occupancy disc trains (`occ/w_mean≈0.81`), residual update (`delta_abs≈0.008`),
  `train_state.pt` written.
- ✅ Resume: `--resume 1` reloads agent+optimizers+RNG, `[resume] continuing from
  global_step=60`, continues the same wandb run.
NOT yet done: the full 400k run's TARGET strict-success. Two runs in flight
(2026-06-12, both resume-capable + wandb): `threading_oracle_residual_seed1`
(the fix, GPU0) and `threading_oraclemap_srcinit_seed1` (plain-IBRL baseline,
GPU1, expected to actor-drift).

### TODO
- [ ] Full run: residual+oracle vs the §21 plain-IBRL baseline (expected to
      collapse) vs frozen base BC (~25%) vs gated `force_c=1.0` ~13%. Does the
      bounded residual + oracle-correct target data beat the frozen BC base?
- [ ] `occ_weight_enabled` 0 vs 1; `action_scale` 0.1/0.2/0.3; `critic_warmup_steps`.
- [ ] If it works with the oracle map, swap in the learned residual dynamics
      model (§21 TODO) for the deployable sim2real version.

---

## 23. Model-Based IBRL (MBPO-on-IBRL) — single-domain sample efficiency

> ⚠️ **部分归档** —— `rl/mb_ibrl_agent.py` **保留**（§30 的 `scripts/diag_dyn_rollout_error.py` / `diag_latent_qres_sensitivity.py` 用其 `DynamicsEnsemble`），但 `train_mb_ibrl.py`、YAML 与 `scripts/` 已归档到 `_archive/`（见 §34）。

### What
A Dyna-style model-based extension of the base IBRL `QAgent` to make it converge
faster / be more sample-efficient on a single domain (NOT a transfer method —
contrast §14/§16/§20/§21/§22 which all use a source→target gap). Keep the IBRL
recipe untouched (rela real replay, flow-BC prior, `mix_rl_rate`, RED-Q
`num_critic_update`, `_act_ibrl` max-Q bootstrap) and ADD:
1. a **probabilistic state-space dynamics ensemble** trained on the agent's own
   real transitions (PETS/MBPO), predicting `(s,a) -> (Δs Gaussian, reward, done)`;
2. short **synthetic Dyna rollouts** branched from real states, actions chosen by
   the SAME IBRL policy (`agent.act`), next-states/rewards from the ensemble;
3. **mixed real+synthetic IBRL updates** — each critic/actor step sees a batch
   that is `real_ratio` real (rela) + the rest synthetic, so the value function
   propagates over far more `(s,a,s')` per real env step.

State-only (`use_state=1`, `state_stack=1`), nstep stays as the IBRL config (the
synthetic transitions are single-step, carrying their own `bootstrap`).

### Where
- `rl/mb_ibrl_agent.py` (NEW):
  - `DynamicsEnsembleConfig` + `DynamicsEnsemble` — K parallel MLPs (`EnsembleLinear`,
    weight `[E,in,out]` + bmm), input `[normalize(s), a]`, output per member
    `Δs_mean[obs] / Δs_logvar[obs] / reward / done_logit`. Learns the **normalized
    next-state delta** (`(s'-s)/std`) so the heterogeneous 37-dim state scales
    don't dominate the Gaussian NLL. PETS soft-clamped learnable logvar bounds.
    State z-score in registered buffers (`norm_state_mean/std`, identity until
    `set_normalization()`). `train_step`: per-member bootstrap resample → Gaussian
    NLL on Δs + logvar-bound reg + `reward_pos_weight`-weighted reward MSE (sparse
    0/1 success → upweight rare r>0) + done BCE. `predict`: random elite member per
    sample, sample Δs from its Gaussian, `s'=s+Δs·std`, `done=sigmoid(logit)>0.5`.
  - `FlatReplayBuffer` — tiny batched GPU ring buffer of single-step
    `(s,a,r,s',bootstrap)`; `add_batch` (batched), `sample` returns a **state-only**
    `Batch` (compatible with `Batch.merge_batches` and the QAgent state-path update),
    `sample_states`, `state_stats`. Used for BOTH the real transitions the model is
    fit on AND the synthetic transitions it generates. (rela can't store an
    arbitrary model-predicted next state, same reason §21's TransitionReplayBuffer
    exists; this one adds a batched insert for the 512×k synthetic writes.)
  - `MBIBRLAgent(QAgent)` — thin subclass holding `self.dynamics` (so it's in
    `state_dict` + moves with `.to()`); critic/actor/IBRL/`value` inherited
    unchanged. `train_dynamics(batch)` (done = `bootstrap<=0`),
    `generate_rollouts(start_states, horizon, stddev, discount)` (branches rollouts,
    done-truncates, **temporarily nulls `self.stats`** because `_act_ibrl`'s
    logging path asserts `bsize==1` for single-env train collection).
- `train_mb_ibrl.py` (NEW, subclasses `train_rl.Workspace` + `train_rl.MainConfig`):
  reuses `_setup_env`/`eval`/`log_and_save`; overrides `__init__` (build
  `MBIBRLAgent`, two `FlatReplayBuffer`s), `warm_up` (also feeds `real_flat`),
  `train` (fits dynamics normalization from warm-up data, then per env step: collect
  → every `model_train_freq` steps train the ensemble `model_train_steps` batches +
  generate `rollout_batch_size`×`rollout_length` synthetic transitions → mixed
  `rl_train`), and `rl_train` (merges real + synth, synth as `batch1` so the merged
  batch is state-only). `load_model` for standalone eval (dynamics unused at eval).
  MBPO knobs on `MainConfig`: `dynamics`, `model_batch_size`, `model_train_freq`,
  `model_train_steps`, `model_warmup_steps`, `rollout_length`, `rollout_batch_size`,
  `synth_buffer_size`, `real_flat_size`, `real_ratio`.
- `release/cfgs/mimicgen/threading_d0_state_mb_ibrl.yaml` (NEW) — mirrors
  `threading_d0_state_ibrl_source_strict.yaml` (state_stack=1, flow-BC prior,
  mix_rl_rate=0.5, RED-Q num_critic_update=5, strict success, 400k) + the dynamics
  + MBPO blocks.
- `evaluate/eval.py` — `--mode mb_ibrl` → `train_mb_ibrl.load_model`.
- `scripts/train_mb_ibrl_threading.sh` (NEW) — `SEED/BC_POLICY/STEPS/REAL_RATIO/
  ROLLOUT_LENGTH/ROLLOUT_BATCH/UTD/GPU` env vars + `"$@"` passthrough.
- `scripts/test_mb_ibrl.py` (NEW) — synthetic-batch smoke test.

### How
```bash
source set_env.sh
bash scripts/train_mb_ibrl_threading.sh                       # MB-IBRL
REAL_RATIO=1.0 bash scripts/train_mb_ibrl_threading.sh        # model-off ablation (= IBRL on the flat real buffer)
ROLLOUT_LENGTH=3 UTD=10 bash scripts/train_mb_ibrl_threading.sh
# raw:
python train_mb_ibrl.py --config_path release/cfgs/mimicgen/threading_d0_state_mb_ibrl.yaml \
    --save_dir exps/mimicgen/threading_mb_ibrl_seed1
# standalone eval (SOURCE, strict-success):
python evaluate/eval.py --weight exps/.../latest.pt --mode mb_ibrl --num_game 100 --strict_success 1
```

### Design choices
- **Reuse IBRL wholesale**: the dynamics ensemble + synthetic buffer are bolted on;
  `QAgent.update` / `_act_ibrl` / the critic/actor are byte-for-byte the base IBRL.
  Synthetic transitions are state-only `(s,a,r,s',bootstrap)` so they feed the
  state-path `update_critic` (which only touches `obs["state"]`) directly, and the
  BC prior (consumes state) bootstraps on the model-predicted next state.
- **Normalized-delta dynamics target**: generative regression on a 37-dim state with
  heterogeneous scales; z-scoring + predicting the delta keeps the NLL well-conditioned.
- **Short rollouts + `real_ratio` not too low + `reward_pos_weight`**: the sparse
  0/1 + `end_on_success` reward makes the reward/done heads' positive class rare, so
  synthetic data can bias Q low. `rollout_length=1` (default), `real_ratio=0.5`
  (keep real reward signal), and upweighting r>0 in the reward loss mitigate this.
- **Model-off ablation = `real_ratio=1.0`** (no synthetic data, IBRL trains on the
  flat real buffer) isolates the model's contribution vs the rest of the pipeline.
- **UTD via the existing `num_critic_update`/`update_freq`** (no new knob): MBPO's
  high update-to-data ratio is just more `rl_train` critic updates per env step,
  now fed by the synthetic buffer.

### Verification status
Smoke-tested 2026-06-16 (`scripts/test_mb_ibrl.py`, all pass):
- ✅ `FlatReplayBuffer` add_batch/sample/wraparound/state_stats.
- ✅ `DynamicsEnsemble` train_step (finite NLL, params move) + predict (shapes,
      bool done, 1-step toy error 0.03); learns a smooth toy dynamics.
- ✅ `generate_rollouts` → synthetic transitions on device, done-truncation fires
      (128 starts × horizon 3 → 233 transitions).
- ✅ mixed real+synth `Batch.merge_batches` (synth=batch1 ⇒ state-only) → finite
      `agent.update`, critic+actor move.
- ✅ `state_dict` round-trip: dynamics IN, BC prior NOT in.
- ✅ YAML parses into the subclassed `MainConfig` (nested dynamics + MBPO knobs).
NOT yet done: the sample-efficiency curve vs vanilla IBRL (`train_rl.py` on the
same base config) + the model-off (`real_ratio=1`) ablation on Threading_D0 source
strict-success; the `rollout_length`/`real_ratio`/`num_critic_update` sweeps.

### TODO
- [ ] **Headline**: success-rate vs env-steps on Threading_D0 source, fixed seed —
      MB-IBRL vs vanilla IBRL (`threading_d0_state_ibrl_source_strict.yaml`) vs
      MB-IBRL `real_ratio=1.0` (model-off). Expect MB-IBRL to hit the same strict
      success in fewer env steps.
- [ ] Sweep `rollout_length∈{1,3,5}`, `real_ratio∈{0.3,0.5,0.8}`,
      `num_critic_update∈{5,10,20}` (UTD), `model_train_freq/steps`.
- [ ] Watch the reward/done model under sparse success: log
      `model/reward_loss`, `model/synth_reward`; if synthetic reward collapses to 0,
      raise `reward_pos_weight` or shorten rollouts.
- [ ] Optional model-based value expansion (MVE/STEVE) target as a switchable
      add-on (use the model only to k-step-expand the TD target) — more robust to a
      biased reward model, smaller speed-up.
- [ ] Elite selection by holdout NLL (currently all members are elites); pixel-mode
      (needs an image dynamics model + encoder).

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
# standalone eval of an adapted ckpt (MPPI — forced via the saved cfg? NO: the
# adapt cfg.yaml is AdaptConfig; eval via evaluate/eval.py --mode simdist reads
# the STAGE-1 cfg with use_planner=0. To MPPI-eval an adapted ckpt, copy it into
# a dir whose cfg.yaml has simdist.use_planner=1, or eval inside the adapt run.)
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

## 27. MPPI 的天花板 = 模型误差 / 任务容错窗口（2026-07-13~14）

### ⚠️ 先读：本节曾被两个 bug + 系统性的小样本误判彻底污染，以下是修正后的版本
1. **`probe_env` 从未 `reset()`** ⇒ domain shift 从未 apply（`restore()` 只写 qpos/qvel，不碰模型）
   ⇒ 所有反事实 rollout 跑在 **SOURCE 物理**上。作废了一整批 oracle 数字。
2. **`evaluate()` 硬编码 `planner.plan`，不看 `use_planner`** ⇒ 所谓的 "policy 裸跑" **实际还是
   MPPI** ⇒ "planner 净贡献 = 精确 0.000" 是 bug 的签名（同一个东西减自己），我却为它编了一套
   机制解释。**是"为什么 training log 只有 0.7-0.8"这个质疑救的场。**
3. **40 局根本不够**：SE≈0.06-0.08，而当天几乎所有推理都活在 ±0.1 量级。一天内被小样本骗了 7 次。
   最惨的一次：40 局给出"planner 把 goodpolicy 拖垮 −0.075"，**300 局的真相是 +0.097（符号反了）**,
   而那条数据是"argmax 主动有害 / winner's curse / 限制搜索半径也救不了"整条论证线的**唯一支柱**。
**教训：① 结果"太漂亮"（精确的 0）时先查工具，别忙着解释；② 和外部记录矛盾时，怀疑自己的测量，
不要把矛盾解释掉；③ 这个项目里任何 ±0.1 以内的结论都必须 300 局。**

### 主结论（全部 300 局）
| 配置 | policy-only | **MPPI（天花板）** | planner 净贡献 | 容错窗口 | 模型 |
|---|---:|---:|---:|---|---|
| source 原样 | 0.713 | **0.973** | **+0.260** | 3.0 mm/边 | source |
| **source + 小环(0.85) + 环摩擦(2.0)** | 0.443 | **0.843** | **+0.400** | **1.8 mm/边** | **source（不变）** |
| target（弱 BC） | 0.417 | 0.850 | +0.433 | 1.8 mm | target-adapted |
| target（goodpolicy） | 0.750 | 0.847 | +0.097 | 1.8 mm | target-adapted |
| **target + ORACLE dynamics** | — | **1.000** (100/100) | — | 1.8 mm | **真实模拟器** |

**⇒ 天花板 = f( 模型误差 / 任务容错窗口 )**
- 容错窗口 ≫ 模型误差（source 原样）→ 误差被物理吸收 → **0.97**
- 容错窗口 ≈ 模型误差（小环 / target）→ 误差暴露 → **0.85**
- 模型误差 = 0（oracle）→ 无论容错多小 → **1.000**

两个**正交的单变量实验**夹死了这个结论：
- **只改容错**（`fx_C`：同一个 source 模型、同一套 source 物理，只把环孔 16→13.6mm + 环摩擦 ×2）
  ⇒ 天花板 0.973 → **0.843**，精确落到 target 的水平。
- **只改模型**（oracle：同一个 target 任务、同一个 planner、同一个 value、同一个 proposal，
  只把 rollout 换成真实模拟器）⇒ 天花板 0.850 → **1.000**（Fisher p<1e-6）。

**⇒ target 的 0.85 天花板既不是 domain gap（`fx_C` 用 source 模型/物理照样 0.843），也不是
proposal 太弱（换成 0.750 的 goodpolicy 仍是 0.847），而是"learned dynamics 的误差恰好与
1.8mm 的容错窗口同量级"。**

### planner 是有用的（我一度全盘搞反）
MPPI **在每个配置下都是净正的**（+0.10 ~ +0.43）。它是一个**proposal 提升器**：把任何 proposal
拉向那个由"模型精度 ÷ 任务精度需求"决定的天花板。净贡献 = 天花板 − proposal，所以 proposal 越差
看起来贡献越大（target 弱 BC +0.43），这只是算术。
**注意 target 两行：proposal 从 0.417 换成 0.750，MPPI 输出几乎不变（0.850 → 0.847）——这就是
"天花板"最直接的证据。**
- **无 proposal（`num_base=0`，32/32 全随机候选）在两个域都是 0.000** ⇒ planner **没有从零规划
  的能力**，它只能在一个已有策略的邻域里精修。

### domain gap ≠ model error（两条独立证据）
- **消除 gap 无效**：`target_distill_v4`（encoder/dyn/value/policy 全在 150k target 轨迹上重训，
  零 gap）稳态仍 **0.83**。
- **带着 gap 也无所谓**：用**物理错误的 source 模拟器**做 rollout（那个 bug 版本意外测到的）
  照样 **1.000**。
⇒ 瓶颈不是 source→target 的 gap，是**"模型是学出来的"这件事本身**。

### ❌ 一整批被推翻的中间结论（记录下来免得重蹈）
1. **"planner 在好 proposal 上主动有害"** —— 40 局 −0.075，**300 局 +0.097，符号反了**。
   连带作废：winner's curse / 误差与动作极端程度相关 / 限制搜索半径也救不了（all-proposal 0.700、
   混合池 0.750 都是 40 局，不可信）。
2. **"SNR = Var_a[y]/Var_a[ε̃] 决定 planning 成败"** —— **证伪**：source 的 SNR **更低**
   （0.752 vs target 1.012）、集内排序**更差**（0.524 vs 0.752），天花板却**高得多**（0.97 vs 0.85）。
   排序质量与成绩脱钩。（H=5 是模型原生 horizon：SNR 1.01 > H=3 的 0.52，信号↑43% 误差↓27%。）
3. **"校准 value 到想象 latent"** —— 4 次全败（0.550 / 0.675 / 0.175 / 0.250 vs baseline 0.825）。
   held-out 集内排序 **0.420 < stock 0.444**。数学上：`argmax` 误差界是 `2·sup|ε|`（**上确界**，
   不是均值），所以 R²/MSE/Spearman 这些平均型指标全部失效；而拟合出来的读出头没有物理先验，
   误差独立且尾部无界，正是 `sup|ε|` 最大化的形态。
4. **"在候选分布上 finetune dynamics"** —— SNR ×3~7（0.55→3.9 / 0.98→3.3，held-out 验证），
   成绩却 **掉 20-32 点**（4 次实验、2 个 horizon、2 个 loss，零例外）。**分布漂移 + argmax 正反馈**：
   把模型在当前候选分布上修准，planner 立刻走到别处，argmax 主动搜索新模型的高估点。
   **Q-learning 是负反馈（Bellman 压缩，有不动点）；planning 是正反馈（打地鼠，无不动点）。**
5. **"decision-aware loss（只惩罚 `Var_a[ε]`，放过 bias）"掉 27.5 点，机制 = 自由 bias 失锚
   （2026-07-15 用 on-policy eps 判决实锤，推翻旧"与 MSE 无显著差异 ⇒ 噪声"的存疑结论）。**
   旧诊断的致命错误:训练 + held-out 指标都在**固定 cache `calib_data2.pt`** 上算,而那套候选是
   **stock planner 一次性采下来冻结的**(`train_value_calibration.py::collect`);dynamics 一被
   var-finetune 改动,planner 的 CEM elites 就漂离这套 cache。所以"held-out `Var[ε̃]` 降 8×、
   SNR 0.52→4.18"测的是 **stock 走过的老路**,成绩测的是 **var 自己搜的新路**——两套分布不是一个。
   `scripts/diag_onpolicy_eps.py`(对 stock/var 各自在**它自己 planner 实际搜出的 on-policy 分布**上
   重算 eps)判决:
   ```
                        Var[eps~]     sup|eps~|   Var_s[eps_bar]   SNR     score
   cache-val(stock路)   0.00437       0.0898      0.00369          0.52
     → var             0.00054(0.12×) 0.0461      0.00680          4.18    ← 假象
   stock @ stock-onpol  0.00024       0.0294      0.00032          1.28    0.825
   var   @ var-onpol    0.00022(0.94×) 0.0264     0.00647(19.9×)   1.16    0.550
   ```
   ① cache 上 var 的全部优势(Var[ε̃] 8×↓、SNR 8×↑、sup 2×↓)**在各自 on-policy 分布上全部归零**
   (0.94×/1.16 vs 1.28,SNR 甚至略降)——纯 in-distribution 假象。② var 在自己路上唯一改变的量是
   **`Var_s[ε̄]` 爆炸 20×**(0.00032→0.00647):var-loss 只压集内方差、完全放开 bias,在固定 cache 上
   不伤 argmax(bias 是集内常数),但把 value 的**绝对水平逐状态标定拆了**;不同候选到达不同终点,失锚
   的 bias 在那些终点不再是常数 ⇒ 重新变成 argmax 最恨的动作依赖误差。③ **MSE"意外更鲁棒"的机制**:
   MSE 连 `ε̄` 一起钉住,绝对形状不散,planner 走到 off-distribution 时误差增长温和。④ 悖论不是测量坏了
   (与 `probe_env`/`evaluate` 真 bug 不同),是**推断链断了**:`held-out 均值型误差↓` ⇏ `planner
   on-policy 的 sup + 绝对标定↑`,被 argmax 分布漂移 + 自由 bias 双重脱钩。候选二(y 的 target 与
   planner 打分是否同一个 value head)已排除——`eps=V̂(f^H)−V̂(encode(s_real))` 用的是同一个 `V̂`,
   只隔离 dynamics 误差。**通用教训:任何"改了模型 → 在旧固定候选集上量指标"的诊断都被分布漂移污染;
   凡是评 planner/argmax 的量,必须在当前模型自己搜出的 on-policy 分布上、且看 `sup|ε̃|`+`Var_s[ε̄]`
   而非均值型的 `Var[ε̃]`/MSE/R²。** 旧数字(H=3 好 12.5 / H=5 差 12.5)是 40 局噪声,作废。
   **⑤ 真正的根因 = value head 零空间(2026-07-15 续,`scripts/diag_latent_fidelity.py` +
   `actsens_clean.py`,四测量锁定,推翻了中途的"宽轮 ε̃"和"plateau 坍缩"两个错误假设)。**
   `Var_s[ε̄]` 只是症状;测 latent 几何保真度直接抓根:
   ```
   ‖f^H − encode(s_real)‖   stock 0.21   var 2.15   (10×, ≈latent 向量长 3.6 的 60%)  ← 几何被摧毁
   Var_a[f^H]  (几何散布)    stock 0.000116  var 0.000874  7.52×   ← 候选散得更开(否掉"动作失敏/坍缩")
   Var_a[V̂(f^H)] (价值散布)  stock 0.005415  var 0.001618  0.30×   ← 价值反而更平
   ```
   `Var_a[f^H]↑7.5×` 而 `Var_a[V̂(f^H)]↓3.3×` ⇒ **var 把动作变化全灌进了 value head 的等值面
   (零空间)**:几何自由乱动、价值却纹丝不动。机制一句话:**var-loss=min `Var_a[V̂(f^H)−y]` 的最优解
   是"把动作变化倒进 value 等值面"——几何(planner 真正 argmax 的东西)被彻底毁掉,而 loss 只看 value
   空间完全察觉不到。** 它报告的"ε̃↓8×/SNR↑"测的是 value head 对几何误差的盲区,不是模型质量。
   **推广:value-aware 的 dynamics 学习对 planning 本质不安全——value 函数的零空间给了模型一个免费
   作弊方向,任何 value-空间 loss(含 decision-aware 的集内方差形式)都能靠它最小化而毁掉状态几何;
   唯一安全的 finetune loss 是状态空间 consistency(= stock 本来就用的)。** 这也判死了两条解法:
   B1(advantage/差分 value)对 MPPI 是恒等变换(CEM 的 topk+softmax 对 per-call 常数平移不变,
   baseline 抵消、选出的动作逐位不变);A(MSE+`Var_s[ε̄]` 正则)最多救回 stock 附近但任何 dynamics
   finetune 都被 §27#4 支配、且 value-空间 loss 仍能钻零空间。**若上 B2 pessimism,必须惩罚状态几何
   不确定性(ensemble 对 `f^H` 的 disagreement),不能惩罚 value 不确定性——后者恰好对零空间几何误差
   全盲(这正是本次失败的形态)。** ⚠️ 复用 `AdaptWorkspace(cfg)` 的临时脚本:它 `__init__` 里
   `sys.stdout=common_utils.Logger(...)` 会**劫持 stdout**,print 结果要么走 `sys.stderr`,要么去
   `<save_dir>/train.log` 里捞(能跑通的 diag 脚本都用 `log()=print(...,file=sys.stderr)`)。
6. **"chunk-Q 能绕过 dynamics"** —— `(s₀,a)→y` 的 held-out 集内排序 0.619，只比 stock 的 0.503
   好一点，远不及 oracle 的 1.0。而 `(s₀,a)→s_H` 在 MuJoCo 里是**确定映射、信息充足**——**学不出来
   而已**。问题从来不是"信息丢失"，是"这个函数太难学"。

### 📊 Horizon（learned dyn, n=40）：H=1 **0.600** < H=3 0.825 < H=5 **0.850**
复合误差随 H 增长，但规划的信息增益增长更快。H=5 是 world model 的原生 consistency horizon。

### 工具
- `scripts/eval_oracle_dyn_mppi.py` —— arms `policy`/`learned`/`oracle`（真实 sim rollout）/
  `random`（J 换随机数）× `--value_agent` / `--q_reduce min|mean` / `--terminal q` / `--num_base`
  / **`--probe_workers N`**（sim-only worker 并行候选 rollout，逐位一致，8 workers ≈ 8.8x）。
- `scripts/diag_source_snr.py` —— source 域的 policy-vs-MPPI + ε 分解；`--ring_size_scale` /
  `--ring_friction_scale` 可在 source 物理上**单独改容错窗口**（关键的单变量实验）。
- `scripts/diag_error_decomposition.py` —— `ε = ε̄(集内常数, argmax 免疫) + ε̃(集内波动, 唯一有害)`。
  ⚠️ 在**固定 cache** `calib_data2.pt`(= stock planner 采下冻结的分布)上算,改了 dynamics 后它
  测的是老分布 —— 见 §27 第 5 条的分布漂移陷阱,评 var-finetune 请改用 `diag_onpolicy_eps.py`。
- `scripts/diag_onpolicy_eps.py` —— **判决"改模型后旧 cache 指标改善是否假象"**。对 stock 与
  var-finetuned dynamics,各自在**它自己 planner 实际搜出的 on-policy 分布**上重算 eps,报
  `Var[ε̃]`+`sup|ε̃|`(argmax 真正的约束)+`Var_s[ε̄]`(bias 失锚签名)+SNR,并对照固定 cache-val。
  实锤了 var-loss 的"cache 上 8× 改善"是 in-distribution 假象、掉分机制是 `Var_s[ε̄]` 爆炸 20×。
- `train_value_calibration.py` / `train_decision_aware_dyn.py` —— 上面 3、4、5 条失败方法的实现。
- ⚠️ `state_only_no_render` **会改变物理**（同一 sim_state + 同一动作，终端状态差 1.5）。原因未查明。
  **别用，并怀疑任何用过它的历史实验。**

### 代码改动（都向后兼容）
- `rl/q_agent.py::value(obs, use_ibrl=False)` —— 新增 IBRL 混合 max-Q value（默认 False ⇒
  SGFT/H2O+/gated 的 `V_src` 语义不变）。用同一 online critic 既选又评（不复用 `_act_ibrl`：
  它用 `critic_target` argmax，与 V 读的 online critic EMA 错开，会破坏 `V_hybrid ≥ V_rl`）。
- `train_rl.py::attach_ibrl_prior()` —— 把 ckpt 训练时的 prior 加载回来（`bc_policy` 或
  `sgft_source_agent` 两类），不碰 `act_method`。
- `train_simdist.py::source_value_ibrl=1`（默认开）—— value 蒸馏目标改为专家的 **hybrid** V
  （与 policy 头克隆的 hybrid 动作对齐）。⚠️ 实测语义正确但**实质无效**（demo-state 上
  Spearman(V_bare,V_hybrid)=0.998）。

### 2026-07-16 悲观 value 的 GO/NO-GO：value ensemble 对主导误差**结构性失明**（实测判死）
针对 MRS.Q（`2601.21306v2.pdf`，min-over-value-ensemble 悲观）问"把 V 做成 ensemble 能不能破
0.85"。在 source replay 上拟合 10 头 MC-return value ensemble（bagging+独立 init，
`scripts/train_value_ensemble.py`），在 target adapt run（`long_h5`，strict 0.80）上跑**真实
deployed MPPI**，每个 committed plan 同时记模型想象终端 `ẑ_H=decode(rollout)` 与真实反事实终端
`s_H`（`scripts/diag_value_ensemble_explain.py`，on-policy ⇒ 躲开 §27#5 的 cache 分布漂移陷阱）。
n=**3875** 对，统计实（Spearman SE≈0.016，非 40 局噪声）。**判死，且比数学预测更狠：**
- **`Spearman(disagree@ẑ_H, 几何误差 ‖ẑ_H−s_H‖) = −0.30`（负相关！）** —— ensemble 在 dynamics
  **越错**的地方**越自信**。这是"流形上合理幻觉"的直接签名：learned dynamics 出错时漂向的是
  value 流形内、所有头一致高分的状态，不是怪异 OOD 态。悲观不只是弱，是**指反了**。
- `Spearman(disagree, |val_err|)=+0.40` 但 `R²=0.12`（只解释 12%），且 **min-push 与 signed
  val_err 相关 −0.07（完全不对准）**：min 把 61% 的高估 +0.245 一律压成 −0.157 的对称低估，净
  |err| 只降 0.030 —— 行为≈**近似常数平移**，而常数平移对 argmax 不变（§27 的 ε̄）⇒ 对 planning
  零贡献。
- 想象态 vs 真实态的分歧 **1.07×**（分不清想象与现实）；**几何误差的 54% 落在分歧最低的一半**
  （瞎的多数）。
⇒ **§3 分解的 ① 项（dynamics 输入误差）是流形上的，任何 value ensemble（source 或 target 训练）
建在 value 流形上都看不见它。value 悲观治的是 ② 项（value 过估）+ 流形外幻觉，都不是 0.85 的
binding 约束（oracle 已证）。把 V 做成 ensemble 破不了天花板，别往 planner 里插。** 下一步若还要试
悲观，只剩"惩罚 **dynamics** ensemble 几何分歧"这一路（同脚本结构可测 `std_i f_i^H` vs 真 ‖f^H−s_H‖），
但 −0.30 这个"误差在自信点"的形态对**任何** ensemble 都是坏兆头，大概率也弱。

### 2026-07-19 oracle + 注入噪声：因果剂量-响应曲线（把"模型误差=天花板"从静态两点变成带标定的因果曲线）
§27 主结论此前靠 oracle(1.00) / learned(0.85) 两个静态点 + `fx_C` 归纳。这里给 **oracle
dynamics 注入受控噪声**、扫成功率下滑,得到因果剂量曲线,并**标定 learned model 的等效误差**。
工具 = `scripts/eval_oracle_dyn_mppi.py` 新增三个 flag(全部向后兼容):
- `--shift {source,target}` —— 切 `eval_env_params["domain_shift"]`(source ⇒ `mode=none`),
  main_env + probe rollout 同域。
- `--oracle_noise_sigma "0,0.02,..."` —— 逗号扫描,单位=逐维 state-std。
- `--oracle_noise_mode {iid,smooth}` + `--smooth_freq`。
注入点在 `plan_oracle` 的真实 rollout 终端态打分前(`real_states = noise_fn(real_states)`)。两模式
**逐维边际幅度相同**(`σ·state_std`),**只差跨候选相关结构**:
- `iid` = 每候选现采独立高斯(用**独立 `torch.Generator`**,不碰全局流 ⇒ 各 σ 的候选采样+env
  reset 逐位一致,干净配对)。
- `smooth` = 一个**冻结的 cos-RFF 随机场** `g(s)=cos((s/std)·W+b)·R`(W/b/R 一次性采样冻结、
  `smooth_freq=0.3` 控相关长度;Rf 经验归一化到逐维 std≈1)。因为一次 planning 的候选都从同一
  `s0` 出发、只差 H=3 小动作,终端态彼此极近 ⇒ `g(s_H)` 在候选团里近似**常数**(不可平均),但沿
  轨迹不同 s0 处不同 ⇒ 系统性扭曲 value 地形。**这是"状态相关、持久、不可平均"误差的模拟。**
σ→needle-mm 换算写进日志(needle=state[9:12];std 整轨道算 target 92.8mm/source 117mm ⇒
**跨域比较按 mm 对齐,不是 σ**)。

**结果(40 局,seed 0,ckpt `threading_simdist_adapt_long_h5_seed1`,H=3/N=32/base10/iters2):**

| target needle 噪声 | iid | smooth |
|---:|---:|---:|
| 0 mm | 1.000 | 1.000 |
| 1.9 | 1.000 | 1.000 |
| 3.7 | 1.000 | 0.975 |
| 5.6 | — | 0.875 |
| 7.4 | **1.000** | **0.725** |
| 9.3 | — | 0.600 |
| 14.9 | **0.700** | **0.175** |
| 29.7 | 0.100 | — |

source(窗口 3.0mm):iid≤9.4mm 仍 1.000;smooth 4.7mm→1.000,**9.4mm→0.875**。
bar:policy target 0.400/source 0.625;**learned target 0.875**/source 0.975;clean oracle 两域 1.000。

**三个判据全部成立:**
1. **⚠️ 结构 ≫ 幅度,且 iid 会误导(纠正我最初套单次-argmax 的反向预测)。** 同 needle-mm:
   7.4mm iid 1.000 vs smooth 0.725;15mm iid 0.700 vs smooth 0.175(**4×**)。iid 要到 ~15mm 才掉、
   30mm 才崩(证明注入非 no-op)。机制:闭环 planner + softmax-over-elites **平均掉随机(iid)、
   平均不掉状态相关的持久场(smooth)**——正是本项目"给 oracle 加噪声"实验**必须用 smooth**、用 iid
   会得出"模型误差无所谓"错误结论的原因。
2. **窗口定律(因果,逐点)。** 同 ~9.3mm smooth:source 0.875 vs target 0.600(z≈2.9,n=40 已显著)。
   宽窗口(3.0mm)吸收误差、窄窗口(1.8mm)扛不住 ⇒ "天花板=误差/窗口"不再只是 `fx_C` 归纳。
3. **⭐ 标定命中。** learned target=0.875;smooth 曲线 **5.6mm→恰好 0.875** ⇒ learned model ≈
   oracle + ~**5.6mm 持久误差**,与 §30 **独立**测得的 learned rollout 误差 **5.75mm** 几乎重合(两独立
   测量对上)。source 上 learned 0.975≈oracle,因 3mm 窗口吸收同等误差。

**结论:learned-dynamics 误差因果地、定量地决定 target 天花板,但仅其"持久/相关"分量;5.6mm 等效误差
砸在 1.8mm 窗口 ⇒ 0.875。** n=40 SE≈0.05-0.08(3.7→0.975 与 4.6→1.000 的小非单调是噪声);paper
级需在 target smooth 5.6/7.4mm + source 9.4mm 补 300 局把 SE 压到 ~0.025。CAVEAT:smooth 是**任务
无关**的通用随机场,复现了"持久/相关/in-support"的**结构性质**,不是真实误差的**具体形状**(真实误差
在接触相更集中、且与动作极端程度相关),要精确复刻 learned 曲线得直接用 learned arm。

### 下一步
- [ ] **要突破 0.85，唯一有据可循的方向是把 `模型误差 / 容错窗口` 这个比值压下去**，而两个分子
      分母都很硬：容错窗口是任务的物理属性（1.8mm），模型误差在接触上有不可约的下界（分段、
      毫米级敏感、近乎混沌）。
- [ ] 若用模型，让它**不做 argmax**（MBPO/Dyna 数据增强、MVE/STEVE value expansion，§23 的
      MB-IBRL 已实现未跑）——那里模型误差进入 TD 目标的**期望**而非 argmax 的**极值**。
- [ ] 已跑通的越过天花板的路：residual on frozen MPPI base ≈**0.96**、permanent prior-margin
      IBRL **0.94-0.97**（§25/§26）——共同点是**用真实反馈的负反馈修正被 argmax 的那个函数**。
- [ ] 重跑被 `probe_env` bug 污染的诊断（`diag_planning_objective.py` 的 J-vs-J*）。

---

## 29. SimDist + value ENSEMBLE（MRS.Q 式 search 悲观）—— baseline arm

### What
把 MRS.Q（Chang et al., ICML 2026, `2601.21306v2.pdf`）的核心药方——**min over an
ensemble of value functions**——做进 SimDist 的 world model，当 **baseline** 存档：
stage-1 在 **source_diverse_v2** 上蒸馏（N=10 个 value 头，全部回归同一个 `V_src`
teacher），MPPI 的终端 value 用 `min_i V_i(z_H)` 排序（悲观），再走 §24 的正常 adapt。

**⚠️ 预期是 baseline 而非解法**：§27（2026-07-16）的 on-policy 诊断已实测判死 value 悲观
——`Spearman(value 分歧, 几何误差 ‖ẑ_H−s_H‖) = −0.30`（**负**！ensemble 在 dynamics 越错处
越自信），min-push 与 signed value 误差相关仅 −0.07（不对准），54% 的几何误差藏在低分歧区。
本 arm 是把该结论在**方法级**（而非事后诊断级）跑实，兼作对标 MRS.Q 的对照。

### Where
- `rl/world_model.py`
  - `WorldModelConfig`: `num_value_heads=1`（默认，**逐位向后兼容**）、`value_reduce="mean"|"min"`、
    `value_bagging=1.0`（每头 Bernoulli keep-prob；**确定性 teacher + 同数据 ⇒ 仅靠 init 分歧
    几乎为 0 ⇒ min≈mean−常数 ⇒ argmax 不变（§27 的 ε̄）⇒ 悲观是构造性 no-op**，所以 bagging
    是让成员真分歧的必需件）。
  - `_build_value_extra()` 建 N−1 个 `value_extra` + `value_target_extra`（**N=1 时是空
    ModuleList ⇒ 零参数 ⇒ state_dict 与旧模型完全一致**）；`value_heads()`/`value_target_heads()`/
    `num_value_heads()`/`value_logits_all()`/`predict_value_all()`。
  - `predict_value(z, target, reduce=None)` 按 `cfg.value_reduce` 归约（N=1 短路到旧路径，
    bit-identical）。`trainable_parameters()` 排除 `value_target_extra.`。
- `rl/simdist_agent.py`
  - `_value_head_loss()` + `_bag()`：所有头回归同一 target，per-head Bernoulli bagging；
    N=1 时**精确退化**为旧表达式。`update_distill` 改用它；TD fallback 的 EMA soft-update 覆盖
    全部头；新增 `distill/value_disagree` 日志（**≈0 就说明成员坍缩、悲观必然无效**）。
  - `reinit_head("value")` / `update_value_only()` 覆盖全部头。
- `scripts/test_value_ensemble.py`（NEW）—— 冒烟测试。

### How
```bash
# stage 1（复用 diversev2 stage-1 recipe，只加 3 个字段）
python train_simdist.py --config_path exps/mimicgen/threading_simdist_diversev2_h5_seed1/cfg.yaml \
  --simdist.world_model.num_value_heads 10 \
  --simdist.world_model.value_reduce min \
  --simdist.world_model.value_bagging 0.5 \
  --save_dir exps/mimicgen/threading_simdist_vens_diversev2_h5_seed1
# stage 2/3（§24 正常 adapt；value 头在两个 phase 都冻结，ensemble 随之冻结）
python train_simdist_adapt.py --stage1_ckpt <上面的>/latest.pt --save_dir <adapt dir>
```

### Verification status
Smoke-tested 2026-07-16（`scripts/test_value_ensemble.py` ALL PASS）：
- ✅ **N=1 逐位兼容**：state_dict key 集合与旧模型完全相同、无 `value_extra` 参数、
      legacy ckpt 严格加载、`predict_value` bit-identical。
- ✅ **真实旧 ckpt**（`threading_simdist_diversev2_h5_seed1/latest.pt`）严格加载进新代码；
      pyrallis 用默认值填充 3 个新字段 ⇒ 旧 cfg 照常解析。
- ✅ N=10：`predict_value_all` [N,B]、`min ≤ mean`、`cfg.value_reduce` 生效、EMA target 不在
      trainable 里；distill 时**全部头都更新**、`value_disagree` 有值；planner 走 min-ensemble。

### 2026-07-16 实测：头**坍缩**，min 退化成常数平移 ⇒ 构造性 no-op（step 10000）
`threading_simdist_vens_diversev2_h5_seed1`（10 头 / min / bagging 0.5）vs 原单头
`threading_simdist_diversev2_h5_seed1`，**逐行相同**：loss 0.8650 vs 0.8632、value 0.5614 vs
0.5660、value_pred_mean 0.5265 vs 0.5303、value_pred_std 0.1584 vs 0.1610。而
**`distill/value_disagree = 0.0059`，只有 value 自身状态间跨度 0.1584 的 3.7%**。
后果（算得出来，不用跑完）：10 个近似 iid 高斯的 min ≈ mean − 1.54σ ⇒ 悲观下压量
≈ **1.54×0.0059 ≈ 0.009**，且在候选间近似**恒定**；而 CEM 的 `topk` 与
`exp(temp·(val−val.max()))` **对常数平移严格不变** ⇒ **对 argmax 精确零作用** ⇒ 跑完 800k 也
必然与单头同分。
**根因 = SimDist 的蒸馏架构本身，不是 bagging 没调好**：value 头回归的是一个**确定性 teacher
`V_src`**，10 个头拟合同一个确定性函数必然收敛到同一处；per-batch Bernoulli 掩码不是真 bagging
（无固定 per-sample 子集），10k 步后各头见的数据分布相同 ⇒ 只加噪声、不产生持久分歧。对比
MRS.Q：它的 value 用 **TD/RL** 训练（bootstrap 目标 + 各自 target 网络）⇒ 分歧自然产生。
**⇒ 蒸馏式 value 架构与 ensemble 多样性天然相斥**（本身是个诚实的发现）。审稿意义上，"坍缩"
这个数据点**已经拿到**（disagree=0.006=3.7%），不需要 800k 步去确认。

### 2026-07-17 坍缩的解药：random prior functions（`value_prior_scale`）
`V_i(z) = f_i(z) + β·p_i(z)`，`p_i` 是每个头一个的**固定随机、永不训练**的网络（在 state_dict
里以便 eval 复现，但被 `trainable_parameters()` 排除）。机制：in-distribution 上 `f_i` 学会抵消
`β·p_i` ⇒ `V_i` 照样拟合 teacher；但**这个抵消离开流形不泛化** ⇒ 各头在 OOD 处持久分歧。
确定性 teacher + 同数据下唯一干净的多样性来源。`value_prior_scale=0` = OFF（默认，逐位兼容）。
- ⚠️ **别用 in-dist `value_disagree` 调 β**：RPF 的设计就是让各头在数据内**一致**，该指标本就
  该小。新增 **`distill/value_disagree_rolled`** —— 在 distill 循环末尾的 **H 步 rollout 预测
  latent** 上测（= planner 实际打分的想象终端那一类点），**这才是决定悲观有没有用的量**：它
  若 ≈ in-dist 那个，min 仍是常数平移 ⇒ argmax 不变 ⇒ no-op。

### 2026-07-17 RPF 也救不了 —— 因为 gap 在 transition，不在 state 边缘分布（最深的根因）
β sweep @10k（`value_disagree_rolled` = H 步 rollout 预测 latent 上的头分歧）：
| β | value | pred_std | disagree(in) | disagree(ROLLED) |
|---|---:|---:|---:|---:|
| 0.5 | 0.5628 | 0.1616 | 0.0040 | 0.0033 |
| 1.0 | 0.5634 | 0.1618 | 0.0050 | 0.0040 |
| 3.0 | 0.5659 | 0.1621 | 0.0104 | 0.0077 |
| 无 prior | 0.5614 | 0.1584 | 0.0059 | — |

RPF **没造出 rolled 分歧**（β=3 仍 0.0077 ≈ 坍缩版 0.0059）。离线探针（β=3 ckpt，按输入分布量头分歧）
锁死了机制：
| 输入 | value_mean | head-disagree |
|---|---:|---:|
| SOURCE (in-dist) | 0.616 | 0.0078 |
| **TARGET demo 态** | 0.724 | **0.0083** |
| SOURCE+N(0,2) 远OOD | 0.357 | 0.0190 |
| gaussian×3 极OOD | 0.350 | 0.0188 |

**TARGET 态分歧 ≈ SOURCE**（0.0083 vs 0.0078）。根因（比 §27 的 −0.30 更本质）：**domain gap 在
`P(s'|s,a)`，不在状态边缘分布 `d(s)`**。target 与 source 的状态高度重叠（同任务同机器人，差别在接触
物理），所以 **target 态对 value ensemble 就是 in-distribution** → RPF 的 in-dist 抵消照样生效 →
无分歧；而 value 只看 `s`、看不到 transition，**结构上测不到这个 gap**。⇒ 任何 value-空间悲观
（bagging / RPF / 独立 TD ensemble 皆然）在"状态边缘分布不变、只有动力学变"的 sim2sim 迁移里都是
结构性失明。三条独立证据（§27 的 −0.30 / β sweep 坍缩 / 本探针 target≈source）同一结论。

### 2026-07-17 β=10/30 收尾：lose-lose，没有甜点区 ⇒ baseline 廉价定论
| β | value(fit) | pred_std | rolled disagree | |
|---|---:|---:|---:|---|
| 3 | 0.566✅ | 0.162 | 0.0077 | fit 好，分歧≈0（坍缩） |
| 10 | 0.603 | 0.315(2×) | 0.0495 | value 开始被 prior 污染 |
| 30 | 0.936 | **137.5💥** | 13.18 | value 彻底崩（真实尺度[0,1]，prior 主导） |

**不存在一个 β 同时给"校准 value"+"有意义分歧"**：β 小→value 准但分歧≈0（坍缩，常数平移，argmax
无关）；β 大→有分歧但 value 被随机 prior 噪声淹没（不再追回报 ⇒ 排不了序，比坍缩更糟）。RPF 是
**单调拿 fit 换分歧、无拐点**——它产的是**无差别先验噪声，不是追踪误差的不确定度**（呼应 §27 的
−0.30 与"gap 在 transition 不在 d(s)"）。
**⇒ SimDist + value ensemble（bagging / RPF 任一）作为 MRS.Q 对标 baseline，结论已用 <1 GPU-小时
的诊断廉价锁定：与单头同分。不需要跑 800k。** 代码（`num_value_heads`/`value_reduce`/`value_bagging`/
`value_prior_scale`，全默认关、逐位兼容）已入库，随时可跑一个 checkpoint 存档，但预测是 null。

---

## 30. latent vs raw 表征：规划 vs 改进的分野（2026-07-17）

### ⚠️ 头号教训：**mm 级预测误差对 planner 成绩零预测力**（一天内骗了我 3 次）
`scripts/diag_dyn_rollout_error.py` / `diag_raw_dyn_adapt.py` 量的 needle 位置 mm-MAE 是
个**看起来天经地义、实则与成绩脱钩**的指标。三次连续误判：
| # | 测到的 | 我的推断 | **实际** |
|---|---|---|---|
| 1 | SimDist decode 往返误差 **53mm** | latent 想象数据全废 | latent MPPI source **0.973** |
| 2 | raw dyn H=1 **1.55mm** ≪ latent 53mm | raw 该有 target headroom | raw target **更差** |
| 3 | adapt 后 held-out 误差涨 **2-7×** | adapt 把模型改坏了 | MPPI **+0.17**（0.538→0.708） |
机制：① planner 只在 latent 里滚动 + 用 value 排序，**decode 精度根本不进它的打分**；
② argmax 只在乎 `sup|ε̃|` 与集内排序，均值型误差与之脱钩（= §27#5 的老结论换个马甲）；
③ held-out 集若不是模型**自己 planner 的 on-policy 分布**，测的就是别人的路（§27#5 的分布漂移陷阱）。
**⇒ 凡是评 planner，一律直接跑 eval；不要用任何预测误差代理。**

### 主结论：latent 与 raw 的优劣**取决于下游是"规划"还是"改进"**
| 下游 | raw（z-scored 37维） | latent（SimNorm 512） | 判决 |
|---|---:|---:|---|
| **model-free 改进**（residual RL） | **0.90→0.98** ✅ | 卡在 base **~0.75** ❌ | **raw 完胜** |
| **model-based 规划**（MPPI, source） | 0.960 | 0.973 | 打平（窗口 3mm ≫ 误差） |
| **model-based 规划**（MPPI, target，**公平对照 300 局**） | 净贡献 **+0.173**（confound 摘除后） | 净贡献 **+0.327** | **latent 胜 0.15（z≈3.8）** |

**机制假说**：SimNorm latent 由 value 目标训出 ⇒ **保住了 value 结构、丢掉了 mm 级几何**。
规划靠 value 排序 ⇒ 受益；策略改进的方向恰是被丢掉的那个几何 ⇒ 被堵死。这解释了 §27 结尾的经验
规律（"越过 0.85 只能靠 raw 空间的 residual / prior-margin"）。

### 证据 A：residual 在 latent 里学不起来（严格单变量，只差特征空间）
`threading_residual_latent_seed1` vs `threading_residual_30k_seed1`（同 base/数据/超参/seed）：
两条都从 base(~0.75) 起步、前 18k 无差别；**~20k 起 raw 起飞（0.85→0.98），latent 永远在 base 附近晃**。
失败模式**不是"没梯度"**：`delta_abs` latent 0.009→**0.068**（快到 action_scale 上限，残差在拼命试），
但 **`q_mean` 死平 0.286→0.289**（raw：0.315→**0.37**↑）。⇒ **critic 认为任何动作修正都不改变价值**
——要修一次差之毫厘的插入，必须知道 sub-mm 错位往哪偏，而那正是 latent 压没的维度。
- 代码：`residual.latent_features` / `latent_dim`（默认 0/0 ⇒ 逐位兼容），`_norm()` 是唯一 hook
  （act/update/TD3 target/AWR/诊断全自动跟着切）；buffer 仍存 raw state，只在前向 encode（冻结）。
  `ResidualSimDistAgent.attach_base_policy` 自动接 base 的 `model.encode` + 校验宽度。
  `scripts/verify_iql_critic.py --features raw|latent`；`train_residual_simdist.py::_assert_feature_match`
  防止 latent ckpt 误装进 raw run。

### 证据 B：latent 作为 value 特征空间**更弱**（受控对照，同数据/配方/评估）
`scripts/diag_latent_qres_sensitivity.py --features latent|raw`（30k adapt rollout，100/28 ep split）：
| | latent | **raw** |
|---|---:|---:|
| Spearman(V, −steps) per-ep | +0.605 | **+0.782** |
| Pearson(Q, γ^steps) | +0.495 ❌ | **+0.655** ✅ |
⚠️ **单 seed，无误差棒**（按 §27 纪律不足以定论）。绝对值远低于 §26 记的 +0.978/+1.000 —— 那是在
**干净专家 demo** 上测的，本表在 **MPPI rollout** 上测，**两组不可比**（这正是补 raw 对照的原因）。
理论侧支持：`encode(s)` 是 s 的确定性函数 ⇒ 数据处理不等式 ⇒ 不可能比 raw 多带信息；SimNorm 更是
主动有损。37 维紧致状态喂 1024×3 MLP，latent 无可贡献。

### 证据 C：latent 的"value 零空间红利"**不转移给新 critic**
`|dV_res|/std`（想象 latent 对 Dyna backup 的污染，H=1）：SimDist 自己的 value 头 **0.03**，
新训的 residual critic **0.06**（2×）。⇒ 那个零空间是**SimDist value 头独有的**（与 dynamics 共同
训练的副产品，§27#5），新 critic 不共享。H=5 时 latent 0.30 vs raw-ensemble **0.17** —— latent 反而更差。
**⇒ "留在 latent 所以想象更干净"是假的。** `rollout_length=1` 被三个独立度量交叉锁死
（mm 尺 H=3 越界 / latent 污染 H=5 撞 0.30 / raw 污染 H=5 涨到 0.17）。

### ⚠️ target 的 raw-vs-latent 仍**未定**：三个 dyn_adapt run 全是 confound
`threading_dyn_adapt{,_tgtval,_goodpolicy}_seed1` 的 **dynamics 完全相同**（同一个
`threading_dyn_diverse_v2_seed1/best.pt`），变的只有 value/proposal —— 分数却从 0.31 爬到 0.65：
| run | 变量 | target MPPI | policy_only |
|---|---|---:|---:|
| `dyn_adapt` | source value + BC(20 demo) | 0.31→0.45 | 0.20 |
| `tgtval` | **换 target value** | 0.47 | — |
| `goodpolicy` | **500 expert demo + bc_steps 5×** | 0.62（12轮均值 0.617） | **0.744** |
⇒ **它们测的是 proposal/value 质量，不是 raw dynamics**（`train_dynamics_adapt.py:193`
写死 `proposal_policy=None`，而 mixture_coef=0.8 ⇒ 80% 候选来自那个临时 BC 头）。
而 latent 的 0.85 有完整蒸馏的 value+policy ⇒ **"raw 0.65 < latent 0.85"不合法**。

### raw dyn 的 target adapt：**有效，但同时在灾难性遗忘**
`goodpolicy` round-0（adapt 前）已是 MPPI 0.538 vs policy_only 0.718（**−0.18**）；adapt 后
0.708 vs 0.726（**−0.02**）⇒ **adapt 把净损失救回 +0.17**。同时 held-out target 预测**全面变差**
（free space needle MAE PRE **1.69mm** → POST **8.37mm**，5×；另一 run 10.17mm，系统性）。
病因：20 轮×1000 步×batch 256 = 512 万样本 ÷ **2 万条 transition buffer = 256 个 epoch**，对上
source 的 10 万条轨迹先验。**但遗忘不是 binding 约束**（MPPI 分照涨）——模型在**自己 on-policy 分布**
上变准（正是 MPPI 打分处），别处遗忘。
**❌ 已否决的修法**：混 source 数据进 adapt（`source_mix_ratio`，代码已入库、默认 0.0 关）——
它在**对抗 adapt 的目的本身**（教模型对同一 (s,a) 同时预测 source 和 target 的 s' ⇒ 学成平均，
而 contact 相正是最要紧处）；且遗忘是不是瓶颈根本没证实。**别开这个开关。**

### ★ 干净对照的判决（2026-07-17, 300 局）：`eval_dyn_arm_compare.py`

**结果（同 value 头 + 同 policy 头 + 同 planner/env/seed，唯一变量 = rollout dynamics）：**
| arm | policy_only | **MPPI** | 净贡献 | SE |
|---|---:|---:|---:|---:|
| **latent** | 0.350 | **0.677** | **+0.327** | 0.027 |
| **raw** | 0.350（同一个头） | **0.577** | **+0.227** | 0.029 |
差 **0.100**，SE_diff≈0.040 ⇒ **z≈2.5 (p≈0.012)** —— 过 300 局门槛，真差异。

**★ confound 摘除（symmetric adapt, 2026-07-17 续）：raw 不升反降 ⇒ 是表征差，非 on-policy 红利。**
质疑：raw-goodpolicy 的 dynamics 是在**别人的分布**（那个 run 的 BC 头 policy_only 0.744 + V_src）上
adapt 的，却拿去给一个 0.35 的 latent 策略规划 —— on-policy-vs-off-policy 混进了"表征"。于是新增
`train_dynamics_adapt.py --proposal_agent <latent ckpt>`：value+proposal 都取自那个 latent agent、
跳过 stage-2 BC，让 raw dynamics 在**和 latent 臂完全相同的分布**上 adapt
（`threading_dyn_adapt_symmetric_seed1`）。300 局重评：
| dynamics | adapt 分布 | MPPI | 净贡献 |
|---|---|---:|---:|
| **latent** | latent 自己 on-policy | **0.677** | **+0.327** |
| raw-goodpolicy | 别人的分布(BC头+V_src) | 0.577 | +0.227 |
| **raw-symmetric** | **latent value+policy(同 latent 臂)** | **0.523** | **+0.173** |
摘掉 confound 后 raw **反而降到 0.523**（比 goodpolicy 还低 0.05，离 latent 差 **0.154, z≈3.8**）⇒
**那 0.1~0.15 是真的表征差，不是 on-policy 红利。** latent 在规划侧确实更强，且 confound 摘除后差距
扩大。（wandb: `dynarm_raw_symmetric_seed1` / project mimicgen / group dyn_arm_compare。代码
`--proposal_agent` 默认 "" ⇒ 现有 run 逐位不变；`eval_dyn_arm_compare.py --use_wb`。）

**两条结论：**
1. **raw planner 其实是有益的（+0.227，正的）。** `goodpolicy` run 的 **−0.02** 纯粹是
   proposal/value 的锅（临时 BC 头 + V_src），**不是 raw dynamics 的锅** ⇒ 三个 dyn_adapt run 的
   confound 完全坐实，"raw target 0.65 < latent 0.85"作废。
2. **但 latent 仍赢 0.10**，即使 raw 拿到一模一样的 value+proposal ⇒ **latent 在规划侧的优势是真的**，
   只是比那个假的 0.23 小一半以上。
**⇒ "隐空间难以表征 ⇒ 规划受害"被证伪；机制假说（latent 保 value 结构、丢 mm 几何 ⇒ 规划受益、
改进被堵）得到支持，且正好解释 §27 的经验规律：SimDist(latent 规划)→0.85，越过 0.85 只能靠 raw
空间的 residual/prior-margin→0.96。**
两臂**共用同一个 value 头 + 同一个 policy 头（proposal）+ 同 planner 参数/env/seed**，
**唯一变量 = rollout 用哪个 dynamics**。公平性关键（两臂都不吃 decode 税）：
- **raw 臂**走外部 hook，但 raw 的 `decode` 是**精确**的（identity encoder 的 denorm 逆）⇒ 无损；
  终端打分 = `latent_value(latent_encode(s_H))`。
- **latent 臂**走**原生**路径（`value_fn=None` → `predict_value(z_H)`；`shared_encoder=1` +
  `proposal=None` → `policy_from_latent(z)`）⇒ **全程不碰 decode**（否则被那 53mm 冤枉）。
`policy_only`（共用 policy 头裸跑）是**两臂都必须跨过的杠**（§27：planner 的价值 = 成绩 − proposal）。
```bash
python eval_dyn_arm_compare.py --arm latent --num_game 300 --shift target
python eval_dyn_arm_compare.py --arm raw --num_game 300 --shift target \
    --dyn_ckpt exps/mimicgen/threading_dyn_adapt_goodpolicy_seed1/model0.pt
```

### TODO
- [ ] 跑完 `eval_dyn_arm_compare.py` 两臂（≥300 局，±0.1 内的结论必须 300 局）——
      **target 上 raw vs latent 的第一个合法判决**。
- [ ] 证据 B 补 3 seed 误差棒（每臂 2-3 分钟）；若误差棒重叠，则 critic 健康度这一轴不能区分两者。
- [x] raw 臂在同 proposal/value 下**仍输 0.10** ⇒ latent 确实更适合规划；同时 §27 的
      "raw target 0.65" **也**确实是 proposal/value 的锅（raw 净贡献其实是 +0.227）。两者都真。
- [x] on-policy confound 已用 symmetric adapt 排除（raw 反降到 0.523，差距扩大到 0.15/z≈3.8）。
- [ ] 多 seed 复现（仍单 seed；三臂方向一致但需 ≥3 seed 上误差棒）。
- [ ] 机制追问：latent 的 +0.10 来自哪？（候选：value 头在自己的 latent 上标定更准 vs
      raw rollout 的 H=5 复合误差 5.75mm 越过 1.8mm 窗口）——决定它是否可迁移。

---

## 31. SimDist + dynamics ENSEMBLE + 几何分歧惩罚（MOPO/PETS 状态空间悲观，2026-07-19）

### What
§27（2026-07-16）判死了 value ensemble 悲观：`Spearman(value 分歧, 几何误差
‖ẑ_H−s_H‖) = −0.30`（value 头在 dynamics 越错处越自信），§29 的 β sweep 又证明
value 悲观在"状态边缘分布不变、只有 transition 变"的 sim2sim 里结构性失明。§27 结尾
明确指出**唯一还值得试的悲观方向 = 惩罚 DYNAMICS ensemble 对 `f^H` 的几何分歧**（状态
空间的不确定度,不是 value 的）。本节实现这条:给 world model 的 dynamics 头做集成,MPPI
planning return 里减去成员对下一 latent 预测的跨头 std,让 argmax 不敢往模型不受约束
(高分歧)的地方走。**从 source 重新蒸馏。**

### 与 value ensemble(§29)的关键区别 —— 为什么这次可能不失明
value 头只吃 `s`,看不到 transition,gap 在 `P(s'|s,a)` 而非 `d(s)` ⇒ 结构性看不见。
dynamics 头吃 `(z,a)` 预测 `z'`,**直接建模 transition** ⇒ 分歧就是几何分歧,正是
argmax 最恨的动作依赖误差(§27 的 ε̃)。in-dist 数据把各头钉在一起(agree),off-dist
(MPPI 搜索到的想象候选)各头因独立初始化/非平稳 encoder 目标而发散(disagree)——这正是
悲观需要的形态。⚠️ 但 §29 的坍缩警告仍在:确定性 teacher + 同数据下,per-batch Bernoulli
bagging 不是真 bagging,可能 in-dist/off-dist 都不发散。所以留了 `dynamics_prior_scale`
(Osband RPF,§29 验证过的干净多样性源)作对冲。**判据 = `distill/dyn_disagree_rolled`
(H 步 rolled 想象 latent 上的分歧)必须显著 > `distill/dyn_disagree`(in-dist);若两者
≈ 且都小 ⇒ 坍缩,惩罚是 no-op,加 prior 或判死。**

### ★ 设计要点:惩罚系数是 EVAL 侧旋钮,训练只产 ensemble
`PlannerConfig.dynamics_disagree_coef` 在 planning return 里生效(`ret -= coef·Σγ^t·
disagree_t`),**不参与训练**。训练只需蒸馏出一个 ensemble 世界模型(`next()` 用集成
MEAN 滚动,policy/value 照常对 mean 蒸馏)。所以**一次 source 蒸馏 → eval 时扫全部
悲观强度**(coef=0 = baseline,>0 = 悲观),不必为每个 coef 重训。且 diversev2 的
in-training eval 用 `use_planner=0`(policy 头直评),ensemble 只影响 consistency 训练
+ 分歧 log;MPPI 惩罚在后续 eval/adapt(`use_planner=1`)才起作用。

### Where（全部 `num_dynamics_heads=1` 默认 ⇒ 逐位向后兼容,旧 ckpt 严格加载）
- `rl/world_model.py`
  - `WorldModelConfig`: `num_dynamics_heads=1` / `dynamics_bagging=1.0` /
    `dynamics_prior_scale=0.0`。
  - `_build_dynamics_extra()` 建 N−1 个 `dynamics_extra` + 可选 `dynamics_prior`
    (N=1 且 prior=0 ⇒ 空 ModuleList ⇒ 零参数 ⇒ state_dict 与旧模型完全一致)。
  - `next_all(z,a)→[N,B,z]`(latent 每头 SimNorm,raw-state 每头残差 z+Δ_i;RPF 时加
    `beta·prior_i`);`next(z,a)` = 集成 MEAN(N=1 短路到旧表达式,bit-identical);
    `dynamics_disagreement(z,a)→[B]` = 跨头 std 的 latent 维均值;`dynamics_parameters()`
    = 全部头;`trainable_parameters` skip 加 `dynamics_prior.`。
  - ⚠️ `next()` 现在 N>1 时返回 MEAN(而非 head-0),这改变了 N>1 的训练滚动语义——但
    N=1 不变,且这正是集成的意图。
- `rl/simdist_agent.py`
  - `_dynamics_cons_loss(z,a,z_next_enc)`:每头 bagged MSE 回归同一 stop-grad 目标
    `encode(s')`,返回 (loss, 集成 mean 用于滚动)。N=1 精确退化为旧 `F.mse_loss(next,·)`。
  - `update_distill` / `update_dynamics_only`(§24 adapt) / `update_finetune` 都改用它
    (训练全部头);finetune/adapt 的 optimizer + grad_clip 用 `dynamics_parameters()`。
  - `update_distill` 新增日志 `distill/dyn_disagree`(in-dist,z0+a_0)+
    `distill/dyn_disagree_rolled`(H 步 rolled z + a_last)。
- `rl/simdist_planner.py`
  - `PlannerConfig.dynamics_disagree_coef=0.0`;`_estimate_returns` +
    `_estimate_returns_batch` 累计 `Σγ^t·disagree_t` 并从 return 减去
    `coef·(·)`。仅当 `coef>0 且 num_dynamics_heads>1` 时计算(否则零开销,N=1 路径不变)。
- `scripts/test_dynamics_ensemble.py`(NEW)—— 冒烟测试。

### How
```bash
# stage 1（source 重蒸馏,复用 diversev2_h5 recipe + ensemble 覆盖;planner 惩罚留到 eval）
python train_simdist.py --config_path exps/mimicgen/threading_simdist_diversev2_h5_seed1/cfg.yaml \
  --simdist.world_model.num_dynamics_heads 5 \
  --save_dir exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1
#   若 dyn_disagree_rolled 坍缩(≈in-dist): 加 --simdist.world_model.dynamics_prior_scale 1.0
# eval / adapt 时扫悲观强度(不必重训):
python eval_dyn_arm_compare.py ... --arm latent   # (需给 planner 传 dynamics_disagree_coef)
# 或 §24 target adapt 后 MPPI eval,cfg 里设 simdist.planner.dynamics_disagree_coef=<λ>
```

### Verification status
Smoke-tested 2026-07-19(`scripts/test_dynamics_ensemble.py` ALL PASS）：
- ✅ N=1 逐位兼容:state_dict key 集合相同、无 dynamics_extra/prior 参数、legacy ckpt
  严格加载、`next()` bit-identical、disagreement≡0;**真实 diversev2_h5 单头 ckpt 严格
  加载进新代码**。
- ✅ N=5:next_all `[5,B,z]`、`next()`==集成 mean、每头 SimNorm 仍是合法 simplex、
  disagreement≥0;raw-state N=4 残差头同样成立。
- ✅ distill/adapt 训练**全部** dynamics 头(所有参数移动);dyn_disagree +
  dyn_disagree_rolled 有值且有限。
- ✅ planner 惩罚:coef>0 严格 ≤ coef=0 的 return 且确有下降;N=1 时守卫关闭无副作用;
  plan + plan_batch 带惩罚产出合法动作。
- ✅ random priors:在 state_dict、不可训练、distill 中冻结、抬高分歧(0.045→0.062)。
DONE 2026-07-20: source 重蒸馏跑满 800k(`threading_simdist_dynens_diversev2_h5_seed1`,
5 头无 prior;末尾 in-dist 分歧 0.0018 / rolled 0.0017,从 0.0044 收缩)。
DONE 2026-07-21: TARGET adapt 启动(`threading_simdist_dynens_adapt_long_h5_seed1`,
与 §24 一致的 dynamics-only adapt),**完全对齐 vens 的 adapt 配方**用于比较:
`--config_path <threading_simdist_vens_adapt_long_h5_seed1/cfg.yaml>` 继承全部超参
(bc_steps 1000 / adapt_interval 1000 / max_transitions 340000 / dyn_update_steps
1000 / eval_every_rounds 1 / num_eval_episode 100 / original_mppi 1 / adapt_value 0
/ dyn_loss mse / eval_policy_only 1),只换 stage1_ckpt + save_dir。adapt 阶段
`dynamics_disagree_coef=0`(与 vens 逐点可比;penalty 是后续 eval 旋钮)。wandb run
`9ge3y0kh`。⚠️ nohup 下 stdout 块缓冲,print 的 `[adapt @ N] target_score=` 行不实时
落 log(tqdm 走 stderr 实时);看分数上 wandb。
比较基准 —— **vens(value ensemble)adapt 曲线(300k+ 段)**:target MPPI ~0.85
(0.80–0.90),policy_only ~0.28 == 普通 SimDist 的 0.85 天花板 ⇒ value ensemble 是
no-op(§29 实测坐实)。
### ★★ 2026-07-21 结果:dynamics ENSEMBLE 打破 0.85 天花板 → 0.96,但功劳是"集成均值",不是悲观
三臂全部 σ=0.1 / 340k / 其余超参逐字段相同(cfg diff 只差 stage1_ckpt+save_dir)。
**dynens 与 plain 已跑满 340k**(vens σ=0.1 到 335k),终段 + 中段两个窗口都一致:

| 臂 | 100–135k MPPI | **300–340k MPPI** | policy_only | 净贡献(终段) |
|---|---:|---:|---:|---:|
| **dynens(dyn ens ×5)** | 0.959 ± 0.004 | **0.965 ± 0.002** | 0.387 | **+0.578** |
| plain(baseline) | 0.824 ± 0.007 | 0.879 ± 0.005 | 0.371 | +0.508 |
| vens σ=0.1(val ens ×10) | 0.722 ± 0.008 | 0.783 ± 0.008 | 0.228 | +0.555 |

dynens 自 ~100k 起就稳在 0.96 并保持到 340k;proposal 与 plain 对齐(0.387 vs 0.371)。
独立的大样本复核(冻结 ckpt,两 seed 合并 n=2000):**0.962 ± 0.004**,与训练内 eval 一致。

**① dynamics ensemble = 迄今第一个在 model-based planning 内部越过 0.85 的东西**(§27 原本
的结论是只有 residual / prior-margin 这类"不在学出来的模型里 argmax"的方法能过)。+0.135,
z≈17。proposal 已对齐(0.387 vs 0.360,差 0.027 解释不了 0.135),stage-1 policy 在 target 上
三臂相当(0.12–0.36 噪声级)⇒ **增益来自规划时的 dynamics 更准,不是更强的 proposal。**
**⚠️ 但 adapt 全程 `dynamics_disagree_coef=0` —— 惩罚根本没开。所以这是"集成均值"效应
(5 头平均 ⇒ rollout 方差降低 ⇒ 模型误差变小),不是悲观。** 正好命中 §27 的因果框架:
天花板 = f(模型误差 / 容错窗口),误差降 ⇒ 天花板升(oracle 误差=0 时是 1.00;§27 2026-07-19
标定 learned ≈ oracle + 5.6mm 持久误差)。集成是**唯一一个真正压低了分子**的干预——对比
§27 里失败的那一整批(校准 value / 候选分布上 finetune / decision-aware loss)全是在**不降低
误差**的前提下折腾,所以全撞墙。

**② value ensemble 确认 no-op(甚至更差)**:vens σ=0.1 只有 0.722 < plain 0.824,但**净贡献
0.478 ≈ plain 0.464** ⇒ 差距全在 proposal(policy_only 0.244 vs 0.360,其 stage-1 的 RPF β=3
把 encoder/policy 弄弱了),**对 planning 本身零贡献**。§27(Spearman −0.30 结构性失明)+
§29(坍缩 ⇒ min 退化成常数平移 ⇒ argmax 不变)在方法级被坐实。

**③ 悲观已测完 ⇒ 无用**(详见下面的 2026-07-21/22 判决):≤0.2 零效应、≥0.33 单调有害。

CAVEAT:单 seed;dynens 尚未跑到 340k(但 ~100k 起已 plateau);每点 100 局(窗口内 36 轮
均值使 SE≈0.004,gap 0.135 远超噪声,但**最终 headline 仍需 300 局**,§27 铁律)。

### ★★★ 2026-07-21/22 判决(`scripts/eval_dynens_ablate.py`,同一冻结 ckpt
### `ablate_frozen.pt`,σ=0.1,original_mppi):**增益来自集成均值;悲观无用(弱则零效应、
### 强则有害)**。headline:5 头 coef=0 = **0.962 ± 0.004**(两 seed 合并 n=2000,远超
### §27 的 300 局门槛)。

**① 归因隔离 —— 增益 100% 来自规划时的集成平均,与 encoder 无关**
同一份权重、同一 encoder/value/policy/planner/env/seed,**唯一变量 = rollout 用几个头**
(`--heads` 在 eval 时截断 `dynamics_extra`):

| rollout | score |
|---|---:|
| **5 头均值** | **0.970 ± 0.017** |
| **单头(head-0)** | **0.120 ± 0.032** |

⇒ 那个"训练时 ensemble 把 encoder 训好了"的替代解释被**彻底排除**。同时暴露一个重要性质:
**成员是强共适应的**——训练时 rollout 沿**集成均值**推进,每个头只需保证"均值对",单独滚
时自身误差沿 H 步复合 ⇒ 单头 0.12 甚至远低于独立训练的单头模型(plain SimDist 0.82)。
所以正确表述是:**+0.15(vs plain 0.82)来自 plan-time ensembling 的方差抵消**,而不是
"每个头都更好"。这直接坐实 §27 的因果律:集成压低了模型误差这个分子 ⇒ 天花板抬升
(§27 2026-07-19 标定 learned≈oracle+5.6mm;5 头平均把这个误差压下去 ⇒ 逼近 oracle 的 1.00)。

**② 悲观(几何分歧惩罚)第一轮 sweep(每点仅 100 局;下界只到 0.33)⇒ 见 ③ 的修正**
先用 `--mode probe` 在**on-policy** target 状态上量准了尺度(躲开 §27#5 的 cache 分布漂移
陷阱):`std_a(return)=0.0383`(argmax 排序的信号)、`std_a(disagree_sum)=0.0115`(惩罚的
跨候选散布)、`mean=0.0239`,**spread/mean=0.48** ⇒ 惩罚**不是常数平移**,理论上**能**拨动
argmax(区别于 §29 value 的构造性 no-op),coef≈3.3 时惩罚与价值信号同量级。据此扫:

| coef | 0 | 0.33 | 1.1 | 3.3 | 10 |
|---|---:|---:|---:|---:|---:|
| score | **0.970** | 0.950 | 0.940 | 0.920 | 0.900 |
| ± | .017 | .022 | .024 | .027 | .030 |

5 点单调下降,单点差(0.970 vs 0.900 = 0.07, z≈2.0)在 100 局下只算边缘,但 5 点全单调
比任一对有说服力。⚠️ **注意:这里 0→0.33 的那 0.02 后来被 ③ 证明是零效应(100 局的噪声
就是 ±0.02),所以"单调下降"只对 ≥0.33 成立;真实形状是"≤0.2 平、≥0.33 降"。**

**③ 小 coef 区(0.05/0.1/0.2)追加实验 + 预注册复现 ⇒ 确认零效应(又一次符号反转!)**
上面的 sweep 最小只到 0.33,小 coef 区没测。补测(500 局/臂,同 seed ⇒ 配对):
`coef 0/0.05/0.1/0.2 → 0.950/0.962/0.968/0.952`,配对 Δ 在 0.1 处 **+0.018 (z=1.68)**,
看着像个峰。**但这是从同一批数据里扫 3 个点挖出来的("三者取最大"在零假设下有 ~26% 概率
给出 z≥1.68),不能直接采信。** 于是做**预注册复现**:换 seed=1、只测 coef∈{0, 0.1}、
n=1500:

| 实验 | n | Δ(coef=0.1 − 0) | z |
|---|---:|---:|---:|
| seed=0(发现) | 500 | **+0.0180** | +1.68 |
| **seed=1(复现)** | **1500** | **−0.0107** | **−1.86** |
| **合并配对** | **2000** | **−0.0035 ± 0.0051** | **−0.69** |

**没复现,而且符号翻转**;合并后与 0 无异 ⇒ **2σ 上排除任何 > +0.01 的增益**。
⚠️ **又一次"小样本符号反转"**——与 §27 那次(40 局 −0.075 vs 300 局 +0.097)同型。教训再
确认:①"扫出来的最优点"必须在**新 seed / 新数据**上预注册复现,否则就是 garden-of-
forking-paths;② 配对(同 seed ⇒ 同初始状态)只把差值 SE 从 0.0128 降到 0.0107(~16%),
在成功率 0.95 的高分区**帮助有限**(方差主要来自哪几局失败,不来自初始状态难易),所以
**靠加局数**才是硬道理。

**⇒ 完整的 coef 曲线:≤0.2 零效应(惩罚散布仅占价值信号的 3%,太弱做不了功),≥0.33 单调
有害(惩罚开始盖过价值信号,把 argmax 带偏)。lose-lose,无甜点区。§27 结尾提出的"唯一还
值得试的悲观方向"到此走完:value 悲观结构性失明(§29),dynamics 几何悲观虽有真信号
(spread/mean=0.48,非常数平移)但仍无增益。悲观这条路整体收尾。**

**机制(尚未实证,别当结论)**:一个自洽的解释是集成均值**已经把误差修掉了**(0.12→0.97),
此后再惩罚分歧只是把 argmax 从本来没问题的候选上推开,拿排序容量去修一个已修好的东西。
要坐实需补测 `Spearman(dyn 分歧, 真实 ‖ẑ_H−s_H‖)`(对比 value 的 −0.30)。

- [x] **300 局 strict 复核** —— 超额完成:coef=0/5 头在两个 seed 上共 **2000 局 = 0.962 ±
      0.004**(seed0 n=500: 0.950;seed1 n=1500: 0.966),与训练内 eval(0.965)一致。
- [x] **多 seed(eval 侧)** —— 两个 eval seed 一致;⚠️ **训练 seed 仍只有 1 个**(stage-1
      蒸馏 + adapt 都是 seed 1),要主张方法级结论仍需 ≥2 个训练 seed。
- [x] **头数 sweep** —— 见下面的 2026-07-22 结果:**≈3 头饱和**;并给出"集成只除掉方差、
      剩下的是偏差"这个机制推论 + 对 10 头 run 的可证伪预测。
- [ ] 机制诊断 `Spearman(dyn 分歧, 真实 ‖ẑ_H−s_H‖)`(对比 value 的 −0.30)——解释为什么
      分歧有信号却换不来增益。
- [ ] 单头 0.12 的追问:是"共适应成员单独滚不行",还是"训练时就没被要求单独准"?可训一个
      `num_dynamics_heads=5` 但 rollout 走单头的对照来分离(目前只知道现象,机制未定)。

**定位**:0.962 与 §25/§26 的 residual(~0.96)/ prior-margin(0.94–0.97)持平,但它是
**纯 model-based planning** 拿到的 —— 第一次不靠"用真实反馈的负反馈去修被 argmax 的那个
函数"就越过 0.85。

### ★ 2026-07-22 头数 sweep(截断,500 局/点,同 seed 配对):**≈3 头饱和**
同一个 5 头 ckpt,eval 时截断到 k 个头做 rollout(`--heads k`):

| heads | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| score | **0.132** | 0.868 | 0.942 | **0.958** | 0.950 |
| ±SE | .015 | .015 | .010 | .009 | .010 |

相邻配对增量:**1→2 = +0.736 (z=36)**、**2→3 = +0.074 (z=4.2)**、3→4 = +0.016 (z=1.5, ns)、
4→5 = −0.008 (z=−0.7, ns)。⇒ **3 个头就到顶,之后是平台。**

**⚠️ 这是"截断"不是"训练 k 头"。** 训练时 rollout 沿 5 头均值推进,成员只被要求"合起来
对";单独/少数几个滚是 off-distribution 的用法(这正是 1 头只有 0.132、**远低于独立训练的
单头 plain 0.82** 的原因)。所以这条曲线回答的是**"规划时要用几个成员才能把方差抵消掉"**,
**不能**读成"训练 3 头就够了"——后者要各自独立蒸馏(每个 ~25h)才能答。

**机制推论(重要,且可证伪)**:1→2 的巨大跃升 + 3 头后完全平台,符合"**集成消掉的是模型
误差里的方差(idiosyncratic)分量**"——方差随 k 以 ~1/k 衰减,3-4 个成员后残余方差已可忽略。
那么**平台高度 0.95–0.96 就是被剩下的 BIAS(系统性误差)顶住的,而平均更多成员对 bias 无效**。
⇒ **预测:真训练的 10 头 ensemble 不会显著高于 5 头(≈0.95–0.96)。** 正在跑的
`threading_simdist_dynens10_diversev2_h5_seed1`(GPU3,800k,ETA ~25h)+ 其 adapt 就是这个
预测的检验:若 10 头明显更高,说明"训练时头数"还有超出 plan-time 平均的额外作用,推翻这个
机制解释;若持平,则**"集成 ⇒ 除方差 ⇒ 抬天花板到 bias 下界"**这条链闭合,同时也说明想再
突破 0.96 必须去治 **bias**(而不是继续加头)。

### ★ 2026-07-22 base 强度 → 适应速度对照(进行中):前期强证据 = 后期不靠 base
问题:adapt 前期极快(0→2k 就 0.90),**后期慢**靠什么?假说:靠 base(proposal)强度?
干净单变量对照(同一 dynens 5 头 stage-1 ckpt / σ=0.1 / dynamics-only adapt / planner 全同,
唯一变量 = stage-2 BC 的 target demo 数 ⇒ base policy 强度):
- **弱 base**:20-demo BC ⇒ policy_only ~0.37(= `dynens_adapt_long_h5`,wandb `orawyua3`)
- **强 base**:500-demo expert + bc_steps 3000 ⇒ policy_only ~0.83(`..._strongbase_seed1`,
  wandb `077btko4`,GPU0,max 160k)

前 5 轮:

| transitions | 强 base MPPI / policy | 弱 base MPPI / policy |
|---:|---:|---:|
| 0 | 0.29 / **0.83** | 0.19 / 0.35 |
| 1000 | 0.54 / 0.81 | 0.59 / 0.36 |
| 3000 | 0.88 / 0.75 | 0.89 / 0.37 |
| 5000 | 0.89 / 0.86 | 0.90 / 0.42 |

前期两条 MPPI 曲线几乎重合(3k 就并轨),尽管 base 强度差 2.3×。**round-0 反直觉再现
(§17 铁律)**:强 base policy_only=0.83,但套上未适应的 source dynamics,MPPI 反被拖到
**0.29** —— MPPI 分数由 **dynamics 精度**定,不由 proposal 强度定(呼应 §27)。

**★ 后期段(结论翻转:强 base 不是重合,是略 WORSE)**:

| 区间 | 强 base MPPI | 弱 base MPPI |
|---|---:|---:|
| 20–60k | 0.917 ± 0.004 | **0.936 ± 0.004** |
| 100–160k | 0.938 ± 0.004 | **0.959 ± 0.003** |

后期强 base **低 0.021(z≈4.2)**。⇒ **"后期靠 base 强度"被否证,且方向相反:更强的 base 略
有害。** 机制 = 上一轮预警的"强但窄"陷阱:500-demo expert 是干净成功轨迹,BC 出的 policy 走
一条**窄**的成功路径 ⇒ MPPI 候选被种子在窄路附近 ⇒ target rollout 的 contact 构型**多样性
下降** ⇒ dynamics 在 off-path 构型覆盖变差 ⇒ 后期修得更慢/更低。弱 base(policy 0.35)探索
更散、contact 覆盖更广,反而后期更高。

**⇒ 完整答案:后期不靠 base 强度(过强+过窄反而伤覆盖),靠的是 target rollout 在 contact 相
的覆盖广度 + dynamics 精修。** 加速后期的正确杠杆是**扩大 contact 相 rollout 的覆盖/多样性**
(探索噪声收集 / disagreement-guided 主动采样),不是换更强的 base policy。§27 天花板律再次
印证:proposal 只挪起点,dynamics 精度定分数。(单 seed;强 base run 截到 160k,弱 base 到
340k;重叠区间对比公平。)
in-training `distill/dyn_disagree_rolled ≈ dyn_disagree`(10k: 0.0040 vs 0.0044;
20k: 0.0030 vs 0.0032)且随训练**收缩**,一度像 §29 的坍缩签名(而且这个 run 用
`bagging=1.0` 无 prior,零多样性机制)。差点误杀。但 in-training 的 rolled 是在**训练
数据轨迹**上滚的(还在分布内),不是 MPPI 真正搜到的 off-dist 区——是弱代理。改用
§29 式**离线 OOD 探针**(`scripts/probe_dyn_disagree_ood.py`,在 30k ckpt 上比较 in-dist
vs 真 OOD 的跨头 std)才是直接测量:
```
in-dist (s,a):        0.00287
OOD action:           0.00506  (1.77x)
OOD state (+0.5 noise):0.00966 (3.37x)   ← 关键
far rollout (15 随机): 0.00530  (1.85x)
```
dynamics 头**确实在 off-manifold 发散**(OOD 状态 3.4x)——定性 **区别于 §29 的 value
ensemble**(那里 target≈source≈OOD 完全不发散,因为 value 只看 s、gap 在 transition)。
dynamics 头吃 (z,a) 建模 transition ⇒ 分歧就是几何分歧 ⇒ off-dist 有信号。**必要前提
(off-dist 分歧存在)满足,让 run 继续。** 教训重申:①"太丑"(疑似坍缩)时也别只看弱
代理就下结论,做直接测量(这次探针救了决策,正如 §27 的 bug 是被质疑救的);② rolled
in-training 指标不等于 MPPI on-policy 的 off-dist 分歧。⚠️ 但绝对量级小(0.003~0.01,
latent per-dim std 0.166),penalty 能否**有意义地**拨动 argmax 仍取决于蒸馏后的
Spearman(disagree, 真 ‖ẑ_H−s_H‖)——那才是真 go/no-go。

### TODO
- [ ] **早诊断(§29 便宜判死纪律)**:读前 ~40k 步的 `distill/dyn_disagree_rolled` vs
      `distill/dyn_disagree`。若 rolled 显著 > in-dist ⇒ 惩罚有信号,继续;若两者 ≈ 且小
      ⇒ 坍缩,加 `dynamics_prior_scale` 重启或判死(同 §29 的 value ensemble 结局)。
- [ ] 蒸馏完成后:**离线验证**分歧是否追踪真实几何误差——在 target adapt run 上跑
      on-policy MPPI,同时记想象终端 `ẑ_H` 的跨头 std 与真实反事实终端 `s_H` 的
      `‖ẑ_H−s_H‖`,算 Spearman(应为 **正**,对比 value ensemble 的 −0.30)。这是 go/no-go:
      正相关 ⇒ 惩罚能把 argmax 从模型高误差区拉开;≈0/负 ⇒ 和 value 一样失明。
      (脚本可仿 `scripts/diag_value_ensemble_explain.py` 结构,把 value-disagree 换成
      dynamics-disagree。)
- [ ] eval 扫 `dynamics_disagree_coef`(0 / 小 / 大),300 局 strict(±0.1 内结论必须
      300 局,§27 铁律),看 target MPPI 能否 > 0.85 天花板 / 或至少 > coef=0 baseline。
- [ ] 若 coef 扫出"lose-lose 无甜点区"(像 §29 RPF 的 β):记录并判死这条悲观路,收尾。
- [ ] 头数 sweep(5 vs 10,MRS.Q 报 ~10 为 value 甜点)+ bagging(1.0 vs 0.5)+ prior。
- [ ] ⚠️ 注意 planning 成本随 num_dynamics_heads 线性增(next_all 每步 N 次 MLP 前向);
      eval mp 若太慢,减 eval 局数或头数。

### ★ 2026-07-25 model-based vs model-free 大样本对决 + "瓶颈=dynamics bias,不是 value 没 adapt"实锤
三问三答,全部 target 域(dynamics-only §12)/strict/hand_init σ=0.1、同协议。

**① dynens(model-based) vs IBRL(model-free) 匹配 2000 局:model-free 显著高 ~2pt。**
| 方法 | TARGET strict | 样本 | SE |
|---|---:|---:|---:|
| model-free IBRL(`threading_ibrl_rlprior_prior_sig01_seed1` model0,直接在 target 在线 + target-snapshots RL prior + warm-start,act_method=ibrl,1M 步) | **98.15%** (1963/2000) | 2000 | ±0.30% |
| model-based dynens MPPI(`threading_simdist_dynens_adapt_long_h5_seed1`,§31 大样本) | **96.2%** | 2000 | ±0.40% |
300 局时差距 z≈2.0(勉强可分),2000 局压噪声后 **z≈3.9 (p<1e-4)** ⇒ 这 ~2pt **真实显著**。定位:dynens 逼近但**未追平** model-free;"model-based 能逼近 model-free"成立,"能达到/持平"不成立(且都单 seed,方法级需 ≥2 seed)。

**② 悲观规划补不上这 2pt(呼应上面的 sweep):** 悲观治**方差**、剩的是**bias**;§31 已实测 ≤0.2 零效应、≥0.33 单调有害、预注册复现证伪小 coef 假峰。lose-lose 无甜点。

**③ ⭐ 证伪"value 没 adapt 是天花板"(用户假设):oracle dynamics + 未 adapt source value = 99.3%。**
dynens 是 `adapt_value=0`(value 头全程冻结在 source 蒸馏的 V_src)。跑 `scripts/eval_oracle_dyn_mppi.py --arms oracle --episodes 300 --shift target --probe_workers 8`(run_dir 默认 `threading_simdist_adapt_long_h5_seed1`,**无 `--value_agent` ⇒ 用模型自己 source 蒸馏的 value 头**,oracle arm 用真实模拟器 rollout):
| 配置 | 成功率 | 样本 | dynamics | value |
|---|---:|---:|---|---|
| **oracle dyn + 未adapt source value** | **99.3%** (298/300) | 300 | 真实模拟器 | source 蒸馏(未 adapt) |
| model-free IBRL | 98.15% | 2000 | — | — |
| dynens learned MPPI | 96.2% | 2000 | learned 5头 | source 蒸馏(未 adapt) |
带**同一个未 adapt 的 value**、只把 dynamics 换成完美的 → **99.3%,不但 > model-free、更比 dynens 高 3pt**。若未 adapt 的 value 是瓶颈,这不可能发生 ⇒ **dynens 距 oracle 3.8pt、距 model-free 2pt 全归因于 learned dynamics 的系统性 bias,不是 value 没 adapt**。机制:① source value 对 target 已近乎最优(§29:gap 在 transition 不在 `d(s)`,target 状态对 value 是 in-distribution,没啥可 adapt);② MPPI 在**偏掉的想象终端** `ẑ_H` 上读 value——adapt value 治不了"在错的点打分"。余量:300 局 oracle 是 99.3% 非满分(2 失败 ep268/278),即便完美 dynamics + source value + 这个 proposal 也非字面 100%(那 0.7% 可能是极刁初始位姿/proposal 偏弱/极少数 value 误排),但 ≪ dynens 的 3.8pt 缺口,不是 dynens 差距的解释。
⚠️ **eval_oracle_dyn_mppi.py 的坑**:`AdaptWorkspace.__init__` 劫持 stdout;脚本 `log()` 走 **stderr**。跑它**不能 `2>/dev/null`**(结果全丢),要 `2>&1` 一起 captured。
**⇒ 三条路(value-adapt / 悲观 / oracle)现在都有 300+ 局实证:只有把 learned dynamics 误差压准这一条能越过 0.96(oracle 极限 99.3%)。**

---

## 32. 新任务 Square_D0（第二个插入任务，测 SimDist 0.85 天花板是否任务特有）+ train_rl prior_margin（2026-07-20/21）

### What
在 Threading 之外加入第二个 MimicGen 插入任务 **Square_D0**（方栓插方孔），
用来检验 §27 的"天花板 = 模型误差/容错窗口"是否只在 Threading 出现。Square
容错窗口比 Threading 宽（方孔间隙大、靠重力落入、接触更简单），假说预测其
SimDist 天花板应**高于** Threading 的 0.85。第一步先建 source 专家：官方 core
数据 → MLP BC prior → IBRL 冲 100% source（target 域设计后续）。

### Where / 数据规格
- 数据：`release/data/mimicgen/core/square_d0/processed_data96.hdf5`（官方 core
  1000 demos，`done_mode 2`，state 模式无相机）。规格：**state=23**
  （eef_pos3+eef_quat4+gripper2+object14，对上 wrapper 的
  `STATE_SHAPE["NutAssemblySquare"]`）、action=7、~151 步/demo（比 Threading 的
  224 短）。`STATE_KEYS["Square_D0"]` 已在 MIMICGEN_D0_TASKS 注册，无需改代码。
- Square **不走** Threading 专属的 `resolve_threading_bindings` / strict_success /
  domain_shift（`mode=none` + strict 关 ⇒ 不解析 binding，不会 raise）。用原生
  MimicGen 成功判据 + 默认 σ=0.02 关节初始化（与官方 demo 同分布）。
- `release/cfgs/mimicgen/square_d0_state_source_mlp_bc.yaml`（MLP BC，1000 demo，
  state_stack=1，3×1024，2000 epoch）→ `exps/mimicgen/square_source_mlp_bc_n1000_seed1`。
  **结果：source 成功率 0.64（best ckpt）/ 0.82（训练峰值）**。
- `release/cfgs/mimicgen/square_d0_state_ibrl_source.yaml`（IBRL，MLP prior，
  state_stack=1 与 prior obs 对齐，无 shift/strict）→ `exps/mimicgen/square_ibrl_source_seed1`。

### ⭐ 结论：一般训练不要用 prior-margin —— 只会拉低 IBRL 最终性能
`prior_margin`（§26/§train_ibrl_simdist）是把 RL actor 的 Q 减一个 margin、
强制 max-Q 早期让位给 prior 的**冷启动/安全**工具，但它有**最终性能代价**：
永久地板让 RL 必须超过 prior 一个 margin 才被选中 ⇒ 封顶了策略改进；退火到低
地板又会在 OOD 处塌（见下）。**普通的 source IBRL 训练直接用原始 IBRL（无
margin、无 add_bc_loss）即可**，别为了冷启动提速去挂 margin —— 得不偿失。

Square 上的实测教训（为什么最后回到纯 IBRL）：
1. **纯 IBRL + MLP prior**：前期噪声性趴 0（60k 才冒头），与 threading 纯 IBRL
   同型（threading 也是 60k 冒头、290k 到 1.0）。前 ~100k eval 低是**正常冷启动**，
   不是卡住 —— 不要像我一样在 200k 就提前判停。
2. **加 prior_margin（init 1.0→地板 0.1，15k 退火，freeze_actor）**：10k eval 确实
   跳到 0.64（prior 水平，证明机制通），但 **20k 就塌回 0.04**。根因是 OOD：
   margin 完全让位给 prior ⇒ RL actor 动作从不被执行 ⇒ critic 学不到它们差 ⇒
   持续高估 ⇒ 地板一到 max-Q 就选了被高估的差动作 ⇒ 崩。**纯 margin + 冻结
   actor 在这个 OOD 问题上自相矛盾。**
3. ⇒ 用户拍板：一般训练**直接用原始 IBRL**，margin 只会拉低最终性能。

（例外/margin 仍适用的场景：§26 的 SimDist-MPPI-base 那种"强 prior + 明确要
安全策略改进 + 接受封顶"的设置。普通 from-scratch source IBRL 不属于此类。）

### train_rl.py 新增（默认关、逐位兼容，留作可选工具）
本次给 `train_rl.py` 加了 `prior_margin_init/steps/final` +
`freeze_actor_during_margin/thresh`（镜像 train_ibrl_simdist 的退火 + 冻结逻辑，
`prior_margin_init=0` ⇒ 纯 IBRL 逐位不变）。现在是 plain train_rl 的可选能力，
但按上面的结论，**默认就该保持关**。

### ⭐ 结果（2026-07-22，100 局标定）：纯 IBRL 到 94%，77% 只是预算不够
| 阶段 | source 成功率(100 局) |
|---|---:|
| MLP BC prior | 63% |
| 纯 IBRL，MLP prior，**400k** | 77% |
| **纯 IBRL，MLP prior，600k** | **94%** ✅ |
| 纯 IBRL，77% agent 当 prior（`sgft_source_agent`+`sgft_shaping=0`），400k | 48% ❌ |

**结论:**
1. **纯原始 IBRL 就到 94%（600k），不需要 margin、不需要 flow prior。** 77% 是
   **400k 预算太短**的未收敛快照（Square IBRL 冷启动前 ~100k 趴 0、后半程才 lift-off，
   到 400k 还在爬 0.60→0.77）；给到 600k 爬到 0.94-0.96 峰值。**教训:别在 400k 就判
   天花板——这个任务的 IBRL 需要 ~500-600k 才收敛。**
2. **用 77% 的 iter1 agent 当 prior 再迭代一轮，反而只有 48%（更差）。** 但这**不是
   "迭代有害"**，而是 iter2 同样是 400k 未收敛 + IBRL 冷启动高方差（iter1 77% / iter2
   48% 只是两条 400k 未收敛轨迹的运气差异）。要公平评迭代得把 iter2 也跑到 600k。实用
   结论:**直接纯 IBRL + 足够步数（≥600k）即可，不必迭代 prior。**
3. 再次印证 §32 主结论:**别用 margin**（当初加 margin 反而 20k 塌到 0.04）。

runs:`square_ibrl_source_seed1`(400k/77%)、`square_ibrl_source_600k_seed1`(600k/**94%**,
Square source 专家用这个)、`square_ibrl_source_iter2_seed1`(iter,48%)。配置
`square_d0_state_ibrl_source.yaml`(+`--num_train_step 600000`)/`_iter2.yaml`。

### TODO
- [x] 纯 IBRL Square source：**600k → 94%**（done 2026-07-22，见上）。→ Square source
      专家 = `square_ibrl_source_600k_seed1/model0.pt`。
- [x] **1M → 93%（done 2026-07-22）：加步数补不上最后 ~6%。** `square_ibrl_source_1000k_seed1`
      (best ckpt 100 局 **93%** ≈ 600k 的 94%)训到 1M 没超过 600k；训练内偶闪 0.98 是 50 局
      噪声。**MLP prior + 纯 IBRL 在 Square source 的硬天花板 ~93-94%**，要更高得换更强 prior
      (flow) 或就是任务残余难度。该 run 保留了 10 个 snapshot（`snapshot_100000..1000000.pt`，
      质量谱 random→expert，1.2GB）供后续 SimDist diverse 数据生成等用。
- [x] **⭐ 更强 prior 抬升 IBRL 上限（done 2026-07-24，500 局定论）：89% → 98%。**
      诊断（`scripts`/scratchpad `diag_square_gap.py`，300 局）先定性了 94% 专家的失败缺口：
      **不是均匀残差**——45% 是 grasp 失败（没抓起）、40% insertion、15% transport，且强烈
      位形依赖：**螺母初始旋转 [45,90)° 成功率仅 62%**（其余桶 90-100%），难手柄朝向抓不起。
      ⇒ ~half 缺口是**抓取覆盖**（更强 prior 能补），~40% 是插入精度残差。于是训了一个
      **RNN-BC**（`StateRnnGmmPolicy`，§5b，多峰 GMM + chunked，`square_source_rnn_bc_n1000_seed1`，
      **标定 83%** vs MLP 63%）当 IBRL prior 重跑纯 IBRL（`square_ibrl_source_rnnprior_600k_seed1`，
      600k）。**500 局 head-to-head：RNN-prior 97.8%（489/500）vs MLP-prior 89.2%（446/500），
      +8.6pt，z≈5.6，p<1e-7 高度显著。** ⇒ **更强专家确实大幅抬升纯 IBRL 上限**（覆盖缺口被补，
      逼近 100%；残 ~2% 是最难插入）。flow 没跑（太慢），RNN-BC 已够。
      **⇒ Square source 首选专家 = `square_ibrl_source_rnnprior_600k_seed1/model0.pt`（97.8%），
      依赖 `square_source_rnn_bc_n1000_seed1`（eval 时加载为 prior，别删）。**
- [x] **RNN-prior 跑满 1M（done 2026-07-25）：92.4%（462/500）< 600k 的 97.8%（z≈3.9）。**
      `square_ibrl_source_rnnprior_1000k_seed1`（+10 snapshot）。与 MLP 的 "1M≈600k" 同理再证
      **纯 IBRL 在 Square ~600k 到顶，过后是噪声平台**；1M 反而略低大概率是 **best-ckpt 按 50 局
      in-training eval 选、末段抖得厉害（0.88-0.98）选到不稳的 ckpt**（非"训练有害"）。首选专家仍
      是 RNN-600k。**通用教训：过了收敛点别再加步数，且 model0=单次 50 局最高分，选择本身有噪声。**
      ⚠️ **§27 教训重演**：MLP-prior 之前记的 "94%" 是 100 局高位噪声（600k/1M 各测一次都恰
      93-94%，连"两个独立 100 局都 ~94%"都骗了人）；**500 局真值 89.2%**。比较一律 500 局。
- [x] **Square target 域设计 + 接口实现 + 定稿（done 2026-07-25）。**
      **接口（默认关、逐位兼容）**：`env/domain_shift.py` 新增 `resolve_square_bindings`
      （SquareNut_main body/geoms、peg1 碰撞 geom、夹爪 actuator）+ `resolve_bindings`
      （task 分派：Threading→threading / Square*→square / 其余 raise）+ `apply_square_peg_shift`；
      2 个新 config 字段 `peg_size_scale`（>1 长方栓=收窄容错窗口，栓 body 固定 ⇒ 成功判据
      不动，类比 ring_size_scale；floor <1.42）、`peg_friction_scale`。mass/nut-friction/
      gripper_kp 复用现有 `mass_scale`/`friction_scale`/`gripper_kp_scale`（apply_contact_shift
      对 square binding 原样可用）。`env/robosuite_wrapper.py` 三处 binding 改分派 + fixed-shift
      按 binding 类型分支（ring→Threading / peg→Square）+ guard hole-mask（Square 无 obj_state_slices）。
      `evaluate/eval.py` 加 `--shift_peg_size_scale`/`--shift_peg_friction_scale`。
      **扫描（97.8% RNN-prior 专家，50 局/档定位 + 500 局定稿）——只有 gripper_kp 是强杠杆：**
      gripper_kp 1.0→0.98 / 0.40→0.78 / 0.30→0.48 / 0.20→0.14 / 0.15→0.06 / 0.05→0.00（cliff
      在 0.2↔0.4）；**mass3+nut_fric0.5 = 1.00（零效应！）**；peg_friction4.0=0.80；peg_size
      1.25(2.75mm/边)=0.98、1.35(1.4mm/边)=0.86（弱）。⇒ **专家强在插入精度、弱在抓取**（印证
      失败诊断），**Threading 的 mass/friction 配方不能照搬到 Square**，软抓（gripper_kp）才是核心。
      **定稿 target（500 局，多因子动力学+几何 gap；source 专家掉到）：**
      - **A（sharp，≈Threading 13%，推荐主 target）**：`gripper_kp 0.25 · peg_size 1.25 ·
        peg_friction 2.0` → **10.6%（53/500）**。
      - **B（moderate，留学习空间）**：`gripper_kp 0.30 · peg_size 1.20 · peg_friction 2.0`
        → **30.2%（151/500）**。
      eval：`--shift_mode fixed --shift_mask_hole_in_obs 0 --shift_gripper_kp_scale 0.25
      --shift_peg_size_scale 1.25 --shift_peg_friction_scale 2.0`。source 专家 =
      `square_ibrl_source_rnnprior_600k_seed1/model0.pt`（97.8%）。
- [x] **Square target IBRL 专家（在 target 域直接训，非收弱源专家的偏 demo；done 2026-07-31）。**
      两个 target 各训一个纯 IBRL（train+eval 都在 target 上，σ=0.1，RNN-BC 源专家当 prior）：
      **target-A（sharp，源零样本 10.6%）→ 23.8%**（500 局）——冷启动受限，prior 太弱（~10%）
      1M 大半趴 0.1；**target-B（moderate，源零样本 30.2%）→ 68.2%**（500 局）成功 lift-off。
      ⇒ sharp target 的弱 prior 是 IBRL 冷启动硬骨头；moderate 可行。config
      `square_d0_state_ibrl_target_{A,B}.yaml`，run `square_ibrl_target_{A,B}_seed1`。
- [x] **⭐ target-B 迭代 bootstrapping 到 94.2%（done 2026-07-31，500 局）。** 用上一轮的
      hybrid 当下一轮 prior（`sgft_source_agent`+`sgft_shaping=0`=纯 IBRL），逐轮推高：
      iter1（RNN-BC 30% prior，1M）**68.2%** → iter2（iter1 68% prior，2M）**~90%(峰0.96)**
      → iter3（iter2 90% prior，4M）**94.2%**（471/500）。σ=0.1 全程。runs
      `square_ibrl_target_B_iter{2,3}_seed1`（model0）。**结论：越强 hybrid 当 prior 逐轮
      bootstrapping 可把 target 专家从零样本 30% 推到 ~94%**（每轮仍从 0 冷启动重学，但更强
      prior 让恢复更快、天花板更高；同 source 的 MLP→RNN prior 迭代规律）。
      → **Square target-B 专家 = `square_ibrl_target_B_iter3_seed1/model0.pt`（94.2%）**。
- [ ] 用 94% 的 target-B 专家做下游：收 target demos（`collect_target_demos.py` 已加
      `--gripper_kp_scale/--peg_size_scale/--peg_friction_scale`，用 B 的 kp0.30+peg1.20+pf2.0）
      + 接迁移方法（SimDist adapt / cotrain）。成功判据栓固定无推动作弊，首轮不加 strict。
- [ ] Square SimDist：stage-1 distill 已跑（`square_simdist_diversev2_h5_seed1`，source
      收敛 ~0.90，但纯 distill 的 target 零样本≈0，需 stage-2 adapt）。用 B 专家收 target demos
      后跑 §24 adapt，测 Square SimDist 天花板 vs Threading 0.85（假说：Square 窗口更宽 ⇒ 更高）。
- [x] Square SimDist 数据：`square_d0_source_diverse_v2`（200k 轨迹=Threading 2×，
      σ=0.1，格式与 threading diverse_v2 一致，见 §32 收集记录）。

---

## 33. 为什么 value adapt 会让 SimDist 变差 —— 根因 = 排序信号只有 0.0068（2026-07-25~28）

### 起因
历史上五条 value-adapt arm **全部**跑崩，且"越训越准、越跑越差"，没有任何离线指标预警：

| 历史 run | 步数 | eval 点 | 收敛分数 | 参照 R | val_mse |
|---|---:|---:|---:|---:|---:|
| `ablate_val_ws_600k`（`encode(s)`×专家Q，warm start） | 600k | 56 | **0.370±0.008** | 0.745 | 3e-4 恒定 |
| `ablate_val`（同上，reinit） | 400k | 21 | 0.378±0.019 | 0.745 | 4e-4 |
| `ablate_dynval_ws_600k`（dyn+value 联合，MC return） | 600k | 59 | 0.173±0.015 | 0.705 | — |
| `ablate_dynvalenc_ws_600k`（再放开 encoder） | 600k | 54 | **0.000** | 0.760 | — |
| `threading_simdist_valueadapt_sig0.1_seed1`（**在线** `adapt_value=1`，MC return） | 20k trans | 21×500局 | **0.000**（第一轮就掉） | 0.604 | — |

在线那条最惨：0.604 → **一轮（1000 步 value 梯度）就到 0.000，之后 20 个点全零**，而 `policy_only` 全程稳在 0.37-0.45 ⇒ 环境/policy/dynamics 都好，**只有 planner 被摧毁**（MPPI 0.000 < 它自己的 proposal 0.40，= 排序退化成噪声的签名）。

### ★ 根因（一个量）
**MPPI 区分候选的信号 = `std_i(γ^H V)` 在候选集内的散布 = 0.0068**，而 V 在状态间的量程 ~0.5 ⇒ **判别信息活在第 4 位有效数字上，且是"同一状态下邻近点之间的差"**。任何离线重拟合的残差都是 0.005-0.033，即**同量级或大 3-5 倍** ⇒ top-64 elite 集合被大面积改写 ⇒ 执行动作（那 64 个首动作的算术平均）变质。

**`val_mse=3e-4` 看着像四位小数精度，实为 RMS 0.0173 = 信号的 2.6 倍。** 所有监控量（MSE/Spearman/单调性/SNR）都是相对 0.5 量程算的，而决定成败的是 0.0068 —— **差两个数量级，结构上不可能预警**。

两个使能前提（先测出来才有上面这句）：
- **CEM softmax 是死的**：`temperature×std = 0.003-0.006` ⇒ elite participation **64.00/64** ⇒ 提交动作 = top-64 首动作的**均匀平均**，softmax 重加权数学上无效。
- **V 占 return 散布的 98%**（reward 项 std 只有 0.0019）⇒ **value 唯一的作用渠道是 topk 成员选择**。

### 因果剂量-响应（E0 系列：目标=现役函数本身，只有拟合精度在变）
| 残差/信号 | 0 | 0.33 | 0.43 | 0.59 | 0.86 | 0.93 |
|---|---:|---:|---:|---:|---:|---:|
| MPPI（200 局） | 0.775 | 0.749 | 0.725 | 0.660 | 0.450 | 0.345 |

`Spearman = −1.000`。⇒ **"相对现役函数的偏离÷信号"单独就足以复现全部崩塌**，不需要任何别的解释。

### 全部干预（每点 200 局，全部跑到收敛；基线"不 adapt" = 0.775）
| 干预 | 收益 | 有效？ |
|---|---:|---|
| 换 teacher（专家 Q ↔ V_src） | ±0.03 | ❌ |
| 换评估点（`encode(s)` ↔ rolled `f^H`） | ±0.02 | ❌ |
| 换成 stage-1 那份 **source** 数据（`source_diverse_v2`） | C +0.05 / **D +0.01** | ❌ |
| 数据量 ×7~13（300→2000/4000 条 episode） | B +0.13 / C +0.04 / D +0.02 | ❌ |
| lr 3e-4→1e-4 | +0.09 | ❌ |
| value 权重 EMA 0.999 | +0.06（**跨 ckpt 跨度压 5.4×**） | ❌ |
| **带新 value 重 adapt dynamics 30k**（pairD 0.550→均值 0.577；对照 pairStock 0.715→**0.788**） | **+0.03** | ❌ |
| **换成"现役函数自身"当目标（E0 自蒸馏）** | **+0.25**（0.749 ≈ 基线） | ✅ 唯一 |

收敛后的量化链条（n=482 决策点，在 stock 的 on-policy 候选集上测）：
| arm | latent × teacher | 均值 | 集内偏离 | **残差/信号** | **top-64 重合率** |
|---|---|---:|---:|---:|---:|
| 不 adapt | — | **0.775** | 0 | 0.00 | 1.000 |
| **E0** | rolled × **现役函数** | **0.749** | 0.0022 | **0.33** | 0.745 |
| C | `encode(s)` × V_src | 0.514 | 0.0046 | 0.67 | 0.529 |
| B | rolled × 专家 Q | 0.516 | 0.0191 | 2.82 | 0.252 |
| D | rolled × V_src | 0.436 | 0.0088 | 1.29 | 0.455 |
| 历史 600k | `encode(s)` × 专家 Q | 0.370 | 0.0152 | 2.23 | 0.287 |

`Spearman(score, top64 重合率) = +0.943`，`Pearson = +0.936`。

### 为什么外部标签有**不可约**下限（加数据/调 lr 都补不上）
1. **目标是另一个函数**：V_src / 专家 Q 定义在**真实状态**上，planner 查的是**想象 latent**（§31 标定二者差 ≈5.6mm 等效）。数据再多也只收敛到 `E[标签|查询点]`，那不是现役那个函数。
2. **rolled 臂的标签甚至不是查询点的函数**：不同 `(s,a)` 映到几乎相同的 `f^H` 却对应不同真实 `s_{t+H}` ⇒ 条件方差不可约。证据：B/D 的 VAL 高于 A/C，且**只有 B/D 剧烈跳变**（跨度 0.275/0.295 vs C 的 0.060）。
3. **现役头的特权不是精度，是"相对自己的残差恒等于 0"**：它在 **source 域数据 + adapt 之前的 dynamics** 上拟合、然后全程冻结（`adapt_value=0`），按任何离线标准都比 C/D **更不匹配**部署场景，却赢 0.26。它是 stage-1 **联合优化**的产物（`consistency_coef=20` 把 `ẑ_{t+1}` 拉向 `encode(s_{t+1})` ⇒ 标签渐近成为查询点的自洽函数；`value_coef=0.1` ⇒ value 只是权重 1/200 的副产品），而 dynamics 的 adapt 又是**带着它在回路里**做的。
4. ⚠️ **最反直觉的一点**：重拟合的头在"对专家 value 的排序一致性"上其实**更好**（想象终端上 Spearman **+0.176** vs 现役头的 **−0.010**，精确的 0），闭环却输 0.26-0.40。⇒ **这里不存在一个"真的 V"要逼近；planner 只需要一个自洽的排序函数，而"不同"本身就是伤害。**

### 与 dynamics adapt 的不对称（为什么那个反而有用）
dynamics 误差改变"想象到哪个状态"，被**任务容错窗口物理缓冲**（§27 剂量-响应：5.6mm 持久误差在 target 上仍 0.875）。**value 排序误差没有任何缓冲** —— 信号就是 0.0068，同量级误差全额进 argmax。

### 若"必须"adapt value（如加宽 needle 初始分布）
唯一没被否证的形态 = **以现役函数为锚的门控残差**（未实测，设计已定）：
```
V_new(z) = V_frozen(z) + g(z)·Δ_θ(z)          # Δ 最后一层 zero-init ⇒ day-0 恒等, top64 重合=1.000
```
- **门控 g** = RND 新颖度（冻结随机网 + 预测器，**在旧区域的查询点 latent `f^H` 上**训，不是 `s0`；latent 模式下 decode 不可靠，§30 往返 53mm），按旧分布 99.5 分位标定成平滑 sigmoid。**必须平滑**：同一候选集里若一部分落新区域一部分落旧区域，`g` 本身就成为集内波动 ⇒ **门控自己消耗预算**，须实测 `std_i(g)`。**固定不学**（联合学 g 会让优化器在旧区域也开门）。
- **不要 value ensemble 的分歧当门控** —— §29 已实测结构性失明（target 态分歧 0.0083 ≈ source 0.0078）。
- **新区域监督 = 锚定 backup**：`V_new(z_t) ← r̂ + γ^H·V_frozen(f^H)`，用 `(1−g(f^H))` 加权、迭代 2-3 轮向外扩张。**绝不用 MC 回报**：稀疏 0/1 下噪声 `≈0.5/√n`，要压到 0.0068 需**每状态 ~5000 条 rollout**（这就是在线 `adapt_value=1` 一轮归零的原因）。
- **信任域正则**：在旧区域候选集上罚 `std_i(γ^H g·Δ)`，**不罚逐状态均值**（`ε̄` 对 CEM 的 topk+softmax 严格不变 ⇒ 免费）。
- 闸门（闭环前先过）：旧区域 **残差/信号 < 0.3**、**top-64 重合率 > 0.85**；新区域 ΔV 显著非零。

### ★ 读数纪律（这轮栽了三次才立起来）
1. **收敛判据**：分数与 VAL_mse **同时**平掉且 ≥3 个连续 eval 点无趋势。违反它栽过三次：E0 10k 0.555→30k 0.725（我据此错误宣布"存在 0.19 不可约残差"）、B 30k 0.515→收敛 0.389、D 30k 0.560→0.411（后两个让我错误排出"latent 轴 > teacher 轴"）。
2. **`best.pt` 系统性乐观 0.06-0.19**（D best 0.630 vs 均值 0.436、B 0.580 vs 0.516）⇒ **一律用多点均值**。
3. **默认开 `--value_ema 0.999`**：跨 ckpt 跨度 0.295→0.055，零代价（不拖慢收敛、不伤 VAL）。低 lr 抬均值 +0.09 但**压不住跳变**（机制不同，可叠加）。
4. **`heldout_mse` 曾是假的**：`latents_and_targets()` 从训练池重采样 ⇒ in-sample，乐观 1.7-9.1×。已改为独立 episode 的 `VAL_mse`。
5. **"数字太丑/太漂亮先查工具"** 又救了两次：`explore_std=0.5` 让 rolled 臂的 latent 与标签错配（B 掉到 0.010）；`200 % 6 ≠ 0` 触发 mp eval 断言。另 `pkill -f <pat>` 会**匹配自己所在的 shell**，把父 shell 连子进程一起杀。

### Where
- `scripts/value_head_recipe_ablate.py`（NEW）—— 重训框架：`--arm {E0,A,B,C,D}`（latent×teacher 的 2×2 + 自蒸馏对照）、`--base_cfg_dir`（换 arch，如 dynens）、`--value_ema`、`--explore_std`、真 `VAL_mse`；每次 eval 校验 `frozen_drift`（value 外所有张量的最大变化）**全程 0.00e+00**。
- `scripts/diag_value_residual_vs_score.py`（NEW）—— 在 on-policy 候选集上测 `std_i(γ^HΔV)/信号` + top-K elite 重合率 + 与现役头的排序一致性。
- `scripts/diag_value_adapt_{ranking,scale,elites,steer}.py`、`diag_value_potential_broad.py`、`diag_elite_action_geometry.py`、`diag_valrecipe_generalization.py`（NEW）—— 七个被**否证**的假设的测量（想象点排序变差 ❌ / softmax 尺度 ❌ / 不信任 proposal ❌ / 单步操舵 ❌ / 局部势能锯齿 ❌ / elite 平均抵消 ❌ / 偏好旋转 ❌）。留着是为了别再重复：**它们全是"精度/排序"类假设，而结构约束早就说明唯一渠道是 topk 成员选择**。
- ckpt：`exps/mimicgen/threading_valrecipe_{A,B,C,D,E0}[_long|_bigdata|_srcdata|_ema|_lr1e4|_s300|_s1500|_s8000]_seed1/`、`threading_repair_pair{D,Stock}_seed1/`。

### ★★ 2026-07-28 重大限定：在 dynens（更准的 dynamics）上，这整个崩塌**几乎消失**
同样的手术搬到 §31 的 dynamics-ensemble 系统（5 头、H=5、target 基线 0.94-0.98、σ=0.1）：

全部跑到 200k 收敛（每点 200 局，EMA 0.999，五点均值）：

| arm（相同手术，仅系统不同） | 现役系统（H=3, 天花板 0.85） | **dynens（H=5, 天花板 0.96）** | 跌幅改善 |
|---|---:|---:|---:|
| 基线（不 adapt） | 0.775 | 0.94-0.98 | — |
| E0（自蒸馏） | 0.749（−0.03） | **0.963**（−0.00），跨度 0.005 | — |
| C（`encode(s)`×V_src） | 0.514（**−0.26**） | **0.930**（**−0.03**），跨度 0.015 | **8.7×** |
| B（rolled×专家Q，残差最大） | 0.516（**−0.26**） | **0.863**（**−0.10**），跨度 0.015 | 2.6× |

两个附加签名：① **残差排序仍保留**（B 残差>C ⇒ B 掉更多，−0.10 vs −0.03）⇒ 机制方向没错，
只是**尺度被 dynamics 精度整体压缩**；② **跨 ckpt 跳变基本消失**（现役系统 0.06-0.28 → dynens
0.015）—— 现役系统那种"每个快照落在不同排序上"的抖动是"需要精细补偿"的表现，畸变小了就没了。

**预测错了一个数量级**（我按"信号会变小 ⇒ 更致命"预测 0.40-0.60）。两处修正：
1. **信号不是变小而是变大 1.64×**：dynens 实测 `std_i(γ^H V)=0.01112` vs 现役 0.00679。
   原因大概率不是集成，而是**我把集成和 horizon 搞混了** —— dynens 用 **H=5**（现役 H=3），
   5 步让候选走得更开 ⇒ 真实价值差更大。**⇒ 更长的 horizon 直接买到更大的判别信号、
   因而对 value 误差更宽容**（这条本身是新的、可用的结论）。两个使能前提在 dynens 上仍
   成立：`temp×std=0.0045` ⇒ participation 仍 **64.00/64**（softmax 死）、V 占 return
   散布 **100%**（reward 项 std 精确为 0，H=5 下更死）。
2. **"分数 = f(残差/信号)"这个单标量关系不能跨系统迁移**：C 的这个比值只从 3.0× 降到
   2.03×（1.5 倍），代价却从 −0.26 降到 −0.02（**13 倍**）。§33 主体那条剂量-响应只在
   **同一系统内**成立。

**统一解释（新，且能回溯解释 §27/§29 的整批失败）**：value 头的"特权"来自它要**补偿
dynamics 想象终端的系统性畸变**（§31 标定现役系统 ≈oracle+5.6mm），而这个补偿关系只有
共适应出来的那个头懂。**dynamics 越准，需要补偿的畸变越少，value 头的特权就越小** ⇒
共适应的重要性**与 dynamics 的误差成正比**。现役系统（5.6mm 畸变、天花板 0.85）里换
value 是致命的；dynens（畸变小得多、天花板 0.96）里几乎免费。预测：oracle dynamics
（§31 实测 99.3%）上换 value 应当更无害。

**⇒ 实践结论按系统分叉**：§33 主体那套"别 adapt value / 只能门控残差"的强约束，是
**天花板 0.85 那个系统**的结论；**在 dynens 上直接重拟合 value 只花 0.02-0.05**，所以
"加宽初始分布 ⇒ 必须 adapt value"这类需求应当**在 dynens 系统上做**，可以用简单得多的
方案（甚至全局重拟合）。

### ★★★ 2026-07-29/30 什么时候 value 才真的必须 adapt —— 答案是 OOD，不是"可达性"
§33 主体证明"别 adapt value"，但没回答"什么情况下不 adapt 就不行"。用同一把尺子
（**oracle dynamics + 冻结的 source value**：模型误差=0 ⇒ planner 唯一还能出错的部件
就是 value 的排序）量了三种 shift，每个 30-60 局：

| 域 | 机制 | **oracle + 未 adapt 的 source value** | value 缺口 |
|---|---|---:|---:|
| §12 target | 均匀变难（接触物理） | **0.993** | — |
| **jam**（NEW，`ring_friction 5.0 × ring_size 0.85`，nominal 物理） | **价值序反转**：同一几何在两域价值相反 | **0.967** | −0.026 |
| **wide1.5**（`needle_scale=1.5`，针 init box ×1.5） | **真 OOD**：整块区域训练中从未访问 | **0.933** | **−0.060** |

**⇒ 只有真 OOD 让 value 失明。** 三臂拆账（wide1.5，n=60）把两笔损失分开：
policy 0.133 → **learned 0.683** → **oracle 0.933** ⇒ **value 的 OOD 代价 0.060**、
**dynamics 在加宽区的额外代价 0.250**。闭环数据独立印证：`..._wide1.5_seed1` 稳态
**0.816** vs `..._nobc_seed1` **0.900**（差 0.084），而 wide1.5 的 **proposal 明显更强**
（policy_only 0.244 vs 0.160）⇒ 差距不是 proposal 的锅。

**三次失败的 trap 尝试（都别再重复）**：
1. **三脚架倾倒**（NEW 旋钮 `tripod_mass_scale`/`tripod_com_height_scale`，默认 1.0）。
   因果验证确认**真不可逆**（从倾倒态重启 `p_recover=0.00` vs 同时刻未倾倒的 CONTROL
   0.29-0.53），source 侧 **0/20 局发生**（最大倾角 2.8°）、target 15%。**但 value 看得
   清清楚楚**：几何匹配后 ΔV=**0.35**、AUC **0.92**。原因 = 倾倒改变 `tripod_pos/quat`，
   而 V 本来就在追这个几何、且它每局随机 ⇒ **"见过的特征取新值"能被正确外推**。
2. **jam（卡死）**：`deep%` 86.7→86.7、`min_dist` 0.122→0.124（几何自然匹配）、
   `steps_deep` 32→195（卡住的签名）、succ 80%→33%（仍可解）。V 的**绝对标定完全错**
   （同一状态 V=0.846，真实 p_success source 0.97 vs jam 0.03），**但排序完好** ⇒
   oracle 0.967。正是 §33 的 `ε̄`（逐状态常数，CEM 的 topk+softmax 严格不变）vs
   `ε̃`（集内波动）之分的又一个实例。
3. **纯可达性域做不出来**：三脚架是 18g 自由体，任何让它更易被扰动的改动都同时增加
   倾倒**和滑移**，而 strict_success 的 `max_tripod_disp=0.04` 把滑移直接判失败 ⇒
   `succ|upright` 85%→35%。`com_h=-8` 更把质心推到物体外 22cm（非物理），三脚架滑出
   **67cm** 却 tip%=0。

**⇒ 修正后的判据**（我一度错误地宣称"OOD 不够、必须改拓扑"，被 wide1.5 的数据推翻）：
> **让 value 失明必须把状态推出训练支撑（整块从未访问的区域）。改变可达性、制造不可逆
> trap、制造价值序反转都只改"度量"（要走多少步），而 V 编码的"哪个状态更接近目标"这个
> 排序取决于 GOAL 而非 DYNAMICS，所以排序活得下来。** 而"初始分布变宽"恰好是最贴近真实
> sim2real 的形态（真机的工件位姿/公差/抓取点都会超出仿真采样范围）。

**⇒ 这也是 §33 那套门控残差第一次有明确用武之地的场景**，而且形态天然对上：`g` = RND
新颖度（"是否超出支撑"的直接度量，正是失明原因）、Δ 零初始化保住旧区域那 0.933、加宽区的
针几步就回熟悉区域 ⇒ 锚定 backup 容易到。

**方法论教训（本轮又栽 3 次）**：① `load_frozen_agent` **强制 `act_method="rl"` 摘掉 IBRL
prior**（`train_rl.py:93`），用裸头测出的 source 专家只有 0.44 且倾倒率 24%，挂回 prior
（`train_rl.attach_ibrl_prior`）后是 **0.85 / 0.0%** —— 三个结论全反转。这个坑在本项目已
咬第三次（§17 v3 数据收集、§27 value 语义、这次）。② "太漂亮的数字先查工具"再次生效：
`dV=0.0000` 精确到四位小数 ⇒ 查出两域快照是**同一批物理状态**（`d<0.14` 时针还没碰到环、
轨迹逐位相同）⇒ 那个"反转测试"是**恒等式**不是测量。③ 代理指标（倾角当 trap 标签）测到
的是**反向因果**（本来要失败 ⇒ 乱动 ⇒ 蹭倒三脚架），必须用经验不可逆性（重启 N 次）。
**闭环 eval 一次又一次比代理指标可靠。**

Where: `env/domain_shift.py`（`tripod_mass_scale`/`tripod_com_height_scale`，默认 1.0 ⇒
逐位兼容）+ `env/robosuite_wrapper.py`（`apply_tripod_trap_shift`）；
`scripts/eval_oracle_dyn_mppi.py --shift jam`（+`--jam_friction`/`--jam_ring`）；
`scripts/calibrate_tripod_trap.py`、`diag_tripod_trap_causal.py`、
`diag_value_trap_blindness.py`、`diag_jam_value_inversion.py`（NEW，四个 trap 诊断）。

### TODO
- [ ] 门控残差在 **wide1.5** 域上的首次实测（目标场景与判据现已确定）：对照三条 ——
      冻结 value（oracle 0.933 / learned 0.683）、全局重拟合（§33 预期 −0.03~−0.10 on
      dynens）、门控残差。闸门：旧区域残差/信号 < 0.3 且 top-64 重合率 > 0.85。
- [x] dynens 上重跑 C + E0 —— done 2026-07-28，见上（预测被推翻）。
- [ ] 收尾中：`threading_valrecipe_C_dynens200k_seed1`（C 延到 200k 拿 5 点均值，把
      −0.02~−0.05 这个区间收紧）+ `threading_valrecipe_B_dynens_seed1`（残差最大的 B 臂，
      测极端情况是否也无害）。
- [ ] 若 B 在 dynens 上也几乎无害 ⇒ "共适应重要性 ∝ dynamics 误差"这条基本坐实，可再用
      oracle dynamics 臂（§31 的 `eval_oracle_dyn_mppi.py` 结构）做端点验证。
- [ ] 门控残差（上面那套）在"加宽 needle 初始分布"场景上的首次实测。**前置**：`env/domain_shift.py` 还没有 needle 初始位姿分布的旋钮（`hand_init_perturb*` 是机器人的），需新增 `needle_init_pos_range`/`needle_init_rot_range`（注意 Threading 用 **freejoint qpos** 设置物体位姿，写 `body_pos` 会被 `_reset_internal` 清掉 —— 同 `apply_hole_offset`）。
- [ ] 多 seed（全部单 seed）。

---

## 34. 仓库瘦身：已归档的实验分支（2026-08-01）

### What
为了让公开仓库聚焦在 **SimDist 跨 domain adapt** 这条主线上，若干已被证伪 /
已停止推进 / 与本课题无关的实验分支，从 git 索引中移出，**移动到本地
`_archive/`**（被 `.gitignore` 排除，不随仓库发布）。

**代码没有被删除，只是不上传。** `_archive/` 完整镜像原路径，恢复任意一项：

```bash
mv _archive/rl/gated_agent.py rl/gated_agent.py      # 单个文件
cp -r _archive/. .                                    # 全部还原
```

### 归档清单（103 个文件 / 984 KB）

| 分支 | 章节 | 归档内容 | 结论 |
|---|---|---|---|
| H2O+ on IBRL | §14 | `train_h2oplus.py` `rl/h2oplus_agent.py` + YAML/scripts | online ratio ≈ no-op |
| GatedGapAgent | §16 | `train_gated.py` `rl/gated_agent.py` + YAML/scripts | 未跑赢 `force_c=1.0` 对照 |
| LWAIL-cross | §18 | `train_lwail.py` `train_icvf.py` `rl/{lwail_agent,icvf}.py` + YAML/scripts | 判别器饱和（占用零重叠） |
| recovery 数据收集 | §19(2) | `scripts/{collect_recovery_data,collect_target_recovery,...}` + YAML | 结论已记录，数据流程停用 |
| 旧 residual（gap-weighted） | §20 | `train_residual.py` `rl/residual_agent.py` + YAML | 被 §25 的 ResFiT 版取代 |
| OracleMap | §21 | `train_oraclemap.py` `rl/oraclemap_agent.py` + YAML/scripts | actor drift 崩溃 |
| Oracle + residual | §22 | `train_oracle_residual.py` + YAML | 同上线，未跑完 |
| MB-IBRL 训练入口 | §23 | `train_mb_ibrl.py` + YAML/scripts（**`rl/mb_ibrl_agent.py` 保留**） | 从未跑；agent 的 `DynamicsEnsemble` 仍被 §30 诊断使用 |
| GAIL sanity | — | `train_gail.py` `rl/gail_agent.py` + YAML | AIL 不适合本任务 |
| weighted BC | §11/§19 | `train_weighted_bc.py` + YAML/scripts | 结论已记录 |
| SGFT / co-train 的 YAML | §10/§11 | 仅 YAML（**代码保留**在 `rl/q_agent.py` / `bc/dataset.py`） | 开关仍可用 |
| upstream 残留 | §1 | `env/{metaworld_wrapper,franka_wrapper,franka_utils,polymetis_controller,drawer,hang,lift,towel,cameras,mocks}.py`、`mw_main/`、`release/cfgs/{metaworld,robomimic_bc,robomimic_rl}` | MetaWorld 被故意跳过（§1）；franka/polymetis 是真机代码，本项目无硬件 |

### ⚠️ 归档时必须保留的三个交叉依赖（踩过）
1. `rl/rebrac_agent.py` —— `rl/residual_rl_agent.py` 导入其 `EnsembleQ`。
2. `train_rebrac.py` —— `train_residual_rl.py` 导入其 `_load_demos_flat`。
3. `rl/mb_ibrl_agent.py` —— §30 的 `scripts/diag_dyn_rollout_error.py` 与
   `scripts/diag_latent_qres_sensitivity.py` 导入其 `DynamicsEnsemble`。

另：按前缀批量筛 YAML 时，`residual` 会误伤 `residual_simdist` /
`residual_srcbase_002` / `residual_rl` 这三个 **SimDist 主线**配置 —— 别删。

### 同时做的清理
- `.gitignore` 新增：`*.log`（966 MB 训练日志）、`*.pt/*.pth/*.ckpt`、
  `*.npz/*.npy/*.hdf5/*.h5/*.pkl`、`_archive/`。仓库体积 1.08 GB → **2.76 MB**。
  `release/data`(72 G) / `exps`(48 G) 本来就已被忽略；`deps/` 是 submodule，
  只上传指针（clone 需 `--recursive`）。
- `evaluate/eval.py` 移除了 9 个指向已归档模块的 `--mode` 分支
  （h2oplus/gated/lwail/weighted_bc/rebrac/residual/oraclemap/oracle_residual/mb_ibrl），
  并补上原本遗漏的 `--mode residual_rl`（`train_residual_rl.py` 一直有 `load_model`
  却没接进分派）。现存 mode：`bc/rl/simdist/residual_rl/residual_simdist/ibrl_simdist/tdmpc2`。
