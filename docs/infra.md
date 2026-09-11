# 基础设施：domain shift · BC heads · eval · 配置 · 坑

跑任何实验都可能要查的：domain shift 框架与旋钮、flow/RNN/diffusion BC head、source/target eval 基础设施（σ 初始化、TopkSaver、CLI flag）、strict success、配置表、已知坑、Threading target 的定义。

> 从 `CLAUDE.md` 拆出（2026-08-14）。速查表、方法论铁律、文件地图仍在 `CLAUDE.md`；已归档分支见 `CLAUDE_ARCHIVE.md`。

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

### TODO
- [ ] `resolve_*_bindings` 目前只覆盖 Threading / Square / Coffee；加新 task 要照写。
- [ ] 可选：视觉域随机化（材质/光照/相机），与当前的接触相 shift 正交。
- [ ] `hole_offset` 的 POMDP 论证、旧 offset-bearing 配方的逐参数调参史见 `CLAUDE_ARCHIVE.md`（当前 target 是 dynamics-only，`hole_offset=0`）。

---

## 4. Flow-matching BC policy

### What
`StateFlowPolicy` — rectified-flow / linear-OT BC head for state-based control.
在主线里的用途：**SimDist 的 MPPI proposal**（§36：换 flow proposal 在弱 WM 上
+0.134，在已到顶的 WM 上 −0.026 ⇒ proposal 只在它 binding 时值钱）。

### Where
- `bc/flow_policy.py` — `StateFlowPolicyConfig` + `StateFlowPolicy`
  - 训练：linear OT `x_t=(1−t)z+t·a`，target `v=a−z`，MSE（**reduction=mean**，
    与 `bc/diffusion_policy.py` 对齐，所以同一 `lr=1e-4` 可直接比）
  - 推理：Euler ODE 10 步；`act()` / `act_chunk()`（后者返回 `[B,Ta,D]`）
  - `update_ema()/apply_ema()/restore_from_ema()`；**EMA shadow 持久化在
    `state_dict()`** 里（`_ema_shadow__` 前缀）⇒ 独立 eval 自动用 EMA 权重；
    旧 ckpt 无这些 key 也能加载（fallback 到 online 权重）
  - **输入/输出归一化**（state z-score + action min/max→[−1,1]），存成 buffer，
    **identity 初始化 ⇒ 逐位向后兼容**；`train_bc.py` 从 dataset 的 train split
    注入（`policy.set_normalization(**dataset.get_normalization_stats())`）
- `bc/dataset.py` — `DatasetConfig.action_horizon`（chunk 长度，1 = 旧的单动作）
- `train_bc.py` — `use_flow` + `flow_policy` 嵌套 config
- `evaluate/` — 每 worker 一个 action queue，`is_first_step` 标记 episode 边界；
  `_policy_act_chunk` 对非 chunk 策略有 fallback ⇒ 旧 BC head 行为不变

### Horizon 约定（对齐 robomimic）
- `pred_horizon` (**Tp**)：网络预测/训练的 chunk 长度，必须 == `dataset.action_horizon`
- `action_horizon` (**Ta**)：每次重预测执行的长度，≤ Tp
- 默认 Tp=16 / Ta=8。⚠️ **SimDist 里当 proposal 用时是 Ta=1**，而
  `finetune_flow_ug.py` 早期的 ckpt 选择跑在 Ta=8 上 ⇒ 选出来的不是 Ta=1 下最好的
  （§36；已加 `--eval_ta`）。

### How
```bash
python train_bc.py --config_path release/cfgs/mimicgen/threading_d0_state_flow_bc.yaml
#   --flow_policy.num_inference_steps 5     # 快一倍
#   --flow_policy.backbone mlp              # 旧 MLP backbone（默认 unet1d）
#   --flow_policy.normalize_obs 0 --flow_policy.normalize_action 0   # 归一化 ablation
```

### Design choices
- backbone 默认 `"unet1d"`（robomimic `ConditionalUnet1D`，FiLM-cond，4.92M，
  与 `bc/diffusion_policy.py` 逐参数一致 ⇒ flow-vs-diffusion 只差训练目标）；
  `"mlp"` 是旧的 6 层 1024 变体（5.94M）。连续时间 t∈[0,1] 先乘 `time_scale=1000`
  再进 `SinusoidalPosEmb`，否则 embedding 塌掉。
- eval 的 chunk queue 放在 evaluate 框架里而非 policy 里 ⇒ policy 无状态、可 pickle。
- CFG（domain-label conditioning）与 low-data 归一化实验见 `CLAUDE_ARCHIVE.md`（§13/§15）。

---

## 5. Source/target evaluation infrastructure

### What
两个**互斥**的机器人初始化噪声旋钮 + 训练中 source/target 双 eval + CLI 覆盖。

### 初始化分布（本项目所有实验的隐藏变量，务必对齐）
- **两个都不传（默认）** = robosuite 内置 **σ=0.02** joint-qpos 高斯（≈22 mm EE std），
  与 MimicGen 采 demo 时同分布。
- `hand_init_perturb_joint_sigma: float` — 直接改这个高斯的 σ。**本项目标准是 σ=0.1**
  （更宽、更难；⚠️ 光换初始分布本身就能吃掉十几个点，见 §40b：Coffee source
  99.8%@σ0.02 → 88.6%@σ0.1）。σ=0 ⇒ 确定性初始化。
- `hand_init_perturb: (rx,ry,rz)` — EE 空间 uniform box + 3-DOF 位置 IK 覆盖
  （sgft 风格）。`(0,0,0)` = 完全确定性。与上一个**互斥**（`__init__` 里 assert）。
- 初始物体位姿分布（如 Square 的 `nut_z_rot` 窄化）走 `InitDistConfig`，见 §41。

### Where
- `env/robosuite_wrapper.py` — 两个 kwarg + 互斥断言
- `bc/dataset.py::DatasetConfig.hand_init_perturb`（只影响 eval env 构造）
- `train_bc.py` — `eval_domain_shift`；训练中 eval 跟随它，训练结束**总是**跑
  source + target 双 eval 并打印 gap
- `evaluate/eval.py` — CLI：`--shift_mode` / `--shift_from_cfg` /
  `--shift_{friction,mass,solref,solimp,gripper_kp,ring_size,ring_friction,needle_size}_scale` /
  `--shift_{pod_holder_size,pod_holder_friction,lid_damping}_scale`（Coffee）/
  `--shift_peg_{size,friction}_scale`（Square）/ `--shift_hole_offset` /
  `--hand_init_perturb[_joint_sigma]` / `--strict_success` 系列 / `--planner_set`

### Design notes
- **训练中 eval 是单模式的**：跟随 `eval_domain_shift.mode`。设成 `"fixed"` 时
  TopkSaver 的 topk 排名反映的是 **target** 分数而非 source。想要"按 source 存 ckpt
  + 结束时看 target"，就把 YAML 里 mode 留成 `"none"`，训练后用 CLI 单独评 target。
- BC 的训练完全不碰 env，所以 `dataset.hand_init_perturb` 与 `eval_hand_init_perturb`
  都只是"eval env 怎么建"的旋钮；拆成两个是为了让 dataset cfg 带一个
  "demo-collection 风格"的默认值，而 eval 时可以覆盖。

### Checkpoint 落盘（`TopkSaver` → `cfg.save_dir/`）
- `latest.pt` 每个 eval 点覆盖；`model0.pt`/`model1.pt` = 训练中 eval 分数 top-2；
  `cfg.yaml` = pyrallis dump 的完整 config（`evaluate/eval.py` 从同目录读它）
- ⚠️ **`model0.pt` 会被后续更好的 ckpt 覆盖** ⇒ 见铁律 14。
- ⚠️ **`model0.pt` = 单次 50/100 局 in-training eval 的最高分**，选择本身有噪声，
  系统性乐观 0.06–0.19（§33）⇒ headline 一律用多点均值或独立大样本重评。

### How
```bash
python evaluate/eval.py --weight exps/.../model0.pt --mode rl --num_game 500 \
    --shift_from_cfg 1 --strict_success 1 --hand_init_perturb_joint_sigma 0.1
```
⚠️ 只传 `--shift_mode fixed` 而不给 `--shift_from_cfg 1` 会构造一个**部分 shift**
（未设的 scale 全 = 1.0），几乎从来不是你要的 target。

---

## 5b. MimicGen-aligned BC baselines (RNN-GMM + Diffusion)

- **`StateRnnGmmPolicy`**（`bc/rnn_gmm_policy.py`）— MimicGen 的官方 baseline：
  **open-loop** LSTM 展开 Tp 步 + 每步 K=5 GMM 头。open-loop（LSTM 每步吃同一个
  编码后的 obs）是为了塞进本仓库无状态的 chunked eval；这是 robomimic 自己支持的
  `rnn.open_loop=True` 模式。**用途：Square 的 IBRL prior**（§32：BC 63%→83% 让
  纯 IBRL 从 89.2% 抬到 97.8%）。⚠️ 但这个杠杆**任务相关**：Coffee 上 MLP 94.4% 与
  RNN 92.4% 无显著差别（§40）—— 先量 prior 的绝对水平再决定要不要换。
- **`StateDiffusionPolicy`**（`bc/diffusion_policy.py`）— DDIM，backbone 可切
  `"unet1d"`（默认，~5M）/ `"transformer"`（`bc/diffusion_transformer.py`，
  real-stanford/diffusion_policy 的 lowdim 配置，n_layer=8/n_head=4/n_emb=256）。
- 三个头共享 `loss/act/act_chunk/update_ema/apply_ema/restore_from_ema` 接口；
  `train_bc.py` 用 `use_flow`/`use_rnn`/`use_diffusion` 三选一（assert ≤1）。
- 训练循环：`eval_period`（epoch 数，跳过时只刷新 `latest.pt` 不动 topk）、
  `val_period` + `dataset.val_ratio`（episode 级切分，固定 seed）。

---

## 5c. Strict success gating (anti reward-hack, Threading_D0 only)

### What
Threading 的原生成功判据是**瞬时、无保护**的 3D 位置测试
（`‖geom_xpos["needle_obj_needle"] − mean(ring geoms)‖ < 12 mm`），策略有两条作弊路：
① 单步穿越（20 Hz 下 0.5 m/s 一步走 25 mm ≈ 球直径）；
② **推环**（三脚架是 18 g 自由体，成功判的是相对距离）。
`StrictSuccessConfig` 在 wrapper 层加门（不改 vendored mimicgen）：只有原生判据通过
**且**门通过的帧才算数，且要**连续 `dwell_steps` 帧**。启用后**同时驱动 success 标志
和每步 reward**，所以 RL 训的是诚实目标。

| 门 | 字段 | 参考值 | 杀掉什么 |
|---|---|---:|---|
| dwell | `dwell_steps` | **3**（@20Hz = 0.15 s） | 瞬时穿越 |
| 环未被推动 | `max_tripod_disp` | **0.04 m** | 推环 / 推出屏幕 |
| 针速度 | `max_needle_speed` | 0.0 = 关 | 高动量硬捅（可选） |

### Where / How
`env/strict_success.py`（`SuccessValidator`，持有连续计数 + reset 后的环心快照）；
`env/robosuite_wrapper.py` 在 domain shift 的 `sim.forward()` **之后**快照环心；
`train_rl.py` / `train_bc.py` 有 `strict_success` 字段（**train 和 eval 用同一个**，
否则 reward 与 metric 不一致）；`evaluate/eval.py`：`--strict_success` /
`--strict_dwell_steps` / `--strict_max_tripod_disp` / `--strict_max_needle_speed`。
**默认 OFF ⇒ 旧 config / 旧 ckpt 的分数不变。**
⚠️ `max_tripod_disp=0.04` 是刻意放宽的（正常接触会轻推），但**调得太高会重新放进
推环作弊**（作弊只需要把环推到针那么远）⇒ 不要超过工作空间尺度太多。
⚠️ 启用后 reward 更稀疏（dwell 把 +1 推迟 K 步）——对 RL 这改变了 MDP，对 BC 只改 eval 计分。
```yaml
strict_success: {enabled: true, dwell_steps: 3, max_tripod_disp: 0.04, max_needle_speed: 0.0}
```
⚠️ **Square / Coffee 不需要**：Square 的栓固定；Coffee 的 machine 有 6.32 kg
（vs Threading 的 18 g）推不动，且 pod 必须落进实体杯壁围出的体积（§40）。
Threading 之外的 task 不解析 binding（`mode=none` + strict 关 ⇒ 不会 raise）。

---

## 6. Quick configuration reference

**Threading_D0（target = §12 的 dynamics-only 配方）**

| YAML | Algo | Train env | Eval env |
|---|---|---|---|
| `threading_d0_state_flow_bc.yaml` | Flow BC | n/a | source + target |
| `threading_d0_state_bc_rnn.yaml` | BC-RNN+GMM (open-loop) | n/a | source + target |
| `threading_d0_state_diffusion[_transformer]_bc.yaml` | Diffusion (DDIM) | n/a | source + target |
| `threading_d0_state_naive.yaml` / `_dr.yaml` / `_dr_oracle.yaml` | RLPD | source / DR / DR-oracle | target |
| `threading_d0_state_ibrl_shifted.yaml` / `_newtarget.yaml` | IBRL | target | target |
| `threading_d0_state_simdist.yaml` | **SimDist**（latent WM + MPPI） | source | target |
| `threading_d0_state_simdist_rawstate.yaml` | SimDist（raw-state 对照，§30） | source | target |
| `threading_d0_state_residual_simdist.yaml` | residual on frozen adapted base（§25） | target | target |
| `threading_d0_state_ibrl_simdist.yaml` | IBRL with the adapted base as prior（§25） | target | target |
| `threading_d0_state_mb_ibrl.yaml`* | MB-IBRL（§23，已归档） | source | source |
| `threading_d0_state_tdmpc2_{source,target}.yaml` | 在线 TD-MPC2 + IBRL prior | source/target | 同 |

**Square_D0（§32/§41；rotw1 = nut z_rot 窄化到 120–180°）**

| YAML | 说明 |
|---|---|
| `square_d0_state_source_{mlp,rnn}_bc.yaml` | prior（63% / 83%） |
| `square_d0_state_ibrl_source.yaml` | 纯 IBRL source（600k → 89.2%；RNN prior → 97.8%） |
| `square_d0_state_ibrl_source_rotw{0,1,2}[_sig01][_snap].yaml` | rotw 窗口对照 + snapshot |
| `square_d0_state_ibrl_target_B_rotw1[_sig01].yaml` | target-B 专家（0.98–1.00） |
| `square_d0_state_simdist_rotw1[_dynens].yaml` | SimDist stage-1（plain / 5 头） |

**Coffee_D0（§40）**

| YAML | 说明 |
|---|---|
| `coffee_d0_state_source_{mlp,rnn}_bc.yaml` | prior（94.4% / 92.4%） |
| `coffee_d0_state_ibrl_source[_sig01].yaml` | 纯 IBRL source（600k → **99.8%**） |
| `coffee_d0_state_ibrl_target.yaml` | target C75kp25（零样本 24%） |

\* 归档分支的 YAML 见 `_archive/`（不随仓库发布），说明见 `CLAUDE_ARCHIVE.md`。

---

## 7. 已知坑 / sharp edges

### 7z. SLURM 长任务编排：自分块链 / 预留窗口 / 监控（2026-09-09，跑 r18 梯子踩出来的）

**① 自分块链没有"最后一个 job id"⇒ 后续任务不能用 `--dependency=afterany:<stage1>`。**
chunk 1 不是最后一块，而最后一块的 id 要等 chunk N-1 跑起来才存在。
`scripts/slurm/pixel_stage1.slrm` 现在有 `THEN_SBATCH` 钩子：**默认空 ⇒ 完全惰性**
（在跑的旧链不受影响），由 marker 文件 `<save>/.then_submitted` 保证**只触发一次**，
两条完成路径（后继的 early-exit / python 之后的 COMPLETE 分支）都会调。
必须加进 `--export` 才能沿链传递。实测两条都跑过：一条 `[then] submitted OK`，
另一条 `[then] hook already fired ... not resubmitting`。
⚠️ `--export` 的值**带空格是安全的**（SLURM 只按逗号切）—— 已用一个 2 分钟探针作业验证：
`THEN_SBATCH=[bash scripts/...sh]` 完整到达，且其后的 `MAX_CHUNK=6` 也没被截断。

**② `--export=ALL` 只带环境变量，从不带 sbatch 的资源/账号 flag。**
所以链上每一跳都要显式转发 `--mem/--time/--cpus/--account`（少了 `--mem` 曾让后继
退回 header 的 72G 并 OOM）。同理，**钩子调起的 adapt/eval 也要转发 `--account`**，
否则 `adapt.slrm`/`eval_simdist.slrm` 里硬编码的 `#SBATCH --account=aip-gigor`
会让加速悄悄漏回原账号。

**③ 可用 steps 不等于 tqdm 的 steps。** 进度只在 `log_per_step` 边界落盘，
walltime kill 会丢掉最后一次保存之后的一切。实测 tqdm 50,249 而 `resume_state.pt`
只有 **50,000**。⇒ 用**可用速率**排预算（224+r18+H=3 是 ~50k usable / 11:59 chunk）。
⇒ 想用维护前的空窗补一块，要让它**明显越过**一个边界并留出 ~169MB 落盘时间，
**别刚好压在边界上**：我按"raw 恰好到 120k"排，结果 120k 那次保存没写完，只落了 110k
（预期 +20k，实收 +10k）。

**④ 维护预留前的空窗怎么安全用**：一次性 chunk，且三件事齐全 ——
`CHUNK == MAX_CHUNK`（走 backstop ⇒ **不排后继**，短 walltime 不会污染真链）、
**不传 `THEN_SBATCH`**（不会误触发钩子）、
`--deadline=<预留开始前>`（**排不进就让 SLURM 取消它**，否则它会在维护后启动，
和真链的 chunk 同时写一个 `resume_state.pt`）。
⚠️ 那个 chunk 的 log 仍会**无条件**打印 "propagating to successor: ..."，
真正算数的是下面那行 `no successor queued (backstop)`。

**⑤ 维护会把 slurmctld 也带走，此时任何"不在 squeue 里就是没了"的判断都是错的。**
`squeue` 直接报 `Unable to contact slurm controller`。一个旧监控据此发出了假的
`[LEFT QUEUE]`。⇒ 控制器不可达时结论是"**未知**"，不是"没了"；用 `sacct` 复核。
⇒ 反过来，**只报状态变化的监控在队列被清空时会完全沉默**，和"还在排队"无法区分
（silence ≠ success）。要么像 `watch.sh` 那样对**每个见过的 id 都查 sacct**
（CANCELLED 会作为新状态触发），要么单独跑一个阻塞到控制器回来的**正向存活检查**。
⚠️ 预留的 `EndTime` 可能是占位符（我们那次写 `Duration=365-00:00:00`，实际停机 ~1 小时）
⇒ **两个方向都不要照它排期**。

**⑥ pixel 跑起来每个 eval 点都会吐两个无害的 teardown traceback**，别当故障：
mp eval worker 在 NFS 上清临时目录的 `OSError: [Errno 16] ... '.nfsXXXX'`，
以及 `MjRenderContext.__del__` 里的 `EGLError(EGL_NOT_INITIALIZED)`（前缀
`Exception ignored in:`）。⇒ 崩溃过滤器要匹配**异常行**并减去这两类，
不要匹配 `Traceback (most recent`（否则每块日志刷十几条，很快就被训练成无视告警）。


**★ 分块续跑会清空 `train.log`（2026-09-02 修复）。** `common_utils/py/logger.py:6,14`
的 `Logger` 默认 `mode="w"`，而 `train_simdist.py:216` 没传 mode ⇒ **每个 resume chunk
都把 train.log 截断重写**，之前所有 `score/source_score` 和 `distill/*` 历史全部丢失。
plain pixel arm 的 0–500k 学习曲线就是这样没的，而且这些指标**不进 stdout**
（`logs/*stage1*.out` 里 grep 不到），所以无处可恢复。已改为
`mode=("a" if cfg.resume_from else "w")`。⚠️ `train_rl.py:373` / `train_simdist_adapt.py:226`
没有 resume 路径，`"w"` 是对的，不要一起改。



- ⚠️ **「render 是单步主要开销」已被实测推翻**（2026-08，L40S，`scripts/bench_render.py`）。
  `env/robosuite_wrapper.py:161-165` 说 224×224 render 是 "the dominant single-step
  cost"（也是加 `state_only_no_render` 的理由）。实测：**无渲染 11.82 ms/step，双相机
  96×96 只 +2.23 ms（1.19×）** —— 物理才是瓶颈（control_freq=20 ⇒ 每控制步 ~25 个 sim
  substep）。⇒ **pixel 模式的 env stepping 只贵 ~19%，不是 2-3×**。
  ⚠️ 但这只适用于 **stepping**：带渲染的**数据生成**实测慢 **2.9×/worker**（每 worker 自己的
  EGL context + 27 KB/step 搬运 + hdf5 写）。两者别混用。
- ✅ **mp eval 带渲染是可用的**：10 worker 聚合吞吐 **9.71×**（理想 10×），无 EGL 争用。
  `state_only_no_render` 是 state 模式的提速开关，不是 pixel 的阻塞项。

- **`pip` 的 shebang 是坏的**（指向不存在的 `/home/Chenyang/anaconda3/envs/ibrl/bin/python3.10`）
  ⇒ 一律用 `python -m pip install ...`。
- **`env/metaworld_wrapper.py` 导入即失败** —— 故意跳过了 mujoco-py/metaworld（§1）。
  franka / polymetis 等真机代码同理，已归档到 `_archive/`。
- **domain shift 的 binding 是 per-task 的**：`resolve_bindings` 只认 Threading /
  Square / Coffee（`CoffeePreparation_D0` 被显式拒绝——它是 Coffee 子类但判据不同，
  套用会静默给出错误 target）。其余 task raise `NotImplementedError`。
- **`set_env.sh` 第 4 行** `conda activate ibrl` 依赖 `~/.bashrc` 里的 conda init，
  否则先 `source /home/Chenyang/anaconda3/etc/profile.d/conda.sh`。
- **`common_utils/rela/episode.cc:90` 的 `bootstrap` 只看位置不看 terminal**
  （episode 最后 `nstep` 步一律 bootstrap=0）⇒ 400 步超时失败被当成吸收终止。
  所有既有 IBRL/residual run 都带这个偏置，为可比性没有改；`verify_iql_critic.py::build_flat`
  是对的（`done = t==T-1 and success`），所以 IQL 预训与 IBRL replay 在这点上一直不一致。
- **恢复 `sim_state` 后继续 step 有 confound**：`sim_state` 不含 `data.ctrl`，也不含 OSC
  控制器 Python 侧的 goal ⇒ 三个 env 会把第一个动作解释在不同参考系下。§27 那条
  "`state_only_no_render` 改变物理"的旧结论已被证伪（同 seed reset + 同动作 40 步逐位
  相同），根因就是这个 confound；§41 的 ProbePool episode 级不可复现也是它。
- **`AdaptWorkspace.__init__` 会劫持 stdout**（`sys.stdout = common_utils.Logger(...)`）
  ⇒ 复用它的诊断脚本要 `print(..., file=sys.stderr)`，或去 `<save_dir>/train.log` 里捞。
  跑 `eval_oracle_dyn_mppi.py` **不能 `2>/dev/null`**（结果全在 stderr）。
- **`pkill -f <pat>` 会匹配到自己所在的 shell**，把父 shell 连子进程一起杀。
- EGL 报错出现在解释器退出时是**良性**的 MuJoCo context 清理（所有保存都已完成）；
  几个 trainer 用 `os._exit(0)` 就是为了绕开它。

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

## 13. ★ 三个任务的 target 权威定义（唯一真源 = SimDist stage-1 的 cfg）

**规则：target 就是 SimDist adapt 实际跑的那个 env，别处一律以此为准。**
`train_simdist_adapt.py:398` 里 `self.eval_env = self.ws.eval_env`，adapt/dyna 臂的
env 完全来自 `stage1_ckpt` 同目录的 `cfg.yaml` 的 `eval_domain_shift` +
`eval_init_dist`。任何 model-free 臂（`train_rl.py`）要和它们比，就必须复制这份定义。

### 权威 stage-1（2026-09-06 核对：按已完成 run 的 `cfg.yaml:stage1_ckpt` 反查）

| 任务 | stage-1（dynens / plain） | 已完成 run 数 |
|---|---|---:|
| Threading_D0 | `threading_simdist_dynens_diversev2_h5_seed1` / `threading_simdist_diversev2_h5_seed1` | 102 |
| Square_D0 (rotw1) | `square_simdist_rotw1_dynens_h5_seed1` / `square_simdist_rotw1_h5_seed1` | 全部 |
| **Coffee_D0 (G5x)** | **`coffee_s1g5x_dynens` / `coffee_s1g5x_plain`** | 31 |

### 三份 shift（`mode: fixed`，σ=0.1 train+eval，`episode_length` 400）

**Threading_D0** — `friction_scale 0.5 · gripper_kp_scale 0.12 · mass_scale 3.0 ·
ring_friction_scale 2.0 · ring_size_scale 0.85`；无 init_dist；`strict_success.enabled = True`

**Square_D0 (rotw1)** — `gripper_kp_scale 0.3 · peg_friction_scale 2.0 · peg_size_scale 1.2`；
init_dist **enabled**，`nut_z_rot = [2.0944, 3.1416] rad = 120–180°`（这就是 "rotw1"），
`needle_x [-0.2,0.05] · needle_y [0.15,0.25] · needle_z_rot [-3.6652,0.5236] ·
tripod_x [-0.1,0.15] · tripod_y [-0.2,-0.1] · tripod_z_rot [0.5236,2.6180]`；
`strict_success.enabled = False`（tasks §41b：Square 线上 strict 三边全 false）

**Coffee_D0 (G5x)** — `friction_scale 0.2 · gripper_kp_scale 0.3 · lid_damping_scale 5.0 ·
mass_scale 8.0 · pod_holder_friction_scale 5.0`；init_dist `enabled: False`；
`strict_success.enabled = False`

### ⚠️ Coffee 有两套 stage-1，名字不提示哪套是 G5x

`coffee_simdist_diversev1_h5_seed1` / `coffee_simdist_plain_h5_seed1` 是 **C50 时代**的，
带的是另一个 target：`friction 0.5 / lid_damping 50 / mass 3 / pod_holder_friction 2`。
`cluster/job_sqcof_dynaibrl.sh` 一度指向它们，会让所有 Coffee 臂跑在 C50 物理上，
和 `exps/coffee_g5x_3arm/` 里 31 个既有 run、`coffee_d0_state_baseline_target_g5x.yaml`、
`coffee_ibrl_target_G5x_seed1` 全部不可比。**判断 stage-1 属于哪个 target，
只能回溯已完成 run 的 `stage1_ckpt`，不能看目录名**（`diversev1` 和 `g5x` 是两套命名）。

### Coffee 的 `coffee_s1g5x_*` = 同一份权重 + G5x 的 cfg 包装

`coffee_s1g5x_dynens/latest.pt` 是指向 `coffee_simdist_diversev1_h5_seed1/latest.pt`
的**符号链接**（plain 同理）。stage-1 在 **source** 域蒸馏，权重与 target 无关；
两个目录的差别只在 `cfg.yaml` 的 `eval_domain_shift`。而 adapt 脚本正是从这份 cfg
读 target ⇒ **选哪个包装目录就是在选 target**。同步到 cluster 时要用 `--copy-links`
把链接实体化。

### model-free 臂的预算口径（与 dyna 臂对齐）

`num_warm_up_episode = 30`（三个任务统一；模板里 Square/Coffee 原为 60，按各自
episode 长度换算是 24k/15k/5k transitions 的差距，Square 那个占 100k 预算的 24%）。
`count_warmup_transitions = 1` 把 warm-up 实际走的 transition 计入 `global_step`，
所以 `num_train_step=100000` 与 dyna 臂的 `max_transitions=100000` 是同一个量。
**preload 的 20 条 demo 两边都不计**。warm-up 用 BC prior 确定性 rollout，
所以 critic warm-start 不影响这批数据。
⚠️ `job_{coffee,square}_baselines.sh` 的 `ibrl` 臂仍是 warm-up 10 且无
`count_warmup_transitions`，跨批比较时要记得。

### 怎么给 model-free 臂复制这份定义

`python3 scripts/make_ibrl_m001_cfg.py` 从上表的 stage-1 抽 env、从各任务的
target-域 IBRL run 抽其余字段，产出
`release/cfgs/mimicgen/{threading,square_rotw1,coffee_g5x}_ibrl_m001.yaml`。
它会打印每份 cfg 的 env 来源，以及 stage-1 与 target 专家 cfg 是否一致
（三个任务现在都 `agree`）。σ 在生成器里钉死 0.1，**不从 stage-1 继承** ——
Threading 的 stage-1 里 σ 是 `None`，继承会做出一个跑在 robosuite 默认初始噪声上的臂。

### `train_rl.py` 的 dataclass 默认全是坑

不传 `--config_path` 就会静默拿到：`domain_shift.mode = "none"`（**source 物理**）、
`q_agent.act_method = "rl"`（**prior 不进 argmax，`prior_margin` 变死代码**）、
critic `10×512` 无 layer-norm / actor `512` dropout 0.5（**全仓库其他 IBRL run 都是
critic 5×1024 + ln、actor 1024 dropout 0**）。三者都不报错。
现在有两道防线：`train_rl.py` 在 `bc_policy` 非空且 `act_method=="rl"` 时抛错；
`load_pretrained_agent` 形状不匹配且 PE 关闭时抛错并指出该对齐哪些旋钮。
