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

## 14. sim2real cell（Threading_RealCell）↔ real2sim arena 对齐（2026-09-18）

**同一个真机 DROID cell，两条线必须共享同一个 arena。** 唯一真源 = `env/peg_insertion.py`
的 `MEASURED_*` 常量（real2sim.md §2e）。`threading_real_cell.yaml` 里写 `arena_from_real2sim: 1`
⇒ `load_sim2real()` 从那里取 `table_above_base`(+0.0207) / `table_full_size`(0.739×1.240×0.03，
碰撞+视觉) / `base_xy`(−0.489,−0.034 相对桌心)；yaml 里再写这三个 key 会被拒。

两条路径都接上了（都读同一 yaml）：
- RL：`Threading_D0` + `sim2real_cell` → wrapper 的 `apply_cell_arena`（包 `_load_model`：先设桌尺寸，
  后覆盖 base xy —— Threading 在内部按 `table_full_size[0]` 推 base x，只改桌子会让机器人倒退）。
- demo 生成 / BC eval：注册类 `Threading_RealCell`（hdf5 里存的 env_name）。它把
  `_s2r_offset_applied/_s2r_arena_applied` 置真，所以再叠 `sim2real_cell` 不会二次偏移。

`workspace_offset` 从 (0, 0.10) 改成 **(0.071, 0.066)** = 旧值 + base 位移 ⇒ 物体**相对机器人**的
位置不变（旧的可达/遮挡检查都是 base 相对的，仍有效）。
相机：zed2i 换成 §2d-x 独立复解（`ground_run2_solved.json`，6-DOF，roll 占位消失），
内参 HD2K 原生 fy=1057.6@1242 ⇒ fovy 60.84°。⚠️ real2sim 的 `render_camera_pov.py` 仍读 §2d 的
`ground_transforms.json`（差 ~30 mm），两边要约定一个解。腕相机 / 桌面倾角 1° 待定。

Gate：`scripts/test_sim2real.py` §I 逐项比对三条路径与 `PegInsertion_D0`（桌面相对 base 高度、
桌心相对 base、碰撞半尺寸、base z）+ 物体相对 base 不变。⚠️ 比对要关 `dr_base_height`（±5 mm）。
preview：`--no_dr` 以前把乘性 DR 置 0 ⇒ 全黑图，已修。

⚠️ **2026-09-17 生成的全部 RealCell 产物属于旧 arena（0.8×0.8 桌、base (−0.56,0)），已作废**：
`release/data/mimicgen/generated/threading_realcell`、`exps/mimicgen/threading_realcell_{flow_bc,mlp_bc,ibrl}_seed1`。

### 14a. sim ↔ 真机的残余差距（2026-09-18 记录，后续 DR 的候选清单）

§14 只保证 **RealCell ≡ PegInsertion_D0**（gate 1e-6）。两者都继承 real2sim 的建模近似，
与**真机**之间仍有下表差距。state 臂的量级是 mm–cm，而 Threading 容错窗口 ~1.8 mm ⇒ 不可忽略。

| # | 项 | sim 现值 | 真机/另一次解 | 量级 | 处理建议 |
|---|---|---|---|---|---|
| R1 | base yaw 取整 | 90° | 91.2°（§2e） | 1.2°，远角 ~13 mm | 可直接建模（不取整）；或 DR ±1.5° |
| R2 | 桌面倾斜 | 水平 | 相对 base z 倾 0.99°（`cell_layout_base.json`） | 桌面范围内 ≤~10 mm 高差 | 可直接建模；或 DR 倾角 ±1° |
| R3 | 两次接地解的分歧 | base xy (−0.489,−0.034)、桌高 20.7 mm、桌 1.240×0.739（§2e） | 独立复算 (−0.493,−0.028)、21.4 mm、1.260×0.737（§2d-x） | 位置 ~7 mm、高度 0.7 mm、尺寸 20 mm | 先与 real2sim 约定一个解；残差作为 base xy / 桌高 DR 的宽度依据 |
| R4 | 绝对尺度未闭合 | 全链米制来自 ChArUco 的 ZED-depth 融合 | 方格 42.43 mm 未经尺子核对 | 若真值 40 mm ⇒ 全链偏大 **6%** | **量，不要 DR**（尺子量一次方格即可闭合） |
| R5 | zed2i 外参两套解 | §2d-x `ground_run2_solved.json` | real2sim 渲染仍用 §2d `ground_transforms.json` | 30 mm / 1.0° | 约定一个；现有 `dr_cam_pos 0.011` / `dr_cam_rot_deg 1.0` 小于这个分歧 |
| R6 | zed2i 内参 | fy 1057.6（抄自 real2sim.md） | 原始 `intrinsics.json` 不在本机；cx/cy 未核 | — | 拿到文件后核对；`dr_cam_fovy_deg 1.0` 已在 |
| R7 | 腕相机 zedm | sim 默认挂载 | `T_ee_zedm` 已解，但 "ee" 的坐标系定义未对齐 | ~150 mm 挂点差 | 先对齐 EE 定义（flange / O_T_EE vs `gripper0_grip_site`），再决定 DR |
| R8 | 视觉外观 | 桌面平涂、放大 5 m；默认 robosuite mount 外形 | 垫子外是光学面包板；底座外形不同 | 仅像素 | 已有 `dr_background_gray` / 桌色 DR；DR 版桌面亮度偶尔接近全白（`figs/sim2real/agentview.png` 第 2 格），可收窄 |
| R10 | reset 关节姿态 | Franka 标准 ready `[0,−π/4,0,−3π/4,0,π/2,π/4]`（`init_qpos`，2026-09-18） | 真机实际 home 未核对 | — | 拿到真机 home 后替换；σ=0.1 的关节噪声以它为中心 |
| R9 | 桌厚 0.03 m | 取自 real2sim | 未测 | 物理无关（只有顶面起作用） | 不处理 |

原则（沿用 `env/sim2real.py` 开头的说明）：**能测的先测准，只对测不准或会漂的量做 DR**；
DR 的宽度取自上表的实测分歧，不要拍脑袋定。R4 属于"能测"，不该用 DR 去吸收
——在 ~1.8 mm 的窗口下，宽到能盖住 6% 尺度误差的 DR 也会把信号淹掉。
现有已接好的几何 DR 只有 `dr_base_height 0.005`（±5 mm）；R1/R2/R3 需要新的旋钮
（base yaw、桌面倾斜、base xy），`env/sim2real.py` 里还没有。

### 14b. reset 姿态改为 Franka ready（2026-09-18）

robosuite 默认 `init_qpos` 是为"桌面在 base 下 112 mm"设计的；在实测 cell（桌面在 base **上** 20.7 mm）
上 grip site 离桌面只剩 **78 mm**，σ=0.1 下 **12/100 次 reset 手指已触桌、最低穿进桌面 44 mm**。
改为 Franka ready 后（同一新 arena，100 次 reset，σ=0.1）：

| | 默认 | ready |
|---|---:|---:|
| grip 离桌面（均值 / 最低） | 78 / **−44** mm | **466 / 343** mm |
| reset 时触桌 | 12/100 | **0/100** |
| 最小关节限位余量（中位 / 最小） | 0.43 / 0.19 rad | **0.69 / 0.45** rad |
| Jacobian 条件数（中位 / 最大） | 12.7 / 39.9 | **8.2 / 8.8** |
| 夹爪朝向 | 偏离竖直 7.3° | 正竖直向下（yaw 相同） |
| P 控制器开到 needle / tripod 上方 10 cm 的步数（中位） | 26 / 16 | 31 / 32 |

⚠️ **实现坑**：`robosuite` 的 `_load_robots()` 每次 hard reset 都从 `robot_configs` **重新实例化** Robot
⇒ 写 `env.robots[0].init_qpos` 只活到下一次 reset（第一版就这么静默失效，统计与默认逐位相同）。
正确入口是 `env.robot_configs[0]["initial_qpos"]`（`apply_init_qpos` / `Threading_RealCell` 都走这里）；
gate §J 检查**第 2 次** reset 的 qpos 以及 σ 噪声以它为中心。

MimicGen 生成成功率 A/B（同一新 arena、同 seed 1000、各 200 次尝试）：**ready 53.0% vs 默认 52.5%**，
平均长度 218 vs 218 步（逐条相同：MimicGen 从起点到第一段的过渡是**固定步数**插值，与距离无关）。
已核对 ready 组生成的 demo 首帧 qpos 确为 ready（不是没生效）。
对照旧记录：旧 arena + 默认姿态是 60%（n=50，SE ±7 点）⇒ 与新 arena 的 ~53% 差异不显著，
arena 改动对生成成功率的影响测不出来。

⚠️ **`scripts/check_demos_strict.py` 曾手工挑字段**构造几何（只抄 table_above_base + workspace_offset）
⇒ 新增 `table_full_size/base_xy/init_qpos` 后，它在旧 arena 上回放新 arena 的 demo，审计给出
**0/100（lenient 也 0）**。已改为「整个 cell + 所有 dr_* 复位」。修后新数据：lenient 100/100、
STRICT 85/100、均长 215（旧 arena 99/83/216）。**任何脚本都别再手工复制 cell 字段。**

### 14c. base 位姿 DR（R1–R3）+ stage-1 数据采集（2026-09-19）

**新旋钮**（`env/sim2real.py`，cell yaml 已开）：`dr_base_yaw_deg 1.5` / `dr_base_tilt_deg 1.0`（绕世界 x、y）/
`dr_base_xy 0.007`，与既有 `dr_base_height 0.005` 一起作为**机器人 base 位姿**的扰动 —— 桌子始终水平不动
（物理上起作用的是二者相对位姿；倾斜桌子会让已放好的物体悬空/穿透）。`any_base_dr()` 触发 OSC 重新锁定。
抽样顺序先 height 后其他 ⇒ 只开 height 的旧配置逐 seed 复现。gate §K：范围内且真在变、桌子不动、reset 不触桌、
零动作 5 步漂移 < 3 mm。⚠️ **IBRL 专家（09-18）训练时没有 R1–R3**，只有 height。

**`generate_simdist_data.py` 以前完全不读 `sim2real_cell`** ⇒ 对 RealCell 专家采集会**静默落在原版 arena**
（0.8 m 桌、base (−0.56,0)、robosuite reset 姿态）。已修：`build_source_env` 从 expert cfg 读 cell（采集时读
**当前** yaml，含当前 DR），`env_cfg.json` 记录 `sim2real_cell.{path, sha256, resolved}`。
冒烟验证：首帧 eef z 均值 1.402（= 新 arena + ready；旧 arena ≈1.01）。

专家独立重评（`indep_eval/`，n=300，strict，σ=0.1）：训练 cell（只有 height DR）**0.957**。

采集：`scripts/collect_threading_realcell_batches.sh`（TPA 批处理脚本的通用版），8 批 × 25k = **20 万条**，
两条链分跑 GPU 2/3，落 `/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_source_diverse_v1/part{b}.hdf5`；
开跑前快照 cell yaml + 专家 sha + git diff，每批后比对 yaml，变了就停。

### 14d. ★ RealCell 的全部随机化（汇总，2026-09-19；stage-1 数据 `threading_realcell_source_diverse_v1` 的口径）

每个 episode 在 reset 时独立抽一次，均为均匀分布（关节噪声除外）。唯一真源 = `release/cfgs/sim2real/threading_real_cell.yaml`
+ 采集命令行（`scripts/collect_threading_realcell_batches.sh`）；每批的 `env_cfg.json` 记录 cell 的 path/sha256/resolved。

**① 初始状态**
| 项 | 设置 | 来源 |
|---|---|---|
| 机器人关节 | 以 Franka ready `[0,−π/4,0,−3π/4,0,π/2,π/4]` 为中心，逐关节 **高斯 σ=0.1 rad** | `init_qpos` + `hand_init_perturb_joint_sigma`（§14b） |
| 物体 | Threading D0 原生采样框（needle 位置框+旋转，tripod 固定），整体平移 `workspace_offset (0.071, 0.066)` | robosuite / §14 |

**② 几何 DR —— 扰动机器人 base 位姿，桌子保持水平不动**（§14c；宽度取自 §14a 的实测分歧）
| 旋钮 | 范围 | 依据 |
|---|---|---|
| `dr_base_height` | ±5 mm | 安装公差 |
| `dr_base_xy` | x、y 各 ±7 mm | R3：两次接地解相差 ~7 mm |
| `dr_base_yaw_deg` | ±1.5° | R1：真机 yaw 91.2° 取整为 90° |
| `dr_base_tilt_deg` | 绕 x、y 各 ±1° | R2：桌面相对 base 倾 0.99° |
扰动后 OSC 重新锁定目标（`any_base_dr()`），gate §K 验证零动作不漂移。

**③ 采集时的策略与动作扰动**（`generate_simdist_data.py`，与 Threading v2 相同）
- 策略混合：每条轨迹 50% 专家（`model0.pt`），否则从 16 个 IBRL snapshot（25k–400k）中抽，全部为 IBRL hybrid（`HYBRID=1 SNAP_HYBRID=1`）。
- 动作噪声：加噪段（1–5 步）与不加噪段（5–10 步）交替；每条轨迹重抽 σ ~ U[0.05, 0.2]。
- `max_len 400`（= episode_length）。

**④ 视觉 DR —— 已接好，对 state 数据无作用**（渲染 pixel 时自动生效）
桌面色调 ±0.02 / 亮度 ×[0.85,1.10]；物体色调 ±0.025 / 亮度 ×[0.85,1.15]；背景灰度 [0.25,0.85]；
灯光漫反射 ×[0.70,1.25]、环境光 [0.02,0.20]、高光 ×[0.40,1.50]、方向 ±0.30、位置 ±0.40 m；
相机位置 ±11 mm、旋转 ±1°、fovy ±1°。

**刻意不随机化**
- 物理参数（摩擦、质量、solref/solimp、阻尼）：`domain_shift.mode = none`（选定的 DR 范围里没有这一项）。
- 桌子尺寸 / 桌高：实测值精确对齐。
- 绝对尺度（R4）：应该去量，不用 DR 吸收。

**专家对 DR 的覆盖**：IBRL 专家（2026-09-18）训练时只见过 ② 里的高度 DR，R1–R3 是采集时才加的。
独立重评（n=300，strict，σ=0.1）：只开高度 **0.957** → 全开 **0.947**，差异在噪声内。
采集混合数据的成功率 ≈ 73.5%（专家 + snapshot + 动作噪声，不是专家水平）。

### 14e. ★ PIXEL stage-1 数据链：重放契约 + 校准腕相机（2026-09-19/21）

**数据集**：`/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_pixel_v1/`
- `part{1..8}.hdf5` = 采集产物，10 万条（8×12500），**只渲染 agentview**（224²，JPEG q90），
  成功率 73.0–74.2%，每条约 1.14 MB。
- `2cam_calibrated_v1/part*_2cam.hdf5` = 补渲染腕相机后的 **36000 条**（4500/shard，
  = `seq_replay_max_transitions` 6.5M 的上限，多了训练也看不到），653 万 transition。
- 同目录另存 `cell_snapshot.yaml` + `expert.sha256` + `git_head.txt` + `git_diff.patch`。

**★ 重放契约（rollout 与渲染解耦）** —— 这是本阶段最重要的基础设施。
`--store_sim_state 1` 让每条轨迹带上：每步 `sim_state`（qpos+qvel，float64，180 B/步）
和每局 `model_state`（1350×float64 ≈ 10.8 KB，= 该局 DR **实际写进模型的值**：base 位姿 /
灯光 / 颜色 / 相机）。渲染是 `(模型字段, qpos)` 的纯函数 ⇒ **任何相机都能事后重渲染，且与
已有帧逐索引对齐**（`scripts/rerender_simdist_data.py`）。开销只占图像字节的 1.9%。
- **存结果不存种子**：种子要求抽样顺序永不变，而顺序已经改过一次。
- gate `scripts/test_rerender.py` 两半都验：重渲染像素与 JPEG 字节**逐位相同**；且
  **跳过 `model_state` 还原会渲出不同画面**（证明还原是有用的，不是摆设）。
- 价值兑现：腕相机外参未定时先只采 agentview，定了之后补渲染 —— **rollout 一条没浪费**。

⚠️ **三个坑（都已修，别再踩）**
1. **`generate_simdist_data.py` 原本完全不读 `sim2real_cell`** ⇒ 给 RealCell 专家采集会
   **静默落在原版 arena**。同理 **`train_simdist.py` 到 2026-09-20 才有 `sim2real_cell`**。
2. **`mj_step` 是先 forward 再积分** ⇒ 一步结束时 `data.qpos` 已更新、渲染读的 `data.xpos`
   还停在积分前。robosuite 的相机 sensor 直接渲染，于是**图像比 qpos 落后一个物理子步**，
   表现为每条边缘一圈 1 px 轮廓（mean |diff| 1.2/255），任何 `(qpos, forward)` 都复现不出来。
   修法 = 在 sensor 渲染前补一次 `forward()`（`attach_render_state_probe`），零额外渲染开销。
   ⚠️ 探针必须**每局 reset 后重挂**：robosuite 每次 hard reset 重建 observables。
3. **`restore_model_state` 会把 `cam_*` 一起还原** ⇒ 用**新**外参重渲染时会被悄悄改回旧相机。
   现役 rerender 在 restore **之后**才写相机（`env/sim2real.py:apply_cameras` 同样语义）。

**校准腕相机**（`2cam_calibrated_v1/calibration.json`，cell = `threading_real_cell_calibrated_wrist.yaml`）
- 采用的解释：`T_ee_zedm` 中的 "ee" = **Franka 法兰**，且 robosuite 的 hand 系需再绕 z 转 **45°**
  才对上（= 早先记的 B@45 假设）。grip 系下 pos (0.0294, 0.0772, −0.0810)、
  quat_cv (0.6718, −0.0927, −0.1266, 0.7239)、**fovy 45.656°**（sim 默认腕相机是 75°）。
- ⚠️ **这个坐标系假设从未用真机照片验证过**，而数据、cell、两个 stage-1 都建立在它上面。
  决定性证据仍是 `scan_run2/pose_*/zedm_left.png` + `render_camera_pov.py`（数据在别的机器上）。
- 一致性已验：该 cell 与采集 cell **非相机字段逐字段相同**；用它渲染 vs 数据集帧
  mean<1 / max<40（JPEG 噪声量级）⇒ 训练期 eval 与训练数据同源。
- ⚠️ 不对称：agentview 帧带每局相机 DR 抖动（采集时抽的），补渲染的腕相机是固定位姿无抖动；
  eval 时两者都抖。

**★★ 腕相机在插入瞬间看不到目标（0.0%）**
120 条成功轨迹的最后 20% 帧，把 ring 几何中心与针尖投影进腕相机：
**校准腕相机 0.0% / 0.0%，sim 默认腕相机（fovy 75）也是 0.0% / 0.0%**（目标落在画面下沿
之外，v≈334–408 vs 画幅 224）。**原版 arena 用 MimicGen source demo 复算同样是 0.0%**
⇒ **不是 RealCell 的退化，这个任务的腕相机一直如此**，published 的 pixel 阶梯也是在这个
条件下训的。合并 agentview 在 224² 下孔只有 3.9×5.9 px ⇒ **两个相机都不解析毫米级对准**，
这是 pixel < state 的一个结构性原因，也意味着**真机上这个相机位提供不了对准信息**（要改
就得物理挪相机，不是改 sim）。

**stage-1 双臂**（2026-09-21 起跑，只差增广，数据/cell/seed/上限全同）
`exps/mimicgen/threading_realcell_pixel_stage1_seed1`（base：仅 shift，GPU0，1.28 it/s ≈ 32.5 h）
vs `..._aug_seed1`（+prop 噪声 2 mm / 色彩抖动 0.3 / 模糊 σ≤2.0，GPU1，1.53 it/s ≈ 27 h）。
eval 改成 **source = 真机 cell 名义物理**（`eval_domain_shift.mode=none`、`eval_source=0`、
σ=0.1 两端一致）⇒ `score/score` 就是 TopkSaver 的选择指标。
⚠️ **与 §47/§48 的 sim2sim 阶梯不可比**（那边全在 dynamics-only target 上）。
⚠️ eval 用 spawn 会**重新 import** 共享代码 ⇒ 跑完前别改 `env/*.py`、`rl/*.py`、`train_simdist.py`。

⚠️ **中间产物 `2cam/`（sim 默认腕相机那批，49 GB）8 个文件全部不可读**（`data` 组打不开，
但文件大小看着正常）—— 已于 2026-09-21 删除。**按大小判断 hdf5 完整性会骗人，要真的打开它。**

### §14f ★ 物体布局翻边 + 灯带 + 物体颜色标定（2026-09-22）

**目标**（owner 定）：ring 放到相机那一侧（robot 左 / +y）、needle 放到右侧，**且手臂要从
ring 的右边插入**，让 agentview 看得见插入过程。ring 只做小位移随机化、不转。

★★ **`layout_tripod_z_rot_deg` 决定手臂从哪一侧插入，是这一块里最要紧的数。**
MimicGen 的插入 subtask 是 **tripod 坐标系**里的刚体变换 ⇒ 从 ring 看过去，**手臂永远站在
方位角 = 这个 yaw 的位置**。相机在 ring 的 **149°** 方向，所以 yaw≈149 = 手臂正好挡在视线上，
yaw≈−30 = 手臂在正对面。

**(a) 可行性**（`panda_joint7` 只有 ±166°）：只搬位置不转 tripod 要求手腕多绕 180° ——
逐帧回放 12 条失败：抓取 **12/12 成功**，插入差 22–136 mm（阈值 12 mm），**37.5% 的时间步
joint7 margin = 0** ⇒ 生成率 5/154 = **3.2%**。

| yaw | −90 | −60 | −45 | −30 | −15 | 0 | +15 | +30 | +45 | +90 | 120 | 135 | 150 | 165 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 生成率 (n=50–60) | 5% | 43% | 38% | **48%** | 45% | 22% | 10% | 10% | 5% | 7% | 24–36% | 40–56% | 40–52% | 47–50% |

两条可行带（≈−60…−15 和 ≈120…165），中间和 ±90 都塌。**needle 的 yaw 无影响**（三档实测）。

**(b) 可见性** —— 8 条成功 demo 的最后 40 帧，224 agentview 上 ring 的 **segmentation 像素数**
（= 策略真正看到的）：

| 布局 | ring px | 全盲帧 | 入射角 |
|---|---:|---:|---:|
| ring 右 / yaw 90（翻边前） | 14.4 | 0% | 53° |
| ring 左 / yaw 135（从相机侧插） | **3.4** | 1% | 52° |
| ring 左 / **yaw −60** | 34.7 | 0% | 57° |
| ring 左 / **yaw −30** | 33.1 | 0% | 50° |
| ring 左 / yaw −15 | 34.4 | 0% | 52° |

入射角四档一样（50–57°）⇒ 差别是**手臂遮挡 + 距离**，不是视角。
⇒ **配方 = `layout_tripod_z_rot_deg: [-30,-30]`**：唯一同时落在可行带里、且手臂正对相机
反面（179°）的值，可见性是翻边前的 **2.4×**、是 yaw 135 的 **10×**。
⚠️ 即便如此，ring 孔在 224 上仍只有 ~5 px —— 这是 §14e 那条角分辨率天花板，没被解决。

**★★★ 三个测量陷阱，全部在这一轮里各骗过我一次**（都已加防线）：
1. **`mj_ray` 会穿过机械臂报「可见」** —— 它只测简化的**碰撞**几何，而画面里挡人的是**视觉**
   网格。它对上表的四个布局一律报 "1.00 可见、零遮挡"。`scripts/preview_sim2real.py` 的
   `visible_fraction` 已换成 segmentation 版 `visible_pixels`（讽刺的是它上一版注释还写着
   「前一版是假阳性机器」，这一版是**假阴性机器**）。
2. **`e.reset()`（裸 robosuite env）跳过 `apply_sim2real`** ⇒ 相机退回 robosuite 默认
   `(0.5, 0, 1.35)`，量的根本不是我们标定的 ZED 位姿。**识别方法：物体变回木头色**
   （cell 的 `object_rgb` 也没应用）。可视化和测量都只能用 **wrapper 的 `env.reset()`**。
3. **reset 之前取的 `m, d` 句柄会失效** —— `hard_reset` 重建 sim，旧句柄冻结在构造时刻，
   由它算出的 `cam_xpos` / 投影 / 入射角全是假的（`sim.render` 用的是 live sim，所以
   **只有像素数那一栏是真的**，这正是当时数字自相矛盾的原因）。
⇒ 现在的测量脚本每次 reset 后重取句柄，并**断言相机位姿等于 cell 的标定值**才肯输出。

**灯带 + 与真机对齐的亮度/阴影**（owner：真机是沿机械臂前后方向的一条灯带，在相机上方）。
MuJoCo 没有线/面光源 ⇒ `light_line_n: 8` 个点光源沿 base **+x** 铺开 `light_line_len: 0.9` m，
各 1/n 亮度。实现上的三个坑：① arena 的 `<light>` **没写 `diffuse`**（MuJoCo 默认 0.7）⇒「只缩放
写出来的属性」= 没缩放 ⇒ 全白。② 默认 `shadowscale=0.6` 的阴影视锥对高处的灯太小 ⇒ 桌面出现
**矩形硬边**；已设 `shadowscale 1.2` + `shadowsize 8192`。③ **`m.nlight` 变了 ⇒
`capture_model_state` 向量长度变了 ⇒ 换 `light_line_n` 后旧 pixel 数据不可重渲染**。
成本 5.4 → 7.1 ms/step（+30%）；n>8 直接 assert。

★★ **亮度和阴影是按真机白板拟合的，不是眼调。两个统计量、两个可分离的旋钮：**

| 统计量 | 旋钮 | 真机 | 拟合后 |
|---|---|---:|---:|
| 板面远/近半区**梯度** | 灯**高度**（关阴影测，隔离 Lambert 项） | +10 | +9 |
| 板面 **p90−p10 跨度** | **`light_shadow_n`**（8 盏里几盏投影） | 52 | 55 |
| 板面**中位数** | **`light_diffuse`**（整条灯带的总 diffuse） | 216 | 216 |

- 灯高：z 1.6→梯度 +19 / 2.2→+13 / **3.0→+9** / 4.0→+6 ⇒ **`light_pos` z 从 1.60 抬到 3.00**。
  真机那条灯带的**等效**高度远高于「相机上方 0.8 m」—— 房间有顶灯 + 墙面反弹，等效源更远。
- ⚠️ **抬灯不能治跨度，反而变差**（72→108）：p10 是被**阴影**压低的，不是被余弦。
- `light_shadow_n` 是**本影深度**的物理旋钮：被 k 盏挡住的点仍收到 (n−k)/n 的光。
  z=3.0 实测 k=3→35 / 4→41 / **5→57** / 6→61 / 8→106（真机 52）⇒ 取 **5**。
- ⚠️ **环境光不是解药**：总 ambient 0→0.30（同时解 diffuse 保持中位数 216）只把跨度 75→71。

**DR 覆盖阴影**：`dr_light_shadow_n: 3` ⇒ 每局 k∈[2,8]，本影深度随机。加上 `dr_light_*`，
24 次抽样实测 **board 216±22 / spread 55±18 / object (108,3,2)**（真机 216 / 52 / (114,1,1)）。

★★★ **这一轮在 DR 里挖出三个 bug，都是加灯带之后才出现的、且全都静默：**
1. **`dr_light_ambient` 是按每盏灯写绝对值** ⇒ 8 盏灯把 `[0.02,0.12]` 变成总量 `[0.16,0.96]`，
   板面被冲白。已按 `nlight` 分摊。
2. **`dr_light_diffuse` 乘的是 arena 快照，不是匹配值** ⇒ `light_diffuse` 这个新旋钮
   **对任何一个 DR 样本都毫无作用**（扫 0.7→1.6 板面中位数纹丝不动 179.5，这才暴露出来）。
   `_matched_pos/_matched_dir` 当初修过同一个坑，diffuse 漏了。
3. **`dr_light_pos` 是逐灯独立抽的** ⇒ 一条灯带被打散成一团点光源，把灯带存在的意义随机化掉了。
   已改成整条灯带**刚体**平移（一次抽样，所有灯共用）。

★ **DR 的均值要对齐真机，而不是 nominal**：`dr_light_dir`/`dr_light_pos` 只会**降低**板面入射
余弦 ⇒ 无论增益多对称，DR 均值都系统性偏暗。曾经的状态是 nominal 216、**DR 均值 179**（暗 17%），
也就是策略看到的平均画面根本不是真机。`light_diffuse` 现在是按 **DR 均值 = 216** 解出来的。
⚠️ 连带：albedo/灯光的**配比**也重平衡过 —— 只靠 albedo 把板面拟到 216 会把它逼到 **0.97**，
增益上界只能设 1.03、DR 均值再被拉低；现在 albedo 回到物理的 **0.85**、灯按同比例调亮
（渲染结果不变），两个增益范围就能对称取 `[0.90,1.10]` / `[0.85,1.15]`。

**并行试布局的办法**：`THREADING_REAL_CELL_YAML` 环境变量覆盖 `Threading_RealCell` 读的 cell
文件（不设 = 逐字节旧行为）⇒ 候选 cell 放 scratchpad，**不用动线上 yaml**（而 `Threading_RealCell`
每次 reset 都重读它，改线上 yaml 会当场打死在跑的 job）。`gen_realcell_demos.py --measure`
现在按 `--shard` 给实验目录命名，`--video` 保留失败并出 playback mp4。

---

## 15. Square_RealCell：把 Square 搬上同一张真机 bench（2026-09-21）

**同一个 DROID cell、同一套相机解与 DR，只换任务与物体。** 唯一真源 =
`release/cfgs/sim2real/square_real_cell.yaml` + `env/square_variants.py`。
arena 仍然来自 `env/peg_insertion.py` 的 `MEASURED_*`（§14），所以 Square / Threading /
PegInsertion 三条线共享一个 arena，gate §L 逐项比对。

### 15a. cell yaml 现在可以 `inherit`

`load_sim2real()` 新增 `inherit: <相对路径>`（子键覆盖父键，`cameras` 整体替换，不支持链式）。
`square_real_cell.yaml` 因此只有 **2 个 task 键**（`workspace_offset` / `object_geom_prefixes`），
相机解、桌色、全部 `dr_*` 都从 `threading_real_cell.yaml` 继承。
**理由**：那些量是「这间屋子」的属性不是任务的属性 —— 重标定一次相机就该同时落到两条线上，
而一份复制粘贴的副本只会在下一次标定时静默过期（正是 §14b 那类漂移）。
⚠️ 连带：任何做 provenance 快照的脚本都要把 **inherit 链上的每个文件**一起哈希。
`pipeline_realcell.sh` 已改（`cell_files()` + `sha256sum -c`）；只存子文件等于没存。

### 15b. ★ Square 不能只靠 wrapper 的 sim2real hook —— 三条，全部静默失败

前两条 Threading 也有（所以要注册类），**第三条是 Square 独有的，而且是致命的**：

1. MimicGen（生成）和 robomimic（`tools/dataset_states_to_obs.py`）都从 hdf5 里的 env_name
   建 env，永远不过我们的 wrapper。
2. `apply_workspace_offset` 打的是 `_get_initial_placement_bounds`，**Threading 每次 hard reset
   重读，Square 不读**：`Square_D0.__init__` 一次性把 bounds 建成
   `SequentialCompositeSampler` 交给 `NutAssemblySquare`，而 `_load_model` 只在
   `placement_initializer is None` 时才重建（`nut_assembly.py:406`），`_reset_internal` 只调
   `.sample()`。⇒ **wrapper 的 offset patch 在 Square 上是 no-op**，螺母会留在 D0 的位置。
3. ★ **peg1/peg2 是 arena 几何，写死在 `pegs_arena.xml` 的世界坐标 `(0.23, ±0.1, 0.85)`；
   `TableArena`/`PegsArena` 都不会让它们跟着 `table_offset` 走。** 桌面从 0.820 抬到 0.9327
   之后，半高 0.1 m 的柱子露出部分从 **130 mm 变成 17 mm** —— 任务不是变难，是**几何上不可能**，
   而且没有任何地方会报错。`Square_RealCell._place_pegs` 把两根柱子重新放到新桌面上，
   并且施加与螺母**相同**的 `workspace_offset`，所以 nut↔peg 的相对几何（即任务本身）不变。
   ⚠️ 改的是 `super()._load_model()` 之后的**合并后 task XML**，因为那份 XML 才是 MimicGen
   存进 hdf5 `model_file`、robomimic 回放时用的东西。

附带：两根柱子的 geom 在 `pegs_arena.xml` 里**没有 name**，所以对一切按 geom 名工作的东西
（cell 的 `object_rgb`、preview 的遮挡报告、逐 geom 的 domain shift）都不存在。`_place_pegs`
顺手给它们命名（`peg1_col0` / `peg1_vis1` / …），这才让柱子能被当成任务物体处理。

### 15c. `workspace_offset = (0.071, −0.034)`：纯 base 位移补偿

```
stock base xy 相对桌心 = (−0.16 − 0.8/2, 0)        = (−0.560,  0.000)
实测 base xy 相对桌心（§2e）                        = (−0.489, −0.034)
位移                                                = (+0.071, −0.034)
```
⇒ 螺母与柱子**相对机器人**的位置与 stock `Square_D0` 逐位一致（gate §L 检查），
stock 的可达性/抓取几何原样保留。

⚠️ **Threading 的 `(0.071, 0.066)` 里多出来的 `(0, +0.10)` 是遮挡修正，不可搬过来**
（cell 文件里已写明「camera-dependent」）。Square 这边 **不需要**：同一台 zed2i 下
preview 报告螺母与方柱都是 `OK`，无遮挡。

**可达性实测**（OSC P 控制伺服到 4 个 waypoint × 5 seed，残差）：

| waypoint | Square_RealCell | stock Square_D0 |
|---|---:|---:|
| nut grasp | 0.2–0.3 mm | 0.6–1.3 mm |
| nut lift +12 cm | 0.7–0.8 mm | 0.7–1.0 mm |
| over peg top +6 cm | **0.1–5.4 mm** | 0.1–1.0 mm |
| peg insert（柱身中段） | 9.2–10.0 mm | 10.0–11.1 mm |

最后一行两边都是 ~10 mm ⇒ 那是**空夹爪撞柱子**的伪迹，不是可达性。
真正的信息是 `over peg top` 一行：RealCell 最差 5.4 mm vs stock 1.0 mm ⇒ 桌面抬高 113 mm 后
柱顶相对 base 从 +38 mm 变成 +151 mm，**余量确实变小了但够用**。

### 15d. rotw1 由**类**携带，不用 `init_dist` 旋钮

`Square_RealCell_RotW1`（`NUT_Z_ROT = (2π/3, π)`，即 120–180°，与
`square_d0_state_ibrl_source_rotw1_sig01.yaml` 逐位一致）。
**为什么不用 `init_dist.nut_z_rot`**：那个旋钮在 `PixelRobosuite`
（`env/domain_shift.py::_apply_square_init_distribution`）里，MimicGen 生成根本不经过它 ⇒
窗口就得在两处各写一遍，而两边都不会报错。现在类名承载完整任务定义，生成 / BC prior /
`train_rl.py` 三处只写 `Square_RealCell_RotW1`，**cfg 里没有 `init_dist` 块**。
不带窗口的 `Square_RealCell` 保留 D0 原生 360°。

### 15e. 圆柱 peg2 已下桌（`ROUND_PEG = False`，2026-09-21 用户决定）

`Square_D0` 的 `single_object_mode=2` 把 RoundNut 传送到 (10,10,10)，**但没动它的柱子** ⇒
peg2 作为干扰物留在桌上，真机上没有对应物；而且被 `object_rgb` 上成同一个 burgundy 后，
两根柱子在渲染里几乎无法区分。现在与 RoundNut 同样处理。
⚠️ 这是**开关不是删除**（要和公开 Square 数据集对比时得拨回来），而且它改的是合并后的 task
XML ⇒ **烘进每一条生成的 demo，不是渲染期选项，改了必须重新生成，不能混用**。

### 15f. 实测数字（2026-09-21）

| 项 | Square_RealCell | 对照 |
|---|---:|---|
| MimicGen 生成成功率（360°，n=50 attempts） | **78%** | Threading RealCell 53%（§14b） |
| MimicGen 生成成功率（rotw1，n=50） | **86%** | — |
| demo 均长 / 最长（rotw1） | 161 / 171 步 | Threading 218 步 |
| agentview 遮挡 | 螺母 OK / 方柱 OK | Threading 需 +0.10 m 才不被臂挡 |
| 孔投影（agentview, 224²） | **14.5 × 17.6 px @ 34.5° 入射** | Threading 环口 3.9 × 6.0 px `UNRESOLVABLE` |
| 孔投影（腕相机, 224²） | 24.1 × 18.3 px @ 18.5° | Threading 1.1 × 11.2 px @ 83.6° |

⇒ **Square 在这台相机上是「看得见」的任务，Threading 不是**（Threading 的插入对齐只能靠腕相机，
而 §14e 实测腕相机在插入瞬间 0.0% 可见）。这条差异值得单独记住，它会决定 pixel 臂的难度排序。

### 15f2. ★ source 专家已训出：**0.96**（2026-09-21，一次跑通）

`TASK=square bash scripts/pipeline_realcell.sh`，全链路无人工干预。

| 阶段 | 结果 | Threading RealCell 对照 |
|---|---:|---:|
| MimicGen 生成 1000 条 | 3.5 min（10 shard 并行） | — |
| 转换 | 全部 `reward match? True`，均长 160 / 最长 171 | 均长 215 |
| MLP BC prior（2000 epoch, 37 min） | 40 个 eval 点均 0.697、末 6 点均 **0.93**、**独立 final eval 0.95** | 0.81 |
| IBRL 400k（2 h 42） | 稳态 190k–400k **0.959**（22 点 × 50 局） | — |
| **独立重评 n=300, σ=0.1** | **0.96** | **0.957** |

IBRL 曲线（每 10k 一点）：前 120k **全 0.00** → 140k 0.46 → 150k 0.90 → 190k 起稳在 0.92–1.00。
⚠️ **前 120k 的 0.00 是 IBRL 的正常冷启动，不是故障**：Threading 同一 pipeline 也是前 50k 全 0.00、
90k 才 0.42。机制 = critic 从零起步时 `act_method=ibrl` 的 argmax 会压过 prior（哪怕 prior 有 0.95）。
判断窗口放在 **90k–150k**，之前的 0.00 没有信息量（铁律 #4）。
⚠️ 别因此去开 `prior_margin` —— tasks §32 实测它会拉低 Square 的最终性能。

⇒ **Square RealCell source 专家与 Threading RealCell 持平（0.96 vs 0.957）**，比 sim2sim 的
Square rotw1（0.98）低约 2 点，与多出来的 base 位姿 DR（R1–R3）一致。

`exps/mimicgen/square_realcell_{mlp_bc,ibrl}_seed1`，IBRL 每 25k 一个 snapshot（供 SimDist 采集）。

### 15f3. `evaluate/eval.py` 的 `[EVAL ENV]` 会把 rotw1 报成 FULL —— 已修

banner 原来只看 `init_dist.nut_z_rot` 这个旋钮，旋钮关着就印 `FULL(0-360)`。
但 §15d 之后窗口是**类**带的 ⇒ 一个真跑在 120–180° 的 run 会被印成全 360°。
**这正是铁律 #17 里废掉 §41 那张 2×2 的同一种错**（banner 是唯一的事后判据，它骗人就没救了）。
已改为旋钮关着时回退去问类的 `NUT_Z_ROT`，印成 `120-180deg(class)`；
拿不到类就仍印 `FULL(0-360)`，且**任何异常只降级这一行打印，绝不影响 eval**。
回归：Threading_D0 与走旋钮的老 Square ckpt 输出均不变（后者无 `(class)` 后缀）。

### 15g. 还没做 / 已知占位

- ~~`object_rgb` 对方柱是占位值~~ **已确认（2026-09-21）**：真机方螺母与方柱和 Threading 零件
  是同一种 burgundy PLA ⇒ 继承的 `object_rgb` 对两件都对，不需要逐件颜色。
- **没有 target 域**：这条是 sim2real，target = 真机本身，不在 sim 里另定 shift
  （区别于 §12/§13 的 sim2sim target）。
- **腕相机**：继承的 `threading_real_cell.yaml` 里是注释掉的（用 sim 默认挂载），
  与 Threading 的 demo 生成 / state IBRL 口径一致。等 pixel 阶段再做
  `square_real_cell_calibrated_wrist.yaml` 继承 `threading_real_cell_calibrated_wrist.yaml`。
- **strict_success 不适用**：Threading/Coffee 专用（§5c），Square 全线 false。
- 一切 `*_RealCell` 类都要在**每个会建 env 的进程**里 import：已加到
  `env/robosuite_wrapper.py`、`scripts/gen_realcell_demos.py`、`tools/dataset_states_to_obs.py`。

### 15i. PIXEL stage-1 数据：与 Threading 的 `2cam_calibrated_v1` 逐项对齐（2026-09-21）

> ⚠️ **数据已于 2026-09-23 删除（cell 大改）。** 本节的**方法与脚本仍然有效**，重采一遍即可；
> 下面的数字属于旧 cell，别再引用。删除的是 `square_realcell_pixel_v1`(133 GB)、
> `threading_realcell_pixel_v1`(211 GB)、`threading_realcell_source_diverse_v1`(6.9 GB)、
> 两个 Threading pixel stage-1 run（日志存档在 `exps/mimicgen/_archive_realcell_pixel_stage1_20260921/`）。
> 作废原因见 §15j。

**配方 = §14e 原样**：agentview-only 采集（224²、JPEG q90、`--store_sim_state 1`）→
用**同一块**校准腕相机从存下的 sim_state 补渲染 → 每 shard 取前 4500 条，8 shard = **36000 条双相机**。
采集参数与 Threading 的 `realcell_pix_gpu{2,3}` 链相同：10 worker/GPU、`PER_WORKER_MIB 3300`、
seed 基数 100、`HYBRID=1 SNAP_HYBRID=1`、p_expert 0.5、动作噪声 σ~U[0.05,0.2]、关节 σ 0.1、
`MAX_LEN 400`、16 个 IBRL snapshot（25k–400k）。

| | Threading | Square |
|---|---|---|
| 采集 cell | `threading_real_cell.yaml` | `square_real_cell.yaml`（inherit 前者） |
| 训练/eval cell | `threading_real_cell_calibrated_wrist.yaml` | `square_real_cell_calibrated_wrist.yaml`（inherit 前者） |
| 腕相机标定 | `2cam_calibrated_v1/calibration.json` | `wrist_calibration.json` = **同一个 `camera` 块逐字复制** |
| 每 shard 采集 | 12500 | **4500** ← 唯一差异 |
| 每 shard 进 stage-1 | 4500（`source_replay_num_data`） | 4500 |

**唯一差异的理由**：`/user_data` 只剩 432 GB（97%），而 Threading 每 shard 采的 12500 条里
stage-1 只读前 4500。所以训练可见的数据完全相同，还省下约 70 GB 用不到的数据。要更多就追加
part（seed 不相交）。「前 4500 条」的组成也等价：merge 按 worker 分片拼接，截断截掉的是
整个 worker，不存在按长度的偏置。

**采集前的验证（全过）**
- `scripts/test_rerender.py --run exps/mimicgen/square_realcell_ibrl_seed1`：两个相机重渲染
  **逐字节相同**，跳过 `model_state` 还原后 mean |diff| 51 ⇒ 重放契约在 Square 上成立。
- 用校准 cell 直接渲染 vs 数据集里补渲染的腕相机帧：mean |diff| **0.70**（JPEG 噪声），
  即训练时 eval 与训练数据同源，与 Threading 的检查结果一致。
- 校准 cell 相对采集 cell **只有 `cameras` 不同**，agentview 逐字段相同（gate §L 已加这条）。
- 采集链的 `threading_real_cell.yaml` 哈希 `b9423238…` 与 Threading pixel_v1 快照**逐字节相同**。

**★ 腕相机在插入阶段看得见 Square**（与 §14e 的 Threading 结果相反）
成功 episode 的最后 20% 帧（n=173），把目标投影进校准腕相机：
**螺母 96% 在画面内，方柱顶端 41%**；Threading 的环与针尖是 **0.0% / 0.0%**。
再加上 agentview 的孔 14.5×17.6 px（Threading 3.9×6.0），**Square 在两个相机上都能解析对准**，
Threading 两个相机都不能。pixel 臂的难度排序应该据此预期：同样的 pipeline 在 Square 上
pixel 与 state 的差距应当小得多。⚠️ 这是 n=11 条 smoke episode 的数，量级可信，不作精确引用。
预览：`figs/sim2real_square/pixel_smoke_sheet.png`（每对 = agentview | 校准腕相机；三行 = 开头/中段/结尾）。

**产物**：`/user_data/Chenyang/ibrl_offload/release/data/mimicgen/square_realcell_pixel_v1/`
`part{1..8}.hdf5`（agentview）→ `2cam_calibrated_v1/part*_2cam.hdf5`（stage-1 读这个）。
同目录有 `cell_chain.sha256`（整条 inherit 链）+ 两个 `.snapshot`、`expert.sha256`、`git_*`。
驱动：两条 `scripts/collect_threading_realcell_batches.sh` 链（`LOG_TAG=square_realcell_pix`，
日志 `logs/square_realcell_pix_gpu{2,3}.log`）+ `scripts/square_realcell_pixel_after_collect.sh`
（等两条链都结束 → 打开每个 part 校验 ≥4500 条且带 sim_state → `rerender_wrist_chunks.py`）。

**结果（2026-09-21 完成，无人工干预）**：采集 75 min（2 条链 × 4 批）+ 补渲染腕相机 19 min。

| | Threading `2cam_calibrated_v1` | Square `2cam_calibrated_v1` |
|---|---:|---:|
| episodes | 36000 | 36000 |
| transitions | 6.53M | **5.36M** |
| 采集成功率 | 73–74% | **78.9%** |
| 平均长度 | ~181 | **149** |
| 相机 | agentview + 校准腕相机 | 同 |
| 大小 | 107 GB | 103 GB |

⚠️ episode 数对齐，**transition 数不对齐**（Square 的 episode 更短，少 18%）。stage-1 的
`seq_replay_max_transitions` 6.6M 上限在两边都不会触发；对照 §45 的 epoch 预算时要按
transition 算，而不是按 episode 数。
清理后 `/user_data` 剩 297 GB（98%）；agentview-only 的 `part*.hdf5`（8×3.9 GB）是补渲染
的源数据，腕相机外参要是再改，靠它们重渲染，别删。

**为此改的共享脚本**（Threading 行为不变）：`collect_threading_realcell_batches.sh` 加
`LOG_TAG`（原来的批日志名写死，第二个任务会覆盖 Threading 的日志）和 inherit 链哈希；
`rerender_wrist_chunks.py` 加 `--expert/--num_shards`（原来写死 Threading 专家 ——
拿它重放 Square 的 state 会重建错的场景；非默认专家写进 manifest，混用时 resume 会被拒绝）；
`test_rerender.py` 加 `--run`。

### 15j. ★ cell 大改（2026-09-23）谁作废、谁不作废

`threading_real_cell.yaml` 换了相机内参（fovy 60.84 → **50.3**）、桌面板（视觉 5×5 m → **实测
0.753 m 方板** + offset）、颜色（按 ZED t=0 帧重新拟合：板 (219,220,218)、物体 (114,1,1)）、
条形光 + 阴影、背景照片 DR、`hide_table_legs`，以及 Threading 的**显式摆放** `layout_needle_*` /
`layout_tripod_*`。`square_real_cell.yaml` 是它的 2 键 override ⇒ 这些全部继承过来。

**逐字段判定**（把新旧 cell 都 resolve 出来对比，而不是读 diff 猜）：

| 作废 | 不作废 |
|---|---|
| **所有 pixel 数据**（两个任务）：相机 + 颜色 + 光照 + 背景全变 | **Square 的 state 线**：demo 1000 条、MLP BC prior、IBRL 专家 **0.96** |
| **Threading 的 state 数据**：`layout_*` 改了复位分布 | Square 的 arena / `workspace_offset` / `init_qpos` / `dr_base_*` 均未变 |

★ **为什么 Square 的 state 线活下来了**：对 Square 而言，改动**全是视觉项**。唯一的几何类改动
`layout_*` 是**按物体名匹配**的（`cell_layout_bounds` 只给 needle/tripod 出 spec），两条应用路径
——wrapper 的 hook（`env/sim2real.py:563`）和 `Threading_RealCell._get_initial_placement_bounds`
——都只覆盖名字对得上的物体。Square 的 sampler 只有 `nut`，所以一条都不命中，仍走
`workspace_offset`。已实测确认。
⇒ **推论**：`layout_*` 现在是 Threading 专用的。哪天要给 Square 也做实测摆放，得新增
`layout_nut_*` 并让 `Square_RealCell` 去读，不会自动生效。

**重采 pixel 数据前要先做的事**：§15i 的脚本原样可用（`LOG_TAG` / `--expert` / inherit 链哈希
都已参数化），但 **cell 稳定之前别开**——这批数据的成本是 1.6 h + 133 GB，而它把当时的相机与
光照逐帧烘死。另外 Square 的 source 专家是在**旧 cell** 上训的：几何没变所以它依然有效，
但如果之后 cell 动了几何（不是视觉），专家要重训。

### 15k. ★ 方柱扰动 `dr_peg_xy` / `dr_peg_yaw_deg`（2026-09-23）

**动机**：Square 的方柱是 arena 几何，任务自己的 sampler 从不碰它 ⇒ 每一局柱子都在同一个点。
真机上柱子的摆放有公差，所以加一个小幅扰动。新旋钮在 `env/sim2real.py`，每次 reset 以
`nominal` 为基准写 `body_pos`/`body_quat`（与 base DR 同构，不会跨局累积）。
`body_pos`/`body_quat` 本来就在 `MODEL_STATE_FIELDS` 里 ⇒ **这次抽样会被 `model_state` 记下，
pixel 事后重渲染能逐帧复现**（§14e 的重放契约）。

**★ 这个扰动对两条线的性质不同 —— 引用收益时必须说明是哪条线**
Square 的 23 维 obs = `SquareNut_pos/quat` + 到 eef 的相对量 + 本体感受，**方柱不在里面**
（对比 Threading：tripod 是 sampler 物体，位姿进 obs）。⇒
- **state 策略**：不可观测的**噪声**，只能靠动作容错吃掉，上限由间隙定；
- **pixel 策略**：方柱在 agentview 里可见，是**可学的信息**。

**间隙**（实测 geom）：孔 45.4 mm 见方（4 根方条内面 ±22.7 mm），柱 32 mm 见方
⇒ yaw 对齐时每边 **6.7 mm**，yaw 偏 45° 时归零。

**剂量-响应**（专家 `model0.pt`，n=200/格，σ=0.1，SE≈0.012）：

| `dr_peg_xy` | yaw | score | Δ |
|---:|---:|---:|---:|
| 0 | 0° | **0.985** | — |
| 2 mm | 0° | 0.970 | −0.015（噪声内） |
| 5 mm | 0° | 0.970 | −0.015（噪声内） |
| 5 mm | 3° | **0.990** | +0.005 |
| 10 mm | 0° | 0.925 | **−0.060**（≈4 SE） |
| 10 mm | 5° | 0.935 | −0.050 |

拐点落在 5–10 mm，**和 6.7 mm 的间隙一致**（机制自洽，不是拟合出来的）。yaw 到 5° 免费 ——
方柱转几度几乎不吃间隙。⇒ **选定 `dr_peg_xy: 0.005` + `dr_peg_yaw_deg: 3.0`**（两份 Square cell
都写了；gate §L 检查两份只差 `cameras`，所以不会漂）。
⚠️ n=200 单组 seed；<0.05 的差读不出来（铁律 #1），别把 0.985→0.970 当成真实退化。

**想要更大的方柱变化怎么办**：小扰动不够，得把方柱位姿**加进 obs** —— 那会改状态维度，
demo / BC prior / IBRL 专家三样全部重来，而且真机上要有方柱位姿的估计来源。是另一个量级的决定。

### 15l. 相机视角的影响：state 为零，pixel 为全部

- **state 线：严格为零。** obs 里没有像素。实证：cell 大改（fovy 60.84°→50°、颜色/光照/背景全换）
  前后同一个专家 n=300 独立重评 **0.96 → 0.9667**。
- **pixel 线：相机就是输入。** 这是 pixel 数据必须重采的原因。Square 的孔投影
  14.5×17.6 px → **18.0×21.6 px**。
- **`fovy: 50` 是为 Threading 选的**（环孔像素 vs 取景余量的拐点，留 62 mm 余量）。
  Square 同口径实测（n=200，全 DR，螺母/柱底/柱顶到画面边）：**最差 44.8 px = 421 mm，
  是 Threading 的 7 倍**。⇒ 对 Square 而言 50° 非常宽裕，**如果想要更多像素，Square 可以单独
  裁得更紧**（它自己的 cell 键即可，代价是真机侧两个任务各用一套裁剪参数）。没有实测支持之前不要改。
- 真机侧配套：50° = 以**主点** (622.555, 352.755) 为中心裁 497 px 再缩到 224，**不是图像中心**，
  且要先去畸变。sim 与真机这一步不一致，pixel 策略直接错位。

### 15m. ★ 重放契约在照片背景/软阴影下一度失效（2026-09-23，已修）

开采前按 §14e 重跑 `test_rerender.py`，**直接崩在 A（max |diff| 212）**。两个独立的根因，
都属于同一类："每局随机、但没被 `model_state` 捕获"的字段。

| # | 字段 | 症状 | 修法 |
|---|---|---|---|
| 1 | 背景照片（写在 `tex_rgb`） | 重渲染画上**另一张**背景 ⇒ 212 | 采集端记录 `bg_index`（逐 episode attr），重渲染端重新画回同一张 |
| 2 | `light_castshadow`（`dr_light_shadow_n` 每局挑哪几盏灯投影） | 阴影集不同 ⇒ 118 | 加进 `MODEL_STATE_FIELDS` |

★ **1 的病根是链路只修了一半**：`apply_sim2real` 早就有 `bg_index` 参数、重渲染端也在用，
**但采集端（`PixelRobosuite.reset`）从不传也不存**。代码里存在一个"可复现"的接口，却没有人
在写入侧调用它 —— 铁律 #26 的同型（钩子在但没人调），而且因为 `tex_rgb` 不在 `MODEL_STATE_FIELDS`
里，`restore_model_state` 看起来"什么都恢复了"，掩盖了它。
⇒ 现在 `PixelRobosuite.last_bg_index` 暴露抽样结果，`generate_simdist_data` 两个写盘函数都落盘，
`rerender_simdist_data` 默认读它重画（`--visual_dr` 时才另抽）。实测 400/400 条都带 `bg_index`。

★★ **判据本身也得改：逐字节相同在开软阴影后不可能成立。** 修完两个 bug 仍剩 max |diff| 7。
诊断方法是**让渲染器和自己比**：同一份源数据连续重渲染两次，彼此就差 max 4–6（0.01% 的像素）。
⇒ gate A 改为**自校准**：要求「源 vs 重渲染」不差于「重渲染 vs 重渲染」（+2/255 松弛，且均值同量级）。
现状 7 / 0.00008 vs 7 / 0.00007，无法区分；而真故障（212 / 118）仍差两个数量级会被抓住。
⚠️ `MODEL_STATE_FIELDS` 一改，`model_state` 向量长度就变，新旧数据不能混用（`model_state_layout`
会对不上）。这次能改是因为那一刻所有 pixel 数据都已删除。

### 15n. 采集就绪状态与成本（2026-09-23 实测）

| 项 | 值 |
|---|---|
| 专家（完整新 cell + 方柱扰动，n=300，σ=0.1） | **0.98** |
| 吞吐（10 worker/GPU，agentview 224 JPEG q90 + 照片背景 + 阴影） | 400 条 / 152 s（含约 40 s 启动）≈ **3.6 条/s 稳态** |
| 单 worker 显存 | ~1.15 GB（`PER_WORKER_MIB 3300` 仍然保守够用） |
| 磁盘 | **2.29 MB/条** ⇒ 8×4500 agentview ≈ 82 GB，加 2cam 约 185 GB |
| 采集成功率 | 77–78% |
| 预计耗时 | 2 条链 × 4 批 ≈ **1.5 h**（与照片背景加入前持平，渲染成本没涨） |

⚠️ **GPU 可用性是当前的硬约束**：GPU 2/3 被别的用户占着（各约 70 GB，已跑一天多），
在上面起 10 worker **7 个直接 CUDA OOM**。GPU 0/1 目前空闲，100 条探针零 OOM。
⇒ 开采前先 `nvidia-smi` 看一眼，并把 `GPU=` 指到真正空的卡；批脚本的 guard 用的是
**整卡** used/total，别人占着时它会一直等而不是报错。

### 15h. 怎么跑

```bash
# 一次性：MimicGen source demo（10 条 human）+ datagen_info
python .../mimicgen/scripts/download_datasets.py --dataset_type source --tasks square \
    --download_dir release/data/mimicgen
python .../mimicgen/scripts/prepare_src_dataset.py \
    --dataset release/data/mimicgen/source/square.hdf5 --env_interface MG_Square \
    --env_interface_type robosuite

# 预览（渲染 + 遮挡/分辨率报告）
python scripts/preview_sim2real.py --task Square_RealCell_RotW1 \
    --config release/cfgs/sim2real/square_real_cell.yaml --out figs/sim2real_square

# gate（§L 是 Square 段）
python scripts/test_sim2real.py

# 全流程：生成 1000 条 -> 转换 -> MLP BC prior -> IBRL
TASK=square setsid bash scripts/pipeline_realcell.sh \
    > exps/mimicgen/pipeline_realcell_square.log 2>&1 < /dev/null & disown
```
