# CLAUDE_ARCHIVE.md — 归档的实验分支笔记

从 `CLAUDE.md` 移出来的部分。**移出的标准 = 与当前主线（SimDist 跨域 adapt 的改进 +
多任务验证）无关**：已被证伪、已停止推进、或只是历史调参过程。内容**一字未改**，
需要时按下表回查；`CLAUDE.md` 里残留的 `§10/§11/§13/…` 交叉引用指向的就是本文件。

大部分对应的**代码也已归档到本地 `_archive/`**（被 `.gitignore` 排除、不随仓库发布），
恢复方式见 `CLAUDE.md` §34。

| 节 | 分支 | 结论 |
|---|---|---|
| §10 | SGFT value transfer on IBRL | 代码保留（`q_agent.sgft_shaping`），YAML 已归档 |
| §11 | IL co-training（source-heavy + 少量 target） | 代码保留（`CoTrainDataset`），YAML 已归档 |
| §13 | CFG on co-training flow BC | 未跑出结论 |
| §14 | H2O+ on IBRL | online dynamics ratio ≈ no-op |
| §15 | Flow BC 归一化 + low-data config | 归一化已并入 §4 主线；low-data 实验未完成 |
| §16 | GatedGapAgent（GT-gap 门控） | 未跑赢 `force_c=1.0` 对照 |
| §18 | LWAIL-cross（ICVF latent 占用匹配） | 判别器饱和（占用零重叠），任务本质不适合 |
| §19a | ReBRAC（offline TD3+BC） | 代码部分保留（`EnsembleQ`/`_load_demos_flat` 被主线依赖） |
| §19b | source-domain recovery 数据收集 | 结论已记录，流程停用 |
| §20 | 旧 residual（gap-weighted） | 被 §25 的 ResFiT 版取代 |
| §21 | OracleMap + occupancy 重加权 | actor drift 崩溃 |
| §22 | Oracle map + residual | 同上线，未跑完 |
| §23 | MB-IBRL（MBPO-on-IBRL） | 从未跑；`DynamicsEnsemble` 仍被 §30 诊断使用 |
| §3-hist | 旧 offset-bearing target 的逐参数调参史 + `hole_offset` 的 POMDP 论证 | 当前 target 是 dynamics-only（§12），`hole_offset=0` |
| §7-old | 2026-05-25 的冒烟测试状态 | 早已被后续实验覆盖 |
| §9-old | 早期 roadmap / 停车场 | 已过期 |

## 34. 仓库瘦身：已归档的实验分支（2026-08-01；笔记 2026-08-14 拆出）

### What
为了让公开仓库聚焦在 **SimDist 跨 domain adapt** 这条主线上，若干已被证伪 /
已停止推进 / 与本课题无关的实验分支，从 git 索引中移出，**移动到本地
`_archive/`**（被 `.gitignore` 排除，不随仓库发布）。

**它们的完整笔记（§10/§11/§13/§14/§15/§16/§18/§19/§20/§21/§22/§23 全文，
以及旧 target 的调参史、早期 roadmap）已于 2026-08-14 拆到 `CLAUDE_ARCHIVE.md`。**
本文件里残留的这些节号交叉引用，都指向那个文件。

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

## §3-hist. 旧 target 配方的调参史（archived）

> 归档自 `CLAUDE.md` §3。**当前 target 见 §12（dynamics-only，`hole_offset=0`）。**
> 保留的理由：这里的逐参数论证（clip-through 风险、摩擦锥、环几何下界）对任何人
> 重新调 target 仍然适用。

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

## §7-old. Verification status（2026-05-25 的冒烟测试，archived）

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

## §9-old. 早期 roadmap（archived）

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
