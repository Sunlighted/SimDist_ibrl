# classifier_test — H2O 双判别器 source/target 动力学分类实验

验证一个想法：仿照 [`/home/Chenyang/H2O`](../../H2O) 的做法，训练两个判别器来区分
source domain 与 target domain 的数据，并通过两者相除得到动力学条件概率之比
`p_target(s'|s,a) / p_source(s'|s,a)`，再看判别准确度。

## 方法（移植自 H2O）

H2O 用两个判别器把「状态-动作边缘分布差异」从「动力学差异」中分离出来：

- `d_sa(s, a)` → 2 类 logits（类 0 = target/"real"，类 1 = source/"sim"）
- `d_sas(s, a, s')` → **advantage** logits；联合分类器用 `softmax(adv + sa_logits)`

这样 `d_sas` 只需建模 `s'` 在 `(s,a)` 之上带来的残差。动力学对数比为

```
log r = [log p_sas(target) − log p_sas(source)] − [log p_sa(target) − log p_sa(source)]
```

把 `(s,a)` 边缘分布约掉，只剩动力学 gap。代码见 `discriminator.py`。

## 数据

| 角色 | 文件 | 轨迹数 | transition 数 |
|---|---|---:|---:|
| source ("sim") | `threading_d0_source_sigma0.1/dataset.hdf5` | **2000** | 120959 |
| target ("real") | `threading_d0_target_dynamics/dataset.hdf5` | 20 | 1963 |

状态用 37 维 `obs/state`，动作 7 维。source 远多于 target，故每个 minibatch 从两域
**各采等量样本**（H2O 同款做法），输入做（两域合并统计的）标准化 + 小高斯噪声防过拟合。

## 运行

```bash
source set_env.sh

# 1) 训练两个判别器（episode 级 8:2 划分验证集）
python classifier_test/train_classifier.py --steps 20000
#   -> classifier_test/discriminator.pt

# 2) 用 checkpoint 在 source(无 shift) + target(dynamics-only shift) 各 rollout 收集 transition
python classifier_test/collect_eval_data.py --num_game 30 \
    --weight exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt \
    --out classifier_test/eval_newtgt.npz
python classifier_test/collect_eval_data.py --num_game 30 \
    --weight exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
    --out classifier_test/eval_source.npz

# 3) 在收集到的 on-policy 数据上评测判别准确度
python classifier_test/eval_classifier.py --data classifier_test/eval_newtgt.npz
python classifier_test/eval_classifier.py --data classifier_test/eval_source.npz
```

`collect_eval_data.py` 默认用**同一个策略**在两个 env 里 rollout，这样两组 transition 的
唯一差别就是 env 动力学（最干净的动力学 gap 测试）。target shift = CLAUDE.md §12 的
dynamics-only recipe（friction 0.5 / mass 3.0 / gripper_kp 0.12 / ring_size 0.85 /
ring_friction 2.0，无 hole_offset）。

## 结果

### 训练域（held-out demo transition）
两个判别器都能到 **~92% 平衡准确率**（`d_sa` 和 `d_sas` 几乎一样）。

### On-policy rollout 数据（关键测试）

| 收集策略 | 域 | n | acc d_sa | acc d_sas | P(target) by d_sas | log 动力学比均值 |
|---|---|---:|---:|---:|---:|---:|
| newtgt model0 | source | 2115 | 0.309 | 0.311 | 0.656 | −0.026 |
| newtgt model0 | target | 2435 | 0.793 | 0.790 | 0.752 | −0.019 |
| **newtgt 平衡准确率** | | | **0.551** | **0.550** | | |
| source model0 | source | 1821 | 0.915 | 0.921 | 0.121 | −0.082 |
| source model0 | target | 10479 | 0.349 | 0.344 | 0.361 | −0.021 |
| **source 平衡准确率** | | | **0.632** | **0.633** | | |

### 结论

1. **判别器无法靠动力学区分 source/target on-policy 数据。** 平衡准确率仅
   0.55–0.63（接近随机的 0.5），远低于训练域的 0.92。

2. **`d_sas` 相比 `d_sa` 没有任何增益**（两列几乎相同），且分离出来的
   **动力学对数比 ≈ 0**。说明判别器学到的几乎全是 demo 的「数据采集 (s,a) 边缘分布」
   差异，而不是真正的动力学 gap——而这个边缘分布差异不会迁移到新策略产生的数据。

3. 预测结果基本由「该 transition 的 (s,a) 与哪个 demo 集更像」决定：source agent 在
   source env 里的 rollout（最像 source demo）被正确判为 source（0.92），但它在 target
   env 里失败的 rollout 仍多被判成 source（只有 0.34 判对）。

**机理**：Threading 的动力学 gap（质量/摩擦/抓取刚度/环孔几何）只在**少数接触步**上体现，
绝大多数 transition 在自由空间、两域动力学几乎一致；逐 transition 的动力学信号本就稀疏，
经 H2O 比值把 (s,a) 边缘约掉后趋近于 0。要让这种分类器真正抓到动力学差异，可能需要：
只在接触/插入阶段的 transition 上训练与评测、去掉输入噪声、或用动作-序列/多步窗口而非单步
`(s,a,s')`。

## 变体：只在接触阶段训练/评测

动力学 gap 只在接触/插入阶段显现，所以加了 `--contact_only`：用 needle body pos
(`state[9:12]`) 到 tripod body pos (`state[23:26]`) 的距离 < 0.15m 作为接触判据
（source 16% / target 30% 的 transition 落入接触阶段）。

```bash
python classifier_test/train_classifier.py --contact_only 1 --contact_thresh 0.15 \
    --steps 20000 --out classifier_test/discriminator_contact.pt
python classifier_test/eval_classifier.py --disc classifier_test/discriminator_contact.pt \
    --data classifier_test/eval_newtgt.npz --contact_only 1
python classifier_test/eval_classifier.py --disc classifier_test/discriminator_contact.pt \
    --data classifier_test/eval_source.npz --contact_only 1
```

### 接触阶段结果

| | held-out demo (sas) | newtgt rollout | source rollout |
|---|---:|---:|---:|
| 平衡准确率 | source ~0.90 / **target 1.00** | **0.52** | **0.62** |
| 动力学对数比 | — | source +0.32 / target +0.06 | source +0.20 / target +0.03 |

- **训练域上接触阶段高度可分**（target held-out 准确率到 **1.00**），且分离出的
  **动力学对数比变成非零**（~0.2–0.36，对比全量训练时的 ≈0）——说明接触过滤确实把动力学
  信号集中了一点。
- **但 on-policy 迁移仍然接近随机（0.52–0.62）**，newtgt 那组甚至更差。原因更深一层：

**判别器抓的主要是「采集策略的行为签名」而不是 env 动力学。** newtgt 策略是在 target
env 上训出来的，它的接触阶段行为天生「target 风格」，无论放在 source 还是 target env
里 rollout，其接触状态都被判成 target（P(target)≈0.9，对 source-env 数据准确率仅 0.06）；
source 策略反之，它在 target env 里失败/卡住的接触 transition 仍多被判成 source（仅 0.33
判对）。即使用**同一个策略**在两 env 收集（newtgt 那组），(s,a) 边缘几乎一致、只有动力学不同，
判别器也无法靠动力学把它们分开。动力学对数比的符号也不可靠（source 接触 transition 反而
给出更高的「target 动力学」比值），进一步说明残差被行为签名污染。

### 总结论

H2O 式双判别器在这个 state-based Threading 设置里**不能稳健地按动力学区分 source/target**：
- 在 demo 同分布数据上准确率高（92%，接触阶段甚至 100%），但这主要来自数据采集分布
  （行为/初始化）差异；
- 迁移到任意新策略产生的 on-policy 数据时跌到接近随机（0.52–0.63），因为预测被采集策略的
  (s,a) 行为签名主导，真正的逐步动力学信号太稀疏（只在少数接触步）、且被 H2O 比值约掉后很弱。

要把它当成可靠的动力学 gap 检测器，需要消除行为签名这个混淆变量，例如：用**固定的同一策略**
（理想是随机/探索策略）在两 env 收集配对数据来训练判别器、或显式用多步窗口/接触力等更直接反映
动力学的特征，而不是单步 `(s,a,s')`。

---

# 反事实配对方案（成功）—— per-transition 动力学 gap 探测器

上面的诊断说明：直接用 mismatched 的 source/target demo 训 H2O 判别器，学到的是**采集策略
的行为签名**而非动力学。修正办法是**消除 (s,a) 分布混淆**——用精确反事实配对让两域共享同一
(s,a)，逼判别器只能靠 s' 区分。面向 sim2real（target 只有固定 demo、不可反复 query；source
是 sim、可任意 reset）。

## 关键机制（cf_utils.py）

只需一个 **source(nominal) env**：对每个 target demo transition `(sim_state_t, a_t, s'_tgt)`
（`s'_tgt` = demo 自己的下一步，本就是 target 动力学走出来的），把 source sim
`set_state_from_flattened(sim_state_t)` 再执行 `a_t` → `s'_src`。(s,a) 完全配对，
`‖s'_tgt − s'_src‖` 就是**纯动力学 gap**。`sim_state`(45 维 qpos/qvel) 在 demo obs 里有；
mass/friction/ring-size 不在 sim_state 里，所以 source 保持 nominal 物理——正是要测的差异。
set_state 重建误差 ~0.003–0.006（噪声底）。四元数有 q/−q 双覆盖，算 gap 前需符号对齐
（`data_utils.state_gap`），否则被伪 2.0 污染。

## 5 步 + 运行

```bash
# 1) 反事实配对（20 条 target demo → 1963 对 (s,a,s'_src,s'_tgt)）
python classifier_test/step1_build_pairs.py
# 2) 画 ground-truth gap ‖Δs'‖ 结构
python classifier_test/step2_plot_gap.py
# 3) 用配对数据训判别器（(s,a) 配对 → d_sa 塌到 0.5，d_sas 学纯动力学）
python classifier_test/step3_train_discriminator.py
# 4) 判别器 gap 分数 vs ground-truth ‖Δs'‖ 相关性验证（held-out demo）
python classifier_test/step4_validate.py
# 5) 泛化：flow BC 在 source 以 sigma=0.1 跑新轨迹，逐条看 gap 定位是否泛化
python classifier_test/step5_generalization.py
```

## 结果

**Step 2（ground-truth gap 结构）**：gap 右偏，中位数 0.0069（≈噪声底）、max 0.13。按中位数
接触 0.0164 > 自由空间 0.0066（2.5×）。最大 gap 由 needle 朝向(quat)、gripper qpos、tripod
位置驱动——是"重 needle(3×)+软抓取(0.12×)+轻环被碰动"的物理。gap 是**局部尖峰**（grasp/lift +
接触时刻），自由空间贴噪声底。

**Step 3（训练）**：`d_sa` loss 精确 = 2·log2 = 1.386 → **完美塌到 0.5**（(s,a) 配对让它无法
区分，正如理论预测）；`d_sas` 学到动力学信号（val 可分性 acc 0.60–0.69 > 0.5，偏弱因多数
transition gap 贴噪声底不可分）。

**Step 4（验证，held-out demo）**：判别器 gap 分数 vs ground-truth ‖Δs'‖ 的 Spearman：
two-sided（`r_tgt−r_src`，sim2sim）**rho=0.495**，one-sided（`|r_src|`，sim2real 现实可用）
**rho=0.330**。高 gap 组分数 0.441 vs 低 gap 组 0.031（14× 分离）。散点图显示接触阶段
（低 needle-ring 距离）的高 gap 点被判别器准确打高分、贴对角线。

**Step 5（泛化）—— probe policy 的选择极大影响结论，暴露 OOD 限制：**

| behavior policy | 训练数据 | source rollout | OVERALL rho |
|---|---|---|---|
| `threading_flow_bc_seed1` | source 1000 demo | 全成功 ~250 步 | **+0.425** |
| `threading_target_only_flow_seed1` | target 20 demo | **全失败 跑满 400 步** | **−0.454** |

- **用能完成任务的 source flow BC**：OVERALL rho=0.425（1467 transition），高 gap 组 0.277 vs
  低 gap 组 0.115，逐轨迹判别器分数在 ground-truth gap 抬升处同步抬升——泛化成立。
- **用计划本意的 target-only flow BC**（20 demo）：该策略太弱，放进 source 完全失败、乱晃进大量
  **OOD 状态**；判别器只在 20 条 demo 的窄 (s,a) 上训过，**把"新奇状态"误当成"高 gap"**，对
  OOD 打高分而真 gap≈0 → rho=−0.454（高 gap 组分数 0.379 反而 < 低 gap 组 0.584）。
- **结论**：判别器在 in-distribution 操作状态上能定位 gap，但**对 OOD 状态会误触发**（混淆
  "novel state" 与 "dynamics gap"）。这是核心待解问题（OOD 鲁棒性）。`step5_targetbc.png` 里
  traj 3/4/5 的分数全程乱跳、真 gap 却平坦，是该失败模式的直接证据。

## 结论

**配对训练让判别器真正学会了定位 per-transition 动力学 gap**（rho 0.33–0.50、高/低 gap
14× 分离、跨新轨迹泛化），与之前 mismatched-demo 判别器的 rho≈0/纯行为签名形成鲜明对比。
核心是 `d_sa` 在配对下塌到 0.5、`d_sas` 被迫学 s' 的纯动力学差异。

**诚实边界（已遵守）**：训练只用 target demo 的配对（假装 target 不可 query，模拟 sim2real）；
ground-truth ‖Δs'‖ 仅用于验证（用 target sim query，sim2sim 才有，不喂回训练）。真到 sim2real，
反事实 reset 会断（真实 demo 无 sim 内部 state），需 state estimation/系统辨识对齐。

## 反事实方案文件
- `cf_utils.py` — set_state 反事实步进、source/target env 构建、demo 加载、TARGET_SHIFT
- `step1_build_pairs.py` … `step5_generalization.py` — 5 步流水线
- `pairs_demo.npz` — 配对数据；`discriminator_cf.pt` — 配对训练的判别器
- `step2_gap.png` / `step4_validation.png` / `step5_generalization.png` — 结果图

---

## 文件（早期 H2O 直接判别实验）
- `discriminator.py` — 网络（H2O `ConcatDiscriminator`）+ 动力学比/分类工具
- `data_utils.py` — 从 hdf5 读取 `(s,a,s')` + episode 级划分
- `train_classifier.py` — 训练两个判别器
- `collect_eval_data.py` — 用 RL checkpoint 在 source/target env rollout 收集 transition
- `eval_classifier.py` — 在收集数据上报告准确度 + 动力学比统计

---

# 条件密度方案（suboptimal source + CFG）—— 诚实的负面/诊断结果

动机：判别器在 OOD/suboptimal 状态上会幻报高 gap（step5/6）。用户洞察：条件密度
`p(Δs'|s,a,c)` + CFG 应能把这块"拉住"——机械臂乱动时 source head 认识它、target head
经 CFG 退回 ∅≈source → DGI≈0（实际真 gap 也≈0）。用新收集的 suboptimal 源域 replay
(`threading_d0_source_replay`，1082 ep/213k 步，含 RL 探索乱动) 作为 c=0 的广覆盖。

## 实现（`density_data.py` / `conditional_density.py` / `train_cdm.py` / `eval_cdm.py`）
- c=0 = replay transition（无需 sim_state，本就是 source 动力学）+ target demo 的反事实
  source-step anchor；c=1 = target demo 真实下一步。只对 Δs'=s'−s 建模，(s,a) 纯 context。
- 两种 head：CondGaussian（μ,logσ）、CondDiffusion（DDPM ε，CFG dropout）。
- 三种 gap 信号对比：判别器 `|log ratio|`；两-head DGI `‖E[Δs'|tgt]−E[Δs'|src]‖`；
  source-only 残差 `‖Δs'_norm − μ_src(s,a)‖`。

## 结果（`eval_cdm.png`）——假设未成立
| 信号 | (A) anchor rho↑ | (C) replay 均值 vs 真gap | (B) flailing 均值 vs 真gap |
|---|---|---|---|
| 判别器 | 0.24 | 0.211 > 0.181 | 喷射 0~3.5（幻报） |
| 两-head DGI | 0.27 | 0.475 > 0.392 | 喷射 0~0.95（幻报） |
| source 残差 | 0.15 | 3.29（被 aleatoric 主导） | 喷射 0~52（幻报） |

**三种方法在 flailing（真 gap≈0）上全部幻报**：分数喷射到从低到极高，大量超过真 gap 的
anchor 分数。条件密度并未自动把 OOD 拉到 0。

## 根因诊断（关键）
1. **OOD 退回 ∅≈source 不会自动发生**：MLP 条件模型在无 target 数据的 OOD 上任意外推；
   CFG dropout 只是让模型"能"产生 ∅，并不强制 c=1 在 OOD 等于 source。
2. **gap 信号 < source aleatoric 方差**：raw GT gap 中位 ~0.007，而源域 Δs' 因动作探索本身
   方差大；"surprise"(NLL/残差) 被 aleatoric 噪声淹没（残差地板 3.3 > gap 信号 1.6–2.0）。
3. **两-head 相减引入 miscalibration 地板**：source head（17 万样本）拟合好、target head
   （1583 anchor）拟合差，均值差在无 gap 区也有 ~0.4 恒定地板。
4. **signed-gap 直接回归塌缩**：`g(s,a)→(s'_tgt−s'_src)` 因目标幅度太小+噪声，MSE 退化为
   处处输出≈0（rho=nan，无判别力）。
5. **"OOD 鲁棒来自 source 覆盖"只在测试分布⊂source 训练分布时成立**：flow-BC+噪声 flailing
   对 replay 训练的 source 模型也是 OOD → 一样幻报。

## 结论与可选方向
反事实配对的判别器/CDM 在 **target 有支撑处**能检测 gap（rho~0.3），但**没有一种方法能可靠地
把"无 gap"外推到 OOD/suboptimal**——因为这要么需要 OOD 处的 target 数据（拿不到），要么需要
gap 可从 aleatoric 噪声中分离（密度法做不到）。可能的真正解法：
- (a) **显式 OOD/密度门控**：估计 (s,a) 在 target 支撑下的密度，OOD 处按设计输出"未知/0"，
  而非寄望模型自发退回。代价：只在 target 支撑内报 gap（覆盖受限，但诚实）。
- (b) **扩大 target 覆盖**（更多/更宽的 demo）——真正的瓶颈。
- (c) **variance-aware gap**：相对 source aleatoric 做 z-score，需大量噪声采样。

## 条件密度方案文件
- `density_data.py` / `conditional_density.py` / `train_cdm.py` / `eval_cdm.py`
- `step8_suboptimal_test.py` — flailing 上对比判别器 vs CDM DGI
- `cdm_data.npz` / `cdm_gaussian.pt` / `cdm_diffusion.pt` / `eval_cdm.png`

---

# 残差参数化 + 对比 + 直接 gap 监督（用户四点方案的实现与权衡）

针对上一节的两个失败模式（OOD 幻报、gap<<aleatoric），实现用户提出的四点改进：
- 一 conditional diffusion（只对 Δs' 加噪、(s,a) 当 condition）：**已是 CondDiffusion 的做法**；
  DGI 改为能量(去噪误差)之差以符合 `|E(d=1)−E(d=0)|`。
- 二 zero-init 残差：`μ(c=1)=μ_base+Δμ`，Δμ 末层零初始化+强 weight decay。
  `CondGaussianResidual`，DGI=‖Δμ(s,a)‖。
- 三 反事实 mixup：**已是 anchor pool**；并用干净的 sim2sim gap 向量 `(s'_tgt−s'_src)/σ`
  直接监督 Δμ（`--direct_gap`，近确定性物理→干净目标非噪声）。
- 四 对比能量 margin（gap-aware，只在高 GT gap anchor 上）。

## 结果与权衡（`train_cdm_res.py`）

| 配置 | in-support rho | flailing OOD vs anchor-hi |
|---|---|---|
| 残差+NLL（res_wd=1e-2）| 0.14 | ✅ 抑制：flailing 0.82 < anchor-hi 2.21 |
| 残差+direct_gap（res_wd=1e-3）| **0.48** | ❌ 爆炸：flailing 18 >> anchor 3 |
| 残差+direct_gap+l2_delta=0.3 | 0.40 | ❌ flailing 3.8 > anchor-hi 3.1 |

## 关键结论
1. **direct_gap（三，用 sim2sim gap 向量监督残差）把 in-support Spearman 从 0.33 → 0.48**，
   是真实进步——清楚地区分 hi/lo gap。
2. **二（隐式 zero-init→OOD 退化）不可靠**：训练后的 MLP 残差在 anchor(demo) 流形外一律
   **外推高值**（flailing、replay 都是 OOD），零初始化只定起点不定外推行为。强正则能压 OOD 但
   同时杀 in-support 信号——无法同时兼得。
3. **四（对比 margin）非约束性**：残差在小 anchor 集上轻易满足 E0−E1≥M，ct→0，没起作用。
4. 连无 gap 的源域 replay 都被 direct_gap 残差误报 ~2.9，说明问题是**anchor(demo) 流形窄、
   残差对任何 off-manifold (s,a) 都外推**——不限于 flailing。

## 仍需的最后一块：显式 support 门控
隐式退化失败 ⇒ 需**显式 OOD/support 门控**：`DGI_final = DGI_raw · gate(s,a)`，gate 在 (s,a)
离 anchor/source 支撑远时→0（可用 base 模型重建误差或 (s,a) kNN 密度，flailing base-err 7.0 vs
in-dist 3.3 可分）。代价：gap 检测只在 demo 流形附近可信——对"部署策略贴近 demo"的 sim2real
可接受。direct_gap(rho 0.48) + 显式门控 是当前最有希望的组合。

## 文件
- `conditional_density.py::CondGaussianResidual` — 残差参数化 + 能量/DGI
- `train_cdm_res.py` — 二/三/四（`--direct_gap` `--l2_delta` `--contrast` `--margin` `--res_wd`）
- `cdm_res_directgap.pt` — best in-support (rho 0.48) 模型
