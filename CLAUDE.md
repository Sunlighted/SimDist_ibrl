# CLAUDE.md — 索引 + 速查

本仓库 = fork 自 upstream ibrl 的 **sim2sim 跨域迁移**研究代码。
**当前主线：SimDist（latent world model + MPPI）的改进，以及在多个任务上的验证。**

**这个文件只放"每次都要知道的东西"**（当前状态 / 方法论铁律 / 去哪找细节）。
详细笔记按主题拆在 `docs/` 下，用到再读；已停止推进的分支在 `CLAUDE_ARCHIVE.md`。
写新笔记时请写进对应的 `docs/*.md`，只有当结论会改变"下次该怎么做实验"时才回填这里。

Last updated: 2026-09-04

---

## 文件地图

| 文件 | 行数 | 什么时候读 |
|---|---:|---|
| **`docs/findings.md`** | ~1415 | **做任何改进决策前**。天花板律（§27）、dynamics ensemble 0.962（§31，⚠️ 末尾补了「增益与 anchor 不可加」）、value 悲观判死（§29）、value adapt 判死（§33）、latent vs raw（§30）、proposal（§36）、容错窗口旋钮（§39）。**里面有一整批已被实测判死的方向，别重复。** |
| **`docs/simdist.md`** | ~1330 | 改 SimDist pipeline 时。§17 本体 · §24 两阶段 target adapt · §25 residual/IBRL on frozen base · §26 省交互变体 · §42 stage-2 的 source 锚 `bc_anchor_coef` · **§43 anchor 形式轴已封闭(别再试换散度)** · **§44 anchor 的价值由 dynamics 精度定(不是 base 分数)+「锚向谁」是独立于 bc_loss 的渠道** |
| **`docs/demo-collection.md`** | ~430 | 要（重）采 stage-2 的 target demo 时。★ 先读 §38。 |
| **`docs/tasks.md`** | ~790 | 加新任务 / 动 Square·Coffee 时。§32 Square · §40 Coffee · §41 Square rotw1 全链条。 |
| **`docs/infra.md`** | ~410 | 查旋钮和 API 时。§3 domain shift · §4 flow BC · §5 eval 基础设施 · §5b BC baselines · §5c strict success · §6 配置表 · §7 已知坑 · §12 Threading target 定义。 |
| **`docs/setup.md`** | ~165 | 装机 / 换机器 / 转数据集时。 |
| **`docs/on-policy-adaptation.md`** | ~190 | **9 组同预算大对照（100k transitions）+ 总结论：on-policy 一致性是 model-based adaptation 的支配性质。** planning ≫ Dyna（同一 world model 0.813 vs 0.157）；value adapt 三种顺序全崩；dynamics 只在 policy 邻域可修；source 锚 +0.081 且方差收缩 4.3×。 |
| `docs/flow_qc_ug.md` | ~110 | （早于本次拆分就存在）flow actor + Q-chunking + UG 的在线 model-free RL 方案书 —— "不再修 SimDist、改把 policy 做强"那条支线。 |
| **`CLAUDE_ARCHIVE.md`** | ~2020 | 只在需要回查已归档分支（§10/11/13/14/15/16/18/19/20/21/22/23）时。代码在本地 `_archive/`，不随仓库发布。 |

---

## 速查

### 三个任务的当前状态（σ=0.1 / target = 各自的 §12/§32/§40 配方；strict 仅 Threading·Coffee，**Square 线上 `strict_success.enabled` 全是 false**，见 tasks §41b）

| 任务 | source 专家 | target 专家 | SimDist adapt | 详见 |
|---|---:|---:|---:|---|
| **Threading_D0** | IBRL 0.98 | model-free IBRL **0.9815**(n=2000) | **dynens 0.962**(n=2000) / plain 0.879 | findings §31 |
| **Square_D0 (rotw1)** | 0.98 | **0.98–1.00** | anchor **0.837**(10seed) / 0.886(d600) | tasks §41 · **§41b 拆账** |
| **Coffee_D0** | 0.998 (σ.02) / **0.938** (σ.1) | **0.982** (C50) / 0.96 (C75) / 0.93 (G5x) | **0.907**(G5x, 2seed) = oracle 0.94 | tasks §40–40e |

Threading target 的参考线阶梯（同一 harness，n≥300）：
`policy 头裸跑 0.37–0.42` < `plain SimDist MPPI 0.88` < `dynens MPPI 0.962`
≈ `residual on frozen base 0.96` ≈ `prior-margin IBRL 0.94–0.97`
< `model-free IBRL 0.9815` < `oracle dynamics 0.993`。

### 三条最重要的机制结论
1. **天花板 = f(模型误差 / 任务容错窗口)**（findings §27，因果剂量-响应已标定：
   learned ≈ oracle + 5.6 mm 持久误差；且只有"持久/相关"的误差有害，iid 被闭环平均掉）。
   ⚠️ §39 补了下限：窗口 < ~1.3 mm/边时，**即使 dynamics 完美也做不到**。
2. **压低模型误差是唯一有效的方向**：dynamics ensemble（5 头平均，§31）把 0.88 推到
   **0.962**，是第一个在纯 model-based planning 内部越过 0.85 的东西；≈3 头饱和
   （剩下的是 bias 不是 variance）。**悲观（value 或几何分歧）全部无效**（§29/§31）。
   ⚠️ **但这个增益与 stage-2 的 source anchor 强互补、不可加**（simdist §44 的 2×2）：
   `anchor λ=0` 时 5 头集成只值 **+0.019**，λ=1 时值 +0.135；反过来 anchor 在 plain base 上
   只值 +0.055、在 dynens 上值 +0.172（交互 t=4.2）。**报任何一边的增益都要写清另一边是开还是关。**
3. **value 不要动**（§33）：MPPI 的候选区分信号只有 `std_i(γ^H V)=0.0068`（量程的
   1.4%），任何离线重拟合的残差都同量级 ⇒ elite 被改写 ⇒ 崩。约束是"value 必须与
   **延续策略**一致"，不是绝对准确（§41 在 oracle dynamics 上独立复现）。
   ⚠️ 限定：dynamics 越准这个约束越松（dynens 上换 value 只花 0.02–0.05）。

### ★ 方法论铁律（每一条都是踩出来的，违反过的都翻过车）
1. **先量噪声地板再谈效应。** 已知三个尺度：单次 20 条 demo 采集的 adapt
   **σ≈0.092** > 同配置 replicate **±0.027** ≈ 基线 seed 跨度 **0.035**。
   ⇒ **<0.05 的"改进"一律不可读**；两次单抽样要差 >0.26 才显著。
2. **±0.1 以内的结论必须 300–500 局。** n=30/40/100 在本项目里**反复翻符号**
   （−0.075→+0.097；0.933→0.824；94%→89.2%）。
3. **事后发现的规则必须换新 seed 预注册复现。** winner's curse 实测抬高 **2.4×**；
   一个 "+0.018 的峰"复现后是 −0.011（符号翻转）。
4. **adapt/训练曲线要等两臂都进稳态再比。** 慢的那条会有瞬时峰：中途窗口给出过
   −0.025 而稳态是 +0.103；另一处 2×2 的交互项从 +0.151 翻到 −0.136。
5. **评 planner/argmax 只能在模型自己搜出的 on-policy 分布上测**，且看 `sup|ε̃|` +
   `Var_s[ε̄]`，不看 MSE/R²/Spearman 这些均值型指标（分布漂移陷阱，findings §27#5）。
6. **离线指标与闭环双向脱钩**：§33「极小的离线残差照样杀死你」，§41「极大的离线排序
   分歧完全无害（84% 偏差是逐状态常数 ⇒ CEM 免费）」。**唯一判据是跑 eval。**
7. **数字太漂亮或太丑，先查工具再解释。** 救过场的三次：`probe_env` 从未 reset ⇒ 反事实
   跑在 source 物理上；`evaluate()` 硬编码 `planner.plan` ⇒ "planner 净贡献精确 0.000"；
   `dV=0.0000` ⇒ 两域快照其实是同一批状态。
8. **`load_frozen_agent` 强制 `act_method="rl"`，摘掉 prior。** 已咬 4 次。要 hybrid 必须
   显式 `source_agent_use_hybrid=1`（默认 0）；确认方式 = 去 `<save_dir>/train.log` 找
   `[SGFT] prior = source HYBRID`。
9. **一次只改一个参数。** planner 超参**跨任务不可移植**（§40c：`init_std` 一个旋钮值 40 点），
   **跨"oracle vs learned dynamics"也不可移植**（§40d）。换设置先重扫 `init_std` /
   `base_policy_std`，并打印 `std_i(γ^H V)/V_level` 与 Threading 的 0.0068 对照。
   ⚠️ `base_policy_std=0` 时 base 候选是**逐位相同的复制品** ⇒ `mixture_coef` 大就等于
   扔掉大部分候选预算（但它同时也是把 CEM mean 钉在 proposal 上的信任域，见 §41）。
10. **代理指标会给反向结论**：`policy_only` 两次给出 z=−5.3/−14 的反向判断；
    一度记为"已验证"的 `contact_steps` 在 n=13 上从 ρ=+0.855 崩到 −0.209。
    **判据必须是最终目标（跑完 adapt 的 target 分数）。**
    ⚠️ 推论（§43）：**两个共线的候选解释，必须专门造一个把它们分开的臂**，否则拟合再好也归错因 ——
    尾部厚度与 `bc_loss` 的 Spearman 是 0.90，单变量干预后尾部的效应量是 **0**。
11. **`nohup` 挡不住 Bash 工具的 timeout(143)**（子进程同进程组）⇒ 长任务必须
    `setsid ... </dev/null & disown`，且不要在同一条命令里跟 `sleep`。
12. **`num_eval_proc` 必须整除 `num_eval_episode`**（mp eval 断言，已踩两次）。
13. **采 demo 必须显式传全部 shift 旋钮**：`collect_target_demos.py` 的 `--hole_offset`
    (默认 0.01) / `--solref_scale` (默认 0.8) / `--solimp_scale` 是**无条件覆盖** ckpt 的，
    不传就静默退回旧 target；几何旋钮与 init_distribution 同理。
14. **`TopkSaver` 会覆盖 `model0.pt`** ⇒ 用它收的数据事后不可考（那个 0.952 之谜）。
    收数据时把 ckpt 快照 / sha256 写进 sidecar。另：`model0.pt` = 单次 50/100 局
    in-training eval 的最高分，系统性乐观 0.06–0.19 ⇒ **headline 一律用多点均值或独立大样本重评**。
15. **`evaluate/eval.py` 的 MPPI eval 自带 ±6 点噪声**：同 ckpt/同 planner/同 seed/同 100 局
    两次测出 86% 与 92%（planner 候选走 torch RNG，只有 episode 级 `np.random.seed` 受控）
    ⇒ **<12 点的 planner 比较不可读**（tasks §40e）。
16. **stage-2 的无正则 BC 会摧毁 proposal，破坏力 target-dependent**：1000 步 × batch256
    ≈ 157 epoch 打在 1633 个 demo transition 上，且**代码里零正则**。`bc_anchor_coef≈1`
    是修法，实测把 round-0 抬 **+0.35~+0.44**（三个 Coffee target 一致）。⚠️ 但 λ=0 的破坏
    幅度差 22 倍（C75kp25 round-0 **0.02** vs G5x **0.44**）⇒ 换 target 必须重测 λ=0。
17. **eval 脚本的环境跟着 ckpt 走 ⇒ 换 agent 会静默换任务。** `train_rl.load_model(weight)`
    连 `env_params` 一起返回，所以 `--weight` 一换，**σ 和 `eval_init_dist` 也跟着变**。
    §41 那张 2×2 就是这么废掉的（π_src 行 σ=null/rotw1，π_tgt 行 σ=0.1/全360°），
    整整一条"换 value 就崩"的结论是假的。**跨 ckpt 比较前必须逐行核对 `[setup]`/`[EVAL ENV]`
    里的 σ / init_dist / strict / planner 全参数**；`eval_square_oracle_mppi.py` 现在会全打印。
    ⚠️ 同理 `bool(cfg_object)` 恒为 True —— 判开关要读 `.enabled`（我自己刚踩过）。
18. **接近天花板时，加法尺度上的 null 不含信息量。** 报效应量必须同时给"剩余差距缩小的比例"
    （或 log-odds）。实例：anchor 在 learned 上 +0.025（基线 0.820）、在 oracle 上 +0.005
    （基线 **0.965**，离 1.0 只剩 0.035）—— 加法看像"机制消失"，但**差距缩小比例两边一样**
    （13.9% vs 14.3%，log-odds +0.18 vs +0.16）⇒ 那个 null 是**算术必然**，不是证据。
    与 §42 强 base 那格同型（那次是 CI 覆盖 0 被读成"无效"）。**要证机制消失，必须先造出余量**
    （例：oracle 上削弱 planner 让基线掉到 ~0.75，模型误差仍为零但余量 0.25）。
19. `evaluate/eval.py` 每次打印 `[EVAL ENV] task= shift= nut_z_rot= sigma= strict= ep_len=`
    —— **两个数只有这行一致才可比**。

### 常用命令
```bash
cd /home/Chenyang/ibrl && source set_env.sh          # conda + PYTHONPATH + MUJOCO_GL
python -m pip install ...                            # 不要直接用 pip（shebang 是坏的）

# SimDist：stage-1 蒸馏 → stage-2/3 target adapt
python train_simdist.py --config_path release/cfgs/mimicgen/threading_d0_state_simdist.yaml \
    --source_agent <src ckpt> --save_dir exps/mimicgen/<run>
python train_simdist_adapt.py --stage1_ckpt <stage1>/latest.pt --save_dir <adapt dir> \
    --planner_set "init_std=0.15,base_policy_std=0.0"      # ⚠️ 非 Threading 必须重扫

# 独立评测（±0.1 内的结论 ≥300 局）
python evaluate/eval.py --weight <ckpt> --mode simdist --num_game 500 --mp 10 \
    --shift_from_cfg 1 --strict_success 1 --hand_init_perturb_joint_sigma 0.1

# 长任务
setsid python train_xxx.py ... > run.log 2>&1 < /dev/null & disown
```

---

## 章节索引（节号在拆分后保持不变，全局唯一）

**`docs/findings.md`**
- **§27** MPPI 天花板 = 模型误差 / 容错窗口。oracle 1.000 vs learned 0.85；只改容错
  （source 物理 + 小环）与只改模型（oracle）两个正交单变量夹死结论；oracle+注入噪声的
  因果剂量曲线（**只有 smooth 有害，iid 被平均掉**）。附一整批被推翻的中间结论。
- **§29** value ensemble（MRS.Q 式悲观）**判死**：蒸馏架构下头会坍缩；RPF 无甜点区；
  根因 = gap 在 `P(s'|s,a)` 不在 `d(s)`，value 只看 s ⇒ 结构性失明。
- **§30** latent vs raw：**规划用 latent（+0.10~0.15）、策略改进用 raw**（latent 里
  residual 学不起来，q_mean 死平）。⚠️ mm 级预测误差对 planner 成绩零预测力（骗了 3 次）。
- **§31** ★ **dynamics ensemble → 0.962**。归因隔离：5 头 0.970 vs 单头 0.120 ⇒ 增益
  100% 来自 plan-time 集成平均，与 encoder 无关；≈3 头饱和；几何分歧惩罚 ≤0.2 零效应、
  ≥0.33 有害；vs model-free IBRL / oracle 的大样本对决。
- **§33** value adapt 判死：信号 0.0068、CEM softmax 本就失效、唯一起作用的渠道是 topk
  成员选择；剂量-响应 Spearman=−1.000；**dynens 上跌幅只有 1/8.7**（共适应重要性 ∝ dynamics 误差）；
  什么时候才真的必须 adapt value（答案：真 OOD，不是可达性）。
- **§36** flow proposal vs UG vs MPPI：**换 proposal +0.138 显著、换搜索 ns**；UG 贵 1.6× ⇒
  被 flow+MPPI 双轴 dominate；GPU 上"宽"免费"深"要钱；proposal 只在 binding 时值钱（交互 0.160）。
- **§39** `needle_size_scale` = `ring_size_scale` 的镜像（只通过插入间隙起作用）；
  **oracle 的 1.000 只在窗口 ≥1.3 mm/边成立**。

**`docs/simdist.md`**
- **§17** SimDist 本体：world model 架构、stage-1 蒸馏配方、raw-state 对照臂、
  offline dynamics trainer。**五个稳定性坑的完整病历**（value 塌成 0、policy detach、
  采集噪声必须 ~0、latent 爆炸 → SimNorm、policy 与 dynamics 抢 encoder → `enc_pi`）。
- **§24** target adapt：stage-2 只训 policy 头、stage-3 只训 dynamics 头，encoder 全程冻结。
- **§25** residual on frozen adapted base（ResFiT，base action 存进 buffer 而非重规划）
  + IBRL-with-same-base 对照臂。
- **§26** 省交互 3 段式（30k adapt + 复用其 replay 做 IQL）；**AWR 在这份数据上必然退化成
  BC（adv_std 0.011 vs Q_std 0.258）⇒ 用 RWR**；margin 崩塌的真正归因。
- **§42** ★ stage-2 的 policy 受**两个竞争条件**约束（靠近 π_src ⊗ target 成功率），
  `bc_anchor_coef=λ≈1` 在弱 base 上 **+0.164 (t=112.6)**、强 base 上测不到 ⇒ 交互 t=5.2。
  ⚠️ 「只对弱 base 值钱」这个推广已被 **§44** 推翻（调节变量是 dynamics 误差，不是 base 分数）。
- **§43** ★ **anchor 形式轴封闭**：现役 MSE **就是** KL（等协方差高斯下 W2/JS/Hellinger/χ² 全是
  同一标量的单调函数 ⇒ 换散度是空操作）。53 标定格 + 12 adapt run 证出**形式只通过一个渠道起作用**：
  `score ≈ 1.065 − 11.1·bc_loss`（R²=0.93，残差 sd 0.010 < seed 噪声）。度量必须是 Fisher（参数空间
  L2 付 3.4× 代价 ⇒ −0.277）；目标必须是 π_src（weight decay 结构性够不到）；**罚 vs 硬约束、尾部形状、
  偏差落在哪个相位全部无因果**。★ 尾部：Spearman 0.90 但干预后效应 0 —— 共线变量必须造臂分开。
- **§44** ★★ anchor 的收益由 **dynamics 精度**定(plain +0.055 vs dynens +0.172,交互 t=4.2)
  ⇒ §42「弱 base 增益大」的推广被推翻(两个 base 的 λ=0 分数几乎相同);**anchor 目标必须是
  π_src 本身** —— 换成数据集记录的 action 在 `bc_loss` 匹配下仍掉 −0.081 (t=2.5) ⇒
  `D_SRC` 是独立于 `bc_loss` 的第二条渠道(§43 只测了后者,因为它把前者标定固定了);
  data-action 的 D_SRC 下界 0.151(靶点 6.3×,λ 拉 100× 不动)= 结构性不可达,与 F2 同型;
  `policy_only` 0.42 → 0.56(π_src anchor) vs **0.23**(data anchor)。

**`docs/demo-collection.md`**
- **§35** 什么属性决定 adapt：成功率**无关**（ρ=+0.20）；4×4 交叉析因（policy 53% /
  dynamics 17% / 交互 30%）；两个 provenance 陷阱作废了一批解读。
- **§37** 抽样方差 + source 覆盖。⚠️ **两条核心结论已被 §38 推翻**，读之前先读 §38。
- **§38** ★ **先量噪声地板**（σ=0.092）；唯一经预注册复现存活的干预 = **用尚未触顶的
  策略采集（+0.139）**；判据不是"策略现在多强"；20+ 个筛选判据全灭。

**`docs/tasks.md`**
- **§32** Square_D0：纯 IBRL 600k → 89.2%，换 RNN-BC prior → **97.8%**；
  **别用 prior_margin**（会拉低最终性能）；target A/B 设计 + 迭代 bootstrapping 到 94.2%。
- **§40** Coffee_D0：source 一轮打通 **99.8%**；更强 prior 的杠杆在这里**不存在**；
  target 用**动力学 gap**选而不是成功率选；**§40c/§40d planner 超参不可跨任务/跨 dynamics 移植**。
- **§40e** ★ Coffee 三 target 横评：**G5x + `bc_anchor_coef 1.0` → 0.907 = oracle 0.94**
  （首次追平完美 dynamics）。**杠杆是 stage-2 的 anchor（+0.40），不是 stage-3 adapt（+0.03）**；
  选 target 看 `stage-1 MPPI 分数`不看 oracle；**难度必须与 gap 同相位且占数据主体**
  （C50 唯一崩掉：难度 86% 在只占 4% 转移的 close-lid 相）。附 6 个被证伪的假设 + 两个测量地板。
- **§41b** ★★ 受控重测推翻 §41 的 2×2：**缺口是 dynamics 不是 value**。同预算 dynamics 误差
  **+0.175 (z=5.1)** ≫ proposal +0.030 > **value +0.015 (ns)**，任务残差 0.015；value 的效应是
  **proposal-conditional**（弱 proposal 上 +0.330，target-BC 过的 proposal 上归零）；oracle 下
  "value 需与延续策略自洽"被证伪（F>E +0.060）；蒸馏 V̂ > source 专家 critic +0.085。
  ⚠️ Square 线上 strict 全 false；anchor 臂 seed2/seed1 的 demo 集不对齐。
- **§41** Square rotw1 全链条；**MPPI 信任域 = `init_std` + `base_policy_std` 两堵墙**；
  planner 的价值完全由 dynamics 精度定（oracle +0.63 vs 训练域内 learned +0.015 ns）；
  蒸馏 V̂ 可直接替代 V_src；⚠️ 两个测量地板（episode 级不可复现、seed 跨度 0.035）。

**`docs/infra.md`** — §3 domain shift 框架与全部旋钮 · §4 flow BC · §5 source/target eval
（σ 初始化、TopkSaver、CLI flag）· §5b RNN-GMM/Diffusion · §5c strict success ·
§6 三任务配置表 · §7 已知坑 · §12 Threading target 的 dynamics-only 定义。

**`docs/setup.md`** — §0 概览 · §1 环境（torch 2.7/cu128、mujoco 2.3.2、submodule）· §2 MimicGen 集成与数据转换。
