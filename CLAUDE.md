# CLAUDE.md — 索引 + 速查

本仓库 = fork 自 upstream ibrl 的 **sim2sim 跨域迁移**研究代码。
**当前主线：SimDist（latent world model + MPPI）的改进，以及在多个任务上的验证。**

**这个文件只放"每次都要知道的东西"**（当前状态 / 方法论铁律 / 去哪找细节）。
详细笔记按主题拆在 `docs/` 下，用到再读；已停止推进的分支在 `CLAUDE_ARCHIVE.md`。
写新笔记时请写进对应的 `docs/*.md`，只有当结论会改变"下次该怎么做实验"时才回填这里。

Last updated: 2026-09-11

---

## 文件地图

| 文件 | 行数 | 什么时候读 |
|---|---:|---|
| **`docs/findings.md`** | ~1415 | **做任何改进决策前**。天花板律（§27）、dynamics ensemble 0.962（§31，⚠️ 末尾补了「增益与 anchor 不可加」）、value 悲观判死（§29）、value adapt 判死（§33）、latent vs raw（§30）、proposal（§36）、容错窗口旋钮（§39）。**里面有一整批已被实测判死的方向，别重复。** |
| **`docs/simdist.md`** | ~2700 | 改 SimDist pipeline 时。§17 本体 · §24 两阶段 target adapt · §25 residual/IBRL on frozen base · §26 省交互变体 · §42 stage-2 的 source 锚 `bc_anchor_coef` · **§43 anchor 形式轴已封闭(别再试换散度)** · **§44 anchor 的价值由 dynamics 精度定 + 唯一充分统计量是 `policy_only`** · **§44b 轴②(在哪些状态上锚)也封闭** · **§47 r18@224 梯子：anchor 复现(+0.432)、ensemble 没有；encoder 的 0.5 是交互；铁律 #14 的选择偏差补充** · **§48 5-seed 重做（跑中）：§44 的三项增广已实现 + `DEMO_OFFSET` 解除同一 draw 的封锁；读法与 Phase 2 已预注册** |
| **`docs/demo-collection.md`** | ~430 | 要（重）采 stage-2 的 target demo 时。★ 先读 §38。 |
| **`docs/tasks.md`** | ~970 | 加新任务 / 动 Square·Coffee 时。§32 Square · §40 Coffee · §41 Square rotw1 全链条 · **★§41c 模型误差决定别的部件重不重要（oracle 列极差 0.07 vs learned 列 0.64）**。 |
| **`docs/infra.md`** | ~490 | 查旋钮和 API 时。§3 domain shift · §4 flow BC · §5 eval 基础设施 · §5b BC baselines · §5c strict success · §6 配置表 · §7 已知坑 · §12 Threading target 定义 · **★§13 三个任务 target 的唯一真源（跑任何跨臂对比前必读）**。 |
| **`docs/setup.md`** | ~165 | 装机 / 换机器 / 转数据集时。 |
| **`docs/on-policy-adaptation.md`** | ~190 | **9 组同预算大对照（100k transitions）+ 总结论：on-policy 一致性是 model-based adaptation 的支配性质。** planning ≫ Dyna（同一 world model 0.813 vs 0.157）；value adapt 三种顺序全崩；dynamics 只在 policy 邻域可修；source 锚 +0.081 且方差收缩 4.3×。 |
| `docs/flow_qc_ug.md` | ~110 | （早于本次拆分就存在）flow actor + Q-chunking + UG 的在线 model-free RL 方案书 —— "不再修 SimDist、改把 policy 做强"那条支线。 |
| **`docs/pixel-progress.md`** | ~112 | **做 pixel/视觉版之前必读 —— 现在是完整实验日志。** 所有已测数字：stage-1 四个 checkpoint（pixel 0.968/0.980 ≥ state）、target adapt 的 2×2（state 0.540 / pixel 0.028；dynamics-only 0.393 vs 0.023）、demo-count 扫（pixel 0.08→0.24，**state 0.360→0.75**）、latent probe（含它**不能**支撑的结论）、编码器 bench、环境坑。 |
| `docs/pixel-gap-closing-prompt.md` | ~230 | 要接着推 pixel 的时候：交给 agent 的完整任务书（Phase A 增广 / B 数据对齐 state / C 224+ResNet-18），含活跃任务共用 checkout 的风险与回归门。
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
19b. **target 的唯一真源 = SimDist stage-1 的 cfg**（infra §13）。adapt/dyna 臂的 env 来自
    `stage1_ckpt` 同目录 cfg（`train_simdist_adapt.py:398`），model-free 臂必须复制它。
    ⚠️ **Coffee 有两套 stage-1，目录名不提示哪套是 G5x**：`coffee_s1g5x_*` 才是
    （31 个既有 run 都用它），`coffee_simdist_*_h5_seed1` 是 C50 时代的另一个 target。
    **判断 stage-1 属于哪个 target，只能回溯已完成 run 的 `stage1_ckpt`，不能看目录名。**
    ⚠️ `train_rl.py` 不传 `--config_path` 就静默拿到三个错的默认值：`domain_shift.mode="none"`
    （source 物理）、`act_method="rl"`（prior 不进 argmax、`prior_margin` 变死代码）、
    critic `10×512`（全仓库其他 IBRL run 都是 `5×1024`+ln）。三者都不报错，已各咬一次。
19. `evaluate/eval.py` 每次打印 `[EVAL ENV] task= shift= nut_z_rot= sigma= strict= ep_len=`
    —— **两个数只有这行一致才可比**。
16. **SLURM 时限要卡在分层线以下**（Killarney：b1 ≤3:00:00 / b2 ≤12:00:00 / b3 ≤1-00:00:00
    / b4 ≤3-00:00:00）。刚好超线极贵：一个 9 分钟的活要了 4:00:00 → 落 b2 → 等了 15h23；
    3 天的活落 b4（42 节点）等 8-15h，而组里中位数是 **1 分钟**。CPU/内存也要小（8-12 核
    / 32G），大申请会缩小「GPU 空闲且槽位匹配」的节点集合。时长一律从 `sacct -o Elapsed,MaxRSS`
    量出来，别猜。
17. **BC prior 的推理成本会乘进 IBRL 的每一步**：IBRL 每次选动作都查 prior，所以 flow
    （10 步 Euler ODE/次）比 RNN-GMM（一次前向）慢一个数量级 —— 实测同样 400k steps，
    **RNN 6h23 vs flow ≥3 天（超时未完成）**。Threading 用 RNN。§32 早就写了「flow 没跑（太慢）」。

18. **`load_pretrained_agent` 是「权重热启动」，不是 resume ⇒ 对 IBRL 等于重启。**
    `train_rl.py:402-433` 只搬 actor+critic 权重，`global_step=0`、replay 空、optimizer
    state 全新，于是 critic 从 25 条几乎全失败的 episode 重新拟合，第一次 eval 之前就把
    继承来的 policy 拖走。实测：target run#1（warm from source，400k）0.00→210k、
    0.02@250k、0.26@370k、best **0.32**；c2 从 run#1 的 0.32 热启动，170k 时又回到
    **0.00** —— 逐点重走 run#1 的曲线，收益为零。⇒ **不要把长训练拆成 400k 的 warm-start
    链**，要么一个长 job（1M ≈ 15 h ⇒ `--time=23:59:00`，b3 84 节点），要么实现真 resume
    （global_step + optimizer + replay 三样都存）。stage-1 已经有真 resume，`train_rl.py` 没有。
19. **`\` 续行的下一行绝不能是 `#` 注释**：注释会终止整条逻辑行，**后面所有 flag 静默丢失**，
    再下一行还会被当成独立命令执行。踩法：pixel adapt smoke 因此退回默认
    `max_transitions=20000` + 默认 eval 规模，2 分钟的 smoke 变成 3 小时的真跑，且
    `rc=0`。注释写到命令上方的 banner 里。全仓扫描脚本见 §本次 commit（当时只有这 1 处）。

20. **「跑在哪一级台阶」由 stage-1 架构决定，adapt 时改不了。**
    `release/cfgs/.../threading_d0_state_simdist_tdmpc2.yaml` **出厂就是
    `horizon:3 + num_dynamics_heads:1`**，即 `plain_h3` ⇒ 参考值只有
    **0.564 ± 0.171**，不是 0.88/0.979。台阶分解（on-policy-adaptation.md）：
    `plain_h3 0.564` --**+0.249 dynens_h5**--> `0.813` --+0.085 未触顶 demo-->
    `0.898` --+0.081 anchor λ=1--> **`0.979`**。
    ⇒ **dynens 是最大的单项干预**，而 `num_dynamics_heads`/`horizon` 在
    `world_model._build` 里建头、`load_state_dict` 是 strict ⇒ **只能重跑 stage-1**。
    用 `*_dynens.yaml`（wm.horizon=5 / wm.num_dynamics_heads=5 / planner.horizon=5，
    对齐 `square_d0_state_simdist_rotw1_dynens.yaml`）。
    ⚠️ `_slow` **不是训练旋钮，是 demo 属性**（§38 未触顶采集），`_anchor` = `bc_anchor_coef=1`。
21. **参考分数是「75–100k 窗口内 100-game eval 的均值」，不是最后一个 eval。**
    协议 = 100k target transitions / 每 1k 一个 eval 点 / 每点 100 局 / strict / σ=0.1
    （on-policy-adaptation.md:3-8）。`adapt.slrm` 旧默认 `max_transitions=20000`
    **少了 5×预算**，且 300 局/点 ⇒ 与任何已发布台阶都不可比。已改默认。
    成本实测：state ~275 s/round ⇒ 7.6 h（11:59 够）；pixel ~480 s/round ⇒ 13.3 h ⇒ 必须 23:59。

22. **分块长任务的 requeue 必须在 python **之前** 用 `--dependency=afterany` 挂好，
    放在 python 后面的任何 requeue 代码都不会执行。** chunk 的设计终止方式就是 walltime
    TIMEOUT，而 walltime kill 会杀掉**整个 job script** ⇒ python 之后的行永远轮不到。
    我为此栽了两次：第一次以为是 rc 判据不对（把 rc 改成 progress 判据），第二次才发现
    **位置**才是根因 —— plain pixel arm 卡在 780000/800000，log 里没有 `requeueing`，
    队列里也没有后继。正确写法：开头先
    `sbatch --dependency=afterany:$SLURM_JOB_ID ... scripts/slurm/xxx.slrm`
    （afterany ⇒ TIMEOUT/FAILED/COMPLETED 都触发），后继任务开头检查
    `global_step >= STEPS` 就秒退，链条自然终止，`MAX_CHUNK` 兜底。
    ⚠️ 别用 `$0` 提交自己：batch job 里 `$0` 是 slurmd 的 spool 副本，随 job 消失。

23. **`WorldModel.__init__` 有三条互斥分支且各自 `return`（`raw_state` :335 /
    `pixel` :369 / `latent` 落到底）⇒ 加在末尾的新功能对前两条臂是"不存在"的。**
    §43 的 aux-state 只把 `self._build_aux_state(...)` 加在落底路径，于是 pixel
    模型上 `self.aux_state` 这个属性**根本没有被创建**，而 `train_simdist.py`
    无条件读它 ⇒ **每一个 pixel run 一启动就 AttributeError**。
    之所以一直没暴露 —— 而且这正是「live jobs share this checkout」最阴的形态：
    在跑的那个 `pixel_dynens` 是在这段（**未提交**的）代码落地**之前**启动的，
    python 只在启动时读文件，所以它**照常跑完出结果**；真正会崩的是它用
    `afterany` 排好的后继 chunk（铁律 #22 的链条）。
    ⇒ 动 `world_model.py` 加任何 head/buffer/属性，**必须三条分支都过一遍**；
    读方一律 `getattr(..., None)`。想让属性对所有分支存在，就在**分支之前**赋
    `None`：赋 None 不注册 parameter/buffer ⇒ state_dict 不变 ⇒ 老 ckpt 仍 strict load。
    ⇒ **dedup gate 只覆盖 state arm**（当时全绿），所以改共用文件后**每条臂都要
    单独 smoke 一次**，不能只看 gate。
    ⚠️ 同理：`num_dynamics_heads`/`horizon` 在 `_build` 里建头且 `load_state_dict`
    是 strict（铁律 #20），所以这类遗漏只会在**运行时、且只在那条臂上**炸。
    自查最快的办法是本地 CPU 直接构造一次（秒级），别等 SLURM：
    `WorldModel(obs_dim=37, action_dim=7, cfg=WorldModelConfig(encoder_type="pixel", ...), image_shape=..., cameras=[...], prop_dim=9)`。

24. **python 的 `print` 进的是 `<save_dir>/train.log`，不是 slurm 的 `.out`。**
    slurm log 里只有 stderr（tqdm、robosuite warning、traceback）。在任务还在跑的时候
    `grep` slurm log 找不到 `[budget]` / `[seq-buffer]` 这类行**不代表没执行**——
    train.log 是块缓冲的，任务结束前可能还是 0 字节。别据此判断代码没生效。

25. **多 seed 的 demo draw 默认是同一批 —— 不传 `DEMO_OFFSET` 的"5 seeds"是假的。**
    `rl/replay.py:206` 和 `load_demos_to_sequence_buffer` 都是**无 seed 地取前 N 条**，
    所以 N 个 adapt seed 会训练在**完全相同的 20 条 demo** 上，误差棒里**没有** §38 的
    单次抽样 σ=**0.092**（本项目最大的噪声源），表格看起来比真相紧得多。
    已加 `demo_offset` / `target_demo_offset` / `adapt.slrm` 的 `DEMO_OFFSET`
    （默认 0 = 逐字节旧行为）。200 条的 target pool ⇒ 10 个互斥的 20-demo draw。
    **分块设计**：seed 内三个 rung 共用同一 draw（delta 配对到 ~0.027），seed 间 draw 不同
    （均值 delta 的误差棒诚实地含 0.092）。这也正是 published state 协议的做法
    （它的 10 个 seed 是 **adapt** seed，stage-1 只有一个）。⇒ 用 5 个 stage-1 seed 反而
    会往我们的误差棒里塞一个 published 那边没有的方差成分。详见 simdist.md §48b。

26. **`dedup_regress` 对"增广坏了"结构性失明** —— 它跑在 aug **关闭**下，只能证明
    default-off 契约。`scripts/test_augmentation.py`（新）才是唯一的防线：
    0 时逐字节 no-op / 开时真的改像素（§44 的病根就是"钩子在但没人调"）/
    **同一 (sample, camera) 的 H+1 帧参数相同、不同 sample 与不同 camera 不同**
    （错了会静默毁掉 dynamics head）/ uint8 往返 clamp 不回绕、shift 仍无损 / 三条分支都跑。
    ⚠️ 再加一条：`WorldModel` 级的 gate **不覆盖训练路径**（§46 d3/d8 两个缺陷都是
    pixel-only + runtime-only）⇒ 改 `update_distill` 必须**本地 CPU 直接跑一次** pixel
    `update_distill`（秒级），别只看 gate。
    ⚠️ `aug_prop_noise` 的单位是**米**（`prop` 未归一化，`pixel_obs_encoder.py:189` 直接拼接）
    ⇒ 0.002 = 2 mm 刚好在 ~1.8 mm 容错窗口之上；看起来很自然的 0.01 是 **1 cm**，直接淹掉信号。

### ★★★ PIXEL 决定（2026-09-09 重写）：用 r18@224，但 **~0.5 是交互，不是编码器主效应**

`enc_type: resnet18_pretrained` + `rl_image_size: 224`，且 **`bc_anchor_coef=1` 必须一起开**。
ViT@96 已退役（只作历史记录）。

**完整 2×2 析因**（encoder × anchor，`dynens_h5` 固定，独立 n=500/格，同一 demo draw，strict/σ=0.1）：

| | no anchor | +anchor | anchor 效应 |
|---|---:|---:|---:|
| **vit96** | 0.338 | 0.266 | **−0.072** (z=−2.5) |
| **r18@224** | 0.334 | **0.766** | **+0.432** (z=+15.2) |
| **encoder 效应** | **−0.004** (z=−0.1) | **+0.500** (z=+18.3) | **交互 +0.504** (z=+12.4) |

⇒ ★★ **anchor 关掉时，r18@224 比 vit96 一点都不强（0.334 vs 0.338）。**
"分辨率 + ImageNet 先验值 ~0.5" 这句**只在 `+anchor` 那一行成立**；encoder 在
`anchor=0` 上的主效应是 **0**。**引用 encoder 收益必须同时说明 anchor 条件。**
（本块旧版把 0.5 记成 encoder 主效应，是错的 —— 已按 §47b 更正。）

⇒ **r18 上的两个梯级结论与 vit96 正好互换**（§47）：

```
state (published, n=10 seeds, 窗口):  0.564 --+0.249 ens--> 0.813 --+0.081 anchor--> 0.979
vit96 (独立 n=500):                   0.218 --+0.120 ens--> 0.338 ---0.072 anchor--> 0.266
r18@224 (独立 n=500):                 0.418 ---0.084 ens--> 0.334 --+0.432 anchor--> 0.766
r18@224 (窗口):                       0.365 --+0.060 ens--> 0.425 --+0.450 anchor--> 0.875
```
- **ensemble 没复现**：+0.060(窗口)/−0.084(独立)，两个量度**符号相反**，且窗口效应小于
  rung A 自己的 SD(0.099) ⇒ 单 seed 上是噪声，肯定不是 +0.249。
- **anchor 复现且成为 pixel 上的支配杠杆**：+0.432(独立)/+0.450(窗口)，两量度一致。
  ⚠️ §42 说 anchor 在**弱** base 上收益最大，而这里**强** base 收益远大 ⇒
  **base 强度解释不了 pixel 的 anchor。**

⚠️ **`num_dynamics_heads`/`horizon` 改不了 adapt 时（铁律 #20）** ⇒ plain_h3 臂必须自己重跑
stage-1（`..._r18_224_plain_h3.yaml`）。
⚠️ 单 stage-1 seed/格；eval 噪声上巨大，但 **stage-1 seed 方差未测**（铁律 #3 仍适用）。

### ★ 铁律 #14 的补充：独立重评**去不掉 TopkSaver 的选择偏差**（§47a）
`model0.pt` 是 in-training eval 点上的 **max**，所以"独立 n=500 重评"只去掉了
in-training 乐观，**选择偏差还在**，而且**溢价随该臂曲线方差放大**：

| arm | 窗口 SD | best−窗口均值 | 独立−窗口 |
|---|---:|---:|---:|
| plain_h3 | **0.099** | **+0.245** | **+0.053** |
| dynens_h5 | 0.054 | +0.095 | −0.091 |
| +anchor | 0.030 | +0.065 | −0.109 |

⇒ **比较两臂前先比它们的窗口 SD。** 差得多时独立 n=500 **不可互换**，delta 只能读窗口
（published state 梯子本来也就是 75-100k 窗口均值 ⇒ 比 delta 时窗口才是同质量度；
独立 n=500 用来报**单个 ckpt 的绝对值**）。**两个都报。**

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
- **§43** ★ PIXEL stage-2 = 主要是 **demo 预算**问题，不是 latent 缺陷（旧版结论已推翻，勿引用）。
  demo 数是主轴：pixel 0.08→0.24 (N=20→200，未饱和)，**state 也从 0.360→0.75** ⇒ 全项目固定
  N=20 本身是瓶颈；bc_steps 无效（8× 步数零收益）；匹配 N 后仍差 0.24 vs 0.75；解冻 encoder
  是**探针不是解药**（MPPI 归零）；⚠️ aux 重建项被论文 ablation 否决（0.90→0.32）；
  **⑧ 我们跑了方法里没有的一个 stage**（Eq.3 只更新 `f_θ`，π 冻结）。
- **§45** ★★ **pixel arm 在 57× 更少的数据上多训了 230×**：论文 **2 epochs**（2.5e7 步），
  我们 state **50**、pixel **461**（0.44M 步）。一条就解释了「source 0.98 / target 0.01」
  ——source 分布被背下来了。修复顺序：**增广（免费、论文有、我们零）→ 砍 epoch/按 target
  早停 → JPEG 存储把 D_sim 扩到 ~1e7 → 224² + ImageNet ResNet-18（分辨率优先于视角数）
  → history encoder + 5 Hz**。§43 的所有测量都是在过拟合的 stage-1 上做的，需重测。
- **§44** ★ 论文 vs 我们的移植：输入/目标函数差异表（3×224² + ImageNet ResNet-18 + history
  encoder + 100k 轨迹 + 4 种增广 + 5 Hz vs 我们 2×96² + from-scratch ViT + 无 history +
  2400 轨迹 + **零增广** + 20 Hz）。⚠️ 100k×3×224² 无法用预先落盘的 offline replay 承载。
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
  π_src 本身** —— 换成数据集记录的 action 掉 0.08–0.17,其 D_SRC 下界 0.151(靶点 6.3×,
  λ 拉 100× 不动)= 结构性不可达,与 F2 同型。
  ★★ **唯一的充分统计量是 `policy_only`(proposal 质量):5 个机制不同的臂上 R²=0.982、
  残差 sd 0.010 < 各臂 seed sd;`D_SRC` R²=0.51、`bc_loss` R²=0.08(⇒ §43 那条 bc_loss 定律
  和本节初稿的"D_SRC 第二渠道"都是共线代理,已修正)。** ⚠️ 只用它解释臂间差异,别用它选参数
  (seed 层面 ρ=−0.50)。交互项分解:79% 来自 planner 的**转换率**(Δscore/Δp:plain 0.42 vs
  dynens 0.90),21% 来自 proposal 抬得更多 ⇒ findings §36 的定量版。
- **§44b** ★ **轴②(在哪些状态上锚)封闭**:按离 demo 支撑集的距离 / NTK 干扰加权,
  3 形式 × 2 度量 × 4 λ = 11 个标定格 **0/11 外移前沿**(全部内移 5.4–22.8%,单调剂量-响应),
  3 seed 闭环 **+0.021 (ns)**。两个前置诊断就预测到了:pool 本来 99.6% 在 demo 支撑集外
  (§42 的原文理由早已自动满足)、NTK 干扰几乎均匀(top10% 只携带 13%,LayerNorm 所致)。
  ⚠️ 但预注册预测(0.73)**反向失败** —— 正是这个负结果逼出了上面 `policy_only` 那个正结论。
- **§44c** ★ 轴②的第二条路径(按 **provenance** 筛 anchor pool)同样封闭,且给出**分解**:
  π_src **就是** `expert_flags==1` 步的 BC(`bc_expert_only=1`)⇒ 筛 expert∧success 把动作源的
  不可约偏移从 0.170 降到 **0.0247**(§44 那个硬下界消失了),但三臂 40k 实测
  **动作质量 +0.028 (ns)、状态覆盖 −0.123、净 −0.095** ⇒ **想要更好的动作标签就得放弃状态
  覆盖,二者不可兼得**(全空间上的 expert 动作回归 = π_src 本身)。
  ⚠️ 连带:`policy_only` 定律加这三臂后 R² **0.982 → 0.752**(残差 sd 0.0335)⇒ 跨机制会漂,
  只能当解释变量。**两次预注册预测连续失败(错 0.113 / 排序反转)⇒ ±0.1 内一律跑满 stage-3。**

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
- **§41c** ★★★ **learned vs oracle dynamics × 4 proposal × 2 value（`base_policy_std=0.1`）：
  模型误差改变了其他部件重不重要。** oracle 列极差 **0.07**（0.980/0.910/0.920，规划增益
  +0.50~+0.92，把裸跑 **0.000** 的 stage-1 头拉到 0.920）vs learned 列极差 **0.64**
  （0.650/0.590/0.130/0.010，弱 proposal 下**规划有害**：裸跑 0.200 → 0.130）⇒ **Δdynamics
  +0.330 / +0.780 / +0.910（z 6.4/17.6/31.5）**。★ learned 列由 **proposal 裸跑分**决定，
  **不是**"离被适配 policy 的距离"（il_ft 判别臂：裸跑 0.500 未被适配 → 0.590 ≈ 被适配的
  0.650，ns；⚠️ 我的预注册预测反向失败，"dynamics 只在 policy 邻域可修"在此不成立）。
  ★ 换 value ≤+0.04（4 读数 2 个为零，异质性 p≈0.09）；V_tgt 配 source-only policy 只有
  0.910，显著低于配另两种（z=−2.0/−2.6）⇒ **关键是 policy 是否仍在 value 的训练分布上**。
  ★ **base 构造三种 6 读数全零**（含新增 `--base_real_rollout`）且贵 30–70% ⇒ 别再投。
  ★ stage-1 头 source 0.960 / **target 0.000**（两套代码互证）⇒ stage-2 是必需品。
  ⚠️ 与 §41b 口径不同（`base_policy_std` 0.0 vs 0.1）不可相减；初稿的「dynamics +0.185~+0.245」
  已作废（base 构造未配对）。
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
