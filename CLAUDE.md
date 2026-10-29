# CLAUDE.md — 索引 + 速查

本仓库 = fork 自 upstream ibrl 的 **sim2sim 跨域迁移**研究代码。
**当前主线：SimDist（latent world model + MPPI）的改进，以及在多个任务上的验证。**

**这个文件只放"每次都要知道的东西"**（当前状态 / 方法论铁律 / 去哪找细节）。
详细笔记按主题拆在 `docs/` 下，用到再读；已停止推进的分支在 `CLAUDE_ARCHIVE.md`。
写新笔记时请写进对应的 `docs/*.md`，只有当结论会改变"下次该怎么做实验"时才回填这里。

Last updated: 2026-09-25

---

## 文件地图

| 文件 | 行数 | 什么时候读 |
|---|---:|---|
| **`docs/findings.md`** | ~1415 | **做任何改进决策前**。天花板律（§27）、dynamics ensemble 0.962（§31，⚠️ 末尾补了「增益与 anchor 不可加」）、value 悲观判死（§29）、value adapt 判死（§33）、latent vs raw（§30）、proposal（§36）、容错窗口旋钮（§39）。**里面有一整批已被实测判死的方向，别重复。** |
| **`docs/simdist.md`** | ~2700 | 改 SimDist pipeline 时。§17 本体 · §24 两阶段 target adapt · §25 residual/IBRL on frozen base · §26 省交互变体 · §42 stage-2 的 source 锚 `bc_anchor_coef` · **§43 anchor 形式轴已封闭(别再试换散度)** · **§44 anchor 的价值由 dynamics 精度定 + 唯一充分统计量是 `policy_only`** · **§44b 轴②(在哪些状态上锚)也封闭** · **§47 r18@224 梯子：anchor 复现(+0.432)、ensemble 没有；encoder 的 0.5 是交互；铁律 #14 的选择偏差补充** · **★★★ §48 pixel 定论（34 run，§48r 汇总）：ensemble +0.258 复现；§44 三项增广倒扣 0.2-0.3；anchor 在任何 λ 下 ≈0；r18 vs vit96 未决；pixel 有三个噪声地板** |
| **`docs/demo-collection.md`** | ~430 | 要（重）采 stage-2 的 target demo 时。★ 先读 §38。 |
| **`docs/tasks.md`** | ~1120 | 加新任务 / 动 Square·Coffee 时。§32 Square · §40 Coffee · §41 Square rotw1 全链条 · **★§41c 模型误差决定别的部件重不重要** · **★§41d 四任务 48 格复现（dynamics 12/12 正；target value 只在 Threading 崩；TPA 是反例；★`learned+V_tgt` 列被 decode 误差废掉）**。 |
| **`docs/infra.md`** | ~915 | 查旋钮和 API 时。§3 domain shift · §4 flow BC · §5 eval 基础设施 · §5b BC baselines · §5c strict success · §6 配置表 · §7 已知坑 · §12 Threading target 定义 · **★§13 三个任务 target 的唯一真源（跑任何跨臂对比前必读）** · **§14 sim2real cell 与 real2sim 共用 arena（`arena_from_real2sim`）· §14d RealCell 全部随机化汇总 · ★§14e pixel 数据的重放契约（存 `sim_state`+`model_state` ⇒ 相机可事后重渲染）+ 校准腕相机 + 「腕相机在插入瞬间 0.0% 可见」** · **★§15 Square_RealCell（同一 bench 换任务）：cell yaml 的 `inherit`、`workspace_offset` 推导、rotw1 由类携带；★§15b Square 的 peg 是 arena 几何 ⇒ 抬桌会把柱子埋掉且不报错；★§15f2 source 专家 **0.96**（= Threading 的 0.957），IBRL 前 120k 全 0.00 是正常冷启动；§15f3 `[EVAL ENV]` 曾把 rotw1 印成 FULL；§15i Square pixel 数据的采集配方（★数据已随 09-23 的 cell 大改删除，方法仍有效）；**★§15j cell 大改后谁作废谁不作废：pixel 全废、Threading state 也废，但 Square 的 state 线（含 0.96 专家）不废 —— `layout_*` 按物体名匹配，Square 的 `nut` 不命中；★§15k 方柱扰动 `dr_peg_xy` 5mm/`dr_peg_yaw_deg` 3°（拐点与 6.7mm 间隙一致；**对 state 是不可观测噪声、对 pixel 才是信息**）；§15l 相机视角对 state 严格为零、对 pixel 是全部，Square 的取景余量是 Threading 的 7 倍；**★§15m 照片背景/软阴影让重放契约失效过：`tex_rgb` 与 `light_castshadow` 每局随机却没被捕获（212/118）；且逐字节判据已不可能，gate 改为「与渲染器自身的重复噪声比」**；§15n 采集就绪状态与成本（0.98 / 3.6 条每秒 / 2.29 MB 每条）**。 |
| **`docs/square_sim2real.md`** | ~210 | **做 Square sim2real 的任何一步之前** —— 采 pixel 数据、跑 stage-1、换机器的完整操作手册（背景与推导在 infra §15）。 |
| **`docs/setup.md`** | ~165 | 装机 / 换机器 / 转数据集时。 |
| **`docs/on-policy-adaptation.md`** | ~190 | **9 组同预算大对照（100k transitions）+ 总结论：on-policy 一致性是 model-based adaptation 的支配性质。** planning ≫ Dyna（同一 world model 0.813 vs 0.157）；value adapt 三种顺序全崩；dynamics 只在 policy 邻域可修；source 锚 +0.081 且方差收缩 4.3×。 |
| `docs/flow_qc_ug.md` | ~110 | （早于本次拆分就存在）flow actor + Q-chunking + UG 的在线 model-free RL 方案书 —— "不再修 SimDist、改把 policy 做强"那条支线。 |
| **`docs/pixel-progress.md`** | ~210 | **做 pixel/视觉版之前必读。★ 顶部的 CURRENT STATE 框是 09-25 的定论，框下的表都是 §48 之前的历史。** 所有已测数字：stage-1 四个 checkpoint（pixel 0.968/0.980 ≥ state）、target adapt 的 2×2（state 0.540 / pixel 0.028；dynamics-only 0.393 vs 0.023）、demo-count 扫（pixel 0.08→0.24，**state 0.360→0.75**）、latent probe（含它**不能**支撑的结论）、编码器 bench、环境坑。 |
| `docs/pixel-gap-closing-prompt.md` | ~230 | 要接着推 pixel 的时候：交给 agent 的完整任务书（Phase A 增广 / B 数据对齐 state / C 224+ResNet-18），含活跃任务共用 checkout 的风险与回归门。
| **`docs/real-teleop-requirements-EN.md`** | ~200 | **发给真机那边的英文需求书**（20 Hz、动作约定、两套栈的 EE 坐标系差异、相机原生内参与不裁剪、摆放、成功标注、逐次 checklist），含已收到 4 条数据的验收结论。 |
| **`docs/real-teleop-stage2.md`** | ~140 | **真机采 stage-2 遥操作 demo 之前必读**：动作约定（20 Hz、相对实测 TCP、左乘、只看夹爪符号）、prop 的两个坑（世界系而非 base 系 / 位置是 TCP 姿态是 hand body）· **★§2b Robotiq cell 起 prop 改成 base 系（`prop_frame: base`，FK(q)，不含标定）—— 上面那条「世界系」只对旧 cell 成立**、相机裁剪、hdf5 契约；采完跑 `scripts/validate_real_demos.py`。 |
| **`real2sim/zed_capture/calib_20261015/SIM_TODO.md`** | ~80 | **★ 10-15 真机之后 sim 侧要做的事（按依赖顺序）**：sim 改 10 Hz、把 sim_ref 限速和滞后放进 sim、手腕相机新位姿、5 条 Robotiq demo 的转换和回放。结果在同目录 `README.md`。**★ 10-16：sim 已是 10 Hz（cell 字段 `control_freq`，cell = `threading_real_cell_20261015.yaml`）；5 条 demo 回放 1/5，瓶颈是 sim 抓取；下垂 = 漏了 ~0.3 kg 负载；动作约定 `ref_inverse`。上机清单 `calib_20261015/NEXT_VISIT.md`。** **★ 10-17：执行层进 sim（`env/exec_layer.py`，10-15 cell 已开）：部署的参考臂/限速/100 Hz/关节阻抗 + 拟合的机械臂参数；不限速（owner 定，sim 和真机一致）；执行层已对齐。** **★ 10-19：抓取接触定了（临界阻尼，真机网格 12/12 逐次吻合），cell 已冻结；训练计划 `calib_20261015/TRAIN_PLAN.md`。** **★ 10-27：stage-1 上真机抓取 100% 但插不进；原因 = 手中针位姿（真机夹手柄尾部/上部、俯仰 +9.8°），sim 里摆真机位姿即复现失败；sim 侧待办 + 数据见 `calib_20261015/real_1027/README.md`。** |
| **`docs/threading-sim2real-camera-plan.md`** | ~150 | **Threading 真机重做相机之前必读**：上次 transfer 失败的诊断（摆放表过期 / 腕相机位姿未验证且插入时 0% 可见 / 全局相机太远 / value 失效是下游症状）+ 腕相机移到手背、全局相机拉近的方案与对齐门槛。 |
| **`docs/new-machine-robotiq.md`** | ~150 | **把 Threading Robotiq 链搬到另一台机器时**：curie 收尾、环境（`$PY` 可覆盖、editable deps 路径）、数据核对、v8 重开 / v6 stage-1 续跑命令、盯什么。 |
| **`docs/real2sim.md`** | ~90 | **真机方向（sim2real via real2sim）的唯一真源**。架构决定 = MuJoCo + `_GaussianRenderer` batched gsplat、**不用 PolaRiS/Isaac**（全部实测：PolaRiS 3.98 steps/s、Isaac 固定地板 173 ms/step vs gsplat 558 tps/GPU@2cam·763k）；vertical slice 已通（PolaRiS 机器人 splat MIT+metric，丢 `panda_link0`，per-body 校正 + 4 个坑）；任务 = 方形 peg-in-hole ~3 mm/边（§39 地板 1.3 mm）；阶段 A–F 现状表。job 提交用 `aip-rhinehar`（FairShare 远好于 aip-gigor）。 |
| **`CLAUDE_ARCHIVE.md`** | ~2020 | 只在需要回查已归档分支（§10/11/13/14/15/16/18/19/20/21/22/23）时。代码在本地 `_archive/`，不随仓库发布。 |

---

## 速查

### 三个任务的当前状态（σ=0.1 / target = 各自的 §12/§32/§40 配方；strict 仅 Threading·Coffee，**Square 线上 `strict_success.enabled` 全是 false**，见 tasks §41b）

| 任务 | source 专家 | target 专家 | SimDist adapt | 详见 |
|---|---:|---:|---:|---|
| **Threading_D0** | IBRL 0.98 | model-free IBRL **0.9815**(n=2000) | **dynens 0.962**(n=2000) / plain 0.879 | findings §31 |
| **Square_D0 (rotw1)** | 0.98 | **0.98–1.00** | anchor **0.837**(10seed) / 0.886(d600) | tasks §41 · **§41b 拆账** |
| **Coffee_D0** | 0.998 (σ.02) / **0.938** (σ.1) | **0.982** (C50) / 0.96 (C75) / 0.93 (G5x) | **0.907**(G5x, 2seed) = oracle 0.94 | tasks §40–40e |
| **PegInsertion_D0** (sim2real) | IBRL **0.95** (n=500, `--mode rl`) | 真机，未测 | stage-1 跑中 | real2sim §2d-z/§2d-w |

sim2real（RealCell）线的 source 专家（同一张 DROID bench，n=300 独立重评，σ=0.1）：
**Threading_RealCell 0.957**（strict） · **Square_RealCell_RotW1 0.96**（strict 不适用，infra §15f2）。

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
   ⚠️ **这是 state 的结论。pixel 上不成立**（§48）：anchor 关着时 5 头集成就值 **+0.258**，
   而 anchor 在两种增广、四个 λ 下都 ≈0 ⇒ pixel 上没有「互补」可言，ensemble 单独就是杠杆。
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
27. **位置吻合不代表姿态吻合。** `robot0_right_hand` 与 polymetis 法兰位置差 **0.5 mm**，
    姿态却差 **绕 z 的 45.03°**（14 pose 上恒定到 0.000°）。§2d-y 只核对了位置就下结论
    「同一个系」。⇒ 在 sim 里放腕相机必须 `T_sw_hand @ Rz(45.03°) @ T_ee_zedm`；漏掉它
    夹爪占 23.8% 画面（真值 4.6%）。**核对坐标系必须同时给平移和旋转。**
28. **材质 albedo 不能直接对着「观测到的亮度」标定** —— 我连续错了两次（先写下这条
    caveat，然后照样犯）。唯一有效的回路是 **渲染 → 量 → 修**：板子 albedo 两轮收敛到
    `0.186 0.202 0.536`，peg 的 `0.95 0.95 0.90` 本来就是对的（渲染 209 vs 视频 212）。
    ⚠️ 「MuJoCo 里物体偏暗」先查**光照**再改材质：robosuite 默认单盏近垂直平行光，
    竖直面 N·L≈0.27 ⇒ 0.95 的 albedo 渲出中灰。真实 cell 是桌子左缘上方一条灯带
    （`env/peg_insertion.py` STRIP_*，5 盏点光）。
29. **两个候选解释共用同一个判据时，那个判据是瞎的**（铁律 #10 的几何版）。
    「aspect 比例」分不开**平板 board** 和**一片平的桌面** —— 两者都是 [1,1,0.18]，
    所以 aspect 测试放行了一个纯桌面的 crop，ICP 还给了 3.61 mm 的漂亮残差。
    破局的是**颜色**（与形状近乎正交）。⇒ 造判据时先问「它能不能把我怕的那两种情况分开」。
30. **ICP rms 不是充分统计量。** trimmed ICP 丢掉最差 20%，所以一坨**过胖**的点云可以把
    薄薄的 mesh 埋进去，拿到 **1.06 mm** 的残差。verdict 必须同时要求一个**尺度无关**的
    形状检验（aspect）通过。另：**mesh 要采样表面**，不能用三角形顶点 —— 一个 box 只有
    8 个唯一顶点，ICP 根本约束不住（改成 40k 表面采样后 13 mm → 3.6 mm）。
31. **`scene_scan5_norobot_v2.ply` 已经是米制 board 系，别再套 `scene5_world_transform.json`。**
    那个 similarity 属于旧的 raw-COLMAP `sfm_scan5/scene.ply`，而两者就放在同一个目录里。
    套上去房间缩小 0.1023× ⇒ 渲出一团黑底白刺。判据：v2 不做任何变换时最密 z 桶在
    **0.73 m**、xy 跨度 1.83×1.58 m（= 桌面）。**我为此编了一小时的错误理论**（「v2 在
    归一化系」「需要重新配准」），真正解决问题的是 `md5sum` 对上了 `pov_correctK`。
    ⇒ **有人说「这个东西我见过它正常工作」，先去找那次能跑的调用，别继续分析工件本身。**
32. **`expert_hybrid=0` 是 Threading 的选择，不是普适的。** peg 上 2×2 实测：
    σ_max 0.2 时 hybrid 0/1 = **14.8% / 62.5%**；σ_max 0.1 时 = **10.0% / 68.2%**。
    **BC prior 是全部效应，噪声幅度几乎无关**（去掉 prior 后降噪反而更差）。
    不带 prior 的 expert 从 0.95 崩到 **0.108**。铁律 #8 的第五次。
33. **PHOTOREAL obs 的成本要重新量，别沿用 state 臂的数。** 3DGS 合成帧
    **37,049 B/transition**（state 臂 22,943，×1.66 —— 真实纹理不好压）、
    **~130 transitions/s**（×2.9 慢）。⇒ buffer 从 5.5M 降到 3.3M 才是 126 GB。
    ⚠️ `source_replay_num_data` 必须从**真 shard** 的 mean length 反推，不能用 smoke：
    15 条轨迹给 131.9，30400 条给 **153.7**，按前者会超 cap 14%，而
    `_evict_for_transitions` 从**前面**弹 ⇒ 静默丢掉最早的 shard，8 个生成 seed 塌成 7。
34. **gsplat 在每个进程 JIT 编译，多 worker 会抢同一个半成品缓存目录。** 2 worker 实测
    一个成功、另一个 `ImportError: cannot import name 'csrc'`。⇒ fork 之前先串行 pre-build
    （`gen_source_replay.slrm` 已加）。
35. **别往「有作业正在用」的 venv 里 pip install。** 装 `plyfile` 把 numpy 拉到 2.2.2，
    与所有按 numpy 1.x 编译的模块（cv2 / numba / `rela`）ABI 不兼容。在跑的 5 个 job 因为
    早就 import 完了而幸存，但它们排着 5 个 eval 依赖作业 —— 任何一个在那 3 分钟里启动都会崩。
    ⇒ **先 `squeue`，再装；能用 `--no-deps` 就用**（triton 那次就没出事）。
36. **monitor 的过滤条件必须能区分「做完了」和「链断了」。** 我按 job name 盯 stage-1 的
    chunk 链，但续跑的 chunk 继承的是脚本头里的 `#SBATCH --job-name`，不是我在首次提交时
    传的那个 ⇒ chunk 1 一结束就报 `[chain-done]`，而这**和铁律 #22 真断链时的输出一模一样**。
    ⇒ 盯**状态**（`resume_state.pt` 的 global_step + 还有没有 chunk 在队列里），别盯名字。
37. ⚠️ **ZED HD2K 只有 15 fps，而我们按 20 Hz 训练。** HD2K/HD1080/HD720 的 FOV 相同
    （宽高比例都是均匀缩放，只有 VGA 不是），但**帧率**不同。世界模型学的是 50 ms 的
    `p(s'|s,a)`，15 Hz 部署就是 66.7 ms，**33% 的持久动力学误差**，而 stage 3 只修 dynamics
    头 ⇒ 正好撞在 findings §27 的天花板律上。**未决**：HD1080@30fps 抽到 20 Hz（要按该模式
    重读内参、crop 变 1078 px），还是按 15 Hz 重生成。
    ✅ **2026-10-15 已决：Threading Robotiq cell 用 10 Hz**（相机 HD720@30fps，每步 3 帧）；sim 按
    `control_freq` 10 重训，旧的 20 Hz 数据和 ckpt 不能沿用。依据和 sim 侧清单见
    `real2sim/zed_capture/calib_20261015/{README,SIM_TODO}.md`。部署走 `policy_server.py --action_map sim_ref`
    （每步从实测 q 重算目标会被关节静摩擦卡住，x 只走 1–4 %/步）。

38. ★ **训练用什么观测，eval 就必须用什么观测 —— 否则 loss 全正常、分数恒为 0。**
    `train_simdist.py:398-413` 的 train/eval env 都是从 **config** 建的，而 `splat_obs`
    当时只在生成侧按程序传 ⇒ 用 3DGS 合成帧训出来的 encoder 被拿去在**纯 MuJoCo** 画面上
    评测，source 和 target 双双 **0.000**，而 consistency/policy/value 全在正常区间。
    实测同一个 ckpt：**splat 0.960（48/50）vs MuJoCo 0.000（0/50）**。
    ⚠️ 那 13 h 没白跑，只是 eval 瞎了；`splat_obs` 现在是 config 字段，两个 env 都吃。
    ⇒ 推论：**「安全的默认值」在另一侧可能就是静默的分布错配**（这里 `splat_obs=None`
    正是我为了让 MuJoCo 路径逐字节不变而特意加的）。
    ⇒ 另：**别凭印象判断 loss「太小」**。我怀疑 consistency 0.0016 是 latent 塌了，
    查了健康的 Threading pixel 臂是 0.0013 / 0.0022 —— 完全正常。先找同类基线再下判断。

16b. **别假定 `aip-rhinehar` 更快 —— 每次都要量。** real2sim 那条「job 提交用
    `aip-rhinehar`（FairShare 远好于 aip-gigor）」只在写下它的那一刻成立。
    2026-09-15 实测同一个 200G/23:59 的 stage-1 chunk：**gigor 19:46 vs rhinehar 次日
    09:07，gigor 早 ~13 小时**。fairshare 只是 priority 的一项，两个账号还落在不同的
    backfill 队列里。⇒ **两个账号各排一份，比较 `squeue -o %S` 的真实估计，留快的、砍慢的**
    （同一 save_dir 的两份必须只留一个 writer，见 infra §7z ④）。
    ⚠️ `sbatch --test-only` 的估计**不可信**：它给过 09-16 00:34，真排进去后是 09-17 05:49。
    只有真正排队后的 `%S` 才有参考价值，而且它还会随 backfill 反复漂。
    ⇒ 扇出多个 job 时**跨账号交替提交**，别把整批押在一个队列上。
    ⚠️ 此后又测了两次，**gigor 三连胜**：09-15 早 17 h、09-23 早 **11.7 h**（同一批 adapt，
    22:00 vs 次日 09:41）⇒ 至少在 2026-09 这段时间 **gigor 是默认的快队列**，但仍然每次量。
    ⇒ 最省事的量法：`scontrol update job=<id> Account=<另一个>` 挪一半过去、等 backfill
    重估 `%S`、再挪回来 —— job id 不变，afterany 依赖链全部保留（已验证）。

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

### ★★★ PIXEL 决定（2026-09-25 重写，依据 §48 的 34 个 5-draw run —— 取代 09-09 版）

**配方 = r18@224 + 只用 random shift 增广 + `dynens_h5` + anchor 可关。**
`enc_type: resnet18_pretrained` · `rl_image_size: 224` · `aug_shift_pad: 8`，
**`aug_color_jitter` / `aug_blur_sigma_max` / `aug_prop_noise` 全部 = 0** ·
`num_dynamics_heads: 5` + `horizon: 5` · `bc_anchor_coef` 0（λ≤1 无害，**λ=3 有害**）。

| 杠杆 | 效应（5 draw，窗口/独立一致） | 判决 |
|---|---:|---|
| dynamics ensemble | **+0.258** / +0.156 | ✅ 复现 published +0.249 —— pixel 上唯一的正向大杠杆 |
| §44 的 jitter+blur+prop-noise | **−0.21 ~ −0.28** | ✅ **倒扣**，只留 random shift |
| source anchor（= KL 系数） | **≈0**；λ∈{0.1,0.3,1,3} 无一过预注册门槛 | ✅ 无可靠效应（§48q） |
| r18@224 vs vit96 | ? | ⚠️ **未决**（见下） |

最好的实测配置：**0.714 ±0.183 窗口 / 0.562 ±0.162 独立**（shift-only + dynens，anchor 关）。
⚠️ **r18@224 比 vit96 好不好现在是未决的**：下面那张 2×2 里 encoder 的 +0.500 **整个住在
+anchor 那一行**，也就是被 §48n 拆掉的那个 +0.432。anchor=0、draw `[0,20)` 上 r18 = 0.286
（独立）vs vit96 0.338（n=1）⇒ 没有优势；但 `[0,20)` 恰好是 r18 五个 draw 里最差的。
**要定论需 vit96 × 5 draw。** 在那之前继续用 r18（全部基础设施都按它建），但别再引用「encoder 值 0.5」。
⚠️ 任何 pixel 对比：**必须传 `DEMO_OFFSET`、≥5 个 draw**（铁律 #25）——三个噪声地板
（draw 使 delta 跨 **0.58**、固定配置单 run **±0.14**、eval 抽样），单 draw/单 run 的数只值 ±0.15。

<details>（以下为 09-09 版，**已被 §48 推翻，保留作病历**）</details>

~~用 r18@224，但 ~0.5 是交互~~；~~`bc_anchor_coef=1` 必须一起开~~ —— 两句都已作废。

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

### ★★★ §48 撤回了 §47 的两个梯级结论（2026-09-18，5 adapt seeds，全部跑满 100k）
§47 是 **n=1**。加了 `DEMO_OFFSET`（铁律 #25）后重跑 3×5，两个结论**双双翻转**：

| delta | §47 (n=1) | §48 (n=5，配对) | 判决 |
|---|---|---|---|
| ensemble B−A | +0.060 窗口 / −0.084 独立 | **+0.258 ±0.184 窗口 / +0.156 ±0.136 独立** | **复现**（5/5 seed 同号，两量度同号） |
| anchor C−B (full-aug) | **+0.450 / +0.432** | **−0.068 ±0.064** | **无效果** |
| anchor K−L (shift-only) | — | **−0.003 ±0.212 (0.0σ)** | **无效果** |
| anchor λ 扫 0.1/0.3/1/3 (§48q) | 从未在 pixel 上调过 | **+0.053/+0.130/+0.099/−0.081**，无一过门槛；λ=3 在 2/3 draw 有害 | **null 不是没调参** |

★★★ **anchor 在两种增广条件下都是 0。** §48m 的完整 2×2（4 格 × 5 个互斥 draw，
20 个 run 全部跑满 100k）：

| | anchor OFF | anchor ON | anchor 效应 |
|---|---:|---:|---:|
| **shift-only stage-1** | 0.714 ±0.183 | 0.711 ±0.100 | **−0.003 (0.0σ)** |
| **full-aug stage-1** | 0.502 ±0.091 | 0.434 ±0.079 | **−0.068 (2.4σ)** |
| **增广效应** | **−0.212** | **−0.277 (4.2σ)** | |

★★ **真正的大效应是增广，而且是负的：−0.21 ~ −0.28，两个 rung 都中招**（主效应，
不是与 anchor 的交互）。⇒ §45 把"增广（免费、论文有、我们零）"排在修复顺序第一位，
**实测是倒扣 0.2-0.3**。random shift/crop 不在此列（§47/§48 每条臂本来就有）。
⚠️ 没测出来的是：三项（jitter / blur / prop-noise）里哪一项、以及是不是只是**强度选大了**
（jitter 0.3 / blur σ≤2.0px@224 / prop 2mm）⇒ 下一步是单项剂量扫，不是全盘否定增广。
⚠️ 也**没有**把"增广"与"stage-1 seed"分开（每个条件只有一个 stage-1）；4.2σ 排除了
adapt 噪声，但 stage-1 seed 方差在本项目**仍然未测**（§47d 的老坑）。

★★★ **§47 的 +0.450 已被完整拆解，而且它没有 bug**（§48n）：
`rung B draw[0,20)` §47=0.425 vs 我们 Ls1=**0.419（几乎逐点复现）**；
`rung C` §47=0.875 vs 我们 Ks1=**0.732（差 0.143 = §48l 的单次 run 噪声）**；
⇒ **+0.450 = (+0.312 这个 draw 的真实效应) + (~0.138 单次 run 噪声)**，
而 **+0.312 是 5 个 draw 里唯一为正的那个**（其余 −0.074/+0.058/−0.043/−0.268，均值 −0.003）。
⇒ §47 把**一个 draw 上测得的正确数字**当成了效应本身。
⚠️⚠️ 而 `[0,20)` 正是**本项目此前所有 pixel 实验用的那个 draw**（`rl/replay.py` 无 seed 取前 N，
铁律 #25）⇒ **所有单 draw 的 pixel 结论都坐在同一个非随机抽样上。**

- ensemble 的 +0.258 基本**正中** published 的 +0.249，也越过去偏差后的 +0.20–0.22；
  A/B 的窗口 SD 相近（0.053 vs 0.060）⇒ §47a 的可互换条件**成立**，两量度都能读。
  ⇒ **「ensemble 在 pixel 上没复现」作废** —— 那是 n=1 的假象（n=1 的 SE ≈ 0.19）。
- **「anchor 是 pixel 上的支配杠杆」也作废。** 5 draw 上均值 ≈ 0（shift-only −0.003 / full-aug −0.068）。
- ⚠️ **教训：§47 里 z=+15.2 的那个"最稳"结论才是牺牲品。** z 是对着 **eval 抽样噪声**
  算的，它对 **draw / seed 方差一无所知**，而 §38 的 0.092 支配后两者。
  ⇒ **任何 n=1 的 z-score 都不能当作"稳"的证据**，无论它多大。
- ✅ 已隔离（§48k/m，20 个 run 的 2×2）：**anchor 的 null 与增广无关**，两种增广条件下都是 0。
- ⚠️ **§47b 的 encoder×anchor 交互 (+0.504) 仍未重跑，且现在更可疑**：它的 r18 格
  (+0.432) 就是被 §48n 拆掉的那个数。下面这个 PIXEL 决定块赖以成立的正是那一行 ⇒
  **在 vit96×5seed 臂跑出来之前，下面的 2×2 只能当单 draw 结果读。**

⇒ **r18 上的两个梯级结论与 vit96 正好互换**（§47，★ 已被 §48 撤回，保留作病历）：

```
state (published, n=10 seeds, 窗口):  0.564 --+0.249 ens--> 0.813 --+0.081 anchor--> 0.979
vit96 (独立 n=500):                   0.218 --+0.120 ens--> 0.338 ---0.072 anchor--> 0.266
r18@224 (独立 n=500):                 0.418 ---0.084 ens--> 0.334 --+0.432 anchor--> 0.766
r18@224 (窗口):                       0.365 --+0.060 ens--> 0.425 --+0.450 anchor--> 0.875
```
- ~~**ensemble 没复现**~~ **（§48 已撤回：5 seed 下 +0.258，与 published 等同）**：
  §47 读到 +0.060(窗口)/−0.084(独立)，两量度符号相反 ⇒ 当时只能说"单 seed 上是噪声"。
  真相是这个 delta 本来就存在，n=1 看不见它。
- ~~**anchor 复现且成为 pixel 上的支配杠杆**~~ **（§48 已撤回：5 seed 下 −0.068）**：
  §47 读到 +0.432(独立)/+0.450(窗口)，两量度一致 —— 但"两量度一致"只排除了选择偏差，
  **排不掉 draw/seed 方差**。这是本项目最贵的一次 n=1 误判。

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

# ★ 新机器/新 clone 必做一次（否则 release/data 的 wipe 事故会重演，setup.md「Data-loss guard」）
git config core.hooksPath scripts/githooks            # pre-commit 拦截被 gitignore 的路径 / 绝对 symlink
git config --add simdist.storagelink "release/data=<your>/simdist/data"
./scripts/check_storage_links.sh                      # 随时可单独跑；exit 1 = 要人看

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

### 真机 (FR3 cell) —— 机器人工作站
```
Host tisl
  HostName 100.66.83.240      # Tailscale; Killarney 登录节点可直连
  User tisl-robotics
  Port 2233
```
- 认证：密钥（owner 的 `~/.ssh/id_rsa`）或密码 —— **密码不进仓库**，问 owner。
- 机器 `tisl-ws-22-5`，RTX 3090，Ubuntu 20.04。ROS 2 Humble 只在容器
  `ros2_devcontainer-gello-ros2-1` 里（host 网络，**无 torch / 无 GPU**，`~/simdist/gello_software` 挂在 `/workspace`）。
- 臂 = **FR3**（不是 Panda；运动学相同、关节限位不同），只经 `joint_impedance_controller`
  控制：暂停 `gello_publisher`（SIGSTOP）后自己发 `/gello/joint_states`，100 Hz，**0.5 s 看门狗**。
  范本 = 容器内 `scripts/reset_arm.py`。相机 = `publish_zed_uvc.py`，**原始未去畸变**帧 + 出厂 K,D。
- 策略 runner：`real2sim/deploy/`（README），host 端部署在 `~/simdist/deploy_simdist`
  （conda env `simdist`）。别往 host 已有的 conda env / 容器里装东西。

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
- **§41d** ★★★ **四任务 48 格复现（Square/Coffee/Threading/TPA，统一小预算 + held base，n=100/格）。**
  ★ **dynamics 项 12/12 格全为正**，均值 +0.43~+0.67（唯一跨任务同号且有量级的效应）。
  ★ **target critic 不是普遍更好**：Square/Coffee 测不出（|Δ|≤0.08），**Threading 把 oracle
  planner 打崩（0.97→0.42 / 0.98→0.41 / 0.98→0.22）**，TPA 弱 proposal 行 −0.280 (z=−4.5)；
  机制同 §40c（target value 在 CEM 状态上 OOD），但**这里 dynamics 是完美的** ⇒ 不是共适应。
  ★ 「oracle 让 proposal 无关」在三个 400 步任务上极差仅 0.01–0.07，**TPA（ep_len=1000）是反例
  （0.480）**，且那里不是裸跑分决定 oracle 分 ⇒ 平均值会抹平这个定性差异。
  ★★ **`learned+V_tgt` 整列不可用**：外部 critic 迫使解码终端 latent，decode 往返本身值
  −0.13~−0.58 且 policy 相关到能翻符号 ⇒ 我曾把它读成「换 value −0.200、交互 z=3.43」并配了
  §33 的机制解释，**全是解码伪迹、已撤回**（预注册的 decode 控制格救了它）。
  ⚠️ 全部单 seed；预算未按任务重扫（违反铁律 #9 ⇒ 绝对值偏低）；Threading 的 tgt-only BC
  用的是另一套 demo（原集随 release/data 丢失）；**Coffee G5x 的 strict 其实是 false**。
  ⚠️ `eval_coffee_oracle_mppi.py` 的 `TARGET_B` 常量是 **C75kp25 不是 G5x** ⇒ 必须传
  `--shift_from_cfg`，否则静默评错 target（旧 Coffee oracle 数就是这么错的）。
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
- **§15** ★ **Square_RealCell**：同一张真机 bench 换任务。**§15b 三条「只靠 wrapper hook 会静默失败」**
  （其中 ③ Square 独有且致命：peg1/peg2 是 `pegs_arena.xml` 里写死世界坐标的 arena 几何，抬桌
  113 mm 后柱子露出部分 130→**17 mm**，任务几何上不可能且零报错 —— 搬任何新任务上 real cell
  都要先查「任务的目标物是不是 arena 的一部分」）；§15a cell yaml 的 `inherit`（连带：provenance
  快照必须哈希整条 inherit 链）；§15c `workspace_offset=(0.071,−0.034)` 是纯 base 位移补偿，
  **Threading 那 +0.10 的遮挡修正不可搬**；§15d rotw1 由**类**携带而不是 `init_dist` 旋钮
  （生成不过 wrapper ⇒ 旋钮会让 demo 和 RL 两处各写一遍）；§15f 生成 86%/161 步、
  **孔投影 14.5×17.6 px 对 Threading 的 3.9×6.0 `UNRESOLVABLE`**。

**`docs/setup.md`** — §0 概览 · §1 环境（torch 2.7/cu128、mujoco 2.3.2、submodule）· §2 MimicGen 集成与数据转换。
