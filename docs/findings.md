# 机制结论：天花板律 · ensemble · value · 表征 · proposal

为什么 SimDist 卡在某个分数、什么有用什么没用。**做任何改进决策前先读这里**，里面有一整批已被实测判死的方向（别重复）。

> 从 `CLAUDE.md` 拆出（2026-08-14）。速查表、方法论铁律、文件地图仍在 `CLAUDE.md`；已归档分支见 `CLAUDE_ARCHIVE.md`。

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

### ★ 2026-08-03 换一个 proposal 复现 dynens > plain:+0.103(z=9.6),且 plain 曲线不稳
§31 主表的 dynens-vs-plain(+0.086)只在 default demo(`threading_d0_target_dynamics`)那一个
proposal 上做过。换成 **snapB_400000 那批 demo**(§35 的 snapshot 系列,一个"弱 base":
policy_only 只有 ~0.20-0.24)重做同一对照,两臂各跑满 **100k transitions**:

| 窗口(每点 100 局) | dynens(5 头) | plain(1 头) | Δ | z |
|---|---:|---:|---:|---:|
| round-0(adapt 前) | 0.020 | **0.280** | −0.260 | — |
| 0–5k | 0.464 ± 0.046 | 0.246 ± 0.026 | +0.218 | — |
| 10–20k | 0.589 ± 0.019 | 0.439 ± 0.016 | +0.150 | +6.0 |
| 20–30k | 0.616 ± 0.018 | 0.553 ± 0.016 | +0.063 | +2.6 |
| **30–40k** | 0.609 ± 0.015 | **0.634 ± 0.012** | **−0.025** | −1.3 |
| 50–60k | 0.643 ± 0.016 | 0.573 ± 0.014 | +0.070 | +3.3 |
| **60–100k 稳态** | **0.660 ± 0.007** | **0.556 ± 0.008** | **+0.103** | **+9.6** |

**① 复现成立**:+0.103 与 default demo 上的 +0.086 同量级 ⇒ **dynens 的优势不是 demo/proposal
特异的**。两臂唯一变量 = stage-1 的 `num_dynamics_heads`(5 vs 1);stage-2 用同一批 20 demo /
同 1551 transitions(BC loss 0.00664 vs 0.00642),adapt 侧继承**同一份 cfg**
(`dynens_adapt_long_h5/cfg.yaml`,只 CLI 覆盖 `stage1_ckpt`)。
**② ⚠️ 中途窗口骗了我一次(§27 同型)**:30k 时 Δ 已从 +0.218 单调收窄到 +0.063、30–40k 甚至
**反超 −0.025**,当时我判断"可能只是 plain 收敛慢、不能下结论"。真相是 **plain 有一个瞬时峰
(best 0.68 @ 37k)然后回落并稳在 0.556**,dynens 从 ~40k 起稳在 0.66。**⇒ 早期"差距收窄"是
plain 的暂态峰,不是天花板追平;必须等两臂都进稳态段再比。**
**③ 新增的独立证据 —— plain 不稳、dynens 稳**:末 10 点 plain 0.46–0.62 剧烈抖动、dynens
0.60–0.72。与 §31「集成除掉的是模型误差的方差分量」直接吻合(单头没有跨成员平均 ⇒ rollout
方差全额进 argmax ⇒ 分数随 ckpt 抖)。
**④ round-0 反直觉再现**:plain 的 proposal 更**弱**(0.170 vs 0.230)、adapt 前 MPPI 却更**高**
(0.280 vs 0.020)⇒ 未适配的 5 头 source dynamics 对 MPPI 比单头更糟,增益**全部**来自 adapt
阶段。只看 round-0 会得出完全相反的结论。
**confound 披露**:两臂 `policy_only` 有系统差(dynens ~0.24 vs plain ~0.20,因 stage-1 encoder
与 policy 起始权重不同,BC loss 再接近也消不掉)。按 §27(固定 dynamics 时 proposal 跨
0.11→0.83 只让 MPPI 在 0.80–0.86 间动)这 0.04 解释不了 0.103,且 ④ 的方向相反。单 seed。
Runs: `threading_simdist_dynens_adapt_snapB_400000_seed1`(GPU3) /
**`threading_simdist_plain_adapt_snapB_400000_seed1`**(GPU2,NEW),wandb project mimicgen。
⚠️ 该 base 整体偏弱(dynens 只到 0.66,远低于 default demo 的 0.965)—— 与 §35 的
`contact_steps` 结论一致,不影响本对照(两臂共用同一批 demo)。

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

### ★★ 2026-08-28 Square 上的受控重测：value 的效应是 **proposal-conditional**（见 tasks §41b）

同一把尺子（oracle dynamics + 只换一个部件，rotw1/σ=0.1/n=200 每格）在 Square 上量出来的结果，
给 §33 这套结论加了一条**新的、与"dynamics 误差"正交的边界**：

| proposal 裸跑 | "把 value 换成 target 训的" 的收益 | z |
|---|---:|---:|
| π_src **0.135** | **+0.330** | 8.4 |
| SimDist stage-2 policy 头 **0.595** | **+0.015** | **0.67 (ns)** |

⇒ **"value 值不值钱"主要由 proposal 有多好决定**，不是由 value 本身决定。proposal 差时，
更好的 value 能把搜索从烂候选里拽出来；proposal 一旦 target-BC 过，value 就没东西可补。
这解释了为什么 Threading 上"别 adapt value"如此干脆（那里 proposal 一直是 stage-2 BC 出来的，
oracle + **冻结** source value 就有 99.3%），也解释了 Square 上一度看起来"value 至关重要"。

**同一批实验的完整拆账**（proposal 固定为 SimDist stage-2 policy 头）：
`dynamics 误差 +0.175 (同预算, z=5.1)` ≫ `proposal +0.030` > `value +0.015 (ns)`，任务残差 0.015。

两条对 §33 的直接修正：
1. **§41 那张"换 value 就崩到 0.42"的 2×2 作废**（两行跑在不同的 σ 和不同的初始分布上）。
   干净口径下反对角是 **0.925 / 0.940**，是**提升**不是崩塌。
2. **在 oracle dynamics 下，"value 必须与延续策略自洽"被证伪**：同一个弱 proposal 配
   *更强但不自洽* 的 value 反而好 +0.060 (z=2.2)。与 §33 不矛盾 —— §33 的机制是
   "value 的特权来自补偿 dynamics 畸变"，**dynamics 完美时没有畸变要补，自洽性就不值钱**。
   ⇒ 自洽性重要 **∝ dynamics 误差**；value 可改进的余量重要 **∝ proposal 的差距**。两条轴。

⚠️ 顺带作废：Square 上所有 value-iteration 的判决（`square_value_iter{,2}.py` / `kill_test.py`）
都建在 σ=null 的 `square_ibrl_rotw1_seed1` 上 —— 那里余量 ≤0.06 而噪声地板 ±0.04，
**信噪比 1.3:1**，而且 kill test 的前提"失败只占 ~10%"在 σ=0.1 下（失败 40%）根本不成立。

---

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

## 36. flow proposal vs UG vs MPPI —— 拆开"更强的 proposal"和"更好的搜索"（2026-08-02）

### 起因
memory `project_flow_ug_simdist.md` 记的结论是"flow BC 是更强 base policy(58>42)，
**UG 能在 flow 之上再破**(58→79)"，并把 79 与"SimDist 完整 adapt MPPI ~58%"对比。
两个问题：① 那个 58% 的 SimDist baseline 在 log 里对不上（`threading_simdist_adapt_
seed1/train.log` 的 500 局 σ0.1 in-training eval 在 28k–40k 段是 0.66–0.74）；
② 0.79 用的 WM 其 dynamics 是被 **MPPI over MLP 头**采的数据 adapt 出来的，flow 只在
部署时换上去 ⇒ **"换 proposal"和"换搜索"两个变量混在一起**。本节把它们拆开。

### ★ 主表（全部 500 局，同一 WM `threading_simdist_adapt_seed1/model0.pt`、
### 同 harness、target §12 / strict / σ0.1）

| proposal | 搜索/精修 | strict success | 延迟 ms/步 |
|---|---|---:|---:|
| MLP（WM 自带头） | MPPI 原配方(100base+250rand, 3 iters, H=3) | 0.686 ± 0.021 | **5.2** |
| **flow** | **MPPI 原配方** | **0.824 ± 0.017** | 28.6 |
| flow | UG @ coeff 0.3 | 0.790 ± 0.018 | 46.3 |
| flow | 无（Ta=1 裸跑） | 0.580 ± 0.028 (n=300) | 22.6 |

**效应分解：**
- **换 proposal（MLP→flow，搜索固定 MPPI）：+0.138，z=5.1** ← 唯一显著的大效应
- **换搜索（MPPI↔UG，proposal 固定 flow）：+0.034，z=1.37** ← **不显著，两者等价**
- 有无搜索（flow-only → flow+任一搜索）：+0.21 ~ +0.24，z≈7

⇒ **"UG 能在 flow 之上再破"这句话对了一半：破的是"搜索"，不是"UG"。** 同一个 flow
proposal 换成普通 MPPI 一样到 0.824。**UG 相对 MPPI 没有任何特殊性，而且贵 1.6×**
⇒ flow+MPPI 在两个轴上都 dominate flow+UG，是这条线该选的配置。
0.824 也贴着 §27 的天花板律：proposal 和搜索方式都换过仍卡在同一处 ⇒ 剩下的约束在
dynamics 误差，不在 proposal 也不在搜索。

### ⚠️ 又一次小样本翻车（§27 铁律第 N 次生效）
| arm | n=30 | **n=500** | 偏差 |
|---|---:|---:|---:|
| flow + MPPI | 0.933 ± 0.046 | **0.824** | −0.109（>2× 该 SE） |
| MLP + MPPI | 0.767 ± 0.077 | **0.686** | −0.081 |
两个 arm 的 n=30 都系统性偏高，且 flow 那个超出自身 SE 两倍以上。**n=30 的预览一度
让我判断"MPPI 明显强于 UG、要翻盘"，500 局的真相是打平。** 任何 ±0.1 内的比较必须 500 局。

### ★ 为什么 MPPI 反而比 UG 便宜（微基准，GPU 上"宽"免费、"深"要钱）
| (a) flow 采样 10 步 ODE | B=1 | B=100 | B=350 |
|---|---:|---:|---:|
| ms | 23.94 | 25.64 | 28.36 |
| ms/条 | 23.94 | 0.256 | 0.081 |

| (b) WM rollout H=3 | 前向 ms | 前向+反向 ms |
|---|---:|---:|
| B=1 | 0.88 | **2.61** |
| B=350 | 1.32 | 3.11 |

搜索部分：**UG = 10 串行 ×(H 前向+反向)@B1 = 26 ms；MPPI = 3 串行 ×(H 前向)@B350 = 4 ms。**
6.6× = **3.3×(串行深度 10 vs 3) × 3.0×(要不要反向) ÷ 1.5×(宽 350 的代价)**。
最能说明问题的一句：**MPPI 每步评估 1050 条候选 rollout，UG 只评估 10 条，但 MPPI 只花
UG 六分之一时间。** 这些小网络是 kernel-launch/延迟受限、不是 FLOP 受限 ⇒ **付"宽度"的
算法便宜、付"串行深度"的算法贵**；UG 结构上是深度算法（引导必须嵌进串行采样器），
MPPI 结构上是宽度算法。
**两个推论**：① 想让 UG 便宜只有 `num_inference_steps`（10→5 同时砍 flow 采样+guidance，
约 46→25 ms），没别处可省；② **MPPI 的候选数几乎免费**（B=350 前向才 1.32 ms），
350→1000 大约只多 1–2 ms —— 这条线上最便宜且没试过的旋钮。

### horizon 扫描（cost only；WM 按 H=3 蒸馏，H>3 精度上是 OOD）
拟合：**MPPI ≈ 1.4 + 1.79·H ms；flow+UG ≈ 31.4 + 3.99·H ms**（H=1/3/5/10 实测
3.2/7.3/10.3/19.3 vs 42.1/52.4/56.4/75.5）。**UG 的斜率反而陡 2.2 倍、截距还高 30 ms
⇒ 两条线永不相交**，渐近比稳定在 ~2.2×。H=20/40 那两点看着"收敛到 1.4×"是假象——
**UG 的 rollout 被 flow 的 `Tp=16` 截断了**。这也是一个结构性限制：**UG 的规划视野被
flow 的 `Tp` 卡死，MPPI 没有这个上限。**

### 图像输入会不会改变结论？（分析 + 一个实测常数，pixel 模式两条线都未实现）
两条线都是 **encode 一次 → 之后全在 latent 里跑**（MPPI 的 rollout 是 latent 的；
flow 的 obs cond 算一次给 10 个 ODE 步复用，UG 的梯度对动作求、不回传到 encoder）
⇒ **编码器是共同的加法常数，不是区分项**。实测本仓库 `networks/encoder.py::ResNet96`
（1.21M，96×96）：B=1 单视角 2.80 ms、两视角批在一起 2.46 ms、B=350 = 28.04 ms。
加 ~2.5 ms 到两边：H=3 时 MPPI 7.3→9.8、UG 52.4→54.9，**比值从 7.2× 压到 5.6×，
但 ~45 ms 的绝对差不变**（换更重的 ResNet18@224 比值还能压到 ~2.7×，绝对差依旧）。
唯一可能翻盘的是**像素空间 rollout**（dynamics 预测图像）：MPPI 要 350×H 张
(28 ms/深度 ⇒ H=3 约 84 ms)，UG 要 10×H 张但 B=1（2.8 ms/张 ⇒ 也约 84 ms）——
batch-1 单张比 batch-350 贵 35×，恰好抵消 MPPI 350× 的宽度，**两边一起变成不可用，
不会让 UG 反超**。真正因图像而变的是精度侧：encoder 进入 §33 的共适应回路，且 §30
"改进用 raw、规划用 latent"里 **raw 这个选项没了**。

### ★ 修正 §35 之外的两条旧结论
**① "flow+UG 采集的数据 adapt dynamics 更慢" —— 不成立（未对齐 transition 造成的误判）。**
`train_flow_ug_dynadapt.py`（`exps/mimicgen/threading_flow_ug_dynadapt_seed1`）与
`threading_simdist_adapt_seed1` **同起点**（`threading_simdist_tdmpc2_stage1_seed1/
latest.pt`）、都只训 dynamics 头、各用自己的部署策略采集+评测：

| transitions | MPPI 驱动(eval=MPPI,500局) | flow+UG 驱动(eval=flow+UG,200局) |
|---:|---:|---:|
| 0 | 0.594 | 0.655 |
| 3000 | **0.116** | 0.395 |
| 10000 | 0.654 | 0.515 |
| **20000** | **0.704** | **0.720** |
| 0–20k 均值 | 0.518 | **0.590** |
| 末 5 点均值 | 0.691 | 0.679 |

**两者本质相同**：15k 起完全重合，终点 0.704 vs 0.720（差 0.016 在噪声内）。均值那
0.072 全来自 1k–8k 暂态——**MPPI 的早期暂态反而更糟**（掉到 0.116）。memory 旧记的
"0.72 < 76.6 所以慢"是拿 **20k 比 40k**，不合法。（一个未测的假说：MPPI 那 250 条随机
候选确实探到 source-dyn 错的地方、采集信号强，但同一批随机候选在还没修好的 dynamics
上做 argmax 也最容易被 model exploitation 打崩 ⇒ 早期掉得更狠。）
**⇒ 要隔离"base policy 作为 dynamics 老师"的贡献，需要交叉评测**（同一评测策略 ×
transition 对齐地测两个 WM）。现只有一半：flow+UG 测 MPPI-adapt 的 WM **@40k** = 0.766。
**② "SimDist 完整 adapt MPPI ~58%" 作废** ⇒ 正确值 **0.686 ± 0.021**（`model0.pt`，
500 局，同 harness）；训练内平台 0.66–0.73。

### Where（代码改动，全部向后兼容）
- **BUG FIX** `rl/simdist_planner.py:231` + `rl/flow_ug_agent.py::_base_seq` ——
  `base_seq_fn` 原来靠 `states.shape[0]==1` 猜调用方是 `plan()`(要 3 维) 还是
  `plan_batch()`(要 4 维)，但 **mp eval 里最后一个环境未结束时 `plan_batch` 也是 B=1**
  ⇒ 返回 3 维、与 4 维随机候选 `torch.cat` 失败（`eval_origmppi_*.log` 当初就死在这，
  所以 flow+MPPI 这格一直没有数字）。改为 hook 契约固定 batched、由 `plan()` 自己
  squeeze。全仓库只有 `flow_ug_agent` 设过这个 hook，其它路径逐位不变。
- `eval_simdist_mlp_arm.py`（NEW）—— 用与 `eval_flow_ug.py` **完全相同**的 env_params /
  strict / σ / mp-eval 协议评 SimDist 的 MLP arm（`--use_planner 0` 则是 policy-only 参照），
  自动从 adapt run 的 cfg 追到 stage-1 cfg。填的就是主表第一行。
- 微基准（scratchpad，未入库）：`bench_latency.py` / `bench_horizon.py` /
  `bench_flowmppi.py` / `bench_breakdown.py` / `bench_pixel.py`。

### 遗留的 confound（结论的边界）
1. **flow 拿到了 MLP 头没有的"挑选优势"**：flow 是 `finetune_flow_ug.py` 8000 步里按
   9 个 ckpt × 100 局 best-by-eval 挑的（best_step=7000）；SimDist stage-2 是固定
   `bc_steps=1000`、无挑选（而按 §35，1000 步对所有 base 都已过训）。所以 **+0.138 里
   有多少是"flow 架构本身"、多少是"多挑了一次"，没拆开**。
2. **挑 ckpt 的执行模式与部署不一致**：`finetune_flow_ug.py:97` 建 agent 时没覆盖 Ta
   ⇒ 选择过程跑在 flow 原生 **Ta=8** 开环（曲线只有 0.02–0.24），部署/主表是 **Ta=1**。
   Ta=8 vs Ta=1 差极大(0.150 vs 0.580) ⇒ **step 7000 未必是 Ta=1 下最好的 ckpt**。
3. 全部单 seed、单 WM ckpt、单 flow ckpt。
4. 2×2 有一格**结构上不存在**：没有"MLP + UG"——UG 是对**迭代式生成采样器**去噪轨迹的
   引导，确定性 MLP 一次前向没有可插入的中间状态。所以只能靠 flow 行内部隔离搜索方式。

### ★★ 2026-08-03/04 搬到 dynens 上 + 预注册的剂量-响应：**proposal 只在它 binding 时值钱**
把上面的 flow-proposal 换头搬到 §31 的 dynens 5 头 WM（H=5）上，跑了两个天花板差异极大
的 adapt run。**关键设计**：两者共享同一个 stage-1 ckpt（`dynens_diversev2_h5_seed1/latest.pt`）
+ 同架构 + 同 planner 原配方 + 同 eval 协议，**只差 stage-2 用哪 20 条 demo** ⇒ 架构内部的
干净对照，不是跨架构比较。全部 **500 局 / arm**，target §12 / strict / σ0.1，同一 harness。

| WM | ckpt | MLP+MPPI | **flow+MPPI** | **Δ** | z |
|---|---|---:|---:|---:|---:|
| §36 弱 WM（`adapt_seed1`, H=3） | model0 | 0.686 ± .021 | 0.824 ± .017 | **+0.138** | 5.1 |
| **`dynens_adapt_snapB_400000`**（H=5） | latest | 0.706 ± .020 | **0.840 ± .016** | **+0.134** | 5.2 |
| **`dynens_adapt_long_h5`**（H=5） | ablate_frozen | **0.952 ± .010** | 0.926 ± .012 | **−0.026** | −1.7 |

**预注册（跑之前写下的）**：long_h5 Δ∈[−0.03,+0.02] 点估计 0.00；snapB Δ∈[+0.05,+0.20] 点估计
+0.12；交互 >0.05 且 z>2。**三项全部命中**（实测 −0.026 / +0.134 / 交互 **0.160, z=5.3**）。
⇒ **"flow proposal 值 +0.14" 是错的表述。正确的是：proposal 只在它是 binding 约束时才值钱，
值多少 ≈ 把 proposal 补到该系统天花板的那段距离。** 两个 0.69–0.71 的系统各涨 ~0.135 到
0.82–0.84；0.95 那个已到顶，换更强 proposal 反而略负（方向与 §31 的 strong-base 对照一致：
"强且窄"的 base 略有害）。long_h5 的 MLP 基准 0.952 ± .010 与 §31 用另一脚本 n=2000 测的
0.962 ± .004 一致 ⇒ **两把尺子交叉验证通过**。

### ★ §36 的两个 confound 现在有数字了（都不足以解释掉效应）
**① 挑选优势**：flow best-of-9 = 0.840 vs flow **fixed-1000**（与 SimDist stage-2 固定
`bc_steps=1000` 匹配、无挑选）= 0.794 ± .018，差 **+0.046 (z=1.88, 边缘)**。
⇒ **+0.134 里 ~+0.088 (66%) 是 flow 本身、~+0.046 (34%) 是"多挑一次"**；匹配预算下 flow
仍赢 **+0.088 (z=3.23)**。confound #1 关闭。
**② `base_policy_std`**（flow 的候选本来就随机、再加 0.1 噪声 ⇒ base 池比设计更宽）：
去掉噪声后 0.798 ± .018（vs 0.840，Δ=−0.042, z=−1.73 边缘），**但仍比 MLP 基准高 +0.092
(z=3.39)** ⇒ flow 的增益来自提议质量、不是噪声撑宽了候选池。副产品：**flow 自身的采样
多样性还不够**，之上再加 0.1 噪声还能赚 ~0.04（边缘，需更多局）。
**③ Ta 选择模式**：`finetune_flow_ug.py` 新增 `--eval_ta`（默认 0 = 旧行为）。在 snapB demo
上用 **Ta=1** 选择，曲线 0.30→**0.590@4000**（旧的 Ta=8 选择只有 0.02→0.24、best 在 7000）
⇒ 坐实"原 ckpt 不是 Ta=1 下最好的"。snapB 上 flow-only **0.59 vs MLP policy-only 0.257**（2.3×）。

### ⚠️ 一个被证伪的判据：**oracle 探针不能用来 gate proposal 实验**
设计时我写的 gate 是"若 oracle≈0.99 ⇒ 缺口是 dynamics 误差 ⇒ flow 也救不了、Δ≈0"。
snapB 实测 oracle **0.963**（窄预算 N=32/base10/H=3/iters=2，n=300；同预算下 learned 0.383、
policy 0.257 ⇒ **learned→oracle = +0.580, z=19**），按那个 gate 应当预测 Δ≈0，**而 B2 实测
+0.134**。⇒ **gate 错了，两个轴被我当成一个**：
- `oracle − learned` 量的是 **dynamics 贡献**（模型若完美还能多拿多少）；
- `flow − MLP` 量的是 **当前 proposal 有没有把这个不完美模型能给的榨干**。
一个系统可以**同时**是 dynamics-bound（+0.58）**和** proposal-improvable（+0.134）——模型有
误差时更好的 proposal 让 argmax 一开始就落在好 basin，模型误差可乘之机更少，这不需要模型变准。
**⇒ 想知道 proposal 是否 binding，只能直接换 proposal 试；别用 oracle 探针代理。**
（幸好没真按 gate 停掉 B2。另注：同一 snapB 上窄预算 learned 0.383 vs 原配方 0.706 ⇒
**planner 预算本身值 +0.32**，与 §27"planner 是 proposal 提升器"一致。）
⚠️ B0 的 `state_std collected (needle-pos std 1.2 mm)` 是坏的（§27 整轨道口径是 target
92.8mm）——大概率因 snapB 的 policy 太弱、采 std 的 rollout 早早失败。本次 `oracle_noise_sigma=0`
未用到注入所以无害，但**别在 snapB 上跑 §27 那套 oracle+噪声剂量实验**，mm 换算是错的。

### Where（本轮新增，全部向后兼容）
- `finetune_flow_ug.py`: `--eval_ta`（选择 eval 的 Ta，0=旧行为）+ `--save_fixed_step N`
  （额外存 `flow_fixed<N>.pt` = 无挑选对照）。两个默认关 ⇒ 旧 run 逐位不变。
- ckpt：`exps/mimicgen/threading_flow_snapB_seed1/{flow.pt(best@4000), flow_fixed1000.pt}`
  （从 `threading_flow_ug_seed1/flow_source.pt` 出发，FT 在 `threading_d0_snapB_400000` 的 20 条 demo 上）。
- ⚠️ **每个 WM 必须配 demo-matched 的 flow**：snapB 的 stage-2 用 `threading_d0_snapB_400000`、
  long_h5 用 `threading_d0_target_dynamics`。把 long_h5 的 flow 拿去测 snapB 会把"换 demo 集"
  和"换 proposal"混在一起。两个 flow 都从同一 `flow_source.pt` + 同超参出发，只差 `--target_data`。

### TODO
- [x] 拆开 confound 1（挑选优势）与 Ta 选择模式 —— done 2026-08-03，见上（+0.088 是 flow 本身）。
- [x] flow proposal + MPPI 搬到 dynens 5 头 WM —— done，见上（long_h5 −0.026 / snapB +0.134）。
- [ ] **最便宜的没试旋钮**：MPPI 候选 350→1000（约 +1–2 ms）能否把 snapB 的 0.840 再推高
      （long_h5 已到顶，别在那试）。
- [ ] `num_inference_steps` 10→5/3 对 flow+MPPI 分数的影响（能把 28.6 ms 砍到 ~15 ms）。
- [ ] snapB 的 0.840 是不是它自己 dynamics 的天花板？测法 = 在**原配方**预算下跑 oracle 臂
      （窄预算的 0.963 不可直接比），或继续加强 proposal 看还涨不涨。
- [ ] dynamics-adapt 的交叉评测 2×2（同一评测策略 × transition 对齐），隔离"谁采的数据
      是更好的 dynamics 老师"。**这轮明确没做**（原计划的 Phase 2）。
- [ ] 全部单 seed（单训练 seed、单 flow ckpt）；eval 侧 500 局。方法级结论仍需 ≥2 训练 seed。

---

## 39. 针变粗 = 环变小的镜像旋钮；oracle 的 1.000 存在窗口下限（2026-08-07）

### What
新增 `needle_size_scale`（**默认 1.0 ⇒ 逐位兼容**）：把针杆横截面（标称 10×10 mm）整体放大，
与 `ring_size_scale`（把 16 mm 环孔缩小）构成**同一个物理量的两侧**——插入间隙
`clearance = (ring_opening − needle_cross_section)/2`。设计与 `ring_size_scale` 完全对称：
只缩放 `needle_obj_needle{,_vis}` 的 x/z 半尺寸，**不动 y（长度）、不动 geom_pos** ⇒
`geom_xpos["needle_obj_needle"]`（`threading.py:_check_success` 唯一读的量）不变 ⇒
**12 mm 成功判据不移动**。质量/惯量**故意不随体积缩放**（保持与 `mass_scale` 可分离）。

**两个可分离性检查（都通过）**：
- **不影响抓取**：从 20 条 target demo 实测夹爪抓取相指间距 **38.7 mm**（p5–p95: 38.7–38.9）
  ⇒ 夹的是 **handle（40 mm 立方）**，不是针杆。针 10→12 mm 与夹爪无接触。
- **不影响成功判据 / 不累积**：4 次 reset 后 shaft 稳定 12.000 mm、len_y 120.0 mm、
  `geom_pos=[0,-0.02,0]`、成功半径 12.0 mm、mass 22.80 g 全部不变（`geom_size` 在
  `snapshot_nominal` 里，每次 reset 从 nominal 重新组合）。

### ★ 主表（全部 n=40，同一 WM `threading_simdist_adapt_long_h5_seed1/latest.pt`，
### oracle = 真实模拟器 rollout + **模型自己 source 蒸馏的 value 头**（未 adapt），
### strict / σ=0.1 / MPPI N=32,base10,H=3,iters=2；除 TF 外都是 §12 target 接触物理）

| # | clearance | ring | needle | policy | learned | **oracle** |
|---|---:|---:|---:|---:|---:|---:|
| TA | 2.00 mm/边 | 1.00 | **1.20** | 0.425 | 0.875 | **1.000** |
| base | 1.80 mm/边 | 0.85 | 1.00 | 0.400 | 0.875 | **1.000** |
| TC | 1.30 mm/边 | 0.85 | **1.10** | 0.125 | 0.675 | 0.975 |
| TD | 1.30 mm/边 | **0.7875** | 1.00 | 0.200 | 0.600 | 1.000 |
| TE | 1.05 mm/边 | 0.85 | **1.15** | 0.000 | 0.225 | **0.400** |
| **TB** | **0.80 mm/边** | 0.85 | **1.20** | 0.000 | 0.000 | **0.000** |
| TF | 0.80 mm/边 | 1.00 **(source 物理)** | **1.44** | 0.000 | 0.000 | **0.200** |

**⇒ 直接回答"针变粗 1.2"**：
- **叠加在现 target 上（保留 ring 0.85）= TB = 0.80 mm/边 ⇒ 三个 arm 全 0.000**，任务崩掉，
  **不是可用的 target**。
- **替换（ring 回 1.0，纯针变粗）= TA = 2.00 mm/边 ⇒ oracle 1.000 / learned 0.875 /
  policy 0.425**，与现 target（1.80 mm/边）**逐项相同**。

### ① 两个旋钮只通过"间隙"起作用（互换性，两组独立配对一致）
- **TC vs TD 精确匹配 1.30 mm/边**（一个走针侧、一个走环侧）：0.125/0.200、0.675/0.600、
  0.975/1.000 —— 三项差都 ≤0.075，落在 n=40 的 SE（≈0.06–0.08）内。
- **TA vs base**（2.00 vs 1.80）：learned **0.875 = 0.875**、oracle **1.000 = 1.000**。
⇒ 可以按需从任一侧设置容错窗口；`needle_size_scale` 是 `ring_size_scale` 的等价替代。
⚠️ n=40，按 §27 铁律 ±0.1 内不下定论；结论强度靠"两组独立配对方向一致"，不靠单对。

### ② ⭐ 新边界：**oracle 的 1.000 只在 ≥1.3 mm/边成立**
§27 记的是"模型误差=0 ⇒ 天花板 1.000，与容错窗口无关"（target 1.8 mm/边 oracle 0.993）。
本节把**分母**一路压下去，发现该结论有下限：
`1.80→1.000, 1.30→0.975/1.000, 1.05→**0.400**, 0.80→**0.000**`。
⇒ 窗口窄到某个程度后，**即使 dynamics 完美也做不到**，binding 约束从"模型误差"换成
**执行/控制精度**（OSC + 20 Hz + 这个 proposal/planner 预算能达到的对准精度）。
§27 的"天花板 = f(模型误差 / 容错窗口)"应加限定：**仅在窗口 ≳1.3 mm/边（本系统）时，
分子→0 才能把天花板推到 1.0**。

### ③ TB 的 0 不是 proposal 塌造成的（confound 已排除）
§27 记过"无 proposal（`num_base=0`）在两个域都是 0.000 ⇒ planner 不能从零规划"，
本来会让"TB 的 policy=0.000 ⇒ planner 必 0"成为竞争解释。**TE 否掉了它**：
TE 的 proposal 同样是 **0.000**，learned 仍有 **0.225**、oracle **0.400**。
⇒ **"proposal 得分为 0" ≠ "没有 proposal"**——一个自己完不成任务的 policy 依然能播出有用的
候选种子。所以 TB 的崩塌来自窗口本身。

### ④ 0.80 mm/边不是几何不可行，是**窗口 × target 接触物理的交互**
同样 0.80 mm/边，换成 **source 物理**（TF：nominal 夹爪 kp、nominal 摩擦、nominal 质量，
只把针放大到 14.4 mm）⇒ oracle **0.200 ≠ 0**。
⇒ 几何上穿得过去；是 target 的**软夹爪（kp 0.12）+ 环摩擦 2.0 + 3× 质量**把它推到 0。
（针穿过环需走 10 mm 环厚，0.8 mm/边 ⇒ 角度容差仅 atan(0.8/10)≈4.6°，vs base 的 ≈10°。）

### Where（全部默认关、逐位兼容）
- `env/domain_shift.py`：`DomainShiftConfig.needle_size_scale`（默认 1.0）、
  `resolve_threading_bindings` 新增 `needle_geom_ids`（碰撞 geom **+ `_vis` 双胞胎**——
  与 ring 不同，针变粗是物体的**真实可见属性**，不是要藏起来的隐藏 gap，所以渲染应同步）、
  `apply_needle_shift()`（含可行性表；`>=1.60` 在 ring 1.0 下即不可行）。DR 路径**不**采样它
  （与 `tripod_*` trap 旋钮一致）。
- `env/robosuite_wrapper.py`：fixed-shift 分支里 `apply_ring_shift` 之后调用。
- `evaluate/eval.py`：`--shift_needle_size_scale`。
- `scripts/eval_oracle_dyn_mppi.py`：`--needle_size_scale` / `--ring_size_scale`
  （<0 = 不覆盖），在 `--shift` 之后叠加，日志打印换算好的 mm 与 clearance，
  并在 needle ≥ ring 时告警。⚠️ 注意与既有的 `--needle_scale`（加宽针的**摆放box**）区分。
- `scripts/collect_target_demos.py`：`--ring_size_scale/--ring_friction_scale/
  --needle_size_scale` 加进显式覆盖表（否则按 §38 的坑，采 demo 时会静默退回 ckpt 的几何）。

### Caveats
- **n=40**（SE 峰值 ≈0.08）。headline（TB=0.000、TA=base）是 0/1 边界所以无歧义；
  TC-vs-TD 的 ≤0.075 差**不构成结论**，仅"两组配对方向一致"。
- **learned / policy 两个 arm 都是在 base target（ring 0.85 / needle 1.0）上适配的**，
  部署到改过窗口的域属 out-of-domain；oracle arm 的 rollout 不受影响（真实模拟器），
  但其 value 头与 proposal 仍来自 base。所以本表读作"**同一个系统在不同窗口下**"，
  不是"每个窗口各自训练后的天花板"。
- 单 seed、单任务（Threading_D0）。

### TODO
- [ ] TC vs TD 互换性补到 300 局（learned arm 即可，不必跑 oracle）把 ±0.075 收紧。
- [ ] 若要用"针变粗"造新 target：**TA（ring 1.0 + needle 1.2）**是与现 target 等难度的
      干净替代；想更难就走 TC/TD 档（1.30 mm/边），**不要用 TB（0.80）**。
- [ ] ② 的新边界值得单独定位：在 1.05–1.30 之间加密（1.1/1.2 mm/边）找 oracle 从 1.0
      脱落的确切拐点，并查是"对准精度"还是"抓取内针位姿抖动"限制的（后者可用
      gripper_kp sweep 分离——TF 已给出方向性证据）。
