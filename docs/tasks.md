# 多任务：Square_D0 · Coffee_D0

第二/第三个环境的 source 专家、target 域设计、SimDist 链条。加新任务时照这个流程走。

> 从 `CLAUDE.md` 拆出（2026-08-14）。速查表、方法论铁律、文件地图仍在 `CLAUDE.md`；已归档分支见 `CLAUDE_ARCHIVE.md`。

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

## 40. 新任务 Coffee_D0（第三个环境）—— source 域一轮打通到 99.8%（2026-08-08）

### What
接入第三个 MimicGen 任务 **Coffee_D0**（把咖啡胶囊插进 pod holder **并合上盖子**），
走 §32 在 Square 上验证过的配方：官方 core 数据 → MLP BC prior → **纯原始 IBRL**。
一轮跑通，无需任何调参或额外杠杆。

### 环境规格（实测，非从文档抄）
| 项 | 值 |
|---|---|
| **state 维度** | **66** = 9 prop + 57 object-state |
| object-state 57 | 4 个 body（pod / machine / pod_holder / lid）× (pos3+quat4+to_eef_pos3+to_eef_quat4)=56 + hinge_angle 1 |
| action | 7（OSC_POSE + gripper） |
| 官方 rollout horizon | **400**（`mimicgen/__init__.py:106`） |
| 数据 | core 1000 demos，mean 223 步（min 210 / max 249），223,130 transitions，100% 成功 |

**两阶段任务**（Threading/Square 都是单阶段）：
1. 抓 `coffee_pod`（MG subtask 1，`subtask_term_signal="grasp"`）
2. 插进 pod holder **+ 关盖**（MG subtask 2）。盖子在 `_reset_internal` 里被
   **强制打开到 120°**，必须关到 **<15°**。

**成功判据** `_check_success = _check_lid and _check_pod`（纯瞬时几何，无 dwell）：
- lid：`hinge_angle < 15°`
- pod：水平中心距 < `holder_r − pod_r` = **5.2 mm**（holder 29.5 / pod 24.3）；
  且垂直方向 pod 整体在 holder 底面之上、lid 底面之下

**⭐ reward-hack 风险远低于 Threading**（所以首轮没上 strict_success）：
`coffee_machine` 总质量 **6.32 kg**（Threading 的 tripod 只有 **18 g**）⇒ 推不动，
没有"推环作弊"这条路；pod 必须落进实体杯壁围出的体积，不像 Threading 的 12 mm 球
可以穿模掠过。随机策略 5/5 局零虚假成功。

**初始分布（D0 最窄）**，30 次 reset 实测：
| | D0 | D1 | D2 |
|---|---|---|---|
| coffee_machine | x=0, y=−0.1, z_rot=−30° ⇒ **spread 0 mm，完全确定** | x/y 各 10 cm，z_rot −30°~+60° | 移到右侧 |
| coffee_pod | 实测 **54×55 mm** 方框，z_rot 固定 | 25×13 cm | 移到左侧 |

pod→holder 水平距离 ~27 cm，holder 高于桌面 **17.4 cm**（要抬起来插）。

### ★ 结果（全部 500 局 / source / seed 1，同一 harness）
| | 500 局 | SE | vs prior |
|---|---:|---:|---:|
| MLP BC（prior） | **94.40%** (472/500) | ±1.03% | — |
| RNN-GMM BC | 92.40% (462/500) | ±1.19% | — |
| **IBRL `model0`** | **99.80%** (499/500) | ±0.20% | **+5.4pt, z=5.1** |
| IBRL `latest` | **97.80%** (489/500) | ±0.66% | +3.4pt, z=2.8 |

`model0` 是"单次 50 局最高分"挑的、必然乐观，所以诚实区间是 **[97.8, 99.8]**
（`latest` 这个未经挑选的点也**显著**高于 prior，所以"IBRL 有真实策略改进"这一条
不依赖于挑 ckpt）。⚠️ 与 §32 Square 不同的是，这次 100 局数字**站得住**
（BC 100 局 0.94 → 500 局 0.944），没有重演 Square 那次 94%→89.2%。

### ⚠️ §32 的"更强 prior"杠杆在这里**不存在**
Square 上 MLP prior(63%) → IBRL 89.2%，换 RNN-GMM prior(83%) → IBRL 97.8%（+8.6pt, z=5.6）。
Coffee 上 **MLP 94.40% vs RNN 92.40%，z=1.27 不显著，MLP 反而略高且更便宜**（单步、无 chunk）。
⇒ 不是 RNN 不行，是 **MLP 根本不是瓶颈** —— 任务对 BC 来说已接近饱和。
**判据**：先量 prior 的绝对水平，只有它明显低于任务上限时"换更强 prior"才是杠杆。

### 训练曲线：包络上升 + 剧烈震荡（三次被稀疏采样骗到）
600k 步 / 5h22m。50 局 in-training eval 的形状：
`0.00(10k) → 0.18(100k) → 0.02–0.90 剧烈震荡(200–250k) → 0.86–0.92(300k)
→ 0.70–0.94(350k) → 0.86–1.00(400k+) → 末段 6 点均值 0.977`
`bootstrap_bc`：0.097 → 0.33(150k) → **0.199(500k)**（先升后降 = critic 先学会靠 prior、
actor 再接管），是比分数更平滑的进度指示器。

**⚠️ 方法论教训（本轮连栽三次，全部同一个根因）**：我先用**每 50k 采一点**的监控读曲线，
依次得出「稳步上升」→「崩溃了（0.32→0.02）」→「卡在 0.4 上不去」三个结论，**全是混叠假象**
—— 改成每次打印**最近 5 个点**后真相是 0.02–0.90 的震荡带、包络稳定上升，且 230k 就摸到过 0.90。
连带作废了我据此编的一个机制解释（「train 0.91 vs eval 0.44 说明 critic 系统性高估 RL actor」
—— 对齐同一时刻后两者是吻合的 0.90/0.90、0.70/0.76，那个 gap 是跨时刻比出来的）。
**⇒ 读任何训练曲线都必须看窗口不看点；`±0.1` 内的判断在 50 局 eval 上没有信息量（SE≈7%）。**
另：平台期仍会偶发单点塌陷（540k 的 0.24，train 同期 0.94→0.79 再回来）。

### Where（3 个新 config，全部 source 域、无 shift、无 strict）
- `release/cfgs/mimicgen/coffee_d0_state_source_mlp_bc.yaml` → `coffee_source_mlp_bc_n1000_seed1`
- `release/cfgs/mimicgen/coffee_d0_state_source_rnn_bc.yaml` → `coffee_source_rnn_bc_n1000_seed1`
- `release/cfgs/mimicgen/coffee_d0_state_ibrl_source.yaml` → `coffee_ibrl_source_seed1`
  （纯 IBRL：**无 prior_margin、无 add_bc_loss**；600k 不是 400k —— §32 Square 在 400k 只有 77%）
- 数据：`release/data/mimicgen/core/coffee_d0/processed_data96.hdf5`（9.87 GB —— 转换器默认带了
  `agentview` + `eye_in_hand` 图像，对 state 训练无影响，以后做 pixel 模式可直接复用）

**⇒ Coffee_D0 source 专家 = `exps/mimicgen/coffee_ibrl_source_seed1/model0.pt`（99.8%）。**

### TODO
- [ ] **source 域基本已解决 ⇒ 研究价值在别处。** 三个方向：
      ① **D1**（machine 也随机，代码零改动、只换 `task_name`）；
      ② **自定义 target 域** —— 但 `resolve_bindings` 对 Coffee 会 raise，需新写
      `resolve_coffee_bindings`。Coffee 的天然旋钮很干净：pod/holder 的 **5.2 mm 径向间隙**
      （对应 Threading 的 `ring_size_scale` / §39 的 clearance 口径）、**盖子铰链的阻尼/摩擦**
      （Threading/Square 都没有的第二阶段）、`gripper_kp`（§32 在 Square 上验证过是最强杠杆）；
      ③ 先诊断 BC 那 5.6% 的失败分布（抓 pod / 插入 / 关盖），再决定旋钮往哪拧 —— §32 在
      Square 上正是这个诊断（45% grasp / 40% insertion）指出了"更强 prior 能补一半缺口"。
- [ ] **§27 天花板律的跨任务检验**：Coffee 的容错窗口 **5.2 mm 径向** vs Threading 的
      1.8 mm/边（§39 口径）⇒ 预测 Coffee 的 SimDist 天花板应**明显高于** Threading 的 0.85。
      这是"天花板 = f(模型误差 / 容错窗口)"第一次能在第三个任务上被证伪。
- [ ] 全部单 seed；方法级结论需 ≥2 训练 seed。

### 40b. Coffee_D0 TARGET 域定稿（2026-08-08）—— 用**动力学 gap**选，不是用成功率选

#### ⚠️ 先说方法上的修正：成功率是错的筛选口径
第一轮我按"专家成功率掉多少"扫旋钮，得出 `mass_scale` / `friction_scale` /
`pod_holder_friction_scale` "无效"（全部 100%）。**这是错的口径**：一个旋钮完全可以大幅
改变 `P(s'|s,a)` 而专家依然做得成。改用直接测量后它们各自贡献 0.015–0.044 mm/步的散度，
叠进配方把 p90 抬了近一倍。**要造 dynamics gap 就必须直接量 gap。**

`scripts/diag_coffee_dynamics_gap.py`（NEW）：从 demo 取真实 `(sim_state, action)`，在
source 与 target 各走**一步**，量 pod / lid / eef / gripper 的散度（物理单位,不是 66 维范数
—— 避免四元数符号污染且可解释）。两臂都 `qacc_warmstart=0` 冷启动 **+ `controller.
reset_goal()`**（§35 记的 OSC goal 不在 sim_state 里的 confound，这里显式消掉），
⇒ 唯一差异是模型参数。**控制臂 `source` 实测 0.001 mm**（必须≈0，否则整张表作废）。

#### ★ 定稿 target = `C75kp25`
| 字段 | 标称 | target | 改的是什么 |
|---|---:|---:|---|
| `gripper_kp_scale` | 1000 | **0.25** → 250 | 手指执行器增益（严格说是**执行器**gap） |
| `lid_damping_scale` | 0.005 | **75** → 0.375 | 盖子铰链粘性阻尼（**只打第二阶段**） |
| `mass_scale` | 6.58 g | **3.0** → 19.7 g | pod 质量 + 惯量 |
| `friction_scale` | 1.0 | **0.5** | pod 滑动摩擦 |
| `pod_holder_friction_scale` | 1.0 | **2.0** | 杯壁摩擦（μ 取 max ⇒ 插入更涩） |

**σ=0.1 下的基线（500 局，`coffee_ibrl_source_seed1/model0.pt`）**：
source 域 **88.60%** → target **24.00%**，gap **64.6 点**。
（σ=0.02 时是 99.8% → 39.2%；**换初始分布本身就吃掉 11 点**，target 掉 15 点 ⇒ 宽初始分布
与 domain gap 有轻微复合。之后一律 σ=0.1，读数请拿 88.6% 当分子参照，别把初始分布的代价
记成 domain gap。）

候选对照（全部 500 局；gap = pod 单步散度 mean/p90 mm）：
| 配方 | kp | lid | mass | pfric | hfric | gap mean/p90 | 成功率(σ=.02) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | .20 | 100 | – | – | – | 0.167/0.302 | 11.8% |
| B | .20 | 75 | – | – | – | 0.162/0.249 | 26.6% |
| C50 | .30 | 50 | 3 | .5 | 2 | 0.188/0.336 | 73.0% |
| C75 | .30 | 75 | 3 | .5 | 2 | 0.195/0.353 | 46.4% |
| **C75kp25** | **.25** | **75** | **3** | **.5** | **2** | **0.198/0.362** | **39.2%** |

两条从对照里读出来的性质：
1. **`lid_damping` 50→75 几乎不改 gap（0.336→0.353）却改 26.6 点成功率** —— 因为它只作用在
   close-lid 相，而该相只占 4% 的转移。**难度分档用它，不牺牲 gap。**
2. **gap 旋钮不可加**：`kp` 单独测 gap 是 0.128 mean，但在已有 mass/摩擦的组合里再从 0.30
   收到 0.25 只换来 +2.5% p90（噪声内），成功率却掉 7 点。

#### ⛔ 两个被实测判死的旋钮（别再试）
- **`pod_holder_size_scale`（收窄杯口）= NO-OP**。0.995→0.94 全部 100%（干涉量已到 1.09 mm）。
  根因不是"行程不够"：成功带上沿允许 pod 中心到 0.9953，此时 pod 最宽的那圈（中心+22 mm）
  在 1.0173、而杯沿在 1.002 ⇒ **pod 最宽处根本不需要穿过收窄的杯壁**，真正要过的是锥形下半部
  （那个高度半径 ~23.4 mm）。要咬住需 scale≈0.93，而那已是"完全插不进"的悬崖。无可用窗口。
  ⚠️ 另注：`env.pod_size[0]`=24.30 mm 是 MujocoObject 规格的包围盒，比**真实 mesh 最大半径
  24.56 mm 小 0.26 mm**；而常被引用的 5.2 mm 是**成功判据容差**（用 holder 外半径 29.5 mm），
  **不是物理间隙**（真实物理间隙只有 0.41 mm）。三个数别混。
- **`solref_scale`**：Coffee 的 pod 标称 timeconst **0.001 s**，已在 MuJoCo 的
  `2*timestep=0.004 s` 钳制线**以下** ⇒ 缩放落在钳制/语义不明区。实测有 0.031 mm 效果但机制
  未查清。**不进配方。**（对比 Threading 的针是 0.02 s，那里 solref 才是真旋钮。）

#### Where
- `env/domain_shift.py`：`DomainShiftConfig` +4 字段（`pod_holder_size_scale` /
  `pod_holder_friction_scale` / `lid_damping_scale` / `lid_frictionloss`，全默认 1.0/0.0）、
  `resolve_coffee_bindings()`（显式拒绝 `CoffeePreparation_D0`：它是 Coffee 子类但多 mug/drawer
  且判据不同，套用会静默给出错误 target）、`apply_coffee_shift()`、`snapshot_nominal` 新增
  `dof_frictionloss`（restore 处用 `in` 保护，旧 snapshot 仍可载）。
- `env/robosuite_wrapper.py`：fixed-shift 分支加 `elif "pod_holder_geom_ids" in bindings`。
- `evaluate/eval.py` +4 个 `--shift_*` flag；`scripts/collect_target_demos.py` +4 个显式覆盖
  （§38 的坑：不在显式表里的字段会被静默退回 ckpt 旧值）。
- **逐位兼容已验证**：`mode=fixed` 全默认 vs `mode=none`，11 个模型字段 max|Δ| = **0.000e+00**；
  旋钮开启后只动 lid 那 1 个 dof + 恰好 64 个杯壁 geom；`r_diff` 恒为 0.0052。

#### Runs（σ=0.1 起为标准）
- `release/cfgs/mimicgen/coffee_d0_state_ibrl_source_sig01.yaml` → `coffee_ibrl_source_sig01_seed1`
  （400k，`snapshot_period 20000` ⇒ 20 个质量谱 ckpt，供 `generate_simdist_data.py`）
- `release/cfgs/mimicgen/coffee_d0_state_ibrl_target.yaml` → `coffee_ibrl_target_seed1`
  （target C75kp25，1M，prior = source **hybrid**）
- ⚠️⚠️ **`source_agent_use_hybrid: 1` 必须显式写，默认是 0**。`load_frozen_agent` 会 pin
  `act_method="rl"` 摘掉 source 自己的 BC prior ⇒ target 竞争的是 source 的**裸 RL 头**（一个
  source 从不执行的策略）而非 99.8% 的 hybrid。**这是 §35 记录的同一个坑第 4 次咬人**（本轮
  第一次起 target run 就踩了，warm-up 阶段发现并重起）。确认方式：去 run 自己的
  `<save_dir>/train.log` 找 `[SGFT] prior = source HYBRID` —— nohup 的 stdout 会缓冲这行。
- CAVEAT：当前 target run 的 prior 是 **σ=0.02** 训的（用户选择先跑起来）。
  `coffee_ibrl_source_sig01_seed1` 完成后换上重跑可消除该 confound。

#### TODO
- [ ] target IBRL 曲线 + 500 局定稿；对照 24.0% 零样本起点。
- [ ] 用 σ=0.1 的 source 专家换掉 prior 重跑 target IBRL（消除 σ 错配）。
- [ ] §32 式迭代 bootstrapping（用上一轮 hybrid 当下一轮 prior）能否把 target 推到 90%+。
- [ ] 收 target demos → SimDist adapt，检验 §27 天花板律在第三个任务上是否成立。
      ⚠️ 我原先"Coffee 窗口 5.2 mm 比 Threading 的 1.8 mm 宽 ⇒ 天花板更高"的预测**作废**
      （量错了：那是成功判据容差不是物理间隙；真实间隙 0.41 mm，比 Threading **窄**）。
- [ ] 全部单 seed。

### 40c. ⚠️ planner 超参不可跨任务移植 —— `init_std` 一个旋钮值 40 个点（2026-08-10）

**现象**：在 Coffee target 上跑 oracle 探针（真实模拟器 rollout + source 的
`V=min-ensemble Q(s,π(s))` + source proposal，**模型误差=0**），用 §17 在 Threading 上调出来的
planner 默认值 `init_std=0.15 / num_samples=32 / iterations=2`，成绩只有 **50%**（C75）/
**38%**（C75kp25）。把 `init_std` 改成 **0.40**（**其余不变、零额外算力**）：

| C75 oracle（proposal+value 全是 source） | init_std | samples×iters | 成绩 | 失败 |
|---|---:|---:|---:|---|
| 基线 | 0.15 | 32×2 | **50.0%** | close-lid 24 |
| **只加半径** | **0.40** | 32×2 | **90.0%** | close-lid 4 |
| 只加预算 | 0.15 | **128×4** | 84.0% | close-lid 8 |
| 两者都加 | 0.40 | 128×4 | **94.0%** | close-lid **0** |

重测其它域同样翻盘：**C75kp25 38%→90%、C50 92%→98%**。
⇒ **半径是免费的、预算要 8× 算力才拿更少 —— 先调 `init_std`。**

**为什么会踩**：§17 收窄 `init_std` 的论证是"Threading 稀疏 reward 下 `V_src` 在专家成功流形上
近乎平坦，随机候选会稀释好的 proposal"。那个论证依赖 Threading 的 value 地形。Coffee 实测
`std_i(γ^H V)/V_level = 0.033`，是 Threading 的 **0.0068（≈1.4%）的 ~2.4 倍** —— value 地形完全
不同，收窄的前提不成立。**我照搬了旋钮，没照搬对它的前提做检查。**

**作废的一整串推论**（都是基于窄搜索的数字）：
"Coffee 的 source value 跨不过 domain"、"难度被 close-lid 相垄断所以 oracle 救不回来"、
"C50(92%) 与 C75(50%) 有本质区别"（真值 98% vs 90%，差距从 42 点缩到 8 点）。

**本轮死掉的假设**（记下来免得重试）：① value 盲（同一 source value 在 C50 拿 98%）；
② value 饱和（target 专家的 value 区分度反而**大 2.5 倍**：std/level 0.082 vs 0.033）；
③ 聚合信号不足（C50 与 C75 的 std/level **完全相同**都是 0.0331）；④ 视野太短（H=10 即使配足
128×4 预算仍是 **0.000**，长视野在这里真的有害）；⑤ proposal 缺这个动作模式（source proposal
配宽搜索就到 94%）。

**一个独立于 §33 的新结论**：把终端 value 换成**该域 95.8% 的 target 专家**的 value，
oracle 从 50% 崩到 **2.0%**（失败 49/49 全 close-lid），而它的跨候选区分度**更大**（0.082 vs
0.033）—— 不是分不出，是**分得很确信但方向错**。§33 把这类崩溃归因于"value 与 learned dynamics
共适应，换掉就破坏共适应"；**这里 dynamics 是完美的真实模拟器、不存在共适应，换 value 照样崩**
⇒ "planner 需要的是一个**自洽**的排序函数，'不同'本身就是伤害"这条比 §33 的共适应解释更普适。
（机制假说未验证：target 的 `V=min-ensemble Q(s,π_tgt(s))`，π_tgt 在 source-proposal 采出的候选
状态上 OOD，min-ensemble 的悲观被放大。）

**另一个反直觉观察（未解释，两个都 500 局）**：σ=0.1 训练的 source 专家域内更强
（93.8% vs 88.6%）、零样本迁移却更差（**13.6% vs 24.0%**，C75kp25 上）。⇒ **一次比较里绝不
可混用这两个专家。**

**操作准则**：换任务/换 target 后，**先扫 `init_std`（0.15/0.3/0.4/0.6）再下任何"天花板"结论**；
并把 `[signal] std/level` 打出来与 Threading 的 0.0068 对照，它是 value 地形是否可比的快速判据。

---

## 41. Square_D0 rotw1 全链条 + MPPI 信任域的两堵墙（2026-08-08~12）

### What
Square 上从 source 专家一路打通到 SimDist target adapt 的完整可比链条，全部锁在
**rotw1 初始分布**（nut z_rot 窄化到 120–180°）+ **σ=0.1**。副产品是一条通用结论：
**MPPI 在 learned dynamics 上的信任域由两个旋钮合取构成，缺一不可**。

### rotw1 是什么、为什么
Square_D0 的螺母 z_rot 是**全 360°**，而 Threading_D0 只有 **60°**（−120°~−60°，且三脚架
完全固定）。把 Square 窄化到同宽的 60° 窗口 ⇒ 学习显著加速（→0.9 从 520k 降到 170k）。
⚠️ **窗口位置很重要，不只是宽度**（我最初的推论"位置不重要"被证伪）：三个同宽 60° 窗口的
BC prior 成功率差 3 倍（0–60° 26% / 120–180° 72% / 240–300° 78%），因为 handle 单侧外伸、
**没有任何旋转对称性**（周期 360°，不是 90°），朝向直接决定 IK 顺不顺。选 **120–180°**
（handle 朝机器人侧，∠56°）。⚠️ §32 记的"[45,90)° 成功率 62%"是**折叠到 90° 的错误口径**
（`yaw % 90`），不可用于设计。

### 资产（全部 rotw1 + σ=0.1，可直接复用）
| | 路径 | 成绩 |
|---|---|---|
| source 专家 | `square_ibrl_rotw1_sig01_snap_seed1`（8 snapshot） | source 0.98 |
| source diverse 数据 | `square_d0_source_rotw1_diverse` | 200k traj，`expert_hybrid=1 snapshot_hybrid=0`（§17 v4 配方） |
| **target-B 专家** | `square_ibrl_target_B_rotw1_sig01_seed1`（6 snapshot） | target **0.98–1.00** |
| target demo | `square_d0_target_B_rotw1_demos` | 50 条，98% 接受率，75–88 步 |
| stage-1 plain | `square_simdist_rotw1_h5_seed1` | source 0.93 |
| stage-1 dynens | `square_simdist_rotw1_dynens_h5_seed1` | 跑中（5 头） |

**target-B 专家单轮 bootstrapping 就到 ~1.00**（§32 全 360° 上要三轮迭代才 94.2%），
关键是 prior 的 **σ 和 window 都对齐**。⚠️ 需 `source_agent_use_hybrid=1`（新增字段，默认 0）
——`load_frozen_agent` 一直 pin `act_method="rl"`，所以 §32 说的"用上一轮 hybrid 当 prior"
**其实一直是裸 RL 头**，那段描述不准确。

### ★★ MPPI 信任域：两个旋钮，init_std 主导（⚠️ 已按稳态数据修正）

> **修正记录（2026-08-14）**：本小节初版基于 **20–40k 中途窗口**，结论是"交互 +0.151 超可加、
> 只修一个仍低于 policy 裸跑"。C/D/E 跑满 100k 后**全部翻转**——根因是 **D（只改 init_std）
> 一直在爬、远未收敛**：0.416(20-40k) → 0.517 → 0.639 → **0.737**(80-100k)，最终追平 C。
> 又一次栽在"没等稳态就比"（§31 记过同型）。以下是**稳态(60–100k)**的正确版本。

| 稳态 60–100k | init_std 主效应 | base_std 主效应 | 交互 |
|---|---:|---:|---:|
| 20–40k（作废） | +0.158 | +0.291 | +0.151（超可加） |
| **60–100k** | **+0.294** | **+0.128** | **−0.136（次可加）** |

**⇒ `init_std` 才是主导旋钮**（不是初版说的 base_policy_std）；两者**次可加**，单修 init_std
已能拿到大部分收益。**"只修一个仍低于 policy 裸跑"是错的**（D=0.687–0.737 ≫ policy_only 0.55）。
**但 `base_policy_std=0` 仍值 +0.128，且它让收敛快得多**（C 在 20k 就到 0.78，D 要 80k+）。

### 原始的 2×2 数据（下表 20-40k 列即上面被修正的那个窗口）
SimDist 的 `apply_original_recipe`（`--original_mppi 1`）设 `init_std=1.0` + `base_policy_std=0.1`。
在 Square 上这两项**同时**是灾难。adapt 2×2（100 局/点，20–40k 窗口，其余逐字段相同）：

| | `base_policy_std 0.1` | **`base_policy_std 0.0`** |
|---|---:|---:|
| **`init_std 1.0`** | **0.333**（原生） | 0.548 |
| **`init_std 0.15`** | 0.416 | **0.782** |

（上表为 20–40k；**稳态 60–100k 为 A 0.324 / D 0.687 / E 0.521 / C 0.747 / B(light6) 0.802**，
见上面的修正记录。）round-0（adapt 前，100 局）：0.170 / 0.310 / 0.290 / 0.620，交互 +0.190
—— **round-0 的超可加是真的**（那时没有 adapt 可以补救），但 **adapt 会把单修臂救回来**，
所以稳态交互变负。⇒ 正确表述：**adapt 能补上一部分信任域漏洞，但要多花 4 倍的 transition**。

**机制**：`base_policy_std=0` 时那 100–410 条 base 候选是 policy rollout 的**完全相同副本**
（policy 确定性 + 同一 z0）。这看似浪费，实为**信任域**：它们在 top-k elite 里构成压倒性投票，
把 CEM 的 mean 钉在 proposal 上。加 0.1 噪声后它们变成各异的变体，于是 value+learned dynamics
挑"**在模型里看起来最好**"的那条 = 模型误差的最大化方向。`init_std=1.0` 则让随机候选铺满
[-1,1]，全在分布外。**两个旋钮是两堵墙，补一堵搜索就从另一堵漏出去** —— 信任域是**合取**不是加法。
**dynamics adapt 补不了这个漏洞**（round-0 的交互结构在 100k adapt 后完整保留）。

**⇒ Square 的最小充分 planner 配方**（`train_simdist_adapt.py --planner_set`，新增；
`evaluate/eval.py --planner_set` 同语义）：
```
--original_mppi 1 --planner_set "init_std=0.15,base_policy_std=0.0"
```
另外 4 项（temperature 2.0 / iterations 2 / num_samples 128 / num_elites 16）在 20–40k 只值
~0.03，但**稳态下值 +0.055**（`light6` 0.802 vs `both2` 0.747，C 从 0.791 回落而 B 没有）——
仍在基线 seed 跨度 0.035 的两倍以内，暂记为"小而可能真实"，需第二个 seed⇒ **改"怎么排序"的旋钮没用，改"候选池覆盖
哪里"的才有用**。⚠️ Threading 用原生配方是成功的，所以这不是 `apply_original_recipe` 本身坏，
是**它与某些 world model 不兼容**；换任务必须重扫这两项。

### ★ planner 的价值完全由 dynamics 精度决定（Square 上的干净对照）
| dynamics | proposal 裸跑 | MPPI | planner 净值 |
|---|---:|---:|---:|
| **oracle**（真实 sim, target-B） | 0.24 | **0.87–0.92** | **+0.63** |
| **learned**（source, 训练域内！） | 0.92 | 0.935 | **+0.015 (ns)** |
| learned（target, adapt 后） | 0.55 | 0.78 | +0.23 |

**在自己的训练域内、proposal 已 0.92 时，learned dynamics 的 MPPI 榨不出任何东西**
（12 个配置全扫过，最好 0.935，两 seed 复现 0.935/0.930，vs policy 头 0.920，z=0.6 不显著）。
规律单调：任何加强搜索的方向都更差，极限就是完全信任 proposal。**要 source 接近 100% 只能改
dynamics，不能改 planner。** §27 天花板律在第二个任务上独立成立。

### ★ 蒸馏出的 V̂ 可直接替代 V_src，且离线保真度**双向骗人**
target-B + oracle dynamics，200 局：**V̂ 0.845±0.026 vs V_src 0.870±0.024（z=0.7）**，
失败**都是 100% insertion**（grasp/transport 全 0）。但离线指标说 V̂ 是个糟糕近似：
```
集内 Spearman +0.325 · top-8 elite 重合 38% · perturb/signal 0.688
bias spread 0.121 vs perturb 0.024  ⇒ 84% 的偏差是逐状态常数偏移
```
按 §33 剂量-响应（0.207 → 掉 0.18）ratio=0.688 该是灾难，实际零损失 —— 因为常数平移对 CEM 的
`topk+softmax` 严格免费（§33 的 `ε̄`）。**⇒ MPPI 的 +0.63 不来自候选间的精细排序**，只来自
粗粒度结构 + elite 平均的平滑；精细排序打乱的是近似并列项。temperature 独立印证：`temp=2` 时
有效 elite 数 **7.99/8**（softmax 完全失效 = 均匀平均），调到 50 让它降到 4.4，**分数不变**。
**这是 §33 的镜像**：那边"极小离线残差照样杀死你"，这边"极大离线排序分歧完全无害" ——
同一条教训的两个方向，离线指标与闭环**双向**脱钩。

### ⛔ 用 target 数据调 value：判死
⚠️⚠️ **下面这张 2×2 已于 2026-08-28 被受控重测推翻，作废，见 §41b。** 它的两行跑在
**不同的初始分布和不同的 σ** 上（π_src 行 σ=null/rotw1，π_tgt 行 σ=0.1/全 360°），
干净口径下反对角根本不塌（0.925 / 0.940，z≈9 的**提升**）。本段其余关于在线迭代拟合的
记录同样受影响：那些实验全部建在 σ=null 的 `square_ibrl_rotw1_seed1` 上，那里余量 ≤0.06
而噪声地板 ±0.04 ⇒ **信噪比 1.3:1，判决无效**。
2×2（oracle dynamics，50 局/格）：
| | π_src (裸跑 0.24) | π_tgt (裸跑 0.96) |
|---|---:|---:|
| **V_src** | **0.900** ✓ | **0.420** ✗ |
| **V_tgt** | **0.420** ✗ | **1.000** ✓ |
对角 0.90/1.00，反对角 0.42/0.42。**最强签名在右上**：π_tgt 自己 0.960，配 V_src 被拖到 0.420，
且 grasp 失败从 0 涨到 15。⇒ **约束是"value 必须与延续策略一致"，不是 value 的绝对准确度**
（V_tgt 客观更准，插进来掉 0.48）。这比 §33 的"共适应补偿 dynamics 畸变"更普适——这里 dynamics
是 oracle、不存在共适应，换 value 照样崩。
在线迭代拟合也判死：MC 一次拟合就 0.940→0.000（残差/信号 4.0）；换 TD(0)+信任域闸门+revert
后残差降 190×（ratio 0.007–0.04）、8 轮 400 局均值 **0.905 ≡ 基线 0.900**；最后在**失败子集**上
做高功效判死（λ∈{0,0.15,0.5,1.0}），λ=0.15 救回 10/12 与**什么都不做的对照 9/12 无法区分**，
λ=0.5 只有破坏（成功局 19→13）。

### ⚠️⚠️ 两个测量地板（比任何效应都大，先量再谈）
1. **episode 级不可复现**：λ=0（逐位相同的 value）+ 相同 seed，12 个失败局有 **9 个变成成功**。
   定位实验（同 seed 连跑两次 + 中间插一局无关的）显示**第一个动作 t=0 就已不同**。根因 =
   ProbePool worker 的 `sim.get_state()` **不含 `data.ctrl` 和 OSC 控制器的 Python 侧 goal**，
   `restore()` 倒不回去（§35 记过的同一个 confound）。**作废**：任何 per-episode 配对推理。
   **仍成立**：多局聚合的期望估计（本节所有 ≥0.1 的差距）。
2. **基线的 seed 间跨度 = 0.035**（同配置 200 局，seed0 0.845 vs seed1000 0.880）⇒
   **任何 <0.05 的"改进"都在地板以下**。

### 两次预注册复现（一败一成，正好对照）
| 干预 | 发现(n=50) | **复现(新 seed, n=200/臂)** | 判决 |
|---|---:|---:|---|
| `num_elites 8→3` | +0.070 | **−0.030** | ❌ 符号翻转（§38 的 forking-paths） |
| `init_std 0.30`（oracle 臂） | +0.075 | **+0.040** | ✅ 复现（0.920 vs 0.880） |
区别正是机制：**elites 改的是"怎么排序"（已证不是瓶颈），init_std 改的是"候选池覆盖哪里"。**

### 三个 provenance 坑（都已修）
1. `collect_target_demos.py` 的 `--hole_offset` 默认 **0.01**、`--solref_scale` 默认 **0.8**，
   **无条件覆盖** ckpt shift（§38 的老坑还在）⇒ Square 必须显式传 `0 / 1.0 / 1.0`。
2. **reset 分布（rotw1 窗口）从 ckpt 的 `eval_init_dist` 隐式继承、CLI 上不可见、sidecar 不记录**
   ⇒ 已补进 `collect_target_demos.py` + `generate_simdist_data.py` 的 `env_cfg.json`。
3. `train_simdist.py` **原本完全没有 init_distribution 通路** ⇒ 数据是 rotw1、env 却建成全 360°，
   `source_score` 不可比。已补 `train_init_dist`/`eval_init_dist`（默认关、旧 cfg 逐位兼容）。
4. stage-1 cfg 的 `eval_domain_shift` 曾误继承 **target-A**（kp0.25/peg1.25）而非 target-B
   （kp0.30/peg1.20）—— 对蒸馏无影响（纯离线），但下游 adapt 会继承错。已修。
`evaluate/eval.py` 现在每次都打印 `[EVAL ENV] task= shift= nut_z_rot= sigma= strict= ep_len=`，
**两个数只有这行一致才可比**。

### Where（新增/改动，全部默认关、逐位兼容）
- `env/domain_shift.py`：`InitDistConfig.nut_z_rot`（默认 `(0,2π)`）；`apply_init_distribution`
  改为按 env class 分派。⚠️ **Square 必须先判**：它也有 `_get_initial_placement_bounds`，但
  Square 在 `__init__` 就把 sampler 建死了（`nut_assembly.py:406` 只在 None 时重建），
  monkeypatch 那条路对它**无效**；Square 走 `placement_initializer.samplers["SquareNutSampler"].rotation`
  直接赋值（跨 hard reset 持久，`reset()` 不碰 `rotation`）。
- `train_rl.py`：`source_agent_use_hybrid`（默认 0）+ eval 路径同步。
- `train_simdist.py`：`train_init_dist`/`eval_init_dist`。
- `train_simdist_adapt.py` / `evaluate/eval.py`：`--planner_set "k=v,..."`（在 `original_mppi` 之后生效）。
- `scripts/eval_square_oracle_mppi.py`（NEW）：Square 版 oracle arm（source value + 真实 sim rollout
  + `--simdist_ckpt` 用蒸馏 V̂ + `--value_weight` 换 value）。
- `scripts/square_value_iter{,2}.py`、`scripts/square_value_kill_test.py`（NEW）：value 迭代与判死。

### TODO
- [ ] dynens stage-1（跑中）→ adapt（planner 用 `both2`）。判据：plain 的 **0.78**
      （= oracle 0.92 的 85%）；§31 Threading 先例是 plain 0.879 → dynens 0.965。预测 **0.85–0.90**。
- [ ] C/D/E 三条 adapt 跑满 100k 补齐 2×2（现为 20–40k 窗口）。
- [ ] 修 `ProbePool.restore()`（存/恢复 `data.ctrl` + 控制器 goal）并加"同 seed 两次必须逐位相同"
      的断言；修好后 per-episode 配对才可用。
- [ ] §38 提醒：demo 用的是**已触顶**的 600k ckpt（标准 SimDist 做法）；snapshot 都在，
      想榨那 ~0.14 可换 `snapshot_200000/300000` 重采。
- [ ] 全部单 seed（训练侧）。

---

### 41b. ★★ 受控重测：Square 的缺口是 **dynamics**，不是 value —— §41 的 2×2 作废（2026-08-28）

#### 起因：`eval_square_oracle_mppi.py` 的环境来自 `--weight` 的 cfg
`train_rl.load_model(args.weight)` 顺带带出 `env_params`，所以 **换 proposal 会静默换环境**。
旧 2×2 因此在两个轴上同时被污染：

| 行 | ckpt | σ | `eval_init_dist` |
|---|---|---|---|
| π_src | `square_ibrl_rotw1_seed1` | **null（=0）** | enabled，120–180° |
| π_tgt | `square_ibrl_target_B_iter3_seed1` | 0.1 | **disabled，0–360°** |

⇒ 旧的 `0.900 / 0.420 / 0.420 / 1.000` 全部作废。**新增的 `[setup] planner ... | hand_init sigma= | strict= | ep_len=`
日志行就是为了这个** —— 两个数只有这两行一致才可比。

#### 干净口径（全部：rotw1 120–180° · σ=0.1 · strict=**false** · oracle dynamics · 同一 planner
`N=32 (base 10) H=3 iters=2 elites=8 temp=2.0 init_std=0.15 base_policy_std=0.0` · n=200/格）

| 格 | proposal（裸跑） | value | **oracle MPPI** |
|---|---|---|---:|
| A | π_src (0.135) | V_src | **0.595 ± 0.035** |
| B | π_src (0.135) | V_tgt@600k | **0.925 ± 0.019** |
| E | π_tgt@300k (0.510) | V@300k（**自洽**） | **0.885 ± 0.023** |
| F | π_tgt@300k (0.510) | V@600k（更强，**不自洽**） | **0.945 ± 0.016** |
| C | π_tgt@600k (0.975) | V_src | **0.940 ± 0.017** |
| D | π_tgt@600k (0.975) | V_tgt@600k | **0.985 ± 0.009** |

资产：source = `square_ibrl_rotw1_sig01_snap_seed1`（σ=0.1、source 域、rotw1）；
target = `square_ibrl_target_B_rotw1_sig01_seed1`（σ=0.1、target 域、rotw1，6 个 snapshot）。
**弱 π_tgt = `snapshot_300000.pt`，独立 200 局 = 0.510**（训练中 50 局的 eval 在同一段
给出 0.78/0.84/**0.50**/0.82/**0.58**，摆幅 0.34 ≈ 5×SE ⇒ 训练曲线不可用来标 snapshot 强度）。

#### ★ 结论 1：反对角不塌，两个单换都是大幅**提升**
`只换 value` **+0.330 (z=8.4)**、`只换 proposal` +0.345 (z=9.0)、`两个都换` +0.390 (z=10.9)。

#### ★★ 结论 2：value 的效应是 **proposal-conditional**，不是主效应
同样是"换成 target 训的 value"：

| proposal 裸跑 | Δ | z |
|---|---:|---:|
| π_src **0.135** | **+0.330** | 8.4 |
| SimDist stage-2 policy 头 **0.595** | **+0.015** | **0.67** |

⇒ **那个 +0.33 是"proposal 太烂"的替身**：proposal 差时更好的 value 能把搜索从烂候选里拽出来；
proposal 一旦经过 target BC，value 就没东西可补。这条把三处观测统一起来 ——
Threading（proposal 本来就好 ⇒ oracle+冻结 V = 99.3%，value 不是瓶颈）、
Square 弱 proposal（value 看着至关重要）、Square 真实系统（value 归零）。

#### ★ 结论 3：F > E ⇒ 在 oracle dynamics 下"value 必须与延续策略自洽"**被证伪**
同一个弱 proposal，配**更强但不属于它**的 value 反而更好 **+0.060 (z=2.2)**。
⚠️ 这不推翻 findings §33，而是给它划边界：§33 的机制是"value 的特权来自补偿 dynamics 畸变"，
**dynamics 完美时没有畸变要补，自洽性自然不值钱**。

#### ★★★ 结论 4：完整拆账 —— dynamics ≫ proposal > value
proposal 固定为 `sq_dynens_anch_d400000_seed2/latest.pt` 的 stage-2 policy 头（裸跑 0.595），
`--simdist_base_rollout 1` 让 base 候选与 SimDist 一样是闭环 rollout ⇒ 两边只差
"scoring 用真实 sim 还是学出来的模型"：

| 项 | 值 / Δ | z |
|---|---:|---:|
| G2 value = source 专家 critic | 0.855 ± 0.025 | — |
| **G1 value = 冻结蒸馏 V̂** | **0.940 ± 0.017** | — |
| G3 value = target 专家 critic | 0.955 ± 0.015 | — |
| **L_small** learned + **同一小预算** | **0.765 ± 0.030** | — |
| **L_big** learned + adapt 原配方 (N=350/H=5/iters=3/elites=64) | **0.825 ± 0.027** | — |
| **dynamics 误差（同预算）** G1−L_small | **+0.175** | **5.1** |
| dynamics 误差（vs 原配方） G1−L_big | +0.115 | 3.6 |
| 搜索预算 L_big−L_small | +0.060 | 1.5 (ns) |
| **value** G3−G1 | **+0.015** | **0.67 (ns)** |
| proposal → target 专家 D−G3 | +0.030 | 1.77 |
| 任务残差 1−D | +0.015 | — |

**⇒ 上界不是 0.94。** 0.94 = "现有 pipeline（stage-2 proposal + 冻结 V̂）+ 完美 dynamics + 小预算"；
把 value 和 proposal 也换成 oracle 版本是 **0.985**，任务残差只有 0.015。
**dynamics 是唯一有量级的一项（0.115–0.175），value 最小且测不出来** ⇒
"加大数据量重训一个 target value"这条路的上限已经用最强形式测过了，就是 **+0.015**。

#### 两条附带发现
1. **蒸馏 V̂ 比 source 专家 critic 好 +0.085 (z=2.8)**。§41 记的"V̂ ≈ V_src 打平"是
   σ=0 + 专家 proposal 的口径，换到 σ=0.1 + SimDist proposal 就不成立。⚠️ 这里 dynamics 是 oracle，
   **不能再用"共适应补偿畸变"解释**，机制待查。
2. **L_big 0.825 ≈ 训练中窗口均值 0.828** ⇒ 这条臂的 in-training eval **没有乐观偏差**
   （与铁律 #14 记的 0.06–0.19 不同），图里的数字可以直接用。

#### 口径修正（会影响已发布的图和 CLAUDE.md）
- **Square 线上 `strict_success.enabled` 三边全是 `false`**（source 专家 / target 专家 / stage-1）。
  strict 的判据字段是 `max_tripod_disp` / `max_needle_speed`，本来就是 Threading 专用。
  ⇒ **把 Square 标成 "strict" 是错的**，速查表和三张图的图注都要改。
- ⚠️ provenance：**anchor 臂没有 seed1、多一个 seed2**（demo 集 `s2`），plain/dynens 是 seed1（`s1`）。
  10 seed 均值影响很小，但"arm for arm 同一个 demo 集"在那一格上不成立。配对比较要用 seed101–901。

#### Where（新增，全部默认关、逐位兼容，已回归验证）
`scripts/eval_square_oracle_mppi.py`：
- `--proposal_simdist_ckpt` —— 用 SimDist 的 policy 头当 proposal（env/value 仍来自 `--weight`）。
  自动顺着 cfg 里的 `stage1_ckpt` 找回 `MainConfig`（adapt run 写的是 `AdaptConfig`，见 §24）。
- `--simdist_base_rollout 1` —— base 候选改用 SimDist 的闭环策略 rollout，而不是"一个动作按住 H 步"
  （`base_policy_std=0` 时后者是 `num_base` 份逐位相同的复制品）。
- `[setup]` 新增 planner 全参数 + `hand_init sigma` / `strict` / `ep_len` 三行。
  ⚠️ 顺带修了一个自造的 bug：`bool(env_params['strict_success'])` 对**配置对象**恒为 True。
`evaluate/eval.py` 本来就有 `--use_planner` / `--planner_set`（第 238–252 行），
matched-budget 对照直接可跑；但 `--weight` 必须指向带 `MainConfig` 的目录
（做法：把 stage-1 的 `cfg.yaml` 和 adapt 的权重放进一个临时目录，如 `exps/tmp/sq_matched_budget/`）。

#### TODO
- [x] **P1/P2/P3 其实已经跑完了**（2026-08-28 21:15–23:26 串行，seed101，共用 demo 集
      `square_sat_400000_s101`，n=200/格，`base_policy_std=0.0` 口径）：

      | 格 | stage-2/3 臂 | proposal 裸跑 | **oracle MPPI** |
      |---|---|---:|---:|
      | P1 | plain | 0.560 | **0.900 ± 0.021** |
      | P2 | dyn-ens | 0.740 | **0.965 ± 0.013** |
      | P3 | dyn-ens + anchor | 0.700 | **0.970 ± 0.012** |

      读数：**`P3−P2` = +0.005** ⇒ dynamics 完美时 anchor 不值钱（但基线 0.965、余量只剩 0.035，
      按铁律 #18 这个 null 是算术必然，不算证据）。**`P2−P1` = +0.065** ⇒ dynens 的收益有一部分
      不是 plan-time 平均（oracle 下 dynamics 误差已为零，仍然差 0.065 ⇒ 来自 policy 头/anchor 差异）。
      ⚠️ **provenance：`sq2x2_P*.log` 已被删除，上面的数字只存在于 2026-08-28 会话记录**，
      事后不可核（同 [[feedback-ckpt-provenance-topksaver]] 的教训）。要引用请重跑。
      ⚠️ 之前记成"未完成"是错的 —— 打死驱动的是第一次并发尝试，后来串行重跑成功了。
- [ ] G/L 系列只有 seed2 一个 ckpt，方法级结论需 ≥2 seed。
- [ ] 结论 4 的"上界"仍受小预算限制：oracle 臂每个候选要一次真实 sim rollout，
      复刻 N=350/H=5/iters=3 要 ~27× 算力（200 局约 40h）⇒ 0.940 多半被低估。

---

### 41c. ★★★ learned vs oracle dynamics × 4 种 proposal × 2 种 value —— **模型误差决定了"别的部件重不重要"**（2026-09-14）

#### 口径（全部一致，跨格可比）
Square rotw1 120–180° · σ=0.1 · **strict=false** · planner
`N=32 (base 10) H=3 iters=2 elites=8 temp=2.0 init_std=0.15 base_policy_std=0.1`
· base 种子 **held-action**（除特别标注）· 单 stage-1 seed
（plain `square_simdist_rotw1_h5_seed1`；stage-2/3 = `sq_plain_d400000_seed101`）。
learned 列的 dynamics + V̂ **三行共用同一个 world model**（`sq_plain_d400000_seed101`）⇒ 列内只有
proposal 在变，与 oracle 列精确配对。日志 + GPU trace 全部存档：`exps/tmp/sq_oracle_ladder/`。
⚠️ **§41b 那批是 `base_policy_std=0.0`，与本节不可直接相减**（该旋钮本身实测零效应：
L(plain) learned std=0.0 **0.735** vs std=0.1 **0.735**，n=200 各一次）。

#### ★★★ 主表

| proposal | 裸跑 | WM 见过? | **learned + V̂** | 规划增益 | **oracle + V̂** | **oracle + V_tgt** | **Δ dynamics** | z |
|---|---:|:-:|---:|---:|---:|---:|---:|---:|
| stage-2 头（source 蒸馏 + target-demo BC） | 0.480 | ✅ | **0.650** | +0.170 | 0.980 | 0.980 | **+0.330** | 6.4 |
| il_ft BC（source BC 热启动 + target demo） | 0.500 | ❌ | **0.590** | +0.090 | — | 0.960 | — | — |
| from-scratch BC（只有 20 条 target demo） | 0.200 | ❌ | **0.130** | **−0.070** | 0.910 | 0.990 | **+0.780** | 17.6 |
| stage-1 头（纯 source 蒸馏） | **0.000** | ❌ | **0.010** | +0.010 | 0.920 | 0.910 | **+0.910** | **31.5** |

n：learned 列 100/格；oracle 列 stage-2 行 50、其余 100。

#### ★★★ 结论 1：**oracle 列极差 0.07，learned 列极差 0.64 —— 差 9 倍**
- **oracle dynamics**：规划增益 **+0.50 / +0.71 / +0.92**，把任何 proposal（含裸跑 **0.000** 的）
  都拉到 **0.91–0.98**。⇒ **完美模型让 planner 与 proposal 基本无关。**
- **learned dynamics**：规划增益 +0.17 / +0.09 / **−0.07** / +0.01，proposal 差时**规划反而有害**
  （from-scratch BC：裸跑 0.200 → 规划后 **0.130**）。⇒ **学出来的模型让 planner 受 proposal 支配。**

⇒ **模型误差不只是压低天花板，它改变了其他部件重不重要。** 这是 findings §36「proposal 只在
binding 时值钱」的机制版：**learned dynamics 下 proposal 是 binding 的，oracle 下不是。**
⚠️ oracle 列不是严格平的（stage-2 0.980 vs BC 0.910，z=2.01 勉强显著），但 0.07 vs 0.64 的
对比不受影响。

#### ★★ 结论 2：换 value ≤ +0.04（oracle 下），4 个读数里 2 个为零
| proposal | base | V̂ | V_tgt | Δ | z |
|---|---|---:|---:|---:|---:|
| stage-2 (0.480) | held | 0.980 | 0.980 | 0.000 | 0.0 |
| stage-2 (0.480) | learned-WM rollout | 0.920 | 1.000 | +0.080 | 2.1 |
| from-scratch BC (0.200) | held | 0.910 | 0.990 | +0.080 | 2.6 |
| stage-1 (0.000) | held | 0.920 | 0.910 | −0.010 | −0.3 |

逆方差合并 **+0.037 ± 0.016 (z=2.2)**，异质性 **Q=6.4, df=3, p≈0.09** ⇒ 不是同一个效应量，
**别当一个数引用**。正确表述：「至多 +0.04，比 dynamics 项小一个数量级，且不是每次都测得到」。
★ **V_tgt 列内跨行**：stage-1 行 (0.910) 显著低于另两行（vs W2 0.990 **z=−2.64**；
vs O2c 0.980 **z=−2.01**）⇒ **target value 不是普遍有效的**。
★ **V̂ 列的对照排除了"policy 弱"这个替代解释**：V̂ 列里 stage-1 (0.920) ≈ BC (0.910)，
而 V_tgt 列里两者分化 (0.910 / 0.990) ⇒ 分化只由换 value 产生。
⚠️ **第 2 行是唯一的反例**（同 proposal、同 value，只差 base 构造，给出 +0.080 vs 0.000）。
大概率是 O1 那格 (0.920) 向下波动（O1c 同配置 0.980，n=50 差值 SE 0.038），但"挑掉偏低的格"
是事后解释（铁律 #3）⇒ **O1/O2 需重跑到 n=200 才能定。**

#### ★★ 结论 3：learned 列由 **proposal 的裸跑分**决定，**不是**"离被适配的 policy 有多远"
learned 列的单调（0.650/0.590/0.130/0.010）同时跟着两个共线变量：(a) 离 stage-3 适配所用
policy 的距离；(b) proposal 自己的裸跑分。**专门造臂分开**（铁律 #10 推论）：
**il_ft BC —— 裸跑 0.500 ≈ stage-2 的 0.480，但 WM 从没见过它。**

| 对比 | Δ | z | 判定 |
|---|---:|---:|---|
| LI (0.590) vs L2 (0.650)（被适配过 vs 没被适配过，裸跑相当） | −0.060 | **−0.88 (ns)** | 无差别 |
| LI (0.590) vs LB (0.130)（裸跑 0.500 vs 0.200，都没被适配过） | +0.460 | **7.7** | 巨大 |

⇒ **(b) 胜。** ⚠️ **我的预注册预测失败**：我先判给了 (a)，还写了"和
`docs/on-policy-adaptation.md` 的「dynamics 只在 policy 邻域可修」严格一致" —— **这条推广在
本数据上不成立**，是过度解读。记在这里以免重犯（同 §44b 那次反向失败）。

#### ★ 结论 4：base 候选的构造方式**测不出差别**（三种构造、6 个读数）
`base_seq` 三条路（`scripts/eval_square_oracle_mppi.py:378-415`）：held-action（查 1 次策略、
**不用任何 dynamics**）/ `--simdist_base_rollout`（learned WM 滚 H 步）/ **`--base_real_rollout`**
（新增，真实模拟器滚 H 步）。

| 配对 | dynamics | Δ | z |
|---|---|---:|---:|
| O1c−O1（去掉 base 里的 learned WM） | oracle | +0.060 | — |
| O2c−O2（同上） | oracle | −0.020 | — |
| **W1r−W1（held → 真实 sim，余量 0.09）** | oracle | **−0.060** | −1.3 (ns) |
| **S2r−S2（held → 真实 sim，余量 0.08）** | oracle | **0.000** | 0.0 |
| GATE−L2（rollout → held） | **learned** | **+0.075** | 1.3 (ns) |

6 个读数散在 −0.060~+0.075、**两两反号**，**有余量的两格也是零**。真实 sim 闭环 base 还让
oracle 臂贵 **+30%~70%**（实测 52–55 分 vs 32–42 分 /100 局）。
⇒ **别再投 base 构造这条线**，包括「逐候选闭环 + 噪声进策略输入」那个更重的版本。
⚠️ 三种构造**都是事前加抖动**：`rollout_real` 只重放固定的 `[H,N,A]` 数组，抖动在调用前叠好
⇒ 策略永远看不见噪声、走偏也不纠正。要改必须让 worker 自带策略（CPU 推理以免新增 GPU
context）或逐步 IPC 往返 + 32 份 env 状态。**按本结论不值得做。**
⚠️ 22/32 个随机候选（`:451-452`）**根本不查策略**（第一轮 `mean` 是零向量）。
「policy 生成一个动作然后按住 H 步」只描述 base 那 10 个。
⚠️ **learned 和 oracle 两列必须用同一种 base 构造才能相减** —— 本节初稿的「dynamics
+0.185~+0.245」就是拿 rollout base 的 learned (0.735) 比 held base 的 oracle，**已作废**，
配对后是 +0.330 / +0.780 / +0.910。

#### ★ 结论 5：source 蒸馏的策略在 target 上是 **0.000**，两套代码互证
| ckpt | source | target |
|---|---:|---:|
| stage-1 头（本节 harness） | **0.960** (n=50) | **0.000** (n=100) |
| stage-1 自己的 train.log（`use_planner:0`） | 0.867–0.933 | 0.000–0.033 |

加载路径控制实验通过（同一路径 source 上 0.960）⇒ 0.000 是真实域间隙。
**dynens 那条 stage-1 也是 target 0.000** ⇒ 与 dynamics 头数无关，是 Square shift 的性质。
⇒ stage-2（target demo BC）**不是可选项**。

#### ★ 结论 6：0.48 的 proposal 里几乎全是 **source 先验**
| proposal | 知识来源 | target 裸跑 |
|---|---|---:|
| stage-1 头 | 只有 source 蒸馏 | 0.000 |
| **from-scratch BC** | **只有 20 条 target demo** | **0.180**@n=50 / **0.200**@n=100 |
| il_ft BC | source BC 热启动 + 20 条 target demo | 0.500 |
| stage-2 头 | source 蒸馏 + 20 条 target demo BC | 0.480 |

去掉 source 热启动这**一个**变量，BC 掉 0.30。⚠️ from-scratch 的 `latest.pt` 只有 **0.020**
⇒ 训练后期在 target 上退化，而 `model0.pt` 是 TopkSaver 按 **source** 分数选的
（`eval_target_during_training=0`，见 `scripts/train_target_bc_priors.sh` 自己的 caveat 1）
⇒ 对只见过 target demo 的策略是错配判据。il_ft 跨 seed 摆幅很大（s1 0.620 / s301 0.660 /
s501 0.380 / s101 0.500）⇒ 单 seed 的 proposal 强度不可引用。

#### 论文表述建议（三句，对应 claim 1/2/3）
1. 把 learned dynamics 换成 oracle：**+0.330 ~ +0.910**（随 proposal 变，三行全部 z>6）；
   相比之下任何 proposal 或 value 的替换在 oracle 下都 ≤0.08。
2. V_tgt 并非普遍有效：配 source-only policy 时只有 **0.910**，显著低于它配另两种 policy 时的
   0.980/0.990（z=−2.01 / −2.64）。
3. V̂ 与 V_tgt **在两种保留 source 分布的 policy 上打平**（Δ 0.000 / −0.010），只在纯
   target-demo 的 policy 上落后 0.080 ⇒ **关键是 policy 是否仍处于 value 训练时的分布**。
★ 统一说法：**模型误差改变了其他部件的重要性** —— dynamics 完美时 proposal/value 都不要紧
（极差 0.07 / ≤0.08），dynamics 是学出来的时候 proposal 变成支配因素（极差 0.64）且弱
proposal 下规划有害。

#### TODO
- [ ] **O1/O2 重跑到 n=200** —— 结论 2 的唯一反例，且 §41b 的「value +0.330 / 交互 −0.285」
      也挂在 A 格上（`init_std=0.15` + `base_policy_std=0` 把 CEM 钉在 0.135 的 proposal 上；
      旧 σ=0 扫描里 `init_std=0.30` 给出 oracle 0.920）。**A 格也要按 `init_std 0.30` +
      `base_policy_std 0.1` 重测**，交互项的符号随 A 翻转（A=0.595 → −0.68 log-odds；
      A=0.90 → +1.12）。
- [ ] **H=1 vs H=3 在 oracle 下未测**（H=1 时 base 构造问题彻底消失）。唯一先例是
      Threading + **learned**（findings §131）H=1 0.600 < H=3 0.825 < H=5 0.850, n=40
      ⇒ learned 上 H 大会累积模型误差、oracle 上没有这个惩罚 ⇒ 不可移植；跨任务也不可移植
      （铁律 #9）。测法：W 行（有余量）× H∈{1,3}，n=100。
- [ ] **`learned + V_tgt` 未测**，因为外部 critic 要原始状态 ⇒ 终端 latent 必须过 `decode()`，
      而 `shared_encoder=1` 下 plan 阶段**从不调用 decode**（`rl/world_model.py` decode 注释）
      ⇒ 解码误差只落在那一格、正好污染要测的 Δ。脚本里已加硬断言拒绝这个组合。
      要测必须先单独标定 decode 误差。
- [ ] 全部单 stage-1 seed；结论 2 的正效应那一半（BC 行 +0.080）只有一个 seed。
- [ ] oracle 列 stage-2 行是 n=50（差值 SE 0.034 而余量 ≤0.04）⇒ 该行内部 delta 本质不可读。

#### Where（新增，默认关、逐位兼容）
`scripts/eval_square_oracle_mppi.py`：
- **`--arms learned`** —— 与 oracle 臂共用全部 CEM 代码，只把 `rollout_real` 换成 `wm.next()`
  滚 H 步、终端值用 `wm.predict_value(z_H)` 从 latent 直接读（**无 decode**）。
  **验证门**：`--simdist_base_rollout 1` + n=200 复现 `evaluate/eval.py` 的 0.735 ⇒ 实测
  **0.725（145/200）**，200 局只差 2 局 ⇒ 移植正确。与 `--value_weight` 互斥（硬断言）。
- `--proposal_bc_ckpt` —— 用 train_bc.py 的 BC 当 proposal（无 world model ⇒
  `--simdist_base_rollout` 自动失效并告警；`base_policy_std=0` 时告警 10 个 base 同一份）。
- `--base_real_rollout` —— base 种子在**真实模拟器**里闭环滚 H 步（主进程多 1 个 EGL context，
  计入并发预算）。优先于 `--simdist_base_rollout`。
- 修了一处误导日志：`VALUE = ...` 行的 `(proposal from ...)` 之前不认 BC 分支，会写成 "the expert"。
资产：`exps/mimicgen/square_bc_tgtonly_rotw1_seed101`（from-scratch target-demo BC，
配方 = `scripts/train_target_bc_priors.sh` 但 `--load_model none`）。

#### ⚠️ 本轮的运维教训（已写进 [[gotcha-gpu-density-mppi-runs]] 第 11–13 条）
4 格并发 oracle eval 把 nvidia 驱动打死（41 个 D 态、`nvidia-smi` 自己 D 了 57 分钟、
别的用户的任务一起死、整机重启、4 格实验全废）。三个具体错误：
① 用 `nvidia-smi --query-compute-apps | wc -l` 数 context —— 它**只列有 CUDA context 的进程**，
probe worker 是 EGL-only，**从来不在这个列表里**；② §41b 的补救写成"串行**或**降 workers"，
我降了 workers（16→6）却把并发格数从 3 提到 4–5 —— **并发 run 数才是紧约束**；
③ 当天新写的 `--base_real_rollout` 在主进程多建一个 EGL context，只在 `probe_workers=2` +
机器空闲时 smoke 过就直接上 4 格并发。
**串行重跑全部成功**（单格显存足迹实测仅 7–8 GB/卡 ⇒ 显存从来不是约束）。
现在起任何 GPU 批量任务都先挂 `gpu_trace`（30s 采样显存/util/context），存档在
`exps/tmp/sq_oracle_ladder/gpu_trace*.log` 当基线。
---

### 41d. ★★★ 四任务复现 41c 的 3×2×2（48 格）—— **dynamics 12/12 为正；target value 只在 Threading 崩；"oracle 让 proposal 无关"有一个反例（TPA）**（2026-09-17）

#### 协议（四个任务同一把尺子）
每任务 3 proposal × 2 value × 2 dynamics + 2 个 decode 控制 = 14 格，n=100/格，**串行**。
planner 统一用 Square 调出来的小预算 `N=32 (base 10) H=3 iters=2 elites=8 temp=2.0
init_std=0.15 min_std=0.02 momentum=0.1 base_policy_std=0.1`，base 一律
**held-action、且在 RAW state 上查 proposal**（不经 decode）。
⚠️ **这个预算没有按任务重扫，违反铁律 #9**（§40c 实测 `init_std` 在 Coffee 上值 40 点）
⇒ **绝对值系统性低于各任务的真实水平**，可比的是 within-task 的 Δ。用户明确选了统一口径 (a)。

| 任务 | stage-1（= src only proposal） | plain adapt（= src+tgt proposal + world model + V̂） | target 专家 critic | ep_len | strict |
|---|---|---|---|---:|---|
| Square | `square_simdist_rotw1_h5_seed1` | `sq_plain_d400000_seed101` | `square_ibrl_target_B_rotw1_sig01_seed1` | 400 | false |
| Coffee G5x | `coffee_s1g5x_plain` | `coffee_g5x_3arm/cof_plain_seed1` | `coffee_ibrl_target_G5x_seed1` | 400 | **false** |
| Threading plain_h5 | `threading_simdist_diversev2_h5_seed1` | `threading_simdist_adapt_long_h5_seed1` | `threading_ibrl_rlprior_prior_sig01_seed1` | 400 | **true** |
| TPA (`ThreePieceAssembly_D0`) | `tpa_simdist_diversev1_h5_seed1` | `tpa_adapt100k_anch0_seed1` | `tpa_ibrl_target_qsrc_seed1` | **1000** | false |

⚠️ **Coffee G5x 的 `strict` 是 false**，与速查表写的「strict 仅 Threading·Coffee」不符（真源是
`coffee_s1g5x_plain/cfg.yaml`）。Threading 保持 strict=true，与它已发表的数字同口径。
⚠️ **Threading 的 "tgt only" BC 训在 `threading_d0_sat_160000_s1` 上**，不是该臂 stage-2 真正用的
`threading_d0_target_dynamics`（后者随 `release/data` 被 untrack 而本地丢失，commit 1cf5569）
⇒ Threading 这一行与它自己的 "src+tgt" 行**不是同一批 demo**。其余三任务的 from-scratch BC
都用各自 stage-2 的同一套 demo。

#### ★★★ 48 格全表

| 任务 | proposal | 裸跑 | learned+V̂ | learned+V̂ 经 decode | learned+V_tgt | **oracle+V̂** | **oracle+V_tgt** |
|---|---|---:|---:|---:|---:|---:|---:|
| Square | src+tgt | 0.480 | 0.650 | 0.390 | 0.450 | 0.980 | 0.980 |
| Square | tgt only | 0.200 | 0.130 | 0.210 | 0.230 | 0.910 | 0.990 |
| Square | src only | 0.000 | 0.010 | — | 0.010 | 0.920 | 0.910 |
| Coffee | src+tgt | 0.160 | 0.430 | 0.000 | 0.000 | 1.000 | 0.980 |
| Coffee | tgt only | 0.030 | 0.110 | 0.000 | 0.000 | 0.990 | 0.950 |
| Coffee | src only | 0.090 | 0.470 | — | 0.000 | 0.980 | 0.960 |
| Threading | src+tgt | 0.380 | 0.830 | 0.250 | 0.080 | 0.970 | **0.420** |
| Threading | tgt only | 0.000 | 0.420 | 0.270 | 0.020 | 0.980 | **0.410** |
| Threading | src only | 0.140 | 0.700 | — | 0.130 | 0.980 | **0.220** |
| TPA | src+tgt | 0.090 | 0.250 | 0.120 | 0.090 | 0.920 | 0.880 |
| TPA | tgt only | 0.000 | 0.000 | 0.000 | 0.000 | **0.440** | 0.160 |
| TPA | src only | 0.040 | 0.160 | — | 0.090 | 0.920 | 0.920 |

**四任务平均**（src+tgt / tgt only / src only）：
裸跑 `0.278 / 0.058 / 0.068` · learned+V̂ `0.540 / 0.165 / 0.335` ·
**oracle+V̂ `0.968 / 0.830 / 0.950`** · oracle+V_tgt `0.815 / 0.628 / 0.753`。

**四任务中位数**（`plots/value_dynamics_dumbbell.py` 的正图用的是这个，`--stat mean` 出均值版）：
裸跑 `0.270 / 0.015 / 0.065` · learned+V̂ `0.540 / 0.120 / 0.315` ·
**oracle+V̂ `0.975 / 0.945 / 0.950`** · oracle+V_tgt `0.930 / 0.680 / 0.915`。

⚠️⚠️ **均值与中位数在这里不是口味问题，它们支持相反的结论。** n=4 的中位数 = 砍掉两端，
而被砍掉的恰好是本节的两个发现：

| 量 | 均值 | 中位数 | 中位数丢掉了谁 |
|---|---:|---:|---|
| oracle+V̂ `tgt only` | 0.830 | **0.945** | **TPA 的 0.440** = 结论 3 的唯一反例 |
| oracle+V_tgt `src+tgt` | 0.815 | **0.930** | **Threading 的 0.420** = 结论 2 的主要证据 |
| oracle+V_tgt `src only` | 0.753 | **0.915** | Threading 的 0.220 |
| **oracle+V̂ 跨行极差** | **0.138** | **0.030** | ⇒ 中位数会说「oracle 下三行都 ~0.95」 |
| **换 value（`src+tgt` 行）** | **−0.152** | **−0.045** | ⇒ 中位数会说「换 value 只花 4.5 点」 |

这些不是测量误差型离群点：每格都是 n=100 的独立测量，且各有机制（Threading 的 V̂ 本就近乎最优，
findings §31 ③ 的 99.3%；TPA 是 1000 步多阶段任务）。⇒ **引用中位数时必须同时给出跨任务的
min–max**，正图就是这么画的（中位数标记 + 全距线 + 每任务散点，被中位数排除的点仍留在页面上）。

#### ★★★ 结论 1：dynamics 项 **12/12 格全为正**，均值 +0.43 ~ +0.67
| proposal | Square | Coffee | Threading | TPA | 均值 |
|---|---:|---:|---:|---:|---:|
| src+tgt | +0.330 | +0.570 | +0.140 | +0.670 | **+0.427** |
| tgt only | +0.780 | +0.880 | +0.560 | +0.440 | **+0.665** |
| src only | +0.910 | +0.510 | +0.280 | +0.760 | **+0.615** |

这是 48 格里唯一**跨任务、跨 proposal 一致同号**的效应，也是唯一有量级的。与 findings §27/§31、
`project_tpa_oracle_dynamics` 同向。⚠️ 加法 Δ 跨任务差 4.8×（0.14–0.91）；要跨任务汇总请用
"剩余差距缩小比例"（三个配对预算任务在 §41c 前身那轮是 0.86–1.00），不要平均加法 Δ。

#### ★★ 结论 2：**target critic 不是普遍更好** —— oracle 下均值 −0.15~−0.20，但几乎全由 Threading 贡献
| proposal | Square | Coffee | **Threading** | TPA |
|---|---:|---:|---:|---:|
| src+tgt | 0.000 | −0.020 | **−0.550** | −0.040 |
| tgt only | +0.080 | −0.040 | **−0.570** | **−0.280** |
| src only | −0.010 | −0.020 | **−0.760** | 0.000 |

⇒ Square/Coffee 上**测不出来**（|Δ|≤0.08），Threading 上**把 oracle planner 打崩**
（0.97→0.42、0.98→0.41、0.98→0.22），TPA 的弱 proposal 行 **−0.280 (z=−4.5)**。
机制与 §40c 同型（target 专家的 value 在 CEM 搜出的状态上 OOD：「不是分不出，是分得很确信
但方向错」），而这里 **dynamics 是完美的**，所以不能用"与模型误差共适应"解释。
与 findings §31 ③ 一致：Threading 上 oracle + **冻结的 source V̂** 本就 99.3%，换掉只能是净损失。
★ **效应量出现在有余量处**（TPA 的 0.44 那格），贴天花板的格子测不出来 —— 铁律 #18 的又一实例。

★★ **判别格：Threading 的崩塌是 critic 本身，不是 prior**（2026-09-17）。
被质疑点：`eval_oracle_dyn_mppi.py:659-672` 会给 value 专家**重新挂上它训练时的 prior**，
`--value_ibrl 1`（默认）时 V = `max(Q(s,π_actor), Q(s,π_prior))`。核查后**四个任务的 target 专家
都经 `sgft_source_agent` 挂了冻结 RL agent** ⇒ 函数形式一致；**差的是 prior 的来源**：
Square/Coffee/TPA 是各自的 **source** 专家，**Threading 是 target snapshots**
（`threading_ibrl_target_snapshots_rlprior_seed1`）。于是加测 `--value_ibrl 0`（裸 actor、prior 不进 max）：

| Threading `src+tgt`，oracle | 分数 |
|---|---:|
| 蒸馏 V̂ | 0.970 |
| target critic，带 prior（`value_ibrl=1`） | 0.420 |
| **target critic，裸 actor（`value_ibrl=0`）** | **0.290** |

⇒ 去掉 prior **更差**（−0.13），prior 不是元凶；**崩塌来自 target 专家的 Q 本身在 CEM 状态上排序错**。
预注册判据是 `<0.70 ⇒ critic 本身`，自动链据此退出，未重测其余 9 格。
⚠️ 我中途一度断言"Threading 的 value 函数形式和另三个任务不同"，**错了**（四个都挂 prior），已更正。
⚠️ 为这个检查给 `eval_square_oracle_mppi.py` / `eval_coffee_oracle_mppi.py` 加了 `--value_ibrl`
（原先硬编码 `use_ibrl=True`；默认 1 = 此前所有格的行为，逐位兼容）。
⚠️ 该格用时 4 小时 39 分（低成功率 ⇒ episode 跑满）。

★★ **同谱系的另两个 Threading target critic 也崩**（2026-09-17，`src+tgt` 行、oracle、n=100）：

| target critic | 训练 | oracle+V_tgt |
|---|---|---:|
| 蒸馏 V̂（对照） | — | **0.970** |
| `rlprior_prior_sig01`（表里用的） | 1M 步 | 0.420 |
| 同上，`value_ibrl=0` | — | 0.290 |
| `sig01_snapshots` | **200k 步** | **0.270** |
| `target_snapshots_rlprior`（谱系祖先） | — | **0.410** |

⇒ **训练长度（200k vs 1M）、prior 参不参与 max，都不是原因**；这条谱系的 critic 一律把
0.97 打到 0.27–0.42，而这些策略自己的成功率是 0.94–1.00。
⚠️ **边界**：Threading 上所有强的非 flow-BC target 专家**同出一支**
（热启动 + prior 都来自 `threading_ibrl_target_snapshots_rlprior_seed1`）⇒ 这三格**不是独立重复**。
要证"Threading 的 target critic 普遍如此"，需要一条独立谱系
（现存唯一候选是 flow-BC 那支 `threading_ibrl_newtgt_sig01_snap_seed1`，从 source IBRL 热启动），
或重训一个。**未做。**

★★ **多重比较：12 格里唯一为正的 Square 格经不起校正**（2026-09-17）。
四个负效应（Threading 三行 + TPA `tgt only`）在 Bonferroni（12 格，p<0.0042 ⇔ |z|>2.87）下全部存活；
**Square `tgt only` 的 +0.080 是 p=0.0083 ⇒ 校正后不显著**（BH-FDR 下勉强压线第 5 名）。
再叠一层：铁律 #15 记的 planner RNG 自带 **±0.06** 的复现噪声，而这个效应只有 +0.08。
⇒ **"target value 在 Square 上是正的"不可引用**；诚实的表述是
**「四任务 12 格中，target critic 在 4 格显著有害、8 格测不出，没有一格可信地有益」**。

★ **一致性核查（回应"其他任务是不是和 Square 一致"）**：逐项对过 target 配方 / σ / init_dist：

| 任务 | critic 训练 target == 评测 target | 评测 σ | critic 训练 σ | init_dist | prior 来源 |
|---|---|---:|---:|---|---|
| Square | ✅ | 0.1 | 0.1 | 匹配 | **source** 专家 |
| Coffee | ✅ | 0.1 | 0.1 | 匹配 | **source** 专家 |
| Threading | ✅ | 0.1 | 0.1 | 匹配 | ⚠️ **target** snapshots |
| TPA | ✅ | 0.1 | ⚠️ **None** | 匹配 | **source** 专家 |

⇒ **没有任何配置差异指向 Square**（它四项全对）。查出的两处不一致都在别的任务上，且都不解释
Square 的正号：① **TPA 的 critic 训练在无扰动（σ=None）下、却在 σ=0.1 上打分**
（`eval_oracle_dyn_mppi.py:464` 硬编码评测 σ=0.1）—— 这会让 TPA 的 −0.280 有一部分是
start-distribution 错配，需要注明；② Threading 的 prior 来自 target snapshots 而非 source 专家 ——
但上面 `value_ibrl=0` 的格已经证明 prior 不是崩塌的原因。

#### ★★ 结论 3：「完美 dynamics 让 planner 与 proposal 无关」**有一个反例**
列内跨行极差（proposal 敏感度）：

| 任务 | 裸跑 | learned+V̂ | **oracle+V̂** |
|---|---:|---:|---:|
| Square | 0.480 | 0.640 | **0.070** |
| Coffee | 0.130 | 0.360 | **0.020** |
| Threading | 0.380 | 0.410 | **0.010** |
| **TPA** | 0.090 | 0.250 | **0.480** ⚠️ |

三个 400 步任务上极差 **0.01–0.07**（oracle 把裸跑 0.000 的 proposal 分别救到 0.92 / 0.99 / 0.98）；
**TPA（`ep_len=1000`、多阶段装配）是唯一例外**：同样 oracle，stage-1 头（裸跑 0.040）到 0.920，
而 from-scratch BC（裸跑 0.000）只到 **0.440**。
⚠️ 且 TPA 上**不是裸跑分决定 oracle 分**（0.040→0.920 vs 0.000→0.440）；区别像是 proposal 的
**来源**：前两行是 world model 自己的 policy 头（共享 encoder、动作与蒸馏数据同族），
BC 是独立 MLP。**未做判别臂，勿当结论。**
⇒ **四任务平均的 0.145 把这个定性差异抹平了**，任何引用平均值的图注必须点出 TPA。

#### ★★ 结论 4：`learned + V_target` 这一列**不可用** —— 被 decode 误差主导
外部 critic 只吃原始状态 ⇒ learned 臂必须把终端 latent 解码；而 `shared_encoder=1` 下
planner **正常从不调用 decode**（`rl/world_model.py` decode 注释）⇒ 误差只落在这些格上。
专门的控制格（同一 proposal、同一 V̂，只多一次 `decode→encode`）量出：

| decode 往返的代价 | Square | Coffee | Threading | TPA |
|---|---:|---:|---:|---:|
| src+tgt 行 | −0.260 | −0.430 | −0.580 | −0.130 |
| tgt only 行 | **+0.080** | −0.110 | −0.150 | 0.000 |

**解码代价 ≫ 换 value 的效应，而且 policy 相关到能翻符号**（Square 的两行 −0.26 / +0.08）。
口径对齐后（`L+V_tgt − L+decode`）：

| | Square | Coffee | Threading | TPA |
|---|---|---|---|---|
| src+tgt | +0.060 (z=0.9) | 0.000 | **−0.170 (z=−3.3)** | −0.030 (z=−0.7) |
| tgt only | +0.020 (z=0.3) | 0.000 | **−0.250 (z=−5.4)** | 0.000 |

⇒ 只有 Threading 在扣除解码后仍显著为负（与结论 2 同向）。
★★ **教训**：我一度把 Square 的 `learned` 列读成「换 value −0.200 / +0.100、交互 z=3.43」，
并配了 findings §33 的共适应解释 —— **全部是解码伪迹，已撤回**。预注册的控制格救了这个结论。
⇒ **画图/写论文时这一列要么不画，要么必须与 decode 控制并列。**

#### 工具与移植（本轮新增，全部默认关、逐位兼容）
- `scripts/eval_coffee_oracle_mppi.py`：`--arms learned` · `--proposal_simdist_ckpt` ·
  `--proposal_bc_ckpt` · `--decode_roundtrip` · **`--shift_from_cfg`**（从 stage-1 cfg 读 target）
  · 补了 `[setup] planner` 行 · 修了过时的 "(proposal still the expert)" 日志。
  ⚠️⚠️ **该脚本的 `TARGET_B` 常量是 C75kp25（kp 0.25 / lid 75 / mass 3），不是定稿的 G5x
  （kp 0.30 / lid 5 / mass 8 / friction 0.2 / pod_holder 5）** ⇒ **不传 `--shift_from_cfg` 就会静默
  评在另一个 target 上**。这就是「Coffee 的旧 oracle 数不太对」的根因。
- `scripts/eval_oracle_dyn_mppi.py`（Threading + TPA）：`--proposal_bc_ckpt`（走
  `rl/simdist_planner.py:128` 的 `proposal_policy` 钩子，并接到 `policy` 臂）·
  **`--temperature/--init_std/--min_std/--base_policy_std/--momentum`**（原先**未暴露**，
  两臂静默沿用各 run 自己的 cfg ⇒ 统一预算根本不成立）· **`--base_held`**（两臂都用 raw-state
  held-action；原有的 `--base_open_loop` 只管 oracle 臂，且第一步仍过 `_policy_rollout(z0)` ⇒
  BC 行独有解码误差）· `--decode_roundtrip` · **`--demo_path_override`**。
  ★ **回归门**：不带任何新参数跑默认路径，复现 findings §27 的 `policy 0.400 / learned 0.875`
  ⇒ 实测 **0.380 / 0.840**，通过。
- `--demo_path_override` 的用途：`AdaptWorkspace` **构造时**就加载 target demo buffer，
  而 Threading plain_h5 的那套 demo 已丢失 ⇒ 无法构造。该 buffer 只在训练侧被用
  （stage-2 BC / stage-3 batch / `_anchor_far_weights`），**评测从不采样**，所以覆盖它
  不会改变任何评测数字，只是让 workspace 能建起来。
- 资产：`{threading_bc_tgtonly_s160,coffee_bc_tgtonly_g5x,tpa_bc_tgtonly}_seed1`
  （from-scratch target-demo BC，配方 = `train_target_bc_priors.sh` 但 `--load_model none`）。
- 日志 + GPU trace 全部存档：`exps/tmp/x4_grid/`（77 个文件），另有
  `exps/tmp/cof_grid_crashed/`、`exps/tmp/thr_grid_aborted/` 存失败那轮。

#### 未决 / 边界
- [ ] **全部单 seed**（每任务一个 stage-1 + 一个 adapt + 一个 BC）。方法级结论按项目惯例需 ≥2 seed。
- [ ] **预算未按任务重扫**（见协议一节）⇒ 绝对值偏低，尤其 Coffee。
- [ ] **TPA 的 tgt only 行在 learned 列全为 0**（地板），该行的 learned 侧不含信息。
- [ ] 结论 3 的机制（"proposal 来源"而非"裸跑分"）**缺判别臂**。Square 上那个判别臂
      （`LI`，il_ft BC 裸跑 0.500 未被适配 → 0.590 ≈ 被适配的 0.650）判的是 **learned** 列，
      不能外推到 oracle 列或别的任务。
- [ ] Coffee 的 `LRNGATE`（原配方 learned）实测 0.410 vs 该 run 日志里的 0.550 ——
      在铁律 #14 的训练中 eval 乐观偏差（0.06–0.19）范围内，但没有独立复核。

---

### 40e. ★ Coffee 三个 target 横评 —— **G5x + `bc_anchor_coef 1.0` 追平 oracle**（2026-08-22）

#### 定稿：Coffee 的 target 用 **G5x**
```
gripper_kp_scale 0.30 / lid_damping_scale 5.0 / mass_scale 8.0
friction_scale 0.2   / pod_holder_friction_scale 5.0
```
（`lid_damping` 近标称是**故意的**，见下面 ③。cfg：
`release/cfgs/mimicgen/coffee_d0_state_ibrl_target_g5x.yaml`）

| target | stage-1 MPPI<br>(source-only dyn) | anchor λ=1 后<br>round-0 | **100k adapt** | oracle | **距 oracle** |
|---|---:|---:|---:|---:|---:|
| C50 | 56% | 0.86 | 0.62 | 0.98 | **−0.36** |
| C75kp25 | 24% | 0.46 | 0.71 (2 seed) | 0.90 | −0.19 |
| **G5x** | 51% | **0.87–0.88** | **0.907** (2 seed) | 0.94 | **−0.03** |

**G5x 的 0.907（seed1 0.896 / seed2 0.918）与 oracle 0.94（n=50, SE≈3.4%）统计不可区分**
—— Coffee 线上第一次 learned world model 追平完美 dynamics 的天花板。

#### anchor λ 响应（每个 target 各自的形状）
| λ | C75kp25 (2 seed 均值) | **G5x** (2 seed 均值) |
|---:|---:|---:|
| 0 | **0.19** ⚠️ | 0.653 |
| 0.3 | 0.68 | 0.859 |
| **1.0** | **0.71** | **0.907** |
| 3.0 | **0.41**（崩） | 0.868（不崩） |
| ∞ (noBC) | 0.45 | 0.786 |

**⇒ λ≈1 是跨 target 的安全默认值**（两个 target 上都是峰或并列峰；G5x 上两个 seed 一致胜出）。
但 **λ 响应的形状 target-dependent**：C75kp25 是倒 U（λ=3 崩到 0.41），G5x 是宽平台
（λ∈[0.3,3] 全在 0.86–0.91）。所以"λ=3 一定不行"**不成立**，别外推。

#### 三条会改变"下次怎么做"的结论

**① 真正的杠杆是 stage-2 的 anchor，不是 stage-3 的 adapt。** G5x 上 anchor 把 round-0
从 0.44–0.49 抬到 **0.87–0.88（+0.40）**，在此之上 adapt 只再加 **+0.03**。adapt 的作用是
**把低起点拉到同一平台**（λ=0: 0.44→0.578、0.49→0.728），平台高度由 anchor 决定。
⇒ 有限算力优先调 stage-2，不是 stage-3。

**② 选 target 的判据：看 `stage-1 MPPI 在 target 上的分数`，不看 oracle。**
我当初按"oracle 要高（⇒ value/proposal 够用）"选了 C50 —— 但**"oracle 高"恰恰等价于
"source 的 value/proposal 在 target 上已经够用"，与"gap 大"自相矛盾**。C50 的 stage-1 已有
56%（anchor 后 87%），adapt 无事可做。正确判据是 stage-1 分数**低**。
⚠️ 但它也不是充分判据：G5x 的 stage-1（51%）比 C50（56%）只低 5 点，结局却是 +0.03 vs −0.24。

**③ 难度必须落在"数据主体 + 有 gap"的相位上。** C50 是唯一崩掉的，而它的难度 86% 集中在
**close-lid 相**，那个相位只占 **4%** 的转移、pod gap 只有 0.110 mm（全场最小）。
stage-3 的 on-policy 数据里它几乎不出现 ⇒ dynamics 被另外 96% 主导地重训 ⇒ 对 close-lid
有用的那点结构被冲掉。G5x 的难度和 gap 都在 transport/insert（0.310 / 0.179 mm，占 ~96%）。
⇒ **造 target 时先量分阶段 gap，确认难度与 gap 在同一个相位、且那个相位占数据主体。**

#### 资产
| | 路径 / 值 |
|---|---|
| source 专家 σ=0.1 | `coffee_ibrl_source_sig01_seed1/model0.pt` — 93.8%（+20 snapshot） |
| source diverse data | `coffee_d0_source_diverse_v1/dataset.hdf5` — 100k 条 / 51.9% / 7.2 GB |
| stage-1 world model | `coffee_simdist_diversev1_h5_seed1/latest.pt`（**5 头 dynens**；source-only ⇒ **换 target 不需重训**） |
| **G5x target 专家** | `coffee_ibrl_target_G5x_seed1`（1M；曲线 400k=0.86 拐点 / 首个 1.00 @530k；20 snapshot） |
| **G5x target demo** | `coffee_d0_target_G5x/dataset.hdf5` — 20 条，采自 `snapshot_400000`(0.86)，平均 91.5 步 |
| C75kp25 专家 / demo | `coffee_ibrl_target_seed1`(95.8%) / `coffee_d0_target_C75kp25{,_sat,_weak}` |
| C50 专家 / demo | `coffee_ibrl_target_C50_seed1`(98.2%) / `coffee_d0_target_C50{,_weak200k,_sat1000k}` |

#### 复现命令
```bash
# stage-1 cfg 副本：换 target_demo_path + eval_domain_shift（world model 本身复用）
python train_simdist_adapt.py --stage1_ckpt <s1_G5x>/latest.pt \
  --save_dir exps/mimicgen/coffee_G5x_adapt_anchor1.0_seedN \
  --bc_steps 1000 --bc_anchor_coef 1.0 \
  --bc_anchor_path release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5 \
  --adapt_interval 1000 --max_transitions 100000 --dyn_update_steps 1000 \
  --eval_every_rounds 5 --num_eval_episode 100 --original_mppi 1 --adapt_value 0
```

#### ⛔ 本轮被实测证伪的假设（别重试）
1. **"梯度太小 ⇒ 没东西可学"** —— Coffee adapt 的 `consistency 6e-5 / grad_norm 1e-4`，我据此
   判 C50 无 gap；但 **Threading 全部 118 条 adapt 是同样的数值**却能 0.19→0.94。这个量级只是
   SimNorm latent 空间里 consistency loss 的正常尺度，与能否学到东西**无关**。
2. **"adapt 把模型改坏了"** —— 实测 H 步 latent rollout 误差**降 25–30%**，且两个 held-out
   demo 池同样降（无灾难性遗忘）。模型更准了，planner 分数不涨。
3. **"adapt 破坏 dynamics–value 共适应（§33 的镜像）"** —— 排序保真度实测**没降**
   （spearman 0.124→0.112/0.133/**0.231**，top-6 重合 0.278→0.31–0.39）。
   ⚠️ 但该测量用的是**均匀随机候选**而非 CEM elite 分布，按铁律 #5 属于"在 planner 不访问的
   分布上量它的排序能力"，所以只能算弱证据。
4. **"stage-3 学习率/UTD 太大（过拟合）"** —— A（lr÷10）、B（UTD÷10）、C（延后到 10k 起训）
   三个臂**全部照样在 5k–25k 深跌**。C 最干净：5k 时零梯度步、成绩纹丝不动停在 0.85，训练一
   开始立刻掉 → **不是更新幅度的问题，是方向**。
5. **"结局由 round-0 决定"（`corr(round-0,gain)=−0.72`，134 条 run）** —— C50 从 0.86 跌到
   0.62、G5x 从 0.87 涨到 0.896。**同起点、不同 target、相反结局** ⇒ 那条相关只是相关，
   机制在 target 的相位结构（见 ③）。
6. **"σ=0.1 的宽初始分布是 SimDist 的负担"** —— 同 ckpt 同 planner，σ=0.02 vs 0.1 实测
   64%/60%（stage-1）、91%/92%（anchor round-0）—— **完全无差别**。

#### ⚠️ 两个测量地板（本轮踩出来的）
- **standalone eval 的 σ ≈ ±6 点**：同 ckpt / 同 planner / 同 seed / 同 100 局，两次测出
  **86% 与 92%**。根因 = MPPI 候选采样走 torch RNG，`evaluate/eval.py` 只按 episode 设
  `np.random.seed`，planner 的 `torch.randn` 不受控。⇒ **<12 点的 planner 比较不可读。**
- **adapt 的 seed 间 σ ≈ 0.09–0.13**（与 §38 的 0.092 一致）。C75kp25 上 λ=0.3 vs 1.0 的
  排序**在两个 seed 间翻转**（seed1 差 +17 点、seed2 差 −11 点）⇒ **<0.2 的 adapt 差异需 ≥2 seed。**

#### TODO
- [ ] G5x λ=1.0 补第 3 个 seed；λ=3.0 补 seed2（现只有单 seed 0.868）。
- [ ] G5x 上重扫 planner（现有数字全部用 original 配方；§40d 显示各 ckpt 最优配方不同，
      0.907 可能仍被低估）。
- [ ] 用 ③ 的判据造一个"难度与 gap 同相位、且 stage-1 分数更低"的 target，检验能否复现
      Threading 那种 +0.4 级别的 adapt 增益。
- [ ] `dyn_anchor_coef`（本轮新加的 dynamics 侧 source 锚，`rl/simdist_agent.py` +
      `train_simdist_adapt.py`，默认 0 逐位兼容）**尚未验证** —— 两条 run 起了但被中断。
