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
