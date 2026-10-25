# sim 训练计划（2026-10-19，Threading + Robotiq 真机 cell）

依据：同目录 `README.md`（10-15 → 10-18 的全部对齐结果）。按阶段推进，每个阶段都有**门槛**，过不了就停下查原因，不进入下一阶段
（feedback-gate-rl-on-bc：BC 不过关就不跑 RL）。

## 0. 冻结的接触 / 物理 / 接口（cell = `threading_real_cell_20261015.yaml`）

| 项 | 值 | 依据 |
|---|---|---|
| 控制频率 | 10 Hz | 10-15 决定 |
| prop | base 系，FK(q) | 10-15 |
| 动作 | OSC 增量作用在**参考臂**上（执行层），真机 demo 标签 = `ref_inverse` | 执行层与部署逐步一致（3.5e-5 rad） |
| 执行层 | 参考臂 OSC → 不限速 → 100 Hz 插值 → 关节阻抗；机械臂参数按真机拟合 | 10-18 新数据上样本外成立 |
| 夹爪接触 | 临界阻尼 solref (0.02, 1.0) | 抓取网格 12/12 逐次吻合 |
| 摆放 | needle ±75 / ±50 mm、朝向 [−135, −75]°；tripod (0.521, 0.183) ±25 mm、朝向 [−45, −25]° | 覆盖 15 条真机摆放 + 约 1 cm 的相机不确定度；needle 与 D0 同尺寸（已知能学会） |
| 负载 | 真机 Desk 1.20 kg（不再有静态下垂） | 10-16 |

**这些不再改**；之后只在下面 §DR 里给不确定的量加随机化。改了任何一项，之后的数据和 checkpoint 全部作废（CLAUDE.md 铁律 #37 同类）。

## 阶段 1：state 数据 —— ✅ 完成（2026-10-19），**改用脚本化专家，不用 MimicGen**

**MimicGen 没过门槛**：在新 cell 上生成成功率只有 10%（门槛 30%）。2×2 拆分（50 次尝试一格）：

| | 20 Hz | 10 Hz |
|---|---|---|
| Panda 手爪 | 53–60%（旧管线） | 28% |
| Robotiq | 8% | 10% |

主因是 **Robotiq**：needle 每次都能抓起（19 条失败里 17 条提起了约 19 cm），但它在手里的位姿和 Panda 种子示范不同；MimicGen 把穿环段当成
「末端相对 tripod」的轨迹搬过来，默认手中位姿不变 ⇒ needle 停在离环 2–18 cm 的地方。10 Hz 是次要的（每步只走完指令的约 40%）。执行层不是原因
（关掉执行层同样 10%）。

**改成脚本化专家** `scripts/threading_robotiq_expert.py`：sim 特权状态、闭环。俯视抓柄 → 提起 → 粗对准（冻结手中位姿，参考臂伺服）→
**精对准**（直接对 needle 尖端和轴线闭环，每步只走误差的 0.3，连续 3 步径向 < 1.5 mm、角度 < 1.5° 才插入）→ 沿环轴每步 1.5 mm 插入。
两个踩过的坑：每步重算手中位姿会让「最小旋转」在接触时翻转（目标跳 200 mm）；对实测位姿做比例伺服会因 0.2–0.3 s 的滞后振荡。
成功率 **92/100**（DR 全开，tripod 朝向各段都有）；加 0.05 动作噪声掉到 69/100，所以收集不加噪声。

数据：`scripts/collect_threading_robotiq_demos.py`，8 片并行，**1000 条成功轨迹**（采集成功率 92.6–94.7%），
`/user_data/Chenyang/simdist_data/threading_robotiq_expert_n1000/`，wrapper 格式 + `env_cfg.json`（记录了 cell，BC 评测据此建环境）。

## 阶段 2–3：流水线 `scripts/pipeline_threading_robotiq.sh`（本地 GPU 2，2026-10-19 启动）

> **2026-10-19 结果**：BC-RNN 独立 300 局 **0.34**（过 0.30 门槛；训练中最好 0.44 偏乐观）。
> IBRL 第 1 次（`preload_num_data 10`，照抄旧 Threading 配置）1M 步后独立 300 局只有 **0.257**，低于它自己的 prior：
> 前 375k 步全是 0，之后慢慢爬到 0.3 左右（actor 在 argmax 里压过 prior）。Peg 上同一配方 preload 10 → 0.01、
> preload 1000 → 0.92 ⇒ 第 2 次只改 `--preload_num_data 1000`（`scripts/run_threading_robotiq_ibrl_preload1000.sh`），跑中。
> **2026-10-20**：第 2 次在 475k–600k 步是 0.24–0.58，离门槛很远。下一轮实验（多 seed / 失败重放 / 初始化范围扫描）见 `IBRL_EXPERIMENTS.md`。

## 阶段 2：BC prior（约 1–2 小时）

- RNN-GMM（**不用 flow**：铁律 17，flow 让 IBRL 慢一个数量级）。
- 门槛：独立评测 300 局（σ=0.1 初始化扰动、strict success），成功率 ≥ 0.3。

## 阶段 3：IBRL state 专家（约 0.5–1 天）

- 必须显式传 `--config_path`（铁律 19b：不传会静默拿到 source 物理、`act_method=rl`、错误的 critic 大小）；`expert_hybrid` 按 peg 的经验要开（铁律 32）。
- 确认 `train.log` 里有 `[SGFT] prior = source HYBRID`（铁律 8）。1M 步一个长 job，不要拆成 warm-start 链（铁律 18）。
- 门槛：独立重评 300 局 ≥ 0.9（Franka 手爪旧 cell 是 0.957）。

## 阶段 4：pixel 数据 + SimDist stage-1（约 3–4 天，集群）

> **2026-10-25（exec_obs 线）**：IBRL 平台期的原因是执行层的隐藏状态（参考臂超前实测臂约 14 mm）。修法 `train_rl.py --exec_obs 1`
> （27 维：参考臂与实测臂的末端位姿差、q_ref−q、v_ref、上一步动作），两个 seed 在 325–375k 达到 0.89 / 0.88。
> stage-1 里这 27 维同样进 world model 的 prop（9 → 36）：它在真机上也是本体感知（ReferenceArm 是我们自己的软件）。
> 全流程 `scripts/pipeline_threading_robotiq_stage1.sh`：门槛（300 局 ≥ 0.9）→ 冻结专家和更早的 snapshot →
> 两路相机 8×4500 条 → stage-1（`threading_robotiq_pixel_simdist_r18_224.yaml`）→ MPPI 300 局。
> 「两个相机差约 1 cm / 9°」实为场景相机拟合 needle yaw 的偏差（README 10-17），不是外参误差 ⇒ 按现有相机 DR 采，不阻塞。
> ⚠️ 部署侧 `policy_server` 还要输出同样的 27 维；`train_simdist_adapt.py` 遇到 exec_obs 会直接 assert。

- 用 state 专家采 pixel 数据：cell 的视觉 DR（相机位姿 / 光照 / 背景照片 / 颜色）；存 `sim_state` + `model_state`（重放契约，infra §14e）。
- 配方按 CLAUDE.md 的 PIXEL 决定：`resnet18_pretrained`@224，**只用 random shift 增广**，`dynens_h5`，anchor 关。
- 门槛：source 域（sim）MPPI ≥ 0.9。
- ⚠️ 开始之前先解决 **场景相机和手腕相机差约 1 cm / 9°** 的问题（README「10-18 data in the sim」）：它不影响 state，但会让 pixel 策略看到的
  两路图像和 sim 渲染对不上。

## 阶段 5：真机 target 适配（stage-2/3）+ 真机评测

- target demo：10-18 的 10 条（无下垂）。按 §38 和铁律 #25，至少要 20 条，而且摆放要**散开**（这 10 条都摆在同一个地方）。
- stage-2 / stage-3 配置按 simdist §24 / §44；评测在真机上做（policy_server，不限速，z_min 0.950）。

## DR（阶段 1 起就打开）

| 量 | 范围 | 理由 |
|---|---|---|
| 机械臂 armature / 静摩擦 | ±30% | 拟合残差 j5 / j6 仍有 5–6 mrad（静摩擦方向相关） |
| 夹爪接触 solref dampratio | 0.7–1.0 | 抓取网格只证明了临界阻尼可以，没测出边界 |
| tripod 位姿 | 不另加：摆放范围本身已含约 1 cm 的相机不确定度 | 真机穿环时被推动是 episode 内的接触，由策略闭环处理 |
| 已有的视觉 DR / base DR | 不变 | cell 里已有 |

## 下次上机（不阻塞阶段 1–3）

1. **相机一致性**：把标定板平放在 needle 的位置，机械臂处在抓取姿态，用场景相机和手腕相机分别解算（约 10 分钟）。用来定位那 1 cm / 9° 出在哪里。
2. **20 条以上 demo，摆放在训练范围内散开**：needle 在当前位置附近 ±5 cm、朝向 −135° 到 −75°；tripod 在当前位置附近 ±2.5 cm、
   朝向 −45° 到 −25°（不要超出，超出的摆放 sim 里没训练过）。抓取前悬停 1 s。
3. （可选）更难的抓取网格（沿夹持方向 ±10–14 mm），找到真机抓取失败的边界，用来收窄接触参数的 DR。
