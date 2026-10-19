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

## 阶段 1：MimicGen 生成 state 数据（先做，约半天）

种子：MimicGen 自带的 sim 示范（`release/data/mimicgen/source/threading.hdf5`，10 条，Panda 手爪、20 Hz）。
真机 demo 现在**不能**当种子：在 sim 里能抓起 7/10，但穿环 0/10（真机穿环时 tripod 被推动，开环回放复现不了）。

1. 给 `scripts/gen_realcell_demos.py` 加 `threading_robotiq` 任务（env `Threading_RealCellRobotiq`），并覆盖种子数据 env_meta 里的
   `control_freq 20`（类会拒绝 20 Hz）。执行层开着时 MimicGen 会自动按参考臂位姿算动作（`env/threading_mg.py`）。
2. **冒烟 50 次**，量生成成功率，并按 tripod 朝向分段统计（tripod 朝向决定插入时的 j7，可能顶到 ±166° 限位）。
   - 门槛：整体 ≥ 30%，且没有哪一段朝向 ≈ 0。过不了就先查：Panda 抓取点换成 Robotiq 后的抓取偏移、10 Hz 下的时间缩放、j7。
3. 生成约 1000 条成功轨迹（分片并行）→ merge → 转成 state 数据集。单步耗时：执行层约 19 ms/步（不开执行层 10.5 ms）。

## 阶段 2：BC prior（约 1–2 小时）

- RNN-GMM（**不用 flow**：铁律 17，flow 让 IBRL 慢一个数量级）。
- 门槛：独立评测 300 局（σ=0.1 初始化扰动、strict success），成功率 ≥ 0.3。

## 阶段 3：IBRL state 专家（约 0.5–1 天）

- 必须显式传 `--config_path`（铁律 19b：不传会静默拿到 source 物理、`act_method=rl`、错误的 critic 大小）；`expert_hybrid` 按 peg 的经验要开（铁律 32）。
- 确认 `train.log` 里有 `[SGFT] prior = source HYBRID`（铁律 8）。1M 步一个长 job，不要拆成 warm-start 链（铁律 18）。
- 门槛：独立重评 300 局 ≥ 0.9（Franka 手爪旧 cell 是 0.957）。

## 阶段 4：pixel 数据 + SimDist stage-1（约 3–4 天，集群）

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
