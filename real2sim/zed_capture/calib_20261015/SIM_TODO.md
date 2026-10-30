> ★ **10-30 robot visit (v8 stage 1):** the 10-27 stall is gone; 60k ≫ 70k (70k 0/7, grasps the handle tail → +10/+16/+26° tip-down, beyond the DR). New bottleneck: the first grasp closes askew — handle not fully between the fingers (wrist x 210 px vs sim 183; 70k yaw 21–35° off); the sim contact is too forgiving (lifts off-centre grasps). 60k recovers by re-grasping. v9 list: `real_1030/README.md`.

> ★ **10-27 robot visit (stage-1 exec_obs ckpt): grasps 100 %, never inserts.** Cause = the real in-hand needle pose (grip on the tail/top of the handle, −12 mm / +15 mm, pitch +9.8°, yaw +6.9° vs sim ≈0/0/1.3°/1.4°); placing the real pose in pure sim reproduces the failure. Sim needle also rests flat (tip 15 mm up) while the real tip rests on the table. To do + data: `real_1027/README.md`, `real_1027/results/inhand_real_1027.json`.

# sim 侧准备清单（2026-10-15 真机之后，下次上机之前）

真机结果见同目录 [`README.md`](README.md)；上机清单见 [`../calib_20261014/NEXT_VISIT.md`](../calib_20261014/NEXT_VISIT.md)。
顺序 = 依赖顺序：前一步没做完，后一步的数据都会作废。

## 0. 今天定下来、sim 必须照着改的事实

| 项 | 真机实测 / 决定 | sim 现状 |
|---|---|---|
| **控制频率** | **10 Hz**（100 ms / 步），与用户确认 | robosuite 默认 `control_freq` 20（50 ms），Threading 的 wrapper **没有暴露这个参数** |
| 部署动作映射 | `policy_server.py --action_map sim_ref`：sim 参考轨迹自带 q/qvel，真机跟它 | 训练里没有这一层 |
| 硬限速 | 目标 TCP ≤ 8 cm/s、≤ 0.4 rad/s、领先实测 ≤ 3 cm（`_speed_caps`） | 没有。sim 专家在 \|a\| ≥ 0.2 时每步就超过这个速度 |
| 跟随滞后 | z 约 0.1 s，x 约 0.25–0.4 s；超过最终目标 ≤ 7 mm | 没有（一步之内就到位） |
| 手腕相机 | 新位姿 pos `[-0.07452, -0.03588, 0.01144]`，quat wxyz `[0.92782, -0.15101, 0.03893, -0.33888]`，比 cell 偏 6.2 mm / 0.8°；同一次标定内 ≤ 2.6 mm / 0.3° | cell 里还是 10-13 的旧位姿；DR ±(6, 10, 20) mm / ±(3, 3, 4.5)° |
| 场景相机 | 没动（0.015°） | 不用改 |
| 桌面高度 | 0.0285（10-14 用 GELLO 碰 5 次：0.0287 ± 1.2 mm） | 不用改 |
| 摆放 | needle (0.556, −0.111, −110°)、tripod (0.520, 0.183, −36°)，都在范围内 | tripod 朝向贴着 −35.6° 那一侧 |
| demo | 5 条，10 Hz，21–31 s（210–315 步），夹爪状态是 `observation/gripper_knuckle` | `scripts/convert_real_teleop.py` 遇到 Robotiq / base 系 prop 会直接退出（没移植） |

## 进度（2026-10-16）

- §1 已完成：10 Hz 做成了 **cell 字段** `control_freq`，新 cell 是 `threading_real_cell_20261015.yaml`（继承 10-14，只改了频率和手腕相机位姿），
  并且已经是 `Threading_RealCellRobotiq` 的默认 cell。`scripts/test_sim2real.py` 全部通过（新增 N 节）。
- §3：手腕相机名义位姿已换；DR 宽度没动（等 owner 定）；`check_wrist_needle.py` 没重跑（原始 PNG 在工作站上）。
- §4：转换脚本已移植，回放脚本 `scripts/replay_real_demos_sim.py` 已写好。结果见 README「Sim replay of the 5 demos (10-16)」：
  **跟踪回放 1/5 成功**；**转换出来的动作被 6 mm 下垂污染，不能直接训练**（见下面 §4 新增的那一项）。
- §2 还没做。

## 1. sim 改成 10 Hz（最先做，其余全部依赖它）

- [x] `control_freq` 做成 **cell 字段**（`env/sim2real.py`，构造期字段，和 `gripper_types` 同一种做法）：wrapper 把它传给 `robosuite.make`，
      `Threading_RealCellRobotiq` 类自己也强制使用它（这样直接 `robosuite.make` 的 MimicGen / 回放路径也是 10 Hz，传 20 会报错）。
      默认 None = robosuite 的 20，其他 cell 逐字节不变。cell 而不是 config：频率是这台真机的属性，train / eval / 生成 / 部署都从同一个 cell 读。
- [x] `policy_server.py`：频率改为从 `ReferenceArm`（即 cell）读，策略模式下客户端 `--hz` 不一致就拒绝（脚本测试模式只警告）。
      `convert_real_teleop.py`：`control_hz` 从 cell 读。**还没改**：`record_realcell_rollouts.py --fps 20`、`robotiq_grasp_check.py`。
- [ ] 找出所有写死 20 的地方（原清单）：
      `scripts/convert_real_teleop.py`（`control_hz=20.0`）、`scripts/record_realcell_rollouts.py`（`--fps 20`）、
      `calib_20261014/tools/robotiq_grasp_check.py`、`real2sim/deploy/policy_server.py`（`self.hz = 20.0`）。
      ⚠️ **`policy_server.py` 的 `ReferenceArm` 每步走一个 robosuite OSC 步，必须用和训练相同的 `control_freq` 建环境**，
      否则部署时一步代表的时间不对。
- [x] （`tools/sim_osc_step_1016.py`）同一 |a| 下 10 Hz 和 20 Hz 的**速度**几乎一样，只是每步走得多一倍：
      |a| 0.2 / 0.5 / 1.0 → 10 Hz 每步 x 3.9 / 9.9 / 19.9 mm，z 4.5 / 11.5 / 23.7 mm（cm/s 同数）。真机 sim_ref 是 2.5–7.5 cm/s
      ⇒ |a| ≥ 0.5 时 sim 比真机的 8 cm/s 上限快 1.2–3 倍，§2 的限速必须进 sim。
- [ ] （原文）重新确认 10 Hz 下的 OSC 行为：动作还是 ±5 cm / ±0.5 rad 的增量，但一步有 100 ms，kp 150 能走完更大一部分。
      量一下 \|a\| = 0.2 / 0.5 / 1.0 时每步的位移，和真机 sim_ref 下每步 5–7 mm 对比。
- [ ] `episode_length`：400 步在 10 Hz 下是 40 s，能装下真机 demo（21–31 s），可以保持不变；规划 horizon 5 步变成 0.5 s。
- [ ] 回归门：20 Hz 默认配置下 dedup gate 和每条 smoke 臂都要过（铁律 #23 / #26）。

## 2. 把真机的执行层放进 sim

> **2026-10-17 做完主体**（README「Execution layer in the sim」）：`env/exec_layer.py`，10-15 cell 里已开启。参考臂与部署逐步一致（3.5e-5 rad），
> 限速 / 工作空间限制与 policy_server 共用同一个函数，机械臂参数由开环系统辨识得到，样本外复现了 reseed 的卡住现象。
> **限速：不做（owner 10-17）**，sim 和真机都去掉了（cell 里关掉，policy_server / robot_client 从 cell 读），只留 z_min 底线。
> 结果：5 条 demo 的动作经执行层开环回放 30 s，sim 机械臂离真机实测 p50 8–22 mm ⇒ **执行层对齐了**；抓取没对齐（0/5 穿环，见 §4）。

- [ ] 限速：TCP ≤ 8 cm/s、≤ 0.4 rad/s、领先 ≤ 3 cm，**和 `policy_server._speed_caps` 用同一套数值**（最好共用一个函数）。
- [ ] 滞后：在 OSC 目标和机械臂之间加一阶滞后，τ 的随机化范围约 0.1–0.4 s。x 比 z 慢，可以分轴或者直接放宽范围。
      再加 0–1 步的延迟随机化。
- [ ] 用 `calib_20261015/data/track{20,10}_sim_ref_*`（`steps.jsonl` + `joints.npz`）做对照：在 sim 里重放同样的脚本动作，
      看真机/目标速度比、滞后距离、越过目标量能不能对上（`analyze_tracking.py` 的同一套指标）。
- [ ] （10-16 复核，README「Offline re-analysis」）滞后**按关节分开**：j1–j4 约 0.1–0.2 s，j5/j6 约 0.2 s，j7 约 0.3–0.5 s 且有粘滑；
      TCP 上三个轴都约 0.13–0.16 s。另外实测 TCP **比指令低约 6 mm**（j2 下垂 +10 mrad，负载补偿里没算 Robotiq + ZED-M），
      sim 里要么加一个随位姿变化的下垂，要么让 owner 改 payload 后重测。
- [ ] 人遥操作有 18 % 的步超过 8 cm/s：demo 回放 / MimicGen / BC prior 都要经过同一套限速。
- [ ] 近桌面的慢速带：部署时离 z_min 6 cm 以内每步最多下降 3 mm（`--z_slow_band 0.06 --z_slow_step 0.003`）。
      sim 专家也要遵守，否则 demo 和专家的速度分布对不上。

## 3. cell 更新

- [ ] `threading_real_cell_20261014.yaml`（或者新建 `calib_20261015` 的 cell）：手腕相机的名义位姿换成上面的新位姿。
      注释里写明来源：`data/wrist_run_1015_1915/wrist_fixed_scene.json`，场景相机固定的解法。
- [ ] 手腕相机 DR：**等 owner 决定**。建议 ±(6, 6, 8) mm / ±(1.5, 1.5, 2)°，依据是同一次 ≤ 3 mm / 0.3°、几次之间 6 mm / 0.8°。
- [ ] 用新位姿重跑 `tools/check_wrist_needle.py`（对 `~/simdist/calib/run_1015_1915`），确认夹爪 IoU 约 0.755、needle 没变差。
- [x] （10-17）摆放范围扩大：needle 朝向 ±45°（xy 不变），tripod xy ±40 mm、朝向 [−50, −25]。真机 5 条 demo 里 tripod 有 4 条在旧范围外
      （−33.3 ~ −34.9）；needle 用手腕相机校正后平均 −100.2°，正好在中心。所有角点在场景裁剪内 ≥ 13 px。⚠️ tripod 朝向影响插入时的 j7，生成数据时要看成功率。

## 4. 5 条 demo 转换 + 回放

数据在 `~/simdist/gello_software/demos/threading_robotiq_1015/episode_000{1..5}.h5`（230 MB，**不进 git**）。

- [x] 已移植（2026-10-16）：Robotiq cell 用部署端的 `ReferenceArm` 做 FK 和 prop（和真机策略逐字节同一段代码），
      不需要 `--expert`；输出里另存 `obs/real_joint_pos / real_joint_cmd / real_gripper` 供回放。原始要求：
      - 夹爪状态用 `observation/gripper_knuckle` → `tv.robotiq_real_to_sim`，不要用 `gripper_width`；
      - prop 用 base 系（`apply_prop_frame`，`docs/real-teleop-stage2.md` §2b），不要加旧的 (−0.489, −0.034, 0.912)；
      - 控制频率 10 Hz，图像按 cell 的 fovy 裁剪，手腕用 `wrist_intrinsics.json`。
- [ ] 每条的成功帧 = 最后一帧（操作员是在松开夹爪**之前**按的空格，最后一帧夹爪还是闭着的）。
- [ ] 起始姿态：录制器记的是 "forced"，j2 和 DROID home 最多差 3.6°。回放从实测的 q[0] 开始，不要从 DROID home 开始。
- [ ] 物体初始位姿：从每条第一帧的场景图用 `fit_layout_wrist.py` 第 1 步拟合（needle IoU 约 0.85，tripod 约 0.91）。
- [x] 物体初始位姿：5 条各自第一帧拟合好了，`demo_layout/layout_demos_1015.json`（needle 位置和 layout 照片差 ≤ 5 mm，朝向 −103 ~ −120°）。
- [x] 两种回放都做了（`scripts/replay_real_demos_sim.py`），结果在 README：跟踪回放 1/5；开环 0/5（动作有偏差，见下一项）。
- [x] ★ 动作约定已改为 `ref_inverse`（Robotiq cell 上的默认值）：每一步求出「sim 参考臂（自带 q/qvel，和部署时 sim_ref 一样）走到下一帧 GELLO 指令位姿要发的 OSC 动作」。
      参考臂跟指令的误差 p50 0.08 / p99 3.1 mm；开环回放不再漂（离实测 12–26 mm，= 指令的超前 + 下垂；`cmd_fk` 时是 0.2–0.8 m）。
- [ ] ★ **静态下垂 = 缺了 0.30 kg 负载**（`tools/fit_payload_sag_1016.py`），控制器大概还按 Franka Hand 配的。**请 owner 在 Desk（或 franka_ros2 set_load）
      把末端负载改成 Robotiq 总成（约 1.03–1.1 kg，质心沿法兰轴约 6 cm），然后重测下垂**。改不了的话，备选是在 policy_server 里加前馈补偿（还没做）。
- [ ] ★★ **抓取是主要问题**：needle 扰动 ±2 mm / ±3° 时 sim 只有 22/45 抓起（真机 5/5），而且按 demo 分得很死（demo 0、3 次次抓起，1、2、4 几乎从不）。
      已排除：物体尺寸（真机 4 cm 柄、16 cm 全长 = sim）、初始位置（闭合前差 2–5 mm，成功那条反而差得最大）、needle 质量、扭转摩擦、夹爪闭合时机。
      线索：demo 1、4 闭合前 sim 的 inner knuckle 就碰到柄顶；demo 1 闭合时把 needle 挤出 5–7 cm。
      **需要真机侧回答**：真机抓的时候柄顶是不是也碰到手指上部（侧面拍一张抓取深度的照片最直接）。
- [ ] tripod：不能固定，下次采集**不要碰 tripod**（owner 已同意）。这次 5 条里真机 tripod 被推了 3–42 mm，回放里穿针阶段的失败不能算 sim 的问题。
- [ ] （原文）在 10 Hz sim 里两种回放都做：
      ① 开环回放关节轨迹（`action/joint_positions`）：看抓取和穿环在 sim 里能不能成，**失败的位置就是真机和 sim 的 gap**；
      ② 转成 OSC 动作后再回放：检验动作约定和执行层。
- [ ] 跑 `scripts/validate_real_demos.py`。
- [ ] 量一下：最后 3 cm 的下降速度（之前人约 4 cm/s，sim 专家约 8 cm/s），以及 GELLO 指令领先实测的 7–12°。

## 5. 之后

- [ ] 用能回放成功的 demo 当种子，按 10 Hz 重新生成 MimicGen 数据（之前用 10 个种子），再跑 stage-1（配方见 CLAUDE.md 的 PIXEL 决定）。
- [ ] 根据回放结果定下次的采集方式（摆放怎么分布、速度、成功帧什么时候按），再采剩下的 15–25 条。

## 下次上机要带回来的

- [ ] 剩下的 demo（10 Hz，`run_teleop_session.py --gripper robotiq --rate 10`，录制器已修）。
- [ ] 曝光漂移：拉长到几个小时（这次只隔了 18 分钟，≤ 0.8 %）。
- [ ] 每次开始前拍一张标定板核对手腕相机（支架可能在松动）。
- [ ] 想收窄 `dr_base_xy` / `dr_base_yaw_deg` 的话：在 Execution 模式下用 GELLO 碰**棋盘内角**（不要手托）。
- [ ] 换控制器增益、上 polymetis 都不需要做，见 README 的 Recommendation。
