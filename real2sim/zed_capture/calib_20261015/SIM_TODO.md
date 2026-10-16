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

## 1. sim 改成 10 Hz（最先做，其余全部依赖它）

- [ ] `control_freq` 做成 config 字段，从 `env/robosuite_wrapper.py` `PixelRobosuite` 一直传到 `robosuite.make`，
      train / eval / 数据生成 / MimicGen 全部走同一个字段，默认值保持 20，旧 config 逐字节不变（铁律 #38：两侧必须一致）。
- [ ] 找出所有写死 20 的地方：
      `scripts/convert_real_teleop.py`（`control_hz=20.0`）、`scripts/record_realcell_rollouts.py`（`--fps 20`）、
      `calib_20261014/tools/robotiq_grasp_check.py`、`real2sim/deploy/policy_server.py`（`self.hz = 20.0`）。
      ⚠️ **`policy_server.py` 的 `ReferenceArm` 每步走一个 robosuite OSC 步，必须用和训练相同的 `control_freq` 建环境**，
      否则部署时一步代表的时间不对。
- [ ] 重新确认 10 Hz 下的 OSC 行为：动作还是 ±5 cm / ±0.5 rad 的增量，但一步有 100 ms，kp 150 能走完更大一部分。
      量一下 \|a\| = 0.2 / 0.5 / 1.0 时每步的位移，和真机 sim_ref 下每步 5–7 mm 对比。
- [ ] `episode_length`：400 步在 10 Hz 下是 40 s，能装下真机 demo（21–31 s），可以保持不变；规划 horizon 5 步变成 0.5 s。
- [ ] 回归门：20 Hz 默认配置下 dedup gate 和每条 smoke 臂都要过（铁律 #23 / #26）。

## 2. 把真机的执行层放进 sim

- [ ] 限速：TCP ≤ 8 cm/s、≤ 0.4 rad/s、领先 ≤ 3 cm，**和 `policy_server._speed_caps` 用同一套数值**（最好共用一个函数）。
- [ ] 滞后：在 OSC 目标和机械臂之间加一阶滞后，τ 的随机化范围约 0.1–0.4 s。x 比 z 慢，可以分轴或者直接放宽范围。
      再加 0–1 步的延迟随机化。
- [ ] 用 `calib_20261015/data/track{20,10}_sim_ref_*`（`steps.jsonl` + `joints.npz`）做对照：在 sim 里重放同样的脚本动作，
      看真机/目标速度比、滞后距离、越过目标量能不能对上（`analyze_tracking.py` 的同一套指标）。
- [ ] 近桌面的慢速带：部署时离 z_min 6 cm 以内每步最多下降 3 mm（`--z_slow_band 0.06 --z_slow_step 0.003`）。
      sim 专家也要遵守，否则 demo 和专家的速度分布对不上。

## 3. cell 更新

- [ ] `threading_real_cell_20261014.yaml`（或者新建 `calib_20261015` 的 cell）：手腕相机的名义位姿换成上面的新位姿。
      注释里写明来源：`data/wrist_run_1015_1915/wrist_fixed_scene.json`，场景相机固定的解法。
- [ ] 手腕相机 DR：**等 owner 决定**。建议 ±(6, 6, 8) mm / ±(1.5, 1.5, 2)°，依据是同一次 ≤ 3 mm / 0.3°、几次之间 6 mm / 0.8°。
- [ ] 用新位姿重跑 `tools/check_wrist_needle.py`（对 `~/simdist/calib/run_1015_1915`），确认夹爪 IoU 约 0.755、needle 没变差。
- [ ] tripod 朝向范围 [−45.6, −35.6]：真机这次是 −36.0，贴着边。要么把范围往外扩一点，要么下次采集时把 tripod 摆到 −40° 左右。

## 4. 5 条 demo 转换 + 回放

数据在 `~/simdist/gello_software/demos/threading_robotiq_1015/episode_000{1..5}.h5`（230 MB，**不进 git**）。

- [ ] 把 `scripts/convert_real_teleop.py` 移植到 Robotiq cell：
      - 夹爪状态用 `observation/gripper_knuckle` → `tv.robotiq_real_to_sim`，不要用 `gripper_width`；
      - prop 用 base 系（`apply_prop_frame`，`docs/real-teleop-stage2.md` §2b），不要加旧的 (−0.489, −0.034, 0.912)；
      - 控制频率 10 Hz，图像按 cell 的 fovy 裁剪，手腕用 `wrist_intrinsics.json`。
- [ ] 每条的成功帧 = 最后一帧（操作员是在松开夹爪**之前**按的空格，最后一帧夹爪还是闭着的）。
- [ ] 起始姿态：录制器记的是 "forced"，j2 和 DROID home 最多差 3.6°。回放从实测的 q[0] 开始，不要从 DROID home 开始。
- [ ] 物体初始位姿：从每条第一帧的场景图用 `fit_layout_wrist.py` 第 1 步拟合（needle IoU 约 0.85，tripod 约 0.91）。
- [ ] 在 10 Hz sim 里两种回放都做：
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
