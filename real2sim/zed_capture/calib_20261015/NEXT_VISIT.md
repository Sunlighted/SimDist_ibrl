# 下次上机清单（2026-10-16 写）

背景在同目录 [`README.md`](README.md) 的「Offline re-analysis」「Sim replay」「Follow-up」三节，sim 侧进度在 [`SIM_TODO.md`](SIM_TODO.md)。

## 0. 同步

- [ ] 工作站 `~/simdist/SimDist_ibrl`、`~/simdist/deploy_simdist_1015` 里 `git pull`。
- [ ] 把 `real2sim/zed_capture/calib_20261015/tools/payload_probe.py` 拷到容器的 `/workspace/simdist_runner/`，放在 `robot_client.py` 旁边，它会 import `robot_client.py`。
- [ ] 部署端现在默认用 10-15 cell（`control_freq: 10`）：`robot_client.py` 必须加 `--hz 10`，不然 server 会拒绝。

## 1. 负载辨识（约 15 分钟，决定静态下垂怎么修）

**为什么**：机械臂静止时比指令低约 6 mm（j2 恒定 +10 mrad），说明重力补偿少算了约 0.3 kg。
Franka Hand 时期就已经少了约 0.2 kg（ZED-M 和支架大概从来没算进去）。Robotiq 拆不下来、没法称，
所以改用机械臂自己的力矩传感器来辨识：静止时，控制器的外部力矩估计 `tau_ext_hat_filtered`
正好等于配置里漏掉的那部分重量。换十几种手腕朝向，质量和质心就能分开解出来。

```bash
# 容器里；先让机械臂回 DROID home（reset_arm.py）
python3 /workspace/simdist_runner/payload_probe.py          # dry run：打印控制器当前配置的负载和要走的姿态，不动
python3 /workspace/simdist_runner/payload_probe.py --live --activate \
    --out /workspace/simdist_runner/payload_probe_$(date +%m%d_%H%M).json
```

- 一共 15 个姿态，每个都是 home → 姿态 → home，速度 ≤ 0.3 rad/s，每个姿态停 4 s。总共约 5 分钟。
- 这些姿态在 sim 的 cell 里都检查过：离关节限位留 0.05 rad，机械臂和夹爪最低点离桌面 ≥ 14 cm，没有新增自碰。
  ⚠️ sim 里没有 ZED-M 和线缆：**j7 ±1.5 rad 那几个姿态要盯着线缆**，看着不对就 Ctrl-C（会停住并交还 GELLO）。
- dry run 打印出的 `m_ee` / `F_x_Cee` 是控制器当前的配置，**拍照或复制下来**。
- 带回来的 json 我离线跑：`python calib_20261015/tools/fit_payload_probe.py <json>`。
  能直接给出缺了多少质量和质心位置，并和位置下垂的估计（0.30 kg）交叉核对。
- 现场想看结果：把 json 拷回 host，用 `simdist` env 跑同一个命令（需要 mujoco，不需要 GPU）。

**之后（owner 决定）**：在 Desk 把末端负载改成「当前配置 + 拟合出的缺失部分」，然后**再跑一次 probe**。
修好的判据是拟合出的缺失质量 ≈ 0（≤ 0.03 kg），并且 j2 的静态偏差 < 2 mrad。

## 2. 把漏掉的原始数据带回来

`data/` 被 gitignore 了，下面这些只在工作站上：
- [ ] `~/simdist/deploy_simdist_1015/runs/track{20,10}_sim_ref_*`（sim_ref 跟踪测试；sim 里要用来标定滞后）
- [ ] `~/simdist/calib/run_1015_1915`、`touch_1015_1957`、`layout_1015_2008`（手腕标定、touch、摆放照）

## 3. 脚本化抓取测试（还没写，等确认）

sim 里的抓取比真机脆：needle 抖 ±2 mm / ±3° 时 sim 只抓起 22/45，真机 5/5，而且失败固定出在某几条 demo 上。
人手 demo 每次接近的方式都不一样，分不清原因。要用同一段脚本在真机和 sim 里各抓 ~15 次
（柄正上方下降，指尖离桌面 5 / 10 / 15 mm × 沿夹持方向 −6 / 0 / +6 mm × 另一方向 0 / ±8 mm，闭合、等 0.5 s、提 10 cm），
比较两边的成功率分布。

## 4. 顺手记下来

- [ ] Robotiq 驱动的速度 / 力设置（launch 参数或 driver 配置文件）。
- [ ] 手腕相机支架：开始前拍一张标定板，看有没有又松动（10-10 → 10-15 一直在漂）。

## 5. demo（做完 1 之后）

- [ ] 再采约 10 条，**不要碰 tripod**。录制器设置不变（`run_teleop_session.py --gripper robotiq --rate 10`）。
- 动作约定**不用在真机上改**：`ref_inverse` 是离线用录下的 GELLO 指令算出来的，部署端的 `sim_ref` 本来就是它的正向映射。

## 不要做

- 别往 host 的 conda env 或容器里装东西。
- 写 ZED 的 `white_balance_*_component`（会卡死 ZED-M 的流）。
