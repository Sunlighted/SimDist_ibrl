# 下次上机清单（2026-10-17 写；上一版的负载辨识已做完，见 README「Payload probe」）

> **10-18 进度**：§0 同步 ✅（跟踪脚本改成默认 10 Hz + 1018 客户端）。§1 抓取网格 ✅：真机 12/12 抓起；sim 对比临界阻尼接触 12/12 一致、robosuite 原参数 1/12 ⇒ 用临界阻尼。§2 跟踪 ✅（不限速，z 到 25 cm/s、x 到 17 cm/s 都跟得上，越过目标 ≤ 8 mm）。一次机器人通信急停（communication_constraints_violation）。详见 README「10-18 robot visit」。
> §3 demo ✅ 10 条（`demos/threading_robotiq_1018/`，不进 git，数据完整；抓前悬停 1 s 基本没做到，只有第 7 条有）。
> ⚠️ **10-18 抓取网格第 2 次试抓时参考臂发散**（提起时机械臂突然快速上移，目标被夹到工作区角上）。根因：部署端
> `ReferenceArm.set_state` 只设 Robotiq 的 2 个主动关节，4 个从动关节留在上一步，夹爪一开合就互相穿透（16 mm），
> 接触力把参考臂踢飞。已修（6 个关节一起设，`inner_knuckle = +driver`、`inner_finger = −driver`）；用记录的真机输入重放：
> 修前在第 333 步开始发散、重新对齐 5 次，修后 0 次、提起平稳。exec_layer 等价测试仍通过（2.3e-5 rad）。
> ⚠️ sim 侧：凡是用部署 `ReferenceArm` 且夹爪闭合的离线计算（例如 `convert_real_teleop.py` 的 `ref_inverse` 标签、prop）都可能受过影响，要重跑。

目的：**定下 Robotiq 的抓取接触模型**，然后开始 sim 训练。其余各项只是顺手复查。全程约 1 小时。
背景：sim 和真机的初始化、执行层都已对齐；剩下的是抓取。只改接触参数，sim 能抓起的 demo 就整批换掉，而 5 条人手 demo 定不住它
（README「Object poses」）。dry run 里，robosuite 原接触参数下同一套脚本只抓起 **1/15**，临界阻尼时 **12/15**。真机结果就是判据。

## 0. 同步（5 分钟）

```bash
cd ~/simdist/SimDist_ibrl && git pull
cd ~/simdist/deploy_simdist_1015 && git pull
cp ~/simdist/SimDist_ibrl/real2sim/deploy/robot_client.py ~/simdist/gello_software/simdist_runner/robot_client_1018.py
```
- 新版 client 会从 server 读每步的关节截断：这个 cell 不限速，所以不截断。
- 开始前拍一张标定板（`record_calib.py --no_auto`），检查手腕支架有没有又松动。

## 1. 抓取网格（约 25 分钟，最重要）

needle 放在 10-15 demo 的那个位置附近就行，**不需要精确**：用胶带画个轮廓，每次大概放回去即可（±1 cm、±10° 都可以）。
每次试抓，机械臂都会先悬停在柄的正上方，存一张原始分辨率的手腕图，回来后逐次精确拟合 needle 位姿；sim 用这个位姿逐次复现。

```bash
# host
cd ~/simdist/deploy_simdist_1015
~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py --scripted grasp \
    --task Threading_RealCellRobotiq \
    --cell real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml \
    --grasp_needle 0.551,-0.118,-100 \
    --save_dir runs/grasp_$(date +%m%d_%H%M) --save_every 5

# 容器（teleop session 照常开着，两个相机都要发布）
export CYCLONEDDS_URI=file:///workspace/ros2/cyclonedds_ws.xml
python3 /workspace/simdist_runner/robot_client_1018.py --gripper-state-topic /gripper/joint_states \
    --hz 10 --steps 30000            # 先 dry run：看前几步动作合理，Ctrl-C，重启 server（试次计数清零）
python3 /workspace/simdist_runner/robot_client_1018.py --gripper-state-topic /gripper/joint_states \
    --hz 10 --steps 30000 --live
```

每次试抓（约 15 s）的流程：移到柄的上方 → 悬停 1.5 s（拍照）→ 下降 → 闭合 → 提起 10 cm → 停 1 s → 放回 → 松开 → 抬起 →
**在 server 终端提问**：抓起来了吗？输入 `y`（抓住并提起）/ `n`（没抓起）/ `s`（提起来但滑了），把 needle 放回轮廓里，然后回车，开始下一次。
等待的时候机械臂停在柄上方约 8 cm，手可以伸进去摆 needle。

- 一共 15 次：指尖离桌面 5 / 10 / 15 mm × 沿夹持方向 −6 / 0 / +6 mm，加上另一个方向 ±8 mm 各 2 次，再加 2 次重复。
- 不限速：移动约 8 cm/s，和人手差不多。**手一直放在急停上**。5 mm 那一档指尖很贴近桌面，属于正常。
- 用 **Ctrl-C** 中断（会停住并交还 GELLO），**不要按 User Stop**（会把整个 robot driver 杀掉）。

回来后：`python scripts/grasp_twin.py --real runs/grasp_* [--grip_solref ...]`，逐次比较 sim 和真机的结果，挑出和真机吻合最好的接触参数。

## 2. 跟踪测试（约 10 分钟）

负载已经修好，cell 也不限速了，要重新量一下滞后，用来重新拟合机械臂参数（README「Execution layer」）：
`calib_20261015/tools/run_tracking_test.sh`（它现在用 10-15 cell，不限速）。⚠️ amp 1.0 那一档在空中会超过 20 cm/s，手放在急停上。
带回 `runs/track*`（steps.jsonl + joints.npz）。

## 3. demo（约 20 分钟，约 10 条）

- 第一批修好负载之后的 demo（不再带下垂）。`run_teleop_session.py --gripper robotiq --rate 10`。
- **不要碰 tripod**。**抓之前让手腕在 needle 上方停 1 s**（用来精确定物体位姿，同第 1 项）。
- 摆放尽量散开，覆盖新的范围：needle 朝向 −145° 到 −55°，tripod 朝向 −50° 到 −25°、位置 ±4 cm（cell 第 4 项）。

## 4. 顺手

- 记下 Robotiq 驱动的速度和力设置（launch 参数或配置文件）。
- （可选）录制器同时存右目，以后可以双目三角化。

## 带回来的

`runs/grasp_*`（整个目录：steps.jsonl、hover_trial*_wrist.jpg / _scene.jpg / _state.json）、`runs/track*`、demo 的 h5、标定板照片、Robotiq 设置。

## 不要做

- 往 host 的 conda env 或容器里装东西。
- 写 ZED 的 `white_balance_*_component`（会卡死 ZED-M 的流）。
