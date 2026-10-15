# 下次上机清单（2026-10-15 写）

> **2026-10-15 进度**：§0、§1 已做完，结果见 [`../calib_20261015/README.md`](../calib_20261015/README.md)。
> 冲过头 ≤ 1.6 mm（不是撞桌原因）；真机跟随 ≈ τ 100 ms 一阶滞后；**x 方向每步只完成 1–4 %** ——
> 服务端每步从实测 q 起算目标，偏差推不过静摩擦（09-25 已发现并用 `sim_ref` 修过，但没进主仓库）。
> **下一步：把 `sim_ref` 移植进 `policy_server.py`，用 `calib_20261015/tools/run_tracking_test.sh` 重测，再定 §2。**
> 一键脚本取代了 §1 里的手动命令。
> **10-15 晚**：`sim_ref` 已移植进 `policy_server.py`（默认开，`--action_map reseed` = 旧行为），离线验证 x 能动了；一键脚本现在默认用 `sim_ref`，按 `past tgt mm` 判停。下一步就是在真机上重跑它。

原则：**先量、再定、最后才采**。控制频率没定之前采的 demo 全部作废（CLAUDE.md 铁律 #37）。
每一步后面写了**为什么**；遇到和预期不一样的情况，先停下来把数据带回来，别现场改方案。

## 0. 出发前 / 到了先做

- [ ] **同步代码到工作站**（这批改动还没 commit；commit/push 后在 `~/simdist/SimDist_ibrl`、`~/simdist/deploy_simdist`
      里 pull，或者 rsync 下面这些文件）：
      `env/{sim2real,robosuite_wrapper,threading_variants}.py`、`real2sim/deploy/{policy_server,robot_client}.py`
      （client 还要拷到容器里的 `/workspace/simdist_runner/`）、`calib_20261014/threading_real_cell_20261014.yaml`、
      `calib_20261014/tools/{record_calib.py,analyze_tracking.py,gello_patch/}`。
- [ ] **冒烟（不动机械臂）**：在 host 上起 `policy_server.py --scripted track ...`（见 §1）。
      Threading 需要 `mimicgen`，如果 `simdist` 这个 env 里 import 失败就**停下**，别往 host 的 env 里装东西（铁律 #35），回来再说。
- [ ] 机械臂回 **DROID home**（`reset_arm.py`）；Robotiq **张开**。

## 1. 跟踪滞后 / 延迟测试（决定控制频率）

**做法**：脚本动作代替策略，在空中（桌面以上 ≥ 15 cm；server 自动设 `z_min = 桌面 + 0.15`）先沿 z 再沿 x，
各按 0.2 / 0.5 / 1.0 三档动作幅度走 6 cm → 停 1.5 s → 回来 → 停。20 Hz 和 10 Hz 各跑一次，**其余完全相同**。

```bash
# host（teleop session 照常开着，cameras 要发布）
cd ~/simdist/deploy_simdist
~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py --scripted track \
    --task Threading_RealCellRobotiq \
    --cell real2sim/zed_capture/calib_20261014/threading_real_cell_20261014.yaml \
    --save_dir runs/track20_$(date +%m%d_%H%M) --save_every 1

# 容器：先 dry run（不动，只看链路通不通、打印的 eef 合不合理），再 live
python3 /workspace/simdist_runner/robot_client.py --gripper-state-topic /gripper/joint_states \
    --hz 20 --steps 3000 --joint-log /workspace/simdist_runner/track20.npz
python3 /workspace/simdist_runner/robot_client.py --gripper-state-topic /gripper/joint_states \
    --hz 20 --steps 3000 --joint-log /workspace/simdist_runner/track20.npz --live
# 跑完：把 ~/simdist/gello_software/simdist_runner/track20.npz 挪进对应的 runs/track20_*/joints.npz
# 然后 server 换一个 --save_dir runs/track10_...，client 换 --hz 10、track10.npz，再跑一次
```

现场就能看结果：`python calib_20261014/tools/analyze_tracking.py runs/track20_* runs/track10_* --fig track.jpg`

| 看什么 | 含义 |
|---|---|
| `done/step`（每步实际完成 / 指令） | sim 假设 = **1.00**。明显 < 1 ⇒ 真机跟不上，sim 要加跟踪滞后 |
| `overshoot mm`（指令停了之后还走多远） | sim = **0.0**（本地用假机械臂验证过）。这就是撞桌子的量 |
| `joint delay ms`（100 Hz 命令 → 实测关节） | 纯延迟，sim 的延迟 DR 按它定 |
| `age`（图像帧的年龄） | ROS 接收时间，**下界**；曝光 + USB 还要另加 |

**为什么**：20 Hz 撞桌子有三种解释（sim 专家太快 —— 已量到约人的 3 倍 / 真机跟踪滞后 / 图像延迟），
只有这份数据能分开。在空中做不会撞；同一份日志同时给 sim 的 OSC kp/阻尼、延迟 DR、减速区厚度提供数字。

⚠️ 一直把手放在急停上。`overshoot` 大的那一档跑完就停，不用把三档都跑完。

## 2. 和 owner 当场定控制频率

拿 §1 的表：如果 20 Hz 下 `done/step` 和 `overshoot` 跟 10 Hz 差不多 ⇒ 问题在专家速度，留 20 Hz、sim 里加限速 + 延迟 DR；
如果 20 Hz 明显跟不上 ⇒ sim 整体改 10 Hz。**定不下来就跳过 §6。**

**为什么**：sim、录制器、runner 必须同一个频率；demo 既当 stage-2 target 又当 MimicGen 种子，频率错了两头都废。

## 3. 重标腕相机（标定板 + 双目）

teleop session 要**不发布相机**（`run_teleop_session.py --cameras` 不带名字），容器里开 `joint_tap.py`。

```bash
~/anaconda3/bin/python3 ~/simdist/SimDist_ibrl/real2sim/zed_capture/calib_20261014/tools/record_calib.py \
    --out ~/simdist/calib/run_$(date +%m%d_%H%M) \
    --wrist_intrinsics ~/simdist/SimDist_ibrl/real2sim/zed_capture/calib_20261010/data/wrist_intrinsics.json
```

- [ ] **≥ 15 个位姿**（自动存：停稳 1.5 s 且和已存的都差 ≥ 8°），其中 **≥ 8 个倾斜 ≥ 35°**（窗口上方有计数）
- [ ] 其中几个让 **needle 也在腕相机画面里**（needle 放在名义位置）
- [ ] 启动时确认打印了 `controls locked` 和**没有** `SN....conf not found`（没有 `.conf` 就不能三角化，手动拷进 session 目录）
- 新脚本自动存**右目**（`*_right.png`）。

**为什么**：腕相机现在差 1–1.6 cm（沿光轴）+ ~2°；上次两帧分不清"相机装偏"还是"sim 的 Robotiq 几何不对"。
标定板 PnP 能辨识沿光轴的平移，右目三角化独立复核这个方向。标好后，为兜误差放宽的腕相机 DR（±(6,10,20) mm / ±(3,3,4.5)°）
可以收窄 —— 宽 DR 对 pixel 策略有代价。

## 4. Touch-off（xy + z_min）

同一个 session 状态，Robotiq **闭合**：

```bash
~/anaconda3/bin/python3 .../calib_20261014/tools/record_calib.py --touch \
    --out ~/simdist/calib/touch_$(date +%m%d_%H%M) \
    --points c1,c2,c3,c4,needle_end_a,needle_end_b,zmin
```

- [ ] `c1..c4`：板子上 4 个**已知位置**的点（ChArUco 角点，或贴胶带后用尺子量到板边的距离 —— **记在纸上，拍照**）
- [ ] `needle_end_a/b`：needle 柄的两端（needle 在名义位置）
- [ ] `zmin`：指尖平贴在板面上
- 窗口顶部会提示下一个点；指尖轻触、停稳、SPACE。

**为什么**：prop 改成 base 系后不再依赖 base 标定，但**图像和摆放**还依赖 base↔桌面。现在只有高度
（0.0287 ± 1.2 mm）；xy 两次求解差 ~7 mm，yaw 没量过。量准后 `dr_base_xy` / `dr_base_yaw_deg` 能收窄。
`zmin` 给部署时防撞的硬底线（`policy_server --z_min`）。

## 5. 摆放照 + 曝光漂移

```bash
~/anaconda3/bin/python3 .../calib_20261014/tools/record_calib.py --no_auto --out ~/simdist/calib/layout_$(date +%m%d_%H%M)
```
- [ ] needle、tripod 摆在名义位置，机械臂在 DROID home：SPACE 一次
- [ ] 只有板子（物体拿开）：开始时 SPACE 一次，之后每 ~15 min 一次，到离开前

**为什么**：摆放照核对 sim 的摆放范围、给回放提供初始物体位姿；上次同样锁定曝光、不同时刻亮度差 ~10%，
部署颜色校正要知道这个漂移有多大。

## 6. 采 demo（**只有 §2 定了频率才做**）

- [ ] 20–30 条，DROID home 起，物体在 sim 摆放范围内（`threading_real_cell_20261014.yaml` 的 layout 注释），定下的频率
- [ ] 存原始量：q、dq、`robotiq_85_left_knuckle_joint`、所有时间戳、原生分辨率图像；**不要自己算 prop**
      （也**不要**加旧的 (−0.489, −0.034, 0.912)，见 `docs/real-teleop-stage2.md` §2b）
- [ ] 每条记成功/失败 + 成功的那一帧

**为什么，一份数据用三次**：① SimDist stage-2 的 target demo；② sim 里回放、跑通的作 MimicGen 种子放大到 ~1000 条，
训出带人类速度特征的 BC prior；③ 给 sim 里按高度限速提供参考曲线（人最后 3 cm 下降 ~4 cm/s，sim 专家 ~8 cm/s）。
回放失败的位置本身就是真机这个 target 域的 dynamics/几何 gap。

## 不要做

- 现有策略在 20 Hz 下靠近桌面（原因没查清之前会再撞）
- 写 ZED 的 `white_balance_*_component`（上次把 ZED-M 的流卡死，要拔 USB）
- 往 host 已有的 conda env 或容器里装东西

## 带回来的东西

`runs/track20_*`、`runs/track10_*`（含 `joints.npz`）、`~/simdist/calib/{run,touch,layout}_*` 整个目录、
touch 点的纸面记录照片、和 owner 定下的频率。
