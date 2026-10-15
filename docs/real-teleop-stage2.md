# 真机遥操作采集 stage-2 demo：配置与数据契约

2026-09-21。对象：Threading RealCell 的 **pixel SimDist**（stage-1 = `exps/mimicgen/threading_realcell_pixel_stage1{,_aug}_seed1`）。
目标：真机采的 demo 能**直接**作为 `train_simdist_adapt.py --target_demo_path` 的输入，且动作/观测的**语义与 sim 完全一致**。

下面每一条都来自代码（robosuite `OSC_POSE` / `PandaGripper` / `rl/sequence_replay.py::load_demos_to_sequence_buffer`），
数值取自 sim 实测。**采完先跑 `scripts/validate_real_demos.py`**（§7），它会抓出下面最常见的几类错。

---

## 1. 控制端：动作约定（遥操作设备 → 机器人，必须逐条实现）

动作 7 维，每维 ∈ [−1, 1]：`[dx, dy, dz, ax, ay, az, g]`

| 项 | 必须是 | 真机常见默认（要改掉） |
|---|---|---|
| 频率 | **20 Hz**，每个动作保持 50 ms | DROID 15 Hz。⚠️ real2sim 文档计划的 MPPI 也是 15 Hz —— 与 stage-1 冲突，要么全链 20 Hz，要么全部重训 |
| 平移 | `Δp = 0.05 m × a[0:3]`；目标 = **当前实测 TCP 位置** + Δp | 在**上一步目标**上累加（会积分出漂移） |
| 旋转 | `Δθ = 0.5 rad × a[3:6]`（轴角）；目标 = `R(Δθ) · R_当前实测`（**左乘，base 系**） | 右乘（末端系） |
| 坐标系 | **robot base 系**（名义 cell 下 sim 世界系与 base 系轴向相同、只差平移） | — |
| 控制点 TCP | robosuite `gripper0_grip_site` = 两指尖之间，距 robosuite hand 系 +97 mm | Franka `O_T_EE` 默认 +103.4 mm。设 `F_T_EE`/`NE_T_EE` 对齐到 grip_site，否则旋转会带出额外平移 |
| 夹爪 | **只看符号**：`g > 0` 闭合，`g ≤ 0` 张开，幅值无意义 | 连续宽度指令 |
| 限位 | sim 无限位 | 真机加工作空间限位是必须的，但贴边截断会让"命令 ≠ 执行"，记在 §8 |

**记录的是命令，不是结果**：`actions[t]` = 这一步发给控制器的归一化命令（遥操作设备的输出按上表换算后 clip 到 [−1,1]），
**不是**下一帧与这一帧的实测位姿差。sim 数据存的也是命令。

---

## 2. 观测记录：prop（9 维，⚠️ 两个坑都在这里）

stage-2 的 pixel 臂只从 state 取前 9 维 `prop = [eef_pos(3), eef_quat(4), gripper_qpos(2)]`，**未归一化**直接送进编码器。

| 字段 | 定义（与 sim 完全一致） | 真机换算 |
|---|---|---|
| `robot0_eef_pos` (T,3) | **TCP（grip_site）** 的位置，**sim 世界系**（米） | `p_world = p_base + (−0.489, −0.034, 0.912)` |
| `robot0_eef_quat` (T,4) | **`robot0_right_hand` body** 的姿态（⚠️ 不是 TCP 的），**xyzw** | `R_world_hand = R_base_flange · R_hand_flangeᵀ`，其中 `R_hand_flange` = `2cam_calibrated_v1/calibration.json` 的 `T_sim_hand_reported_flange`（绕 z 45°）。世界系与 base 系无旋转 |
| `robot0_gripper_qpos` (T,2) | 两指关节位置 `[左, 右]`，米 | `[w/2, −w/2]`，w = 夹爪开口宽度（张开 ≈ [0.04, −0.04]）。⚠️ sim 里两指是独立关节、接触时会被推得不对称（\|左+右\| 最多约 3 cm），真机只给一个宽度 ⇒ 真机这两维永远对称，是一个小的分布差 |

⚠️ **坑 1 —— 世界系 vs base 系**：sim 的 prop 在世界系，base 在 (−0.489, −0.034, **0.912**)。
直接存 base 系坐标，z 会差 0.912 m、完全落在训练分布之外，且不报错。
⚠️ **坑 2 —— 位置和姿态不是同一个参考点**：位置是 TCP，姿态是 hand body（两者差 90° 绕 z）。
⚠️ 姿态的 45° 换算和腕相机外参用的是**同一个假设**（B@45，"ee"=法兰），那个假设至今**没有真机照片验证**。如果它错了，腕相机和 prop 姿态会**一起**错。

**参考值（ready 姿态，σ=0）**：`eef_pos = (−0.1821, −0.0340, 1.4058)`，`eef_quat(xyzw) = (1, 0, 0, 0)`，`gripper_qpos = (0.0208, −0.0208)`。
**sim 数据 1%–99% 范围**：`eef_pos` x∈[−0.373, 0.350]、y∈[−0.543, 0.790]、z∈[0.925, 1.473]；`gripper_qpos` 左∈[0, 0.040]、右∈[−0.040, 0]。

**`obs/object` (T,28)：写全零占位。** loader 按 `STATE_KEYS["Threading_D0"]` 拼接 state，必须有这个键；
pixel 臂的 stage-2 BC 只读前 9 维，所以零占位是安全的 —— **但前提是不开** `bc_anchor_far_metric="state"`（它会拿 demo 的
完整 state 算距离，默认关）。

---

## 2b. ★ Robotiq cell（10-14 起）：prop 改成 **base 系**（2026-10-15，owner 选的方案 A）

**上面 §2 的表和「坑 1」只适用于旧的 Franka 手爪 cell。** `calib_20261014/threading_real_cell_20261014.yaml`
设了 `prop_frame: base`：

| 字段 | Robotiq cell 上的定义 |
|---|---|
| `robot0_eef_pos` (3) | TCP（`gripper0_grip_site`）在 **`robot0_base` 系**里的位置（米） |
| `robot0_eef_quat` (4) | `robot0_right_hand` body 在 **`robot0_base` 系**里的姿态，xyzw |
| `robot0_gripper_qpos` (2) | 两个 Robotiq 驱动关节 = `robotiq_real_to_sim(真机 knuckle)` 写两遍：`clip(1.134 × knuckle, 0, 0.556)` |

**为什么。** 真机上 prop 是拿实测 q 过 sim 的 FK 算出来的。在世界系下，它还隐含了「base 相对桌面放在哪」，
而 sim 正在对这个量做 DR（`dr_base_xy` ±7 mm、`dr_base_yaw_deg` ±1.5°、`dr_base_height` ±7.5 mm）。
实测同一个 q 下，世界系 eef 在 8 次 DR 抽样里的散布是 **(9, 27, 18) mm**：
sim 里 prop 等于「末端相对桌面的真值」，真机上却等于「名义 base + FK」，两者之差真机上没人知道。
base 系下，prop 就是 **FK(q)**：sim 和真机是同一个函数，不含任何标定。
base 与桌面的相对关系只通过图像进入策略，这和真机上的情况一致。

**已验证（`scripts/test_sim2real.py` M 节）：**
- 8 次 base DR 抽样下，base 系 prop 的差 ≤ 1e-15；
- `prop == state[:9]`；
- 部署端 `ReferenceArm.prop` 与训练 env 在同一个 q 和夹爪读数下差 1.5e-8。

**参考值**（DROID home，sim 名义）：
- `eef_pos` = (0.3597, 0.0000, 0.3319)
- `eef_quat`(xyzw) = (0.9238, 0.3829, 0, 0)
- 夹爪张开时 `gripper_qpos` = (0, 0)；夹住 40 mm 柄时 = (0.499, 0.499)

### 真机侧要做的

1. **不要自己算 prop，存原始量。** 每个控制 tick 记录：
   - `fr3_joint1..7` 的实测 q（rad，顺序和零位与 sim 一致；DROID home 的帧 0 应在 0.01 rad 以内）和 dq；
   - `/gripper/joint_states` 里 `robotiq_85_left_knuckle_joint` 的值（rad，**一个数**）；
   - 关节和每路相机帧的时间戳（同一个单调时钟）。

   prop 由我们离线用 sim FK 算（重放契约：约定改了只要重算，不用重采）。
   以前那条「加 (−0.489, −0.034, 0.912) 换到世界系」**在这个 cell 上是错的，不要做**。
2. **base 的标定不再影响 prop，但仍然影响图像和摆放。** 相机外参（base 系）、桌面高度、物体摆放都还是相对 base 量的，照常标定。
   需要补一次 **xy touch-off**：
   - 夹爪闭合，用指尖依次轻触板子上 ≥ 4 个已知位置的点（ChArUco 角点或贴胶带做标记，记下每个点的编号）；
   - 每个点停稳后记录 q，以及场景和腕相机各一帧；
   - 再触 needle 柄的两端各一次。

   这样能量出 base→桌面的 xy 和 yaw（现在只有 z：0.0287 ± 0.0012），同时验证 needle 的摆放。工具：`calib_20261014/tools/touch.py`。
3. **部署：**
   - `robot_client.py --gripper-state-topic /gripper/joint_states`：客户端认出 `robotiq_85_left_knuckle_joint` 后只发这一个值，映射在 server 里做；
   - `policy_server.py` 对非 PegInsertion 任务**必须显式给 `--z_min`**（grip site 的 sim 世界系 z）。
     sim 估计：桌面 0.9405，指尖贴桌时 grip site 比最低的指尖高约 10 mm，即 ≈ 0.950。
     请在真机上做一次闭合指尖的 touch-off，dry run 读 server 打印的 `eef[2]`，取该值 + 2 mm。
4. **图像比 prop 旧**（§3b）：请保留每帧的 capture 时间戳，部署前我们要量一次延迟。
5. ⚠️ **控制频率仍未定**（真机 10 Hz vs sim 20 Hz，铁律 #37）。sim、录制器和 runner 必须用同一个频率，定了之后才能重新生成专家和数据。

### sim 侧还没做的（不影响真机采集，只影响转换）
`scripts/convert_real_teleop.py` 和 `scripts/validate_real_demos.py` 还是 Franka 手爪时代的代码（READY 位姿、
指宽、世界系范围）。转换脚本遇到 `prop_frame: base` 或 Robotiq 会**直接拒绝**，不会悄悄写出错的 prop。
等控制频率定了再移植（改用 `env.sim2real.eef_in_base` + `robotiq_real_to_sim` + DROID home）。

---

## 3. 观测记录：两路相机

| 相机 | 真机 | 处理流程 | 结果必须等于 |
|---|---|---|---|
| `agentview` | zed2i，外参 = cell yaml（base 系 pos (0.1222, 0.5014, 0.7721)） | HD2K 左目 → **以光轴为中心**裁成 1242×1242 → 缩放到 224×224 | 竖直 FOV **60.84°**（fy 1057.6 @ 1242） |
| `robot0_eye_in_hand` | zedm，外参 = `calibration.json`（B@45 假设） | 裁成正方形 → 224×224 | 竖直 FOV **45.656°**（fy ≈ 266.07 @ 224） |

- **RGB**（ZED 默认输出 BGRA，要转）、**图像正立**（sim 存的就是正立的）。
- 裁剪必须以**光轴（主点）**为中心：sim 是无主点偏移的对称针孔，偏心裁剪会让像素↔几何的对应整体错位。
- ⚠️ 腕相机的 45.656° 是按 `calibration.json` 算的，**采之前用真机实测的 zedm 内参和裁剪方式确认一遍**这个数。
### 3b. 帧率与时序（⚠️ 当前 agentview 的内参对应的模式跑不到 20 Hz）

**sim**：相机 observable 的采样率 = 控制频率（robosuite `sampling_rate=control_freq`），**每个控制步一帧、与 prop 同一时刻、零延迟**。
**真机**：相机自由运行，和 20 Hz 控制环不同步。每个控制 tick **非阻塞地取最新一帧**，别让控制环等相机。

⚠️ **agentview 现在的内参（fy 1057.6）是 HD2K 模式的，而 ZED 的 HD2K 最高只有 15 fps** —— 跟不上 20 Hz，每 4 个控制步就有一帧是重复的。
★ **2026-09-23 这张表整个重算过** —— 之前 HD720 的 fy≈525.6 是反推的，现在拿到了
**Stereolabs 出厂标定**（按序列号下载，存 `release/cfgs/sim2real/zed_factory_intrinsics.json`）：

| 相机 | hd720 fx / fy | cx / cy | 原生竖直 FOV | 最大居中方裁 |
|---|---:|---:|---:|---:|
| scene ZED 2i (SN 37597132) | 532.445 / 532.470 | 622.555 / 352.755 | **68.13°** | 705.5 px = 67.05° |
| wrist ZED Mini (SN 16478414) | 771.808 / 771.705 | 652.745 / 351.798 | **50.02°** | 703.6 px = 49.01° |

⚠️ 出厂值是**左目、未去畸变**（正是 `publish_zed_uvc.py` 发的那一路）。畸变系数
scene `(-0.0492, 0.0206, 0.0009, -0.0006, -0.0083)`、wrist `(-0.1013, 0.0540, 0.0, 0.0001, 0.0422)`。
**必须先去畸变**：scene 的畸变在 497 px 裁剪框角上约 11 px，缩到 224 后仍有 ~4 px，和环孔同量级。
⚠️ **主点不在图像中心**（622.6 vs 640 / 352.8 vs 360）⇒ 裁剪以**主点**为中心，否则是恒定平移。
⚠️ 最大方裁被 **cy** 卡住（2×352.755），不是被 720/2。

★ **验证**：用这组内参 + 去畸变，把 sim 机器人灌上真机 t=0 关节角再投影，`hand`/`grip` 精确落在
真机腕部和夹爪上；旧的「60.84° + 图像中心」把它们甩到右侧空处（`figs/sim2real/skeleton_hd720.png`）。
⇒ **`scan_run2` 解出的外参仍然有效，之前错的一直是内参。**

**sim 的 agentview FOV 已改为 50°**（不再沿用 HD2K 的 60.845°）。理由是实测的分辨率/取景取舍
（200 次 reset、含完整 DR；完整表在 cell yaml 里）：50° 让 ring 在 224 上的像素数 **34.5 → 53.4**
（孔短边 3.8 → 4.8 px），同时最坏取景余量仍有 **62 mm**；45° 再多 24% 像素但余量只剩 24 mm。

⇒ **实机裁剪尺寸**（以主点为中心，去畸变后裁，再缩到 224）：

| 相机 | sim fovy | 裁剪边长 = `2·fy·tan(fovy/2)` |
|---|---:|---:|
| scene | **50.0°** | **497 px** of 720 |
| wrist | 45.656° | **650 px** of 720 |

⚠️ 腕相机的 FOV **还没按同样的方法选过** —— 它沿用的是 HD2K 的 45.656°，而 hd720 原生是 50.02°。
鉴于 §14e 实测「腕相机在插入瞬间 0.0% 可见」，重选它的 FOV 是一个独立的、值得做的实验。

**时序还有一个 sim 没有的量：图像比 prop 旧。** 曝光 + 读出 + USB 传输通常几十毫秒，再加上"取最新帧"带来的最多一个帧周期。
stage-1 是在零延迟下训的。⇒ **每帧记录相机自己的采集时间戳**（ZED SDK 提供），和控制 tick 的时间戳用同一个时钟，事后就能量出这个滞后。
如果滞后 ≥ 1 个控制步（50 ms），可以在 sim 侧补"延迟增广"：重放契约下按 `frame[t−k]` 配 `prop[t]` 重新组装数据即可，不需要重新 rollout。

- 存储：`uint8 (3,224,224)` 逐帧 **JPEG q90**，与 sim 同一个写法（`scripts/generate_simdist_data.py::_write_image_dataset`，
  数据集属性 `codec="jpeg"`、`frame_shape=[3,224,224]`、`jpeg_quality=90`）。`frame[t]` 与 `prop[t]`、`actions[t]` **同一时刻**。

---

## 4. 每条 episode 的协议

- **复位**：机械臂回 Franka ready `[0, −π/4, 0, −3π/4, 0, π/2, π/4]`，夹爪半开（约 41.7 mm）。
- **摆放** —— 见 §4b，用卷尺从桌边量，不用 base 坐标。
- **长度**：≤ 400 步（20 s @ 20 Hz）。
- **成功标注**：`rewards[t] = 1.0` 在**针穿过环并稳住**的那一帧（sim 的 strict success 要求连续 3 帧、环位移 < 4 cm），其余为 0。
  ⚠️ loader **只保留含 reward=1 的 episode，并在第一个 1 处截断**：失败的 demo 会被整条丢弃，成功帧标早了会截掉后半段。
- 每条 episode 的属性：`success`（bool）、`num_samples`。

---

## 4b. 物体怎么摆（真机上用卷尺量）

俯视图：`figs/sim2real/placement_map.png`（`scripts/make_placement_schedule.py --plot` 生成）。
**所有左右都是机器人视角**（从机器人底座往外看）。操作员如果站在桌子对面面向机器人，左右是反的。
"近边" = 离机器人最近的那条桌边；"中线" = 桌面长边方向的中线。

| 物体 | 参考点 | 位置 | 朝向 |
|---|---|---|---|
| **needle**（平放） | 全长 16 cm 的**中点**（针尖 ↔ 把手外侧面） | 离近边 **24.0–39.1 cm**；在中线**左侧 21.6–31.6 cm**（= 离左桌边 30.4–40.4 cm） | **把手朝远离机器人**，针尖朝机器人；偏角在 **±30°** 内（0° = 把手正对远离机器人，+ = 把手偏向机器人左侧） |
| **tripod**（每条都一样，**固定**） | 环中心**正下方**桌面上的点 | 离近边 **44.1 cm**；在中线**右侧 8.4 cm**（= 离右桌边 53.6 cm） | 环孔的轴线**平行于桌子长边**（机器人左右方向） |

这些数来自 sim 的采样分布（Threading_D0 采样框 + 真机 cell 的 workspace offset），换算到桌边坐标。
base↔桌子配准误差约 7 mm、yaw 取整残差 1.2°（远角约 13 mm）—— 相对 10×15 cm 的采样区可以忽略，摆放精度 ±1 cm 足够。

**怎么做最省事**
1. 用胶带在桌上贴出 needle 中点区域（**深 15.1 cm × 宽 10.0 cm** 的矩形）和 tripod 三只脚的位置。**tripod 每条都必须在同一位置**（sim 里它是固定的），贴死脚位。
2. 每条 demo 按 **`release/cfgs/sim2real/threading_real_placement_seed0.csv`** 的一行摆（`scripts/make_placement_schedule.py --n 50 --seed 0`）：
   它从 sim 同一分布里**分层抽样**（Latin hypercube），50 条就能均匀铺满位置和角度，不会扎堆在中间。
   人手随手摆会集中在"舒服"的位置，而 eval 是整个分布 —— BC 在没覆盖到的角落会崩。
   每行给出：离近边 cm、中线左侧 cm、离左桌边 cm、把手偏角°，外加 base 系坐标（写进该 episode 的 attrs 以便事后分析）。
3. 角度：在区域旁边贴一张画了 0°/±15°/±30° 射线的纸即可，±5° 的精度足够。

**采集前确认**
- **真机零件尺寸要和 sim 一致**：needle 全长 16 cm（4 cm 方块把手 + 1×1×12 cm 针杆），否则"中点"这个参考点就对不上，而且几何本身也不一样了。
- **相机视野里别有 sim 里没有的东西**：ChArUco 标定板、工具、线缆都拿走。sim 里是一张空桌面。
- 每条开始前机械臂在 ready 姿态、夹爪半开；桌面高于 base 原点 20.7 mm（量一次确认没人动过垫子）。

---

## 5. hdf5 布局（stage-2 loader 直接读）

```
data/
  demo_0/
    actions                    (T, 7)   float32   归一化命令，§1
    rewards                    (T,)     float32   成功帧 = 1，§4
    obs/
      robot0_eef_pos           (T, 3)   float32   世界系 TCP，§2
      robot0_eef_quat          (T, 4)   float32   hand body，xyzw，§2
      robot0_gripper_qpos      (T, 2)   float32   [w/2, −w/2]，§2
      object                   (T, 28)  float32   全零占位，§2
      agentview_image          (T,)     vlen uint8  JPEG，attrs codec/frame_shape/jpeg_quality，§3
      robot0_eye_in_hand_image (T,)     vlen uint8  同上
      timestamp                (T,)     float64   控制 tick 的墙钟秒（校验 20 Hz）
      agentview_capture_ts     (T,)     float64   该帧的相机采集时间戳，与 timestamp 同一时钟（量图像滞后）
      robot0_eye_in_hand_capture_ts (T,) float64  同上
    attrs: success, num_samples
  demo_1/ ...
env_cfg.json   (同目录 sidecar，⚠️ 必须含 "env_name": "Threading_D0" —— loader 用它查 STATE_KEYS，缺了直接报错；
                另外记录 cell yaml 路径+sha、calibration.json 的 sha、相机序列号与分辨率档、F_T_EE、控制频率)
```

---

## 6. 采多少

- stage-2 默认 `target_num_data: 20`，但 **demo 数是主轴**（simdist §43：pixel 从 N=20→200 分数 0.08→0.24，**未饱和**）⇒ 建议 **≥ 50，能到 100–200 更好**。
- 要做多 seed 时用 `DEMO_OFFSET` / `target_demo_offset` 取**互不重叠**的 draw（铁律 #25：否则 N 个 seed 训的是同一批 demo，误差棒是假的）。
- 单次 20 条抽样本身的方差就是 σ≈0.092（铁律 #1），比多数可控干预都大 —— 条数越多越稳。

---

## 7. 采前 / 采后的两道检查

**采前：阶跃响应对照。** 在 sim 和真机上发同一串归一化动作（每个平移/旋转轴 +1 连发 10 步再 −1 连发 10 步，夹爪 +1/−1 各 15 步），记录 TCP 轨迹。
- 方向反了、旋转时带出平移、或夹爪符号反了 ⇒ §1 的约定没实现对；
- 每步实际位移的大小与响应时间 ⇒ 跟踪动力学差距，就是 stage-3 要修的量（sim 是 kp=150 临界阻尼的 OSC）。

**采后：`python scripts/validate_real_demos.py --dataset <path>/dataset.hdf5`**，检查键齐全、长度对齐、prop 落在 sim 分布内（抓世界系/base 系之错）、
四元数单位且接近 ready 姿态（抓 xyzw/wxyz 之错）、夹爪范围、动作在 [−1,1]、每条有成功帧、图像可解码且尺寸对、时间戳 ≈ 20 Hz，
并输出一张接触表 `contact_sheet.png` 供肉眼看方向和裁剪。

校验器本身验证过（2026-09-21，用 sim 数据造的规范格式）：合规数据 0 FAIL / 0 WARN；故意存成 base 系坐标、或 wxyz 四元数，都被准确判为 FAIL；规范格式能被 `load_demos_to_sequence_buffer` 完整读入（state 37 维、两路帧 3×224×224）。

---

## 7b. ★ 采数据之前必须先定的（定错了 demo 就白采）

1. **控制频率 —— MPPI 在 20 Hz 下可能算不完。** 实测（RTX PRO 6000 Blackwell，batch 1，pixel r18@224，5 个 dynamics 头，H=5，
   `scripts/bench_inference_latency.py` 的计时法）：

   | planner | 中位数 | p99 |
   |---|---:|---:|
   | **512 样本 × 6 轮（stage-1 cfg 的设置）** | **58.8 ms** | 59.3 ms ⇒ **超出 50 ms** |
   | 512 × 4 轮 | 44.6 ms | 45.2 ms |
   | 512 × 3 轮 | 36.5 ms | 36.8 ms |
   | 350 × 3 轮 | 28.7 ms | 29.0 ms |
   | 只跑 policy 头 | 6.2 ms | 6.3 ms |

   耗时由**轮数**决定（样本数几乎免费，findings §36「宽免费、深要钱」）。以上**不含**相机采集/裁剪/缩放和机器人通讯，
   且真机推理台式机的 GPU 多半更慢。⇒ 两条路：**(a) 20 Hz + 减少轮数**（planner 超参是 eval 时的旋钮，不用重训，
   但铁律 #9：超参不可移植，必须先在 sim 里测减轮数掉多少分）；**(b) 全链改 15 Hz**（stage-1 数据和训练都要重做）。
   **demo 必须按最终部署的频率采。**
   ⚠️ stage-1 cfg 是 `use_planner: 0` ⇒ 训练中看到的 `score/score` 全是 **policy 头裸跑**的分数，MPPI 的分数还没测过。
2. **夹爪换成 Panda hand。** real2sim §6 记录真机当前装的是 **Robotiq 2F-85**，而整条 sim 链（几何、腕相机画面里的手、
   `gripper_qpos` 的语义、TCP 偏移）都是 Panda hand。
3. **用一张真机腕相机照片验证 B@45。** 机械臂停在 ready 姿态拍一张 zedm，与 sim 同姿态的渲染叠加比对
   （`real2sim/scan_pipeline/render_camera_pov.py`）。腕相机外参和 prop 姿态换算都押在这个假设上。
4. **用尺子量一次 ChArUco 方格**（infra §14a R4）：绝对尺度未闭合，最坏整体偏 6%。
5. **阶跃响应对照**（§7）。

## 7c. 采集过程的纪律

- **★ 同时存原始数据**：每步的 **7 个关节角 q**、夹爪宽度、相机**原生分辨率**的帧（至少是未裁剪的）、以及时间戳。
  这是真机版的"重放契约"（infra §14e）：如果 TCP 点、45° 换算、裁剪或 FOV 事后发现要改，可以**用 FK 从 q 重算 prop、
  从原始帧重新裁剪**，而不用重采。只存 224 的裁剪帧和换算后的 prop，任何一个约定错了这批数据就作废。
- **锁定曝光和白平衡**（real2sim 的 scan3 就是被自动曝光漂移毁掉的）；曝光时间要短，压住运动模糊。
- **相机、桌子、垫子在两次采集之间都不能动**。每次开工先用 ChArUco 拍一张解外参（real2sim 有现成流程），和存档值比对并记录。
- **失败的 episode 也要存**（`success=False`）：stage-2 BC 会跳过它们，但 stage-3 和事后分析用得上。
- **操作速度**：sim 专家的 demo 中位数约 180 步（9 s），上限 400 步（20 s）。先练熟再采；**避免长时间停顿**，
  停顿会产生大段零动作，BC 会学成"原地不动"。
- **每条 episode 记录**：操作员、日期、标定文件 sha、相机模式与内参、光照情况、按清单摆放的那一行（§4b）。
- **评估用另一套摆放**：用 `make_placement_schedule.py` 换一个 seed 生成 held-out 摆放，不要在 demo 用过的位置上评估。

## 8. 已知对不齐的（记下来，不是配置能解决的）

- **跟踪动力学**：同一命令 50 ms 内实际走多远取决于阻抗增益、摩擦、负载；sim 只有一种（未加物理 DR）。→ stage-3。
- **夹爪时序**：sim 约 0.4 s 连续爬坡；真 Panda hand 是 grasp + 力控。
- **延迟**：sim 零延迟；真机有感知/推理/通讯延迟。
- **限位截断**：贴边时执行的动作 ≠ 记录的命令。
- **腕相机视野**：插入瞬间环和针尖在腕相机里 **0.0% 可见**（infra §14e）。真机这个相机位提供不了对准信息；要改只能**物理挪相机**，
  但挪了之后 stage-1 的腕相机数据就和真机不一致了（需要按新外参重渲染 + 重训 stage-1）。
- **相机 DR 的不对称**：训练数据里 agentview 带每局相机抖动，腕相机没有。
