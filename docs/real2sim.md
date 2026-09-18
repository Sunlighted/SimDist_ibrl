# real2sim：从真实 DROID cell 构建 source sim，跑完整 stage 1–3

Status 2026-09-15. 方向见 memory `real2sim-droid-plan`；本文件是唯一真源。
硬件：DROID cell（Franka Panda + Robotiq 2F-85 备选 Panda hand，polymetis on NUC，
推理台式机（π0.5 已验证），3× ZED（2 外 + 1 wrist ZED Mini））。cluster 事实见
`/scratch/yifanr/realsim_survey/reports/00-cluster-facts.md`（Vulkan/EGL headless 均可用，
`aip-rhinehar` 的 FairShare 远好于 `aip-gigor` —— 提交 job 用它）。

## §1 架构决定（全部实测支撑，勿凭引用重开）

**保留 MuJoCo/robosuite，把 `_GaussianRenderer`（discoverse-dev，MIT）的 batched gsplat
路径接上来做 photoreal 渲染。不迁移到 PolaRiS/Isaac Sim。**

实测（均为本集群 L40S，raw jsonl 在 `/scratch/yifanr/{polaris_bench,gsbench}/results.jsonl`，
报告与 raw jsonl 的存活副本在 **repo 的 `real2sim/reports/`**）：

| 路径 | 条件 | 数字 |
|---|---|---:|
| PolaRiS as shipped | 1 env, 2 cam 720×1280 | **3.98 steps/s** |
| 其中 Isaac 物理+RTX 的固定地板 | 与分辨率和 splat 均无关 | **173 ms/step** |
| 降分辨率 18×（720p→224²） | full 路径 | 只快 **16%** |
| `_GaussianRenderer` batched | 763k gaussians, 2 cam 224² | **558 tps/GPU**（3 cam 372；≈1100 img/s 恒定，按 cam 数均分） |
| 我们的 MuJoCo 物理 | state step | 11.82 ms（现在物理反而是大头） |

要点：PolaRiS 的瓶颈**不是** splat（只占 251ms 里的 73ms），是 Isaac 每步固定开销 ⇒
"photoreal vs 吞吐"的两难是 PolaRiS 特有的，不是 gaussian splatting 固有的。
gsplat 吞吐在真实重建规模（~730k gaussians）下：**16 env 即饱和**，之后只涨 VRAM
（~12 GB/cam @64env）；成本对 gaussian 数**线性**（1 env 下的亚线性是 GPU 欠载假象，勿引用）。
18M transitions 的 source replay ≈ **9 h/GPU（渲染实测）**，端到端估 4–12 h/node（未实测）。

## §2 Vertical slice：PolaRiS 机器人 splat → robosuite → gsplat（已通）

`real2sim/`（repo 内，从 /scratch/yifanr/gsbench 拷贝的存活副本）：`gs_mujoco.py`（可复用桥，含 per-body 刚性校正——**必须同时旋转
gaussian 的 orientation**，只转 xyz 会让椭球全部错向）、`check_frames.py`、
`calibrate_splat_frames.py`、`render_slice.py`、`splat_frame_calib.json`。
venv `/scratch/yifanr/gsbench/.venv`：mujoco 2.3.2 + robosuite 1.4.1 + numpy 1.26.4
（与生产一致）+ gsplat 1.5.3 + torch 2.7.0+cu128，与 `~/venvs/ibrl` 隔离。

结论：
- PolaRiS `hf-hub/nvidia_droid/SEGMENTED/*.ply` = **MIT、metric（link 尺寸对到 ~1cm）、
  2DGS surfel（loader 自动补第三轴）**。我们真机就是 DROID ⇒ 机器人 splat 直接用。
- 但 link 坐标系 ≠ robosuite body 坐标系 ⇒ 需 per-body 刚性校正（Robotiq 件差 19–39×）。
- **`panda_link0` 必须丢弃**（扫进了 NVIDIA 的底座；它是静止 base，归入背景 splat）。
- 最优配置 IoU 0.733（frontview 512²）；残差部分来自"generic DROID vs robosuite MJCF"
  的模型差，不全是可解变换。**场景/物体 splat 必须自己扫**（他们的厨房不是我们的 cell）。
- 校正精修用**可微渲染**（gsplat 可微，对 silhouette 直接梯度下降），别再用 ICP。

四个坑（都产生过自信的错误答案）：① `geom_pos/geom_size` 不是 mesh 质心/尺寸（是 frame
原点+包围半径），要用 group==1 的 mesh **顶点**；② ICP RMS 分不出对称 alias（178° 翻转也
能 <5mm），判据 = 与 identity 的 RMS 比 + 渲染；③ 单 link IoU 被遮挡不对称性压低，只有
整臂 IoU 可比；④ 相机选错会让测试静默失效（agentview 只拍到 gripper，link1 两配置逐位相同）。

## §2b 真实 cell 扫描 → 场景 splat + 任务几何（2026-09-15，全部在 `/scratch/yifanr/gsbench/`，⚠️ scratch 会被清）

**场景 splat 已建成；peg/board 走的是 mesh 路线，没有（也不需要先有）自己的 splat。**

- **场景**：scan3（手机 AE 漂移，12% 帧糊）作废 ⇒ `ae_lock_check.py`（判 AE lock 要看**连续帧**
  亮度差，整段 std 会把好扫描误杀）⇒ AE 锁定重扫 scan4（2563 帧）→ `extract_frames.py`
  按窗口取最锐帧（400 帧）→ COLMAP（single_camera OPENCV + exhaustive，**400/400 注册**，
  reproj ~0.2px，91k 点）→ 自写 `train_3dgs.py`（gsplat，7000 步，93s）→
  **`sfm_scan4/scene.ply` = 234,975 gaussians，PSNR ~22–25 @960×540**，
  6 张验证渲染在 `scene_renders/`（光学平台/机器人底座/ChArUco 都清晰可辨）。
  ⚠️ 首跑 `torch.optim.Adam(non-leaf)` 崩过一次（参数要 `.detach().requires_grad_()`）。
- **peg + board**：视频扫描（peg.MOV/board.MOV）**过不了质量门**（无纹理 100%、peg 40% 糊、
  相机基本不动 ⇒ SfM 不可用）⇒ 改用**手机 app 直接导出的 mesh**：
  `/home/yifanr/peg_15.stl` + `peg_board_2.stl`（已拷进 `peg_task/meshes/`）。
  STL 无纹理 ⇒ 物体的 photoreal 外观仍缺（见 §4-A 残留项）。
- **任务 MJCF 原型 `peg_task/peg_in_hole.xml`**：board 视觉 = 扫描 mesh，碰撞 = 4 个
  fitted box 墙（`fit_collision.py`：primitive 优先、hull 兜底，outside_p95 + fill 双判据；
  `board_hull_*.stl` 是凸分解备份，xml 未引用）；peg = 扫描 mesh 碰撞。
  **实测间隙（从几何算）：hole 37.9×26.1 mm vs peg 33.0×21.2 mm ⇒ x/y 都 ~2.4 mm/边**
  —— 在 1.3 mm 地板之上、贴近 3 mm 规格。⚠️ 真实 peg 是**长方形**截面（33×21），不是方的。

## §2c scene.ply → 世界系定标（2026-09-15 已解，Phase A 残留①）

`/scratch/yifanr/gsbench/calibrate_scene_scale.py` → `scene_world_transform.json`
（`X_world = s·R·X_colmap + t`）。**不需要知道 ChArUco 的物理尺寸**：同一批 marker 角点
（5X5_50 字典）在 ZED 深度里是米制（scan_run1 的 8 个 pose，60 角点），在 COLMAP 里是
任意单位（189/400 帧检出，68 角点三角化，reproj<3px 门控），配对距离之比即尺度：
**s = 0.1089 m/unit，1755 对的 IQR 2.2%**。姿态：角点平面 = 桌面 → 贴到 sim 桌面 z=0.8，
法向符号用相机重心判；面内轴 = 角点 PCA。⚠️ 面内 yaw/平移是**约定**（board 中心放
(0.20,0.30)）——最终要用机器人底座配准替换（splat 里有 Franka 底座，PolaRiS metric）。
SH 高阶系数未随 R 旋转（viz 够用；stage-1 数据生成前要修）。

## §6 夹爪决定（2026-09-15 owner）：**用 Panda 手**

RL env / demo / BC / IBRL 全部用 robosuite 默认 Panda gripper（现状即是，无需改）。
后果：① **真机要把 Robotiq 2F-85 换成 Panda 手**（scan_run1 的照片里当前装的是 Robotiq）；
② PolaRiS DROID splat 只有 Robotiq 件 ⇒ **手部没有 splat**——渲染里手用 MuJoCo mesh 合成
（render_peg_splat.py 已按缺 body 自动 skip 对应 token），以后要么扫 Panda 手的 splat，
要么接受 mesh 手（手在 wrist 相机里很大，pixel 阶段前要定）。

## §2d-0 scan4 已废弃（owner 2026-09-16：「质量差到没用」）
scan4 派生物全部退役：`sfm_scan4/scene.ply`、`scene_world_transform.json`、以及任何
以 scan4 COLMAP 三角化角点为参照的解。替代链（全部 scan4-free）：
**board_model.json**（scan_run2 的腕 ZED 近拍：14 视角互配准融合，68 角点、
逐角点散布中位 2.6 mm、平面残差 p95 2.7 mm，米制）→ scan5 经
`calibrate_scan5_to_board.py`（Umeyama 含尺度）挂到同一 board 系。

## §2d 接地（grounding）架构：一切经由 ChArUco（2026-09-16）

**锚 = ChArUco 板本身**（board 系由自己的角点定义 ⇒ 板在屋里挪动不碎链，但一个
session 里 scene 扫描 / 相机照片 / 机器人标定必须共享同一次摆放）。三条边：

1. **scene splat → board**：§2c 的流程，逐次扫描自动重解（scan5 进行中）。
2. **外置相机 → board**：**一张照片即可**。ZED Explorer 拍板（rectified，无需 depth）→
   已知米制角点世界坐标 → PnP。实测（相机挪位后的新照片）：31/32 内点、重投影中位
   **1.06 px**；老位置上与 8 次独立 depth-Kabsch 解的散布 **0.36°/5.8 mm** 互证。
   脚本：`scan_pipeline/` 里 `zed2i_new_extrinsic.json` 的生成段（session 开始例行跑）。
   ⚠️ 内参必须用**该分辨率档的原生标定**（ZED 各档不共 FOV —— ×比例缩放会差 17%，见下）。
3. **机器人 base → board**（待做，二选一，A 为主）：
   - **A：重采 scan_run1 式 hand-eye**。现有 8 pose 失败的两个具体原因：wrist ZED 只有
     3/8 pose 看到板 + 旋转多样性不足（Tsai "not enough informative motions"）。
     规格：**12–15 pose、板全程在腕相机视野（0.3–0.6 m）、pose 间腕姿态绕 ≥2 根轴
     各变 ±30–45°、板与机器人固定**。附赠 T_ee→wristcam（pixel 阶段的腕相机反正要）。
   - **B（最简、无相机）：tip-touch**。EE 尖点触板上 4+ 个 marker 角点（记 FK）→
     Kabsch。5 分钟、~2–5 mm；也可作 A 的交叉验证。
   - 解出 T_base→board 后，**一切改为 base 系表达**（scene splat / 相机 / 任务板位姿），
     钉到 robosuite 的 base 上 —— z=0.8 的"板面即桌面"约定退役。板或机器人不动就一直有效。

★ **接地已全部解出（2026-09-16 重采后）**：board model 68 角点/1.5 mm；
zed2i 从自身 HD2K 帧解出（64 角点/3.0 mm，不再依赖单照片）；hand-eye 旋转跨度
15–167°，Tsai/Park/Daniilidis 三法互差 <2 mm ⇒ **T_base→board 一致性 1.0°/11.4 mm
（14 pose）**，base 原点在板面下 2.1 cm（= 垫+夹板厚度，物理自洽）；T_ee→zedm 同轮解出。
`ground_transforms.json` 为唯一真源。★ **「板动过 ~12 cm」是假警报，已撤回**：
是我用 720p 内参 ×(2208/1280) 去解 HD2K 照片 —— **ZED 各分辨率档不共 FOV**（HD2K 真实
fy=1057.6 vs 缩放值 906.6，差 17%，PnP 距离随焦距等比错）。换 native HD2K 内参后
photo 与 in-session 解相差 **36.5 mm**（方法差+深度偏置量级内）⇒ 板与相机都没动，
scene(scan5) 与新接地同一板位、可直接合成。**教训：ZED 内参必须用对应分辨率档的
原生标定，绝不能跨档缩放**（前文 ×(2208/1280) 的说法作废）。板仍建议固定。

★ **scan_run2 第一版的验尸（两条都写进采集清单）**：
① 视觉侧完美（14/14 视角检出、旋转跨度 30–152°、~2 mm Kabsch）——但 **14 份 state.json
的 joints/ee_pos/ee_quat 逐字节相同**：NUC 采集脚本记录的是同一份（陈旧的）机器人状态
⇒ hand-eye 无法进行。**下次采集前先验证相邻 pose 的 `joints` 确实变化**（一行 diff）。
② base 的姿态仍可从"纯平移"数据解出、位置不行 —— 若真机是 freedrive 平移 + 姿态不动，
旋转多样性必须来自**腕关节**主动转（±30–45°，≥2 轴）。备选：tip-touch（4+ 角点）。

代码/数据：`ground_scan_run2.py`（board model + hand-eye + 平移-only 退化路径）、
`calibrate_scan5_to_board.py`、`render_camera_pov.py`（真实相机 POV 四联图：
real|mujoco|splat|composite）、`handeye_ground.py`（含 calibrateHandEye 路径与 per-pose 一致性度量）、
`ground_transforms.json`（老相机位的解 + 失败的 base 解，spread 13°/337 mm = 不可用）、
`zed2i_new_extrinsic.json`（新相机位）。scan_run1 的 `state.json` 里 `ee_quat` 是 **xyzw**。

### §2d-x 独立复算（2026-09-17，另一条 pipeline，结论一致 ⇒ 接地可信）

用 **2D PnP（不用 depth）** 对同一 `scan_run2`（14 pose）重解，并用 splat 侧的桌面拟合
交叉验证。脚本 `scan_pipeline/ground_base_run2.py` → `ground_run2_solved.json` /
`cell_layout_base.json`。**两条独立链的数字互相印证**（左=§2d/§2e，右=本次）：

| 量 | §2d/§2e | 独立复算 | 差 |
|---|---:|---:|---:|
| T_base→board 一致性（14 pose） | 1.0° / 11.4 mm | **0.11° / 6.1 mm** | PnP 比 depth 紧 ~2× |
| base 法兰 vs 工作面 | −20.7 mm | **−21.4 mm** | 0.7 mm |
| 工作面尺寸 | 1.240 × 0.739 m | **1.260 × 0.737 m** | ≤20 mm |
| ChArUco 高于工作面 | +5.3 mm | **+2.3 mm** | 3 mm（均 = 夹板厚，物理自洽）|
| 桌面相对 base z 的倾角 | — | **0.99°** | — |
| zed2i 位姿一致性（14 pose 各自解） | — | **0.41° / 20.0 mm** | — |

要点：
- **ZED 内参陷阱与 §2d 的记载一致，但根因更具体**：不是「各档不共 FOV」，而是
  **`scan_run2/intrinsics.json` 存的就是原生 2208×1242 的标定**（fx 1057.6），而
  `ground_scan_run2.py` 里写死 `width/1280` 又乘了 1.725 ⇒ fx 1824（等于把 92.5° 的相机
  当成 62°）。剂量-响应实测：修正前后 depth-Kabsch 残差 18.4→**3.5** mm(外) /
  32.9→**1.6** mm(腕)，腕 PnP 重投影 9.5→**1.16** px，T_base→board 散布
  170→**6.1** mm。**⇒ 所有读 manifest 的地方都必须用 manifest 自己的 `resolution` 字段缩放。**
- **hand-eye 五法（Tsai/Park/Horaud/Andreff/Daniilidis）互差 ≤0.6 mm** ⇒ 不是单法伪解。
- scan_run2 的 state.json **确实逐 pose 变化**（14 个唯一值，ee_pos sd 125/98/66 mm，
  EE 姿态跨度 12–178°）⇒ §2d 那条「14 份 state 逐字节相同」的验尸属于**更早的一版采集**，
  当前 `/home/yifanr/scan_run2` 是好数据，不要据此再判 hand-eye 不可做。
- ⚠️ **唯一未闭合的绝对尺度**：全链条的米制都来自 `board_model.json`（ZED depth 融合）。
  自洽性极好（平面 0.44 mm rms、marker 边 30.88±0.52 mm、相邻中心 60.0±0.2 mm ⇒
  方格 42.43 mm = 60/√2），但**没有和实物尺寸比对过**。拿尺子量一次 ChArUco 方格
  （或查打印稿）即可把绝对尺度钉死；若真值是 40 mm，则全链条尺度偏大 6%。

## §2e 竞技场（arena）按实测重建（2026-09-16，owner 要求「先做忠实复现」）

**发现**：robosuite 出厂 arena 与真实 cell 差 **133 mm** —— sim 的 base 在桌面
**上方 112 mm**，真机的 base 法兰在工作面**下方 20.7 mm**（机器人拧在光学面包板上，
垫子/海报铺在其上）。带着这个误差，sim 与真机**无法同时**在 base 和工作面上对齐
（可达性也跟着错），所以这是保真度缺陷，不是训练选项。

**实测**（`measure_table2.py` → `cell_layout.json`；先 RANSAC 桌面平面，再在**桌面系**
里量，不在 board 系里量 —— 朴素的 z 直方图曾挑错 slab，给出「桌面在 ChArUco 上方 23 mm」
的自相矛盾结果）：
| 量 | 值 | 备注 |
|---|---:|---|
| board 系 z 与真实竖直方向夹角 | **0.36°** | 标定板确实平放 ⇒ board 系 z 可当竖直用 |
| ChArUco 纸面高于工作面 | **+5.3 mm** | 夹板厚度，物理自洽 |
| base 法兰相对工作面 | **−20.7 mm** | 垫子+海报厚度 |
| 工作面尺寸（垫子覆盖区） | **1.240 × 0.739 m** | 28k 平面内点；**不是**整块面包板 |
| base 在桌面系 | x −0.024 / y +0.490 / yaw −91.2° | base 在长边外侧，面向桌子 |

**落实**（`env/peg_insertion.py` 顶部 `MEASURED_*` 常量）：
`table_full_size=(0.739, 1.240, 0.03)`（长轴换到机器人 y 上）、
`table_offset=(0,0,0.912+0.0207)`（mount 把 base 固定在 0.912，故用**桌高**承载
base↔桌面关系；`base_xpos_offset` 的 z 是 0，改不了 base 高度）、
`set_base_xpos((-0.489,-0.034,0))`。yaw 91.2°→90° 取整，残差 1.2° ≈ 远角 13 mm。

⚠️ 后果：脚本专家从 10/10 掉到 **4/5（n=5，噪声内不可读，20 局在测）** —— 桌面相对 base
升高 13 cm 改变了可达几何与抓取姿态。**§5 的所有旧数字（10/10、172 步）属于旧 arena，
不可与新数字比较**；旧的 1000 条 demo 同样作废（已弃用，未训练）。

## §3 任务：peg-in-hole（方形）

- **间隙 = 天花板的分母**（findings §27/§39）：地板 ~1.3 mm/边（oracle 也过不去）；
  learned dynamics ≈ oracle + 5.6 mm 持久误差；Threading 的工作点是 3 mm/边 ⇒
  **规格 ~3 mm 半径间隙**。工业级 0.1–0.5 mm 在地板以下，必败且与方法无关。
- **方孔方钉**：孔 = 4 个 box primitive，钉 = 1 个 box ⇒ 碰撞精确、物理快（物理现在是
  吞吐大头）。圆孔要 mesh/凸分解，又慢又抖。
- **预抓取/刚接起步**：`obs_stack:1` ⇒ latent 是 Markov 的，夹爪迟滞不可表示（stage 3
  修不了）；去掉抓取相就去掉了唯一结构性修不了的失效模式。
- 间隙**双侧连续可调** ⇒ §27 的 ceiling law 可以在 sim 和真机上直接扫分母（机制主张可测）。

## §4 Pipeline 阶段 × 现状

| 阶段 | 内容 | 现状 |
|---|---|---|
| A 重建 | ZED 扫描（背景/机器人/物体分开）→ COLMAP → **3DGS**（gsplat 格式，非 2DGS）→ ChArUco+ZED 双源定标 → per-body ply → MJCF | **场景 splat 已成**（§2b）；**① scene→世界定标已解**（§2c：scale 0.1089 m/unit，IQR 2.2%）；机器人 splat 用 PolaRiS 的（⚠️ 只有 Robotiq —— Panda 手无 splat，见 §6）；残留：② 物体外观（STL 无纹理）③ scene.ply 只在 /scratch（repo 有备份）④ Panda 手 splat 待扫 |
| B 任务 | 方形 peg-in-hole env class + `resolve_bindings` 分支（`domain_shift.py:732` 对未知 task raise）+ 几何 success（strict 验证器纯几何可移植，阈值须重标）+ **外观 DR**（现在为零；相机外参 DR 必做——外置 ZED 每次装机会动） | **RL env 已通**（§5：`env/peg_insertion.py` = `PegInsertion_D0`，smoke 全绿 + 脚本专家 10/10）；resolve_bindings 分支已加；外观 DR 仍无 |
| C source 专家 | demo → BC prior → IBRL state-only [6:23]，`--snapshot_period 25000` 必传 | **链已提交**（§5b：collect 5465185 → BC 5465186 → IBRL 5465187，2026-09-15）；demo 来源 = 脚本专家 |
| D stage 1 | `generate_simdist_data.py` + gsplat 渲染 → replay（存 **/scratch**，/project 已 98%）→ 离线蒸馏 r18@224+dynens [~38 h，b3 分块] | 生成器/蒸馏齐；缺渲染接线 |
| E stage 2 | 真机 teleop demo → BC + `bc_anchor_coef=1`（pixel 上 +0.432，支配性杠杆） | 缺真机采集侧 |
| F stage 3 | 推理台式机上 MPPI 15 Hz，只训 dynamics 头；`encode(s')` 入 buffer 时缓存 | 缺整个真机部署侧（franka/polymetis 代码已归档不在本地） |

需要人：任务几何最终定稿、扫描 session、teleop demos、机器人时间。
预估 **~2–3 周到第一个 stage-1 run**（阻塞在 A/B 工程，不在算力）。

## §5 RL env（Phase B，2026-09-15 已通）

`env/peg_insertion.py` = `PegInsertion_D0`（robosuite EnvMeta 自动注册，wrapper 已 import；
STATE_KEYS/STATE_SHAPE(37)/GOOD_CAMERAS 已登记；`resolve_peg_insertion_bindings` 已接入
`domain_shift.py`，带 `hole_wall_geom_ids` = 未来的间隙旋钮）。几何 = §2b 实测值硬编码
（hole 37.9×26.1、板厚 41、peg 33.0×21.2×122.9 ⇒ x/y 均 2.41 mm/边）。

设计（2026-09-15 定稿；**peg 上台起步取代了 §3 原来的预抓取设计** —— owner 决定，任务
= 完整 grasp→transport→insert）：peg 直立在台面上（静态稳定，倾覆角 ~9.8°），采样
x∈±0.10 / y∈(0.10,0.20) / z_rot ±30°（长方形 peg 的 yaw 对抓取和入孔都有约束，先窄后宽）；
板 = **静态 fixture**（无 joint，真板是夹死的）、碰撞全 box。成功判据纯几何：tip 在孔柱内
且低于板面 15 mm。episode_length 300，σ=0.1。

踩过的坑（都在 smoke 里有回归项，`scripts/smoke_peg_insertion.py`，CPU 登录节点可跑）：
1. **手指行程轴 ≠ grip_site y**（Panda 差 90°）⇒ 必须从 model 的 `jnt_axis` 算
   （`env.finger_travel_axis()`），否则夹在 33 mm 面上且穿透 6 mm。
2. **夹爪的力要等 servo 目标走完**：action ±1 是斜坡（几步到十几步），夹到贴面≈0.2 N，
   servo 目标到全闭才有 ~10 N/指 ⇒ 脚本专家 close 相位的退出条件是 **ctrl≈0**，
   不是"手指不动了"。grip_depth=35 mm 减小力臂。
3. **姿态伺服的竖直目标必须取"最近的 ±z"**：从台面抓起的 peg 其 +z 朝上，硬编码
   "z 朝下"= 命令 180° 翻转 ⇒ 手腕以饱和速率滚转、peg/手一起翻到 90°（现象看起来
   像"peg 在指间打滑"，其实 handtilt≡pegtilt —— 先分清是谁在转再修）。
4. **转运要分层**：远处走高（板面 +30 mm），到孔上方 15 mm 内才降到 +10 mm ——
   +10 mm 平飞时几度的摆角就能让 peg 下角勾住板沿，把 peg 在指间撬转 90°。
   insert 仍是闩锁式（持续下压 + xy 收紧 ±0.2 + 卡口回退）；对齐门控回撤会在孔口形成极限环。
5. OSC 平移 action 的量纲是 `output_max=0.05 m`：P 增益要乘回去（30 ≈ 实际 1.5 增益）。
6. **物体模型 = 扫描 STL 本体（2026-09-15 owner 要求）**：真 board 是十几个腔的 shape-sorter，
   目标 pocket 离中心 62 mm —— 最初的对称单孔板完全不像。现役：视觉 = `board.stl`/`peg.stl`
   原样（`real2sim/scan_pipeline/peg_task/{board_scan,peg_scan}.xml`，MujocoXMLObject）；
   peg 碰撞 = 它自己的 mesh（凸的）；board 碰撞 = **4 块对同一 mesh 拟合的墙 box**。
   ⚠️ **纯 mesh 碰撞走不通，别再试凸分解**：CoACD threshold 0.02（81 块）仍有 5 块 hull
   凸进 pocket、把口封死（实测穿透 ≡ 插深）——凸分解的凹度地板 ~2–4 mm > 2.4 mm 间隙
   > §39 的 1.3 mm 天花板地板 ⇒ pocket 壁必须解析精确。probe 脚本在 §5 的坑单里救过两次：
   在 pocket 列上扫 (dx,dy,depth) 看 `contact.dist`，穿透≡深度 = 口被封。
   其余腔在碰撞里是实心（视觉照真）——任务只用目标 pocket，无妨。

脚本专家（grasp+insert 全链）10/10（~172 步/局）⇒ demo 采集器 `scripts/collect_peg_demos.py`
（collect_target_demos 同格式：`data/demo_i/...` + `env_cfg.json` sidecar；渲染默认开
agentview+eye_in_hand@96 以备 pixel）。cfg：`peg_insertion_state_source_mlp_bc.yaml` /
`peg_insertion_state_ibrl_source.yaml`（σ=0.1 全对齐，demo 与 RL 同 init 分布 —— 比 Square
的 0.02/0.1 错配更干净）。

### §5b 提交链（**已提交 2026-09-15**：collect 5465185 → peg_bc 5465186 → peg_ibrl 5465187）
```bash
J1=$(sbatch --parsable scripts/slurm/collect_peg_demos.slrm)
J2=$(sbatch --parsable --dependency=afterok:$J1 --account=aip-rhinehar --time=2:59:00 --job-name=peg_bc \
  --export=ALL,CFG=release/cfgs/mimicgen/peg_insertion_state_source_mlp_bc.yaml,SAVE=exps/mimicgen/peg_source_mlp_bc_n1000_seed1 \
  scripts/slurm/bc_prior.slrm)
sbatch --dependency=afterok:$J2 --account=aip-rhinehar --job-name=peg_ibrl \
  --export=ALL,CFG=release/cfgs/mimicgen/peg_insertion_state_ibrl_source.yaml,SAVE=exps/mimicgen/peg_ibrl_source_sig01_seed1,PRIOR=exps/mimicgen/peg_source_mlp_bc_n1000_seed1/model0.pt \
  scripts/slurm/ibrl_source.slrm
```
（历史：2026-09-15 ~14:00–15:30 klogin02 的 `sbatch` 因集群升级半途坏过 —— 23.11.11 的
sbatch + 25.05.9 的 spank_pyxis.so ⇒ "Incompatible plugin version"。管理员当天修复。
若再发生：`SLURM_CONF=<拷贝的 slurm.conf, PlugStackConfig 指向空文件> sbatch ...` 可绕，
我们不用 pyxis/容器。）
