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

## §2f 场景 splat 质量：四个缺陷 + 干净 A/B（2026-09-18，owner 要求「尽力改进 splat」）

**结果（同一 scan5 模型、同一 44 帧 robot-free held-out、同一 1920 长边）**：

| ply | gaussians | held-out PSNR | L1 | 说明 |
|---|---:|---:|---:|---|
| v1，**渲染器实际读到的样子** | 154k | 28.56 | 0.0277 | 迄今所有渲染都是这一行 |
| v1，SH 布局修正后 | 154k | 31.24 | 0.0179 | 布局 bug 值 **+2.68 dB** |
| v2 干净版 + 剪枝 | 953k | **34.24** | **0.0131** | 训练器值 **+3.00 dB** |

⇒ **总计 +5.7 dB**（28.56→34.24），拆成「布局 bug 2.7」+「训练器 3.0」。
30k 曲线单调上升（29.50→34.06），1.1M gaussians 未见过拟合。
脚本：`scan_pipeline/{train_3dgs_v2,eval_splat,prune_unseen,damp_sh,render_zed_pov}.py`。

**四个缺陷**（都在 `train_3dgs.py` v1 里）：
1. **densify 只 clone 不 split，且每 100 步重建 Adam**（动量全丢）。split 才是把大团
   变清晰的操作 ⇒ scan5 停在 154k，而 800k 上限从未接近。v2 用 gsplat `DefaultStrategy`。
   ⚠️ `initialize_state(scene_scale=)` 必须传（阈值是它的比例；默认 1.0 在米制场景下
   会把 >10 cm 的墙/桌 gaussian 全剪掉）。
2. **means 学习率没按场景尺度缩放**（标准 3DGS 是 `1.6e-4×extent`）⇒ 固定 1.6e-4 在
   extent 7.8 的场景里小 ~10×，gaussian 基本没离开 SfM 初值。
3. ★ **SH 布局 bug（静默，影响此前每一个 ply）**：v1 按 **coeff-major** 写 `f_rest`，
   而 `util_gau.load_ply`（= INRIA 约定 = 所有标准 viewer）按 **channel-major** 读 ⇒
   degree≥1 全部被置换。**实测代价 2.68 dB**。v2 写 channel-major。
4. 只跑 7k 步（标准 30k）。
另加：背景改为**实测墙色** `#d4d1cc`（`wall_color.json`，36 个亮-平样本），训练时也用它
⇒ 未观测区域渲染成墙色而非黑洞，且优化器不必去「涂掉」黑画布。

**两个反直觉的实测结论（都差点写成错的）**：
- ★ **per-image 曝光补偿在单条 AE 锁定扫描上是有害的：−4.0 dB**（30.41→34.24 是关掉它）。
  AE 锁了就没什么可补的，而 352×6 个自由参数会吸走真实信号，ply 本身反而更差。
  **只在多机/多次扫描合并时才开，且必须实测。**
- ★ **held-out PSNR 是「插值」指标，不是我们的部署工况。** held-out 取自同一条连续扫描
  （每 8 帧），是训练帧的近邻；而 stage-1 要从 **sim 相机位**渲染，那是**外插**。
  外插的真实表现要看「真实相机位复现」（`render_zed_pov.py`，用 ChArUco PnP 位姿，
  与真实照片同机位并排）。
- 可见性剪枝：352 个训练视角里 **有 140,747 个 gaussian 一个视角都没渲染到**（12.7%），
  剪掉后 held-out **完全不变**（34.241 vs 34.06）⇒ 纯垃圾。但**剪不掉新视角里的黑块**：
  那些 gaussian 在训练视角里是可见的，只是方向锥很窄，SH 在锥外无约束（§2f-b 在测
  「窄支撑就退化成漫反射」）。

**机器人怎么去掉（owner 要求）**：不做 mask、不做 3D 裁剪。做法 = **SfM 保留机器人帧**
（它们是连通组织），**3DGS 训练时按源帧号区间排除**（`--exclude_idx_ranges`；scan5 =
`0-1044,2592-3672`，即机器人 0-14.5% + 跑到别的房间的 36-51%）⇒ 没有任何视角观测过机器人
⇒ 不会长出机器人 gaussian，它原来的位置自然退化成墙色。
⚠️ 排除帧必须**同时**从 held-out 里去掉，否则是在拿「含机器人的照片」惩罚一个刻意不含
机器人的 splat。

**三条扫描合并（robot-free）**：scan5 + aux_scan1 + aux_scan2 → 800 帧联合 SfM。
⚠️ 必须 `--ImageReader.single_camera_per_folder 1`（两台手机不同 FOV；旧 `sfm.slrm` 写死
`single_camera 1`）。mapper 会**碎成 50 个模型**（最大 355/800）—— 区间排除把每条 sweep
切断了 + aux 有 ~68% 帧几乎无纹理。**不要重跑 SfM**：匹配已缓存在 database.db 里，
`colmap image_registrator`（2 轮 + BA + retriangulate）**24 分钟把 355 补到 800/800**
（183,750 点、1.36 px）。board 定标 0.90 mm 中位残差（比 scan5 的 0.98 更好）。

**踩过的坑（都花过一次提交）**：
① `train_3dgs.py` 的 COLMAP 相机参数表把 OPENCV(model 4) 写成 4 个参数（实为 8）⇒
   读完第一个相机后文件指针错位。**单相机模型完全无症状**（循环结束，且前 4 个参数就是
   fx,fy,cx,cy），但 3 相机的合并模型直接 `KeyError`。已修正整张表。
② `util_gau.load_ply` 返回的是**激活后**的值（`exp(scale)`、`sigmoid(opacity)`、归一化四元数），
   自写 reader 返回**原始** ply 值 ⇒ 两者拼接会让场景 gaussian 得到负的线性 scale，
   **整个场景渲染为空**。
③ gsplat 1.5.3：`backgrounds` 形状是 `image_dims+(channels,)`；未批处理的 means (N,3) ⇒
   `image_dims=()` ⇒ 必须是 **(3,)**（(1,3) 会 assert）。且 `DefaultStrategy` 必须配
   `packed=True`（非 packed 路径下单相机的 means2d 被压成 (N,2)，strategy 会 IndexError）。
④ ZED Explorer 静态图是**左右并排立体图**（HD2K = 4416×1242）⇒ 不切一半的话宽度翻倍、
   主点落在基线中点，渲染全黑。
⑤ sim 的 `frontview` 相机在真实 cell 里**位于墙内**（离桌 1.5–2 m，而真实 cell 是个墙角）
   ⇒ 渲出来是一团白色条纹。**stage-1 要用的相机位必须在真实房间里物理存在**。

### §2f-b 插值 vs 外插：**13 dB 的缺口，且主要是几何**（2026-09-18，决定下次怎么扫）

「同一条扫描里每 8 帧留出」是**插值**；stage-1 要从 **sim 相机位**渲染，是**外插**。
真外插测试 = 用 scan5-only 的 splat 去打 **aux 两条扫描的帧**（不同轨迹 + 不同手机 +
位姿是精确注册的，没有 PnP 误差；两边都在同一 board 系，`--only_prefix a1_/a2_`）：

| splat | 插值(scan5 held-out) | 外插→aux1 | 外插→aux2 |
|---|---:|---:|---:|
| v2 clean+pruned | **34.24** | **20.55** | 18.46 |
| v1（SH 布局修正） | 31.24 | 19.16 | 16.28 |
| v2 + SH 全退化成漫反射 | 28.38 | 20.46 | 18.70 |

- ★ **v2 在外插上也赢**（+1.4 / +2.2 dB）⇒ 训练器的收益是真泛化，不是记住训练帧。
- ★ **SH 退化在外插上是中性的（∓0.2），在插值上代价 5.9 dB ⇒ 判死，不要做。**
  连带推翻我自己的假设：新视角的黑块**不是** SH 外插失控，而是**几何**（gaussian 长错
  位置）。⇒ 要清掉它只能靠覆盖或按区域裁剪，不能靠外观项。
  （`damp_sh.py --mode narrow` 的阈值 R≥0.9 命中 **99.9%** 的 gaussian —— 手持扫描里
  每个面片本来就只被一小段方向锥看到，「窄支撑」是常态不是病态。）
- ★ **缺口只有 1.5–2 dB 是光度差**：加逐图 per-channel 增益/偏置后 aux1 20.55→22.10、
  aux2 18.46→20.50；而同一修正用在**同机同扫描**的 held-out 上只值 +0.28（34.24→34.52）
  ⇒ 修正确实只在吃跨设备的 AE/白平衡差，不是白送自由度。**剩下 ~12–14 dB 是几何/细节**
  （近距特写把细节放大到超过采集分辨率、没被观测到的面）。
- ⇒ **下次扫描的规格**：不要再只做一条大范围 sweep，要**按我们真正要用的 sim 相机位去采**
  （agentview 类的距离/俯角 + 腕相机类的近距），围着工作空间做轨道式覆盖。
  **stage-1 的实际画质预期是 ~20–22 dB，不是 34。**
- ⚠️ POV(真实照片) 的 `mean|diff|` **不能用来排序 splat**：相差 6 dB 的两个 splat 在这个
  指标上只差 1（57.8 vs 56.6），因为它被 1.81 px 的 PnP 位姿误差主导。只能看图，别看数。

### §2f-c 三条扫描合并：**受控后仍然更差，别用**（2026-09-18）

| splat | held-out | POV mean\|diff\| | 细节比 |
|---|---:|---:|---:|
| scan5-only v2 clean+pruned | **34.24** | **57.8** | 1.34 |
| merged 800 帧，曝光补偿 ON | 24.95 | 67.2 | 1.07 |
| merged 800 帧，曝光补偿 OFF（对照） | 26.69 | 62.9 | 0.85 |

关掉曝光补偿确实帮了 merged（+1.74 dB，比单扫描上的 +4.0 小 —— 合理，这里确实存在
真实的跨设备曝光差），但**仍然差 scan5-only 7.5 dB** ⇒ 「合并更差」不是曝光补偿造成的。
剩下的嫌疑：**位姿精度**（merged SfM 重投影 1.36 px vs scan5 的 1.01 px；细节比 0.85 < 1
= 比照片更糊，正是位姿噪声的样子）+ aux 帧本身难（特写、~68% 帧近乎无纹理）。
⇒ **生产用 scan5-only**（`real2sim/splats/scene_scan5_norobot_v2.ply`，952k gaussians）。
aux 想用的话，路线是**固定 scan5 位姿、只精修 aux 位姿与逐相机内参**，不要整体 BA。

⑥ 又一个坑：`sbatch` 默认 `--export=ALL` ⇒ 若提交前的 shell source 过 `set_env_drac.sh`，
`PYTHONPATH` 会把 **python3.10 的 Lmod opencv** 带进 gsbench 的 **python3.11** venv，
报 `missing configuration file: config-3.11.py`。gsbench 的 slrm 里都要 `unset PYTHONPATH`。

### §2e-b reset 姿态：arena 修好之后它就坏了（2026-09-18/19）

**症状链**：脚本专家 10/10（旧 arena）→ **4/5**（§2e 修完 arena 之后）→ **20/20**（改 reset 姿态之后）。
中间那次下降**不是 arena 修错了**，是 arena 修对了之后 reset 姿态变得不合法。

**根因**：`env/peg_insertion.py` 从来没设过 `init_qpos`，一直吃 robosuite 的 Panda 默认
`[0, π/16, 0, −π/2−π/3, 0, π−0.2, π/4]`。在**实测 arena**（桌面相对 base 抬高 13 cm）下：
grip site 只在桌面上方 **78 mm**，而立在桌上的 peg 顶端在 **124 mm** ⇒
**夹爪开局在它要抓的物体顶端下面 45 mm**。旧 arena 里桌面低 13 cm，所以同一姿态没问题。

**改法**：`FRANKA_READY_QPOS = [0, −π/4, 0, −3π/4, 0, π/2, π/4]`（Franka 官方 ready，
**也是真机 reset 的姿态** ⇒ sim/real 开局一致）。必须写在 `_load_model()` 里：
`Robot.load_model()` 只在 `init_qpos is None` 时从 robot model 取值，而 `hard_reset=True`
每次 reset 都重跑 `_load_model`，所以覆盖要放在那里才会每次生效。
`hand_init_perturb_joint_sigma` 的噪声由 `Robot.reset()` 加在它上面。

**实测（12 次 reset，σ=0.1）**：grip 高于桌面 **495±40 mm**（最低 415）= 高于 peg 顶端
**+371 mm**；approach 轴离竖直 11±5.6°（标称 0°）；robot/table/board 接触 **0**。
步数 172→191（开局高了 40 cm，下降路程变长），仍远小于 300 的 episode 上限。

**高度全链核对**（`cell_layout_base.json` + sim 实测）：桌面相对 base sim **+20.7 mm**
vs 实测 **+21.4 mm**（差 0.7 mm）；splat 侧独立交叉验证 **+4.6 mm**。peg 静止在桌面
（差 1.0 mm = sampler 的 z_offset）；board 板面 +41 mm = 板厚。
⚠️ 真桌**有 0.99° 倾角**，sim 是水平的（长轴上差 ~13 mm），目前未建模。

**相机可见性（10 次 reset，物体包围盒 8 角点）**：
| 相机 | peg | board |
|---|---|---|
| agentview | 100% | **只有 75% 角点在画面内，0% 的 reset 完整可见** |
| frontview | 100% | 100% |
| 外置 ZED 位姿 | 100% | 100% |
⇒ **agentview 会把 board 切掉**；真正要用的外置 ZED 位姿两个物体都完整可见。
pixel 阶段的相机应该直接放在 ZED 位姿上（`zed2i_extrinsic_correctK.json`），
sim 坐标 `[-0.354, 0.480, 1.691]`（桌面上方 758 mm）。

⚠️ **旧 demo 必须作废**：2026-09-16 采的 1000 条来自旧 arena + 旧 reset 姿态，
已移到 `release/data/peg_insertion/demos_n1000_OLDARENA_donotuse/`。
新链（2026-09-19）：collect 5550284 → BC 5550285 → IBRL 5550286，数据在 `demos_n1000_v2/`。

### §2d-y 腕相机标定在换夹爪后**仍然有效**（2026-09-20，owner 已换成 Panda 手）

**结论：`T_ee_zedm` 不用重标。** 因为 polymetis 报的是**法兰（flange / link8）**，不是 TCP ——
换夹爪不动法兰系。实测（用 scan_run2 的 14 组 `joints` 跑 FK，与同一份 `ee_pos` 比）：

| sim 里的候选系 | 与 `ee_pos` 的差 |
|---|---:|
| `robot0_right_hand`（法兰） | **0.5 mm ± 0.0** |
| `robot0_link7` | 107.0 mm（= link7→法兰的 d=0.107，自洽）|
| `grip_site`（TCP） | 96.5 mm |

⚠️ 唯一会失效的情形：**相机支架本身长在夹爪上**（而不是臂/法兰侧）—— 那换夹爪会物理移动相机。
换完手请目视确认支架没动；动了就重采 14 pose（流程已验证可收敛）。

**两条独立解互证**（都在仓库里，可直接引用）：
| | `T_ee_zedm` 平移 | 来源 |
|---|---|---|
| `scan_pipeline/ground_run2_solved.json` | (33.8, −75.4, 15.5) mm | 2D PnP + TSAI，`T_base_board` 散布 **6.1 mm / 0.11°** |
| `scan_pipeline/ground_transforms.json` | (35.5, −75.8, 14.3) mm | depth-Kabsch + PARK，散布 11.4 mm / 1.04° |
| **两者之差** | **2.1 mm / 0.22°** | ⇒ 可信 |

⚠️ **本节只核对了「位置」，而位置看不见绕自身原点的旋转 —— 见 §2d-z，那里差了 45°。**

### §2d-z 腕相机的独立复核：一个 45° 的坑 + 两个互证的量（2026-09-20，`viz/wrist_handeye/`）

1. **★★ `robot0_right_hand` ≠ polymetis 的法兰系，差一个绕 z 的 `45.03°`**（14 pose 上
   **恒定到 0.000°**）。`robot0_link7` 则是旋转 0.00° / 平移 107.0 mm（link7→法兰的 d），
   互证。§2d-y 的 0.5 mm 是**位置**核对，对这种旋转结构性失明。
   ⇒ **sim 里摆腕相机必须 `T_sw_hand @ Rz(45.03°) @ T_ee_zedm`**；不修则夹爪占 23.8%
   画面（真值 4.6%）。⚠️ 现有代码没踩到它，只因 `render_camera_pov.py` 压根不用
   `T_ee_zedm` —— **stage D 接腕相机的人是第一个会踩的**。
2. **hand-eye 精度：LOO 留一 2.2 mm 中位 / 3.7 mm 最大 / 0.34°**（三种 handeye 解法一致
   到 0.1 mm ⇒ 非过拟合）。全程在真实域，不经 robosuite ⇒ 不受第 1 条影响。
3. **夹爪剪影是第二个独立观测**（夹爪与相机同为法兰刚体 ⇒ 剪影与手臂关节无关，是
   `T_ee_zedm` 的纯检验）。全开下拟合修正 `(-2.10, +0.33, -5.34) mm`，逐轴 SD
   `(0.12, 2.15, 2.01)` ⇒ **只有 x 定得住**。tip 对 tip Δy=0 px、IoU 0.440→0.752。
   与第 2 条同量级 ⇒ 互证；但 **别用它覆盖 `T_ee_zedm`**（y/z 不确定度 = 修正量）。

⚠️ **`viz/pov_shots/` 那组腕相机对比图不能当 hand-eye 检验**：`T_world_zedm_per_pose`
（`ground_scan_run2.py:255`，原注释 "no hand-eye needed"）是**逐 pose 对着板子解**出来的，
再把板子渲染回去 ⇒ 循环。它检验 board model + 内参。
⚠️ **splat 里没有夹爪的原因**：`splat_frame_calib.json` 的 7 个夹爪 token 全是 Robotiq
2F-85，而 `PegInsertion_D0` 用默认 `PandaGripper` ⇒ 自动 skip 全丢；且该脚本让手臂停在
`e.reset()` 构型而非 scan pose 的 joints。**scan_run2 的真机装的是 Panda 手**（外置 ZED
可见，指尖投影吻合），不是 Robotiq。

★ **标定用的真实 ZED 图像已入库**：`real2sim/zed_capture/`（14 pose × 腕/外 左目 PNG +
`state.json` + 原生内参 + Explorer 静态图，**无损、与原件逐字节相同**）。
⚠️ JPEG 不行：q95 会把外置相机的 PnP 解移动 **2.3 mm 均值 / 4.3 mm 最大**，与整条链的
6.1 mm 同量级 ⇒ 有损副本复现不出已发表的数字。深度/置信度 `.npy`（586 MB）**不入库**，
因为标定已全改成纯 2D PnP（depth 拟合只到 18/33 mm，是链条里最弱的一环）。
⚠️ `zed2i_left.png` **每张都有机器人**（14 个不同姿态）⇒ **只能当标定输入，绝不能喂 3DGS**。

### §2d-w PHOTOREAL stage-1：物体/场景 splat 进生成管线（2026-09-22）

**做了什么**：stage-1 的相机观测不再是纯 MuJoCo，而是 **splat 场景 + splat board +
MuJoCo 手臂/peg** 的合成（`real2sim/splat_obs.py`，挂在
`PixelRobosuite._extract_images`；`splat_obs=None` 时逐字节等于旧路径）。

| 部件 | 来源 | 理由 |
|---|---|---|
| 场景（poster 桌面、墙） | **splat** v2 | 占画面大头，材质值模仿不了 |
| board | **splat**（aspect log-err 0.053，ICP **1.58 mm**，25400 gaussians） | 模具凹槽/阴影/印刷细节 |
| peg | **MuJoCo** | splat 判 POOR（aspect 0.316）；它就是个白盒子，材质本来就对（渲染 209 vs 视频 212）|
| 手臂 | **MuJoCo** | PolaRiS 臂 splat 只有 Robotiq，Panda 手根本没扫（§4 A 残留④）|

**物体 splat 配准**（`real2sim/scan_pipeline/register_object_splats.py`）。尺度来自物体
**自己的扫描 STL**（两段 MOV 都没拍到 ChArUco 板）。四个坑都已写进脚本，每个都是踩出来的：
① ICP 要**采样表面**不能用顶点（box 只有 8 个唯一顶点）⇒ 13 mm→3.6 mm；
② verdict **不能只看 ICP rms**（trimmed ICP 把胖点云埋进薄 mesh，给 1.06 mm）；
③ 几何分不开「平板」和「一片平桌面」⇒ 必须加**颜色**门（board 蓝度 p90 +0.435 vs 桌面棕色）；
④ **去平面是逐物体的**：peg 站着要去，board 本身就是一块平板，去了就把它删光（59085→865）。

**门禁**（`scripts/check_splat_obs.py`）：竖直翻转的数据集照样能训练、但绝对迁移不了，
所以用数字判而不是看图。两台相机都 PASS：`|wrapper−expected|` 0.49/0.46 vs
`|flipped|` 66.9/76.4 vs `|mujoco-only|` 100.0/100.9。

**相机**（§2d-z 的延续）：`zedview` fovy **60.763°**、`robot0_zedm_wrist` fovy **45.028°**，
都是**以主点为中心的正方形 crop** 的竖直 FOV（zed2i 1240 px @(1083.7,621.6)；
zedm 1223 px @(1144.6,611.6)）。sim 直接渲 224×224 正方形，所以策略看到的就是这个 crop；
**真机管线必须同样裁**，否则把主点偏移（Mini 的 cx 偏 40.6 px）又塞回每一帧。
⚠️ crop 丢掉水平 ±15.9°（44% 宽度）。查过了：丢掉的边缘**没有**机器人残影，只是没扫到的
糊区；但 **ChArUco 标定板在 crop 内**（scan5 时留在桌上），会出现在每一帧训练图里 ——
真机若不摆这块板，就是一处恒定的外观差，而 encoder 之后永远冻结。**未决**。

**成本与配方**（都是量出来的，别沿用 state 臂）：
`37,049 B/transition`（×1.66）、`~130 transitions/s`（×2.9）⇒ buffer 3.3M = 126 GB；
生成 8 shard × 3800 traj ≈ 1.2 h 每片。噪声配方 2×2 见铁律 #32。
board 抖动容忍度（source expert, n=200/格）：fixed **0.925** · ±5 mm/±2° 0.955 ·
**±10 mm/±5° 0.925（免费）** · ±20 mm/±10° 0.815 ⇒ 随机化臂用 ±10 mm/±5°。
随机化臂的代价：mean length 153.7→160.6，success 66.3%→~64%。

**可移植性**：`deps/GaussianRenderer`（MIT，pin 577efdb）已加为 submodule，
`real2sim/gs_mujoco.py` 改成仓库相对路径，board/scene splat 走 **git-lfs**
（scene ply 225 MB，超 GitHub 100 MB 硬限）。换机器只要 clone + submodule + venv。

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

### §5a ★★★ 目标孔之前一直是错的（2026-09-23 发现并修复）
**真实任务孔 = 板角上的长方形通孔**（从机器人看的板右上角）。`board_scan.xml` 里的碰撞 pocket
（4 块隐形 box 墙之间唯一的空柱）却坐在**椭圆孔**上，尺寸 37.86×26.07 = 椭圆的包围盒
（STL 切片：椭圆 37.4×25.6 @ 中心 (−4.5,−3.8)，旧 pocket 还偏了 4.5/3.8 mm）。
⇒ 长方形 peg 在 sim 里"插进椭圆"：物理只认那个方柱，椭圆只是画上去的视觉 mesh。
⇒ **2026-09-23 之前所有 peg run（IBRL 源专家 v2、三臂数据、stage-1）训练的都是错孔。**
发现方式：真 GELLO demo 回放（`scripts/compare_real_sim_demos.py`）—— 操作员松手点离 sim 目标
74–84 mm；`viz/real_vs_sim/{board_collision_vs_visual,hole_check_real_release}.png`。
- 修复：body 原点 = 角孔中心（mesh 系 (54.86, 58.36) mm），墙按 mesh 的真实外廓重算
  （旧墙相对视觉 mesh 还偏了 ~4 mm），`HOLE_HALF` = 35.56×23.77 mm（切片 35.1×23.3，贯穿
  41 mm 全深，+0.46 mm 切片偏差按椭圆标定）。`board_xy` 平移同一量 ⇒ **物理板位置不变**；
  board splat 的 `obj_calib.t` 同步 −(53.8, −3.8) mm。
- ⚠️ **间隙从 2.4 降到 ~1.3 mm/边 = 正好在 findings §39 的 1.3 mm 地板上**（手机扫描 STL，
  须用卡尺复核真孔与真 peg）。脚本专家仍 20/20（avg 190 步）。
- 重训链（v3，`*_v3.yaml`，v2 cfg 保留以复现旧臂）：`demos_n1000_v3` → `peg_source_mlp_bc_n1000_v3`
  → `peg_ibrl_source_v3`（2026-09-23 提交 5612729 → 5612730 → 5612731，~8 h）。
- 教训：**"collision 从 STL 来"是错觉** —— 视觉 mesh `contype=0`，碰撞是手摆的 primitive。
  改任务几何后必须**把碰撞体画出来叠在视觉上看一眼**，并在真 demo 上核对目标位置。

### §5a2 ★★★ 成功判据可被 hack + 墙是软的（2026-09-23 发现并修复）
MimicGen 数据（`scripts/gen_peg_mimicgen.py`，10 条脚本种子 → 1000，board ±10 mm/±5°）训出的
IBRL 报 **0.86**（独立 n=100），但几乎全是**斜着把 peg 楔进墙里**：成功局墙穿透中位 **8.2 mm**
（max 42）、倾角中位 11.5°、**62 步**成功（专家 191 步）；穿透 ≤2 mm 的合规成功只有 **0.14**。
测量工具 `scripts/verify_peg_ibrl.py`（`--expert 1` = 同 env/同 seed 的脚本专家基线）。
- 根因 ①：`_check_success` 只要 tip 进孔柱且比板面低 **15 mm**（板厚 41）⇒ 刚过口就算。
- 根因 ②：墙用 MuJoCo 默认软接触（solref 0.02 / solimp 0.9–0.95）⇒ 持续推力下连**专家**都能
  压进墙 21–40 mm（它之前只测出 0.53 mm 是因为 15 mm 就停了）。MimicGen 开环重放也有 45% 是挤进去的。
- 修法（`env/peg_insertion.py` 新默认，旧值可传参复现）：墙 `solref (0.004,1)`（= 2×timestep，
  最硬的稳定值）+ `solimp (0.99,0.999,0.001)`、`solmix 1000`（只作用在 peg-墙对，抓取不变）；
  `insert_depth 0.030`；`success_max_tilt 5°`。
- 验证：专家 **0.95**（n=60，穿透 ≤0.43 mm，倾角 ≤0.7°，201 步）；同一个 hack policy **0.84→0.04**；
  smoke `[5]` 加了"刚过口"与"倾斜 10°"两个必须判负的用例。
- 连带：MimicGen 生成率 ~57–70% → **17%**（插入噪声 0.05）/ **35%**（插入噪声 0，n=200 各，
  z≈4.2 ⇒ 0.05≈2.5 mm/步 > 1.26 mm 间隙）；新 demo 穿透 ≤0.18 mm、倾角 ≤3.7°、插深中位 39.6 mm。
- ⚠️ **2026-09-23 之前所有 peg 分数（含 v3 脚本链 IBRL 0.95）都是在旧判据+软墙下测的**，
  引用前先用 `verify_peg_ibrl.py` 复查。重跑链：`TAG=_hard bash scripts/pipeline_peg_mimicgen.sh`。

### §5a3 ★★★ board splat 一直画偏了 62–64 mm（2026-09-24 发现并修复）
photoreal 画面里 peg"插错孔"，但物理上是对的：`obj_calib.json` 的 board `t` 放错了。
`board_obj.ply` 是对着 STL 在 **mesh 系**里配准的（`register_object_splats.py` 对 STL 做 ICP），
所以 `t` 必须等于 `board_scan.xml` 里视觉 mesh geom 的 `pos`（MuJoCo 对同一个 mesh 用的变换）。
- 9/23 之前 `t=0`，而 mesh pos=(−1.06, −62.16) mm ⇒ **y 偏 62 mm**；§5a 的"同步 −(53.8,−3.8)"
  在这个错的基础上又平移一次，偏差仍是 64 mm。
- 测量：只渲染 board splat、对 MuJoCo 视觉 mesh 的剪影做平移网格搜索（`scripts/fit_board_splat_offset.py`），
  最优 = (−51.8, −60.2) mm，与 mesh pos (−54.86, −58.36) 差 <3 mm 且 IoU 相同 ⇒ 用 mesh pos。
  修后 IoU 0.45–0.52 → **0.91–0.97**（`scripts/diag_splat_board_align.py`，jitter 开关一致）。
- 防回归：`SplatCompositor` 初始化时 `_check_board_calib` 断言 `t` 与编译后模型里的视觉 mesh pos
  差 ≤1 mm（旧值会报 61.5 mm 错）。
- ⚠️ **9/24 之前所有 photoreal peg 数据（`peg_d0_src_splat224` 等）与训在其上的 stage-1 都看的是偏了的 board**，须重渲。

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
