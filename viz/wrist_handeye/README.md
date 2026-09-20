# 腕相机 hand-eye 的独立复核（2026-09-20）

起因：有人问「splat 渲染里为什么看不到夹爪」。追下去发现**已发表的那张腕相机对比图
根本不测 hand-eye**，以及 sim 里放腕相机会踩一个 45° 的坑。

## 结论（三条，按可信度排）

1. **★ `robot0_right_hand` 不是 polymetis 的法兰系 —— 差一个绕 z 的 45.03°。**
   14 个 pose 上**恒定到 0.000°**（`frame_check.py`）。§2d-y 只核对了**位置**（0.5 mm），
   而位置对「绕自身原点的旋转」完全不敏感 ⇒ 这个偏移当时看不见。
   `robot0_link7` 的旋转偏移是 0.00°、平移 107.0 mm（= link7→法兰的 d），互证。
   ⇒ **在 sim 里摆腕相机必须 `T_sw_hand @ Rz(45.03°) @ T_ee_zedm`。**
   不修的话相机绕法兰转 45°，夹爪会占掉 **23.8%** 画面（实际 4.6%）。
   ⚠️ 这个坑**不在已提交的代码里** —— `render_camera_pov.py` 压根不用 `T_ee_zedm`
   （见下），所以第一个踩到它的会是 stage D 接腕相机的人。

2. **hand-eye 本身：LOO 留一验证 2.2 mm 中位 / 3.7 mm 最大 / 0.34°。**
   `handeye_check.py`（拟合残差 1.9 mm）+ `handeye_loo.py`（13 pose 重解预测第 14 个，
   TSAI/PARK/DANIILIDIS 三法一致到 0.1 mm ⇒ 没有过拟合）。
   全程在**真实域**内（board 系 + polymetis ee + 图像 PnP），**不经过 robosuite**，
   所以第 1 条的 45° 不影响这两个数。

3. **夹爪剪影是第二个独立观测，量级吻合但精度不够改标定。**
   夹爪与相机同为法兰刚体 ⇒ 剪影**与手臂关节无关**，是 `T_ee_zedm` 的纯检验。
   夹爪全开（owner 确认）下拟合平移修正：`(-2.10, +0.33, -5.34) mm`，
   跨 5 个分割阈值的逐轴 SD `(0.12, 2.15, 2.01) mm` ⇒ **只有 x 被定住**（−2.10 ± 0.12），
   y/z 仍 ±2 mm。修正后 tip 对 tip **Δy = 0 px**、质心 (+4.6, −0.8) px、IoU 0.440 → 0.752。
   ⇒ 与第 2 条同量级（互证），但**不要用它覆盖 `T_ee_zedm`**：y/z 的不确定度和修正量同级。
   ⚠️ 瓶颈是 mask：IoU 随阈值放松单调升（0.69→0.89），因为真值 mask 在长大去贴 sim。
   要再进一步就别用二值 mask，改拟合边缘/梯度图。

## 被推翻的东西（别再引用）

- **`viz/pov_shots/` 那组腕相机对比图不是 hand-eye 检验。** `T_world_zedm_per_pose`
  在 `ground_scan_run2.py:255` 是**逐 pose 对着板子解出来的**（原注释就写着
  "no hand-eye needed"），再从这个位姿把板子渲染回去 ⇒ **循环论证**。
  它检验的是 board model + 内参，不是 `T_ee_zedm`。
- **「splat 里没有夹爪」的原因**：`splat_frame_calib.json` 的 7 个夹爪 token 全是
  **Robotiq 2F-85**，而 `PegInsertion_D0` 用默认 `PandaGripper` ⇒ `body_name2id` 全部
  失败 ⇒ `render_camera_pov.py:141` 的自动 skip 把夹爪整个丢掉。另外该脚本
  `e.reset()` 后直接 `sim.data.qpos[:] = e.sim.data.qpos`，手臂停在 **reset 构型**
  而不是 scan pose 的 joints。
- ⚠️ **`superseded_*`**：sim 端用了 Robotiq85（错，scan_run2 是 Panda 手），或是上面那组
  循环检验图。**`DISPUTED_*`**：断言 scan_run2 机器人上没有夹爪 —— **是错的**，
  Panda 手就在那儿（`11_PROOF_panda_hand_IS_present.png`，把法兰 +z 0.1034 m 的
  指尖投影进外置 ZED，正落在两根深色手指上）。两类都已 gitignore。

## 复现

```bash
source set_env_drac.sh            # CPU/osmesa 即可，不需要 GPU 或 SLURM
python viz/wrist_handeye/frame_check.py      # 45.03° 那条
python viz/wrist_handeye/handeye_loo.py      # LOO 2.2 mm
python viz/wrist_handeye/final2.py           # 剪影拟合 + 两张 side-by-side
```

| 文件 | 内容 |
|---|---|
| `04_fkchain_board_reprojection.png` | 绿=检测角点 vs 红=FK 链预测（2.6 / 14.7 / 26.0 px；板子在 1.2–1.4 m 处，1 px≈0.86 mm）|
| `09_real_finger_mask_...png` | 真值手指 mask：相机与夹爪同为法兰刚体 ⇒ 14 帧里**只有它不动** ⇒ 低时序方差∧暗，无需人工阈值 |
| `11_PROOF_panda_hand_IS_present.png` | 外置 ZED 视角，指尖投影落在真实手指上 |
| `12/13_SIDEBYSIDE_open_*.png` | real / mujoco / overlay（黄=真值 红=拟合 蓝=已发表）|
