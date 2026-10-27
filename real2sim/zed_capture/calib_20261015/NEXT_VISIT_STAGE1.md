# 上机清单：exec_obs stage-1 模型第一次上真机（2026-10-27 写）

目的：**诊断，不是验收。** 拿 sim 里蒸馏出来的 pixel 模型零样本跑真机，搞清楚三件事：
观测对不对得上、差在视觉还是动力学、哪一阶段失败。真正部署走 stage 2/3（真机 demo + on-policy 数据），
**stage 1 不重训**，除非这次发现观测契约本身要改（相机位姿/内参、控制频率、看得出 DR 没覆盖的视觉差异）。

## 模型

| | |
|---|---|
| ckpt | `exps/sim2real/threading_robotiq_pixel_stage1_seed1/model0.pt` + 同目录 `cfg.yaml`（两个都要拷） |
| 观测 | 两路相机 224（`agentview` + `robot0_eye_in_hand`）+ prop 36 维 = FK prop 9 + exec_obs 27（参考臂 vs 实测臂 + 上一步动作） |
| 频率 | 10 Hz，episode 400 步（40 s） |
| 专家 | IBRL exec_obs seed1 @400k，独立 300 局 0.917（`exps/sim2real/threading_robotiq_execobs_expert/`） |

sim 里的独立 300 局（strict，σ=0.1，cell 全部 DR 开）：

| 动作来源 | 分数 |
|---|---:|
| policy 头（不规划） | **0.87** |
| MPPI，ibrl 调过的那套 + `init_std=0.3` | **0.887**（扫描最好） |
| MPPI，original SimDist 那套，H 3/4/5 | 0.873 / 0.853 / 0.87 |
| MPPI，cfg 默认（`init_std` 0.5，H5） | 0.82 |

⇒ 规划在 sim 里约等于 policy 头（差在 300 局的 ±0.02 噪声内），horizon 3–5 无差别。stage-4 门槛 0.9 没过，
但对这次诊断不影响。真机上**两种都跑**：policy 头是主力，MPPI 看规划在真机上是帮忙还是添乱。

## 0. 同步（10 分钟）

```bash
# host
cd ~/simdist/deploy_simdist_1015 && git pull          # 需要 commit 4df5e9f 之后的 policy_server（exec_obs + planner 参数）
cp ~/simdist/SimDist_ibrl/real2sim/deploy/robot_client.py ~/simdist/gello_software/simdist_runner/robot_client_1018.py
mkdir -p ckpts/robotiq_s1      # 把 model0.pt 和 cfg.yaml 拷进来（cfg 里的 sim2real_cell 路径是仓库相对路径，在 deploy 目录里能找到）
```
- 拍一张标定板（`record_calib.py --no_auto`），检查手腕支架没松。
- 先在 host 上起一次 server 看打印：`prop` 维度要是 **36**，`action map sim_ref`，`reference arm control_freq 10 Hz`，
  相机行打印的 fovy/crop 与上次一致。

## 1. 只看不动（15 分钟）

```bash
# host
~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py --ckpt ckpts/robotiq_s1/model0.pt \
    --z_min 0.950 --save_dir runs/s1_dry_$(date +%m%d_%H%M) --save_every 1
# 容器（不加 --live = 不动臂）
python3 /workspace/simdist_runner/robot_client_1018.py --gripper-state-topic /gripper/joint_states --hz 10 --steps 50
```
- 看 `runs/s1_dry_*/obs_*.png`（就是送进网络的 224 图）：和 sim 渲染对比取景、颜色、夹爪占画面比例。
  回来后我和 sim 同位姿的渲染逐张对比。
- `steps.jsonl` 里的动作在臂不动时应该是"往针走"的方向；如果是乱的，先别上 live。
- **z_min 实测**：闭合夹爪，指尖碰桌，读 server 打印的 `eef[2]`，加 2 mm 就是 `--z_min`（sim 估计 0.950）。
  ⚠️ sim 的地板还多一条"闭合夹爪最低点离桌 ≥ 0.5 mm"，server 没有 ⇒ 宁可把 `--z_min` 再抬 2–3 mm。

## 2. 零样本跑（40–60 分钟，主体）

针用胶带轮廓放在上次 demo 的位置（x 0.546–0.556，y −0.120…−0.116，yaw −112…−93°，cell 的 layout 范围内）。
每一局开始前先 `reset_arm.py` 回 cell 的 home，夹爪张开。

```bash
# A. policy 头（主力）
~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py --ckpt ckpts/robotiq_s1/model0.pt \
    --z_min <实测> --save_dir runs/s1_policy_$(date +%m%d_%H%M) --save_every 1
# B. MPPI（sim 最好的那组）
~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py --ckpt ckpts/robotiq_s1/model0.pt \
    --planner_set init_std=0.3 \
    --z_min <实测> --save_dir runs/s1_mppi_$(date +%m%d_%H%M) --save_every 1
# 容器：每局一次
python3 /workspace/simdist_runner/robot_client_1018.py --gripper-state-topic /gripper/joint_states --hz 10 --steps 400 --live
```
- A 先跑 10 局，B 再跑 5–10 局。`--save_every 1` 每步都存图，这批数据之后要当 stage 3 的 on-policy 数据用，别省。
- 每局记一行：成功/失败 + **卡在哪一阶段**（接近 / 抓取 / 提起 / 对准 / 插入）+ 有没有碰桌、re-sync、clamp 警告
  （server 每步的 `warn` 也在 `steps.jsonl` 里）。
- 不限速。**手一直放在急停上**。中断用 Ctrl-C（停住并交还 GELLO），**不要按 User Stop**。
- MPPI 每步延迟：sim 机器上 ~8 ms（3090 上应 < 30 ms）；看 server 回的 `ms.policy`，超过 60 ms 就只跑 A。

## 3. 遥操作 demo（剩下的时间）

stage 2 要真机 demo。照 10-18 的做法（`demos/threading_robotiq_1018/` 那套流程，10 Hz，`docs/real-teleop-stage2.md`），
针的摆放覆盖 layout 范围，**目标 20–50 条**。
- demo 里不需要额外记 exec_obs：遥操作不经过 ReferenceArm，转换脚本会用 demo 的动作把参考臂重放一遍补出这 27 维
  （sim 侧，我来写；同时让 `train_simdist_adapt.py` 支持 exec_obs）。
- 只要保证每条 demo 的 q、夹爪、两路相机、时间戳完整，和 10-18 一样。

## 带回来

- `runs/s1_dry_*`、`runs/s1_policy_*`、`runs/s1_mppi_*` 整个目录（`steps.jsonl` + `obs_*.png`）。
- 每局的手记（成功/阶段/备注）。
- demo 目录。
- 实测的 `--z_min`、标定板照片。
