# 在新机器上接着跑 Threading Robotiq 链（2026-10-29）

curie 的 4 张卡被别人占满（GPU 0/1 是别人的，GPU 2/3 上也有别人的大任务），所以把这条链搬到另一台有 `/user_data` 的机器上。
本文按顺序走：**① curie 上收尾 → ② 新机器装环境 → ③ 核对数据 → ④ 启动 → ⑤ 盯什么**。

背景：sim2real 的目标与门槛见 `real2sim/zed_capture/calib_20261015/TRAIN_PLAN.md`，10-27 真机为什么失败见同目录 `real_1027/README.md`。

---

## 0. 现在的状态（10-29 搬家时）

| 东西 | 路径 | 状态 |
|---|---|---|
| cell（当前） | `real2sim/zed_capture/calib_20261015/threading_real_cell_20261030.yaml` | 针尖着地 + 手中位姿 DR（prob 0.7）+ Robotiq 闭合时序 + **抓取下限 0.048 = 真机 `--z_min 0.960`** |
| v8 demo | `/user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v8{,_execobs}` | ✅ 3000 条，脚本专家 0.988 |
| v8 BC prior | `exps/sim2real/threading_robotiq_rnn_bc_n3000v8_h1000_ema_ep6000_seed1` | ✅ 独立 300 局 model0 **0.75**（门槛 0.3） |
| v8 IBRL ×2 | `exps/sim2real/ibrl_robotiq_v8_fr_chunk1_execobs_seed{1,2}` | ⏳ 刚开训。**`train_rl.py` 没有断点续跑 ⇒ 新机器上从头重开** |
| v8 stage 1 | `exps/sim2real/threading_robotiq_v8_pixel_stage1_seed1`、数据 `/user_data/.../threading_robotiq_pixel_v8` | 还没开始（等 IBRL 过 0.9 自动接） |
| **10-30 更新** | v8 expert = IBRL seed1 @400k（独立 300 局 **0.943**）已冻结；pixel 数据 8/8（36000 条）在 `/user_data`；stage 1 在 curie 上停在 **70k/150k**（GPU 被别人抢到 2.25 s/it） | 断点 + expert 已拷到 `/user_data/Chenyang/ibrl_move_20261029/ibrl/exps/sim2real/threading_robotiq_v8_{expert,pixel_stage1_seed1}`：rsync 回 `exps/sim2real/` 后用 ④ 的 stage-1 命令（`CELL` = 10-30、`FROZEN`/`DEST`/`S1` 换成 v8）续跑，日志要出现 `--resume_from` |
| v6 专家（备用） | `exps/sim2real/threading_robotiq_v6_expert` | ✅ 0.953（10-28 cell，没有地板修正和闭合时序） |
| v6 stage 1（备用） | `exps/sim2real/threading_robotiq_v6_pixel_stage1_seed1`、数据 `/user_data/.../threading_robotiq_pixel_v6`（123 GB） | ⏳ ~13.5k/150k，**有 `resume_state.pt`，可续跑** |

代码在 GitHub `Sunlighted/SimDist_ibrl` 的 `main`（本文所在的提交及之后）。
curie 上的快照（代码 + `exps/sim2real` + conda env + deps）在 `/user_data/Chenyang/ibrl_move_20261029/`。

---

## ① curie 上收尾（新机器开跑**之前**做；同一个 save_dir 只能有一个 writer）

```bash
# curie
cd /home/Chenyang/ibrl
ps -eo pgid,args | grep -E "pipeline_threading_robotiq_(v8|v6|stage1)|train_rl.py.*robotiq_v8|train_simdist.py.*v6_pixel" | grep -v grep
kill -- -<pgid> ...          # 每个 pipeline 的进程组（v8 pipeline、v6 stage-1 pipeline）都要杀，含其子进程
# 最后同步一次（v6 stage 1 的 resume_state.pt 每 10k 步更新）
rsync -a exps/sim2real/ /user_data/Chenyang/ibrl_move_20261029/ibrl/exps/sim2real/
rsync -a logs/ /user_data/Chenyang/ibrl_move_20261029/ibrl/logs/
```

---

## ② 新机器装环境

**最省事（新机器上也是用户 `Chenyang`、家目录 `/home/Chenyang`）：原样放回同一路径，脚本一行不用改。**

```bash
M=/user_data/Chenyang/ibrl_move_20261029
rsync -a $M/ibrl/            /home/Chenyang/ibrl/
rsync -a $M/ibrl_deps/       /home/Chenyang/ibrl_deps/
mkdir -p /home/Chenyang/.conda/envs && rsync -a $M/conda_env_ibrl/ /home/Chenyang/.conda/envs/ibrl/
cd /home/Chenyang/ibrl && git pull          # 拿到 GitHub 上最新（含本文）
```

**家目录不同时**（设 `R=<新的仓库路径>`、`E=<env 路径>`、`D=<deps 路径>`）：

```bash
git clone git@github.com:Sunlighted/SimDist_ibrl.git $R        # 或 rsync $M/ibrl/
rsync -a $M/ibrl/exps/sim2real/ $R/exps/sim2real/              # 训练产物不在 git 里
rsync -a $M/ibrl_deps/ $D/ ;  rsync -a $M/conda_env_ibrl/ $E/
# robosuite / robomimic / mimicgen 是 editable 安装，finder 里写死了 /home/Chenyang/ibrl_deps：
sed -i "s#/home/Chenyang/ibrl_deps#$D#g" $E/lib/python3.10/site-packages/__editable___*_finder.py
export PY=$E/bin/python          # 所有 Robotiq 脚本都认 $PY（默认值才是 /home/Chenyang/...）
```

- ⚠️ env 里 `bin/` 下脚本的 shebang 是坏的：一律 `$PY -m pip ...`，不要直接 `pip`。
- ⚠️ 需要 torch 2.7+/cu128（Blackwell）；换成更老的卡也能用同一个 env。
- `git config core.hooksPath scripts/githooks`（防误提交被 gitignore 的数据，CLAUDE.md「常用命令」）。

**冒烟测试（1 分钟，必须过）：**

```bash
cd $R && source set_env.sh
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF'
import numpy as np, torch
from env.sim2real import load_sim2real
from env.robosuite_wrapper import PixelRobosuite
c = load_sim2real("real2sim/zed_capture/calib_20261015/threading_real_cell_20261030.yaml")
e = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, camera_names=["agentview", "robot0_eye_in_hand"],
                   rl_cameras=["agentview", "robot0_eye_in_hand"], rl_image_size=224, use_state=True, device="cpu",
                   sim2real=c, exec_obs=True)
o, _ = e.reset(); print("exec_layer z_min_base =", e.env.exec_layer.cfg["z_min_base"], "(must be 0.048)")
print({k: tuple(v.shape) for k, v in o.items()}); print("cuda", torch.cuda.is_available())
EOF
```
要看到 `z_min_base = 0.048`、两路 224 图像、`cuda True`。退出时打印的 `EGLError ... MjRenderContext.__del__` 是 robosuite 释放上下文的噪声，无害。EGL 报错 ⇒ 装 NVIDIA 的 EGL（`libEGL_nvidia`），或检查 `LD_LIBRARY_PATH`。

---

## ③ 核对数据（新机器的 `/user_data` 要有这些）

```bash
ls /user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v8/{DONE,dataset.hdf5} \
   /user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v8_execobs/dataset.hdf5
ls /user_data/Chenyang/simdist_data/threading_robotiq_pixel_v6/part{1..8}.hdf5     # 只有续跑 v6 stage 1 才要
```
如果新机器的 `/user_data` 不是同一块盘，从 curie 拷过去（v8 demo 1.5 GB；pixel_v6 123 GB）。
脚本里数据都是绝对路径 `/user_data/Chenyang/...`，所以**挂载点必须同名**。

磁盘：v8 的 pixel 数据会再写 ~120 GB 到 `/user_data/Chenyang/simdist_data/threading_robotiq_pixel_v8`。
内存：每个 stage 1 常驻 ~130 GB（`RAM_BUDGET_GB` 默认 170）；两个同时跑要 ~300 GB。

---

## ④ 启动（一律 `setsid ... & disown`，CLAUDE.md 铁律 #11）

**v8 主线**（A/B/C 已完成会自动跳过；IBRL 目录存在时脚本**不会重开**，所以先挪走半截的那两个）：

```bash
cd $R
for s in 1 2; do mv exps/sim2real/ibrl_robotiq_v8_fr_chunk1_execobs_seed$s{,_curie_partial}; done
GPUS="0 1" setsid bash scripts/pipeline_threading_robotiq_v8.sh > logs/pipeline_robotiq_v8.log 2>&1 < /dev/null & disown
```
- `GPUS` = 两个 IBRL seed 各用的卡；第二张卡之后也跑 v8 的 pixel 采集和 stage 1。
- 之后全自动：IBRL 2M 步 → 从 400k 起每 100k 做独立 300 局评测，有一个 ≥0.9 就冻结到 `exps/sim2real/threading_robotiq_v8_expert` → pixel 数据 8×4500 条 → stage 1（150k 步）→ 300 局 MPPI / policy 评测。

**v6 stage 1 续跑（备用，可选）**：

```bash
CELL=real2sim/zed_capture/calib_20261015/threading_real_cell_20261028.yaml \
RUNS="exps/sim2real/ibrl_robotiq_v6_fr_chunk1_execobs_seed1 exps/sim2real/ibrl_robotiq_v6_fr_chunk1_execobs_seed2" \
FROZEN=exps/sim2real/threading_robotiq_v6_expert DEST=/user_data/Chenyang/simdist_data/threading_robotiq_pixel_v6 \
S1=exps/sim2real/threading_robotiq_v6_pixel_stage1_seed1 GPU=2 \
setsid bash scripts/pipeline_threading_robotiq_stage1.sh > logs/pipeline_robotiq_v6_stage1.log 2>&1 < /dev/null & disown
```
看到 `stage 1 -> ... --resume_from .../resume_state.pt` 才是续跑；`train.log` 是追加写。
⚠️ `CELL` 必须是 **10-28**（v6 的数据和专家都是这个 cell），别用 10-30。

---

## ⑤ 盯什么

| 看哪里 | 正常 | 不正常就 |
|---|---|---|
| `logs/ibrl_v8_fr_chunk1_execobs_s{1,2}.log` | 预热 ~15 min（每局 ~12 s，60 局），之后有 `score` | `Traceback` ⇒ 看最后 40 行 |
| `exps/sim2real/ibrl_robotiq_v8_*/snapshot_*.pt` | curie 上 v6 是 **~70 min / 100k 步**（被抢卡时更慢）⇒ 400k ≈ 5 h | 停在某一步不动 ⇒ 看 GPU 是否被别人占 |
| `logs/pipeline_robotiq_v8.log` 的 `[pipe] gate @` 行 | v6 在 500k 时 0.953 | 1M 都没过 0.9 ⇒ `GATE FAILED`，停，别降门槛 |
| stage 1 进度条 `distill (offline): N/150000` | 无人抢卡时 ~1–1.5 it/s | |
| 最后的 `stage-1 mppi:` / `stage-1 policy:` | 上真机用的就是这个 `model0.pt` | |

判 IBRL 成败看 `gate_eval300_*.log` 的独立评测，**不要看 `model0.pt` 的训练内评测**（铁律 #14，系统性乐观）。

---

## 上真机前（v8 stage 1 出来以后）

- server 的 `--z_min 0.960` 现在和 sim 一致（10-30 cell 的 `z_min_base 0.048` + base 0.912），**不要再改低**。
- 场景相机：10-27 的画面比 sim 随机化范围亮得多（桌面 237–249 vs sim p95 ~190）⇒ 到场先锁 ZED 曝光、拍一帧重标颜色映射（`--color_json`）。
- 部署包流程照旧：`real2sim/deploy/README.md`。
