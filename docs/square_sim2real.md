# Square sim2real 运行手册

**这条线是什么**：把 Square（方螺母套方柱）搬到与 Threading 同一张真机 DROID bench 上，
走 `source 专家 → pixel stage-1 蒸馏 → 真机`。target 不在 sim 里 —— 这是 sim2real，
target 就是真机本身，**不要照着 sim2sim 的 §12/§13 去补一个 target shift**。

**背景与实测结论**在 `docs/infra.md` §15（§15b 三个只靠 wrapper 会静默失败的点、§15f2 专家
0.96、§15j cell 大改后谁作废、§15k 方柱扰动、§15m 重放契约、§15n 采集成本）。
本文只讲**怎么操作**，不重复那边的推导。

---

## 0. 当前状态（2026-09-23）

| 资产 | 状态 |
|---|---|
| source 专家 `exps/mimicgen/square_realcell_ibrl_seed1` | ✅ 已训好，新 cell + 方柱扰动下 **n=300 得 0.98** |
| BC prior `exps/mimicgen/square_realcell_mlp_bc_seed1` | ✅ final eval 0.95 |
| state demo `release/data/mimicgen/generated/square_realcell_rotw1` | ✅ 1000 条 |
| **pixel 数据** | ❌ **待采**（v1 已随 cell 大改删除） |
| pixel stage-1 | ❌ 待跑 |

**注意 cell 改过之后什么作废了**：视觉项（相机 fovy、颜色、灯光、背景、桌面板）全变了 ⇒
pixel 数据作废。但 Square 的几何没变 ⇒ **专家、BC prior、demo 都不用重做**（§15j）。
以后若再改动的是**几何**（桌高、arena、`workspace_offset`、给 Square 加 `layout_nut_*`），
专家必须重训。

---

## 1. 换机器：需要带什么

共享盘上已经放好：`/user_data/Chenyang/square_realcell_collect_bundle/`（1.7 GB）
+ 它自己的 `README.md`。要点：

- **两组 ckpt 都要**：专家（`cfg.yaml` + `model0.pt` + **16 个 snapshot**）和 BC prior
  （`cfg.yaml` + `model0.pt`）。采集用 `HYBRID=1`，专家以 IBRL hybrid 身份 rollout，
  会去读专家 cfg 里的 `bc_policy` —— 少了 prior 直接起不来。
- **`cfg.yaml` 必须和权重同目录**：`train_rl.load_model` 从权重文件的目录读 cfg，只拷 `.pt` 没用。
- **权重要挂到 `<repo>/exps/mimicgen/` 下**：专家 cfg 里的 `bc_policy` / `sim2real_cell`
  都是**相对 repo 根目录**的路径。
- **代码同一个 commit**：`Square_RealCell_RotW1` 这个注册类、方柱扰动、`bg_index` 落盘都在
  `faec797` 之后。commit 号在 bundle 的 `GIT_COMMIT.txt`。
- 不用带数据集：prior 的输入形状来自 env，不是来自 hdf5。

---

## 2. 采 pixel 数据

### 2.1 开跑前的两个检查（各两分钟，能挡住几小时的白跑）

```bash
cd <repo> && source set_env.sh
PY=/home/Chenyang/.conda/envs/ibrl/bin/python

# a) 专家在这台机器上确实是 0.98 —— 几何/cell 对不上会立刻变成 0.00
$PY evaluate/eval.py --weight exps/mimicgen/square_realcell_ibrl_seed1/model0.pt --mode rl \
    --num_game 100 --mp 10 --shift_from_cfg 1 --hand_init_perturb_joint_sigma 0.1
# 期望 [EVAL ENV] task=Square_RealCell_RotW1 ... nut_z_rot=120-180deg(class)，score≈0.98

# b) 重放契约成立 —— 否则腕相机事后补不上，且要等补完才发现
$PY scripts/test_rerender.py --run exps/mimicgen/square_realcell_ibrl_seed1
# 期望 ALL PASS（A 段是「与渲染器自身重复噪声比」，不是逐字节，见 §15m）

# c) 挑一块真正空闲的卡
nvidia-smi
```

### 2.2 采集（agentview only，8 批 × 4500 条）

```bash
DEST=/user_data/Chenyang/ibrl_offload/release/data/mimicgen/square_realcell_pixel_v2
setsid env GPU=<空闲卡号> NUM_PROC=10 PER_WORKER_MIB=3300 GUARD_FRAC=0.75 BATCH_TRAJ=4500 \
  BATCH_IDS="1 2 3 4 5 6 7 8" SEED_BASE=100 \
  RUN=exps/mimicgen/square_realcell_ibrl_seed1 \
  CELL=release/cfgs/sim2real/square_real_cell.yaml DEST=$DEST \
  CAMS=agentview RL_IMAGE_SIZE=224 JPEG_Q=90 STORE_STATE=1 LOG_TAG=square_pix_v2 \
  bash scripts/collect_threading_realcell_batches.sh > logs/square_pix_v2_gpu0.log 2>&1 < /dev/null & disown
```

**为什么只渲 agentview**：`--store_sim_state 1` 让每步带 `sim_state`、每局带 `model_state`，
腕相机就是一次**纯渲染**的补充，rollout 一条都不用重跑（§14e）。开销只占图像字节的 1.9%。

想分两块卡并行就拆 `BATCH_IDS`（如 `"1 3 5 7"` 和 `"2 4 6 8"`），两条链写同一个 `DEST`，
第一条启动的那条负责写 cell 快照。

### 2.3 补渲染校准腕相机

```bash
DEST=$DEST GPUS=0,0,0,0,0,0,0,0 CHAIN_LOGS=logs/square_pix_v2_gpu0.log \
  setsid bash scripts/square_realcell_pixel_after_collect.sh > logs/square_pix_v2_wrist.log 2>&1 < /dev/null & disown
```
它会等采集链结束 → 逐个**打开**每个 part 校验（≥4500 条、带 `sim_state`/`model_state`）→
调 `rerender_wrist_chunks.py`。产物 `$DEST/2cam_calibrated_v1/part*_2cam.hdf5`。
⚠️ 按文件大小判断 hdf5 完整性会骗人（§14e 末尾那 49 GB 全损的教训），所以这里是真的打开。

多链时把 `CHAIN_LOGS` 写成两个日志名（空格分隔）。

### 2.4 成本与预期

| 项 | 值 |
|---|---|
| 吞吐 | ~3.6 条/秒（10 worker/卡）⇒ 8×4500 约 **1.5 h** |
| 显存 | **~2.8 GB/worker** ⇒ 10 worker 约 29 GB |
| 磁盘 | 2.29 MB/条 ⇒ agentview 约 82 GB，加腕相机一遍共约 **185 GB** |
| 采集成功率 | 77–78% |
| 腕相机补渲染 | 36000 条约 20 min |

---

## 3. 训练 SimDist stage-1（pixel）

```bash
$PY train_simdist.py \
  --config_path release/cfgs/sim2real/square_realcell_pixel_simdist_r18_224.yaml \
  --source_agent exps/mimicgen/square_realcell_ibrl_seed1/model0.pt \
  --save_dir exps/mimicgen/square_realcell_pixel_stage1_seed1
```

配置相对 Threading 那份只改了七处（都标了 `### SQUARE`）：`task_name`、`sim2real_cell`、
`source_replay_path`(v2)、`source_agent`、`seq_replay_max_transitions`（Square 轨迹更短，
5.8M 而不是 6.6M）、`strict_success: false`（Threading/Coffee 专用）、`target_demo_path: ''`。

**几个必须知道的**

- **`source_replay_num_data: 4500` 不能换成 `-1`。** `-1` 会把 8 个 shard 全量载入再从**前面**
  淘汰，结果只剩最后一两个 shard ⇒ 8 个生成 seed 塌成 2 个。按 shard 各取 4500 才能保住多样性。
- **`seq_replay_max_transitions` 不能绑死。** 它一旦触顶就从前面淘汰，同样丢掉最早的 seed。
  Square 的 36000 条约 5.5M transition，所以设 5.8M（不触顶）。内存约 105 GB ⇒ 要 `--mem=200G`，
  而且这个数还要覆盖 torch/CUDA 的 host 分配和 10 个 mp eval worker（各带一个 MuJoCo env + EGL context）。
- **eval 打的是 source（真机 cell 的名义物理）+ cell 自己的 DR**，`score/score` 就是
  TopkSaver 选 `model0.pt` 的指标。⚠️ **与 sim2sim 的 pixel 阶梯（simdist §47/§48）不可比**，
  那边全在 dynamics-only target 上。
- **`num_dynamics_heads` / `horizon` 是 stage-1 建模时定死的**（铁律 #20），adapt 阶段改不了。
  这份配置是 `dynens h5`。
- 跑之前别动 `env/*.py`、`rl/*.py`、`train_simdist.py` —— mp eval 用 spawn 会**重新 import**。

---

## 4. 之后：stage 2/3 在真机上

sim 里没有 target。stage-1 产出的是一个在真机 cell 上蒸馏好的 world model + policy 头，
后续适配走真机数据，不要在 sim 里再造 target。

---

## 5. 踩过的坑（按会不会咬到你排序）

1. **别选别人占着的卡。** 另一块卡被占 ~70 GB 时，10 个 worker 有 7 个直接 CUDA OOM。
   批脚本的 guard 判的是**整卡** used/total ⇒ 别人占着时它会一直 sleep，
   **表现是「在跑但不动」而不是报错**。
2. **cell 在采集期间不能改。** `DEST` 首次创建时记下 cell 的 **inherit 链**哈希，之后每批比对，
   变了就停。要换 cell 重采：删掉整个 `DEST` 再起，别在原目录上续。
3. **背景图路径是绝对路径**：`dr_background_image: /user_data/Chenyang/bg_textures/val_256`
   （36500 张）。共享盘上有；换机器若挂载点不同，要改 cell。
4. **改 `MODEL_STATE_FIELDS` 会让新旧数据不能混用**（`model_state_layout` 对不上）。
   只在没有数据依赖它的时候改。
5. **停任务要按进程组杀**：`generate_simdist_data.py` 自己在一个 pgid、批脚本在另一个，
   只杀一个会留下 worker 占着显存。查 `ps -eo pid,pgid,cmd`，两个 pgid 都 `kill -TERM -<pgid>`。
   之后用 `nvidia-smi` 确认显存真的回落。
6. **`nohup` 挡不住 Bash 工具的 timeout**（铁律 #11）⇒ 一律 `setsid ... < /dev/null & disown`。
7. **`num_eval_proc` 必须整除 `num_eval_episode`**（铁律 #12）。

---

## 6. 涉及的文件

| 文件 | 作用 |
|---|---|
| `env/square_variants.py` | `Square_RealCell` / `Square_RealCell_RotW1`（arena + 方柱重定位 + rotw1 窗口 + `ROUND_PEG`） |
| `release/cfgs/sim2real/square_real_cell.yaml` | 采集 cell（inherit Threading 的父 cell，只写 4 个任务键） |
| `release/cfgs/sim2real/square_real_cell_calibrated_wrist.yaml` | 训练/eval cell = 采集 cell + 校准腕相机，**只差 `cameras`** |
| `release/cfgs/sim2real/square_realcell_state_{mlp_bc,ibrl}.yaml` | source 专家那条链（已跑完） |
| `release/cfgs/sim2real/square_realcell_pixel_simdist_r18_224.yaml` | pixel stage-1 |
| `scripts/collect_threading_realcell_batches.sh` | 采集（通用，靠 `RUN`/`CELL`/`DEST`/`LOG_TAG` 切任务） |
| `scripts/square_realcell_pixel_after_collect.sh` | 等采集结束 → 校验 → 补渲染腕相机 |
| `scripts/test_rerender.py` / `scripts/test_sim2real.py` | 两个 gate，改完任何相关代码都要跑 |
| `scripts/record_realcell_rollouts.py` | 录成功轨迹的视频/时序图，排查用 |
