# IBRL 实验：Threading Robotiq cell（2026-10-20，到另一台机器上跑）

> **2026-10-23: everything below this note is history.** The cell changed under it (10-21: grasp-yaw expert fix, closed
> Robotiq linkage, gripper contact + force cap; 10-22: stiff object contact, table-relative floor, table impact = failure
> at 35 N, expert grasp_h 0.020; README "10-21 gripper contact" and the cell yaml). All data / priors / runs named here
> were made on the old cell and are **deleted** (`threading_robotiq_expert_n1000*`, `assets/bc_rnn_seed1`, every run in
> `exps/sim2real` before v5). Current: data `threading_robotiq_expert_n3000_v5`, prior
> `exps/sim2real/threading_robotiq_rnn_bc_n3000v5_h1000_ema_ep6000_seed1/model0.pt`, IBRL
> `exps/sim2real/ibrl_robotiq_v5_2M_fr_chunk{4,1}_seed1` (fail replay 0.5, prior_chunk 4 / 1, 2M steps);
> `scripts/run_ibrl_robotiq_arm.sh` defaults to these. E2 (fail replay) beat E1 on the old cell, 0.823 vs 0.757 (1 seed).

## v5 cell (2026-10-23)

**fr_chunk4 / fr_chunk1** (fail replay 0.5, prior_chunk 4 / 1): in-training eval plateaus at ~0.72 / ~0.69 from
~600k (mean of the 50-episode points 600k–1M). Independent 200 episodes at 500k: 0.575 / 0.53 (prior alone 0.61).
Contact health clean on successes (needle-ring <2 mm in all; table force <35 N).

**Failure anatomy at 500k** (200 episodes each, `scripts/diagnose_threading_robotiq_failures.py`): push the ring
>4 cm 26-29 %, table impact 6-9 % (mostly after the tripod was knocked over), near-but-not-through 2-8 %, grasp/drop
4-5 %. At the FIRST ring push (ring centre moved >10 mm vs step 5; ~25 steps after the tip comes within 5 cm)
successes and failures are indistinguishable: angle 4.8 vs 4.7 deg, shaft-line offset 7.7 vs 8.8 mm, tip speed
0.6 vs 0.7 mm/step, and 82 % of successes also push the ring >=10 mm. The outcome is decided after contact: failures
escalate (96 % >4 cm, 35 % end with the needle back on the table) and wrestle until the 400-step timeout. Renders:
`viz/ibrl_v5_500k_failures/`.

Two new arms on top of fr_chunk4 (one change each), started 10-23 14:20 / 15:40 UTC on GPU 3, 2M steps:
- **fr_chunk4_mid** (`--mid_reset_prob 0.5`): start 5-15 steps before the first ring push of an earlier (mostly
  failed) episode of the policy itself. On-policy, narrow.
- **fr_chunk4_omni** (`--omni_reset_probs fresh=.25,grasped=.25,preinsert=.25,partial=.25 --omni_pool_size 150`):
  UWLab OmniReset-style policy-independent pool (`~/UWLab/.../omnireset`, `MultiResetManager`, 4 reset types at 25 %);
  states from the scripted expert stopped at a random step of the phase + random perturbation, validated.
Both rebuild a state by re-seeding the reset and replaying the actions (DR draws and exec-layer internals are not in
sim_state); bit-exact (0 mismatches). The expert's `ref()` leaves a 1e-13 trace in the exec layer's shadow arm, so
pool states are stored from a clean replay, not from the expert pass.


背景见 `TRAIN_PLAN.md` 阶段 3。现状（curie，GPU 2）：
- BC-RNN prior：独立 300 局 **0.34**。
- IBRL 第 1 次（`preload_num_data 10`）：独立 300 局 **0.257**，前 375k 步全是 0。
- IBRL 第 2 次（preload 1000，seed 1，**还在 curie 上跑**，`exps/sim2real/threading_robotiq_ibrl_preload1000_seed1`）：
  in-training eval 在 475k–600k 步是 0.30 / 0.38 / 0.58 / 0.34 / 0.52 / 0.24，波动大，离 0.9 门槛很远。

> **「IBRL without margin」就是现在的配方**：`threading_robotiq_state_ibrl.yaml` 里 `prior_margin` 是 0，前两次都是原始 IBRL。
> 下面每一臂都是原始 IBRL（`act_method ibrl`、margin 0、RNN prior、preload 1000），每臂只改**一个**旋钮（铁律 #9）。

## 1. 另一台机器的准备

```bash
git pull
# conda env 按 docs/setup.md（torch 2.7/cu128、mujoco 2.3.2）；脚本默认用 /home/Chenyang/.conda/envs/ibrl/bin/python，路径不同就传 PY=...
```

- **BC prior 已进仓库**：`real2sim/zed_capture/calib_20261015/assets/bc_rnn_seed1/{model0.pt,cfg.yaml}`（7.7 MB，
  sha256 前缀 `70bac6a8f0f451cd`）。加载只读同目录的 `cfg.yaml`，不需要数据集。
- **demo 数据不在仓库里**（217 MB）：`/user_data/Chenyang/simdist_data/threading_robotiq_expert_n1000/`（`dataset.hdf5` + `env_cfg.json`）。
  二选一：
  - 拷过来：`rsync -av curie:/user_data/Chenyang/simdist_data/threading_robotiq_expert_n1000/ <本机路径>/`，跑的时候传
    `DATA=<本机路径>/dataset.hdf5`。
  - 重新采（8 片并行）：`scripts/collect_threading_robotiq_demos.py`（用法在文件头）。采出来的不会和原来逐字节相同，
    会多一个变量，**优先拷**。

## 2. 实验臂

每臂都跑 `scripts/run_ibrl_robotiq_arm.sh`：1M 步训练，然后两次独立 300 局评测（strict，σ=0.1）：
- `eval300_own.log`：这一臂自己的初始化范围；
- `eval300_full.log`：完整的 cell 范围（`--sim2real_cell` 会丢掉 overrides）。

```bash
# 每条都要 detach（铁律 #11）；一张卡最多放 2 个 run（curie 上 3 个 run 挤一张卡时，速度从 25 掉到 9 步/秒）
setsid env ARM=<arm> SEED=<s> GPU=<g> EXTRA="<flags>" bash scripts/run_ibrl_robotiq_arm.sh \
    > logs/ibrl_robotiq_<arm>_s<s>.log 2>&1 < /dev/null & disown
```

| 臂 | ARM | EXTRA | seeds | 回答的问题 |
|---|---|---|---|---|
| **E1** 原始 IBRL | `e1_plain` | （空） | 2, 3 | seed 1 是不是运气（加上 curie 的 seed 1，一共 3 个 seed） |
| **E2** 失败重放（Mulligan） | `e2_failreplay` | `--fail_replay_prob 0.5` | 1, 2 | 在失败过的初始状态上多练能不能更快 / 更高 |
| **E3a** 初始化范围 ×0 | `e3_scale0` | `--sim2real_overrides layout_scale=0` | 1 | 摆放固定时 IBRL 学不学得会（排除「任务本身学不会」） |
| **E3b** 初始化范围 ×0.25 | `e3_scale025` | `--sim2real_overrides layout_scale=0.25` | 1 | |
| **E3c** 初始化范围 ×0.5 | `e3_scale050` | `--sim2real_overrides layout_scale=0.5` | 1 | |
| （E3d ×1.0） | = E1 | | | |

一共 7 个 run。3 张卡的分法举例：GPU0 = E1 s2 + E1 s3，GPU1 = E2 s1 + E2 s2，GPU2 = E3a + E3b，E3c 等一个空位。
**E3a 优先**：如果范围 ×0 都学不会，问题在任务 / 奖励 / prior，不在范围，E2 和其它 E3 的结论都要重新读。

### `layout_scale` 是什么
`env/sim2real.py`：把 cell 里每个摆放范围（needle / tripod 的 xy 和朝向）以中点为中心缩放。以 needle 朝向为例：
1.0 → −136..−81°，0.25 → −114..−100°，0 → −106..−104°。⚠️ 0 也不是完全固定：base DR 仍让物体相对机器人有约 ±1.7 cm 的散布，
σ=0.1 的关节初始化扰动也还在。

### 失败重放（Mulligan 的「failure promotion」，arXiv 2610.05882）
`train_rl.py` 的 `fail_replay_*`：
- 训练 episode 失败后，把它的初始状态（物体相对 base 的位姿 + 机械臂关节角）放进一个 FIFO（`fail_replay_cap` 200）。
- 之后每个训练 episode 以概率 `fail_replay_prob` 从这个缓冲区里取一个状态复位。物理 DR 和相机 DR 照常重新抽。
- 重放成功就移出缓冲区（retired），连续失败 `fail_replay_max_tries`（3）次就丢掉（dropped）。
- 复位精度实测：位置 0.0055 mm、姿态 2.8e-7，q 精确。
- `train.log` 里有 `fail_replay/{buffer,replayed_total,retired_total,dropped_total,is_replay}`。

和论文的差别：
- 论文按「最早失败的阶段」排序，我们用 FIFO（没有阶段标注）。
- 论文的「coverage fill」（最远点采样），这里由均匀随机复位代替。
- 论文在失败状态上补人类示范（counterfactual demo）。**没实现**，这是下一步，用脚本化专家在 buffer 里的状态上补 demo 进 replay。
- 论文建在 HiL-IDQL 上，我们是 IBRL。

⚠️ eval 永远是均匀复位，不受 `fail_replay` 影响。所以 E2 与 E1 的比较是公平的。

## 3. 怎么读结果

- **只用独立 300 局比较**，不要用 `model0.pt` 的 in-training 分数：它是 TopkSaver 的 max，偏乐观 0.06–0.19（铁律 #14）。
  也要看 `train.log` 里最后 200k 步 `score/score` 的均值。
- 噪声：300 局的 SE 约 ±0.03；seed 之间的差还没量过，E1 的 3 个 seed 就是用来量它的。**E2 − E1 < 0.1 时，在 E1 的 seed 跨度量出来之前不下结论。**
- 门槛（TRAIN_PLAN 阶段 3）：`eval300_full` ≥ 0.9。
- E3 的读法：看 own 和 full 两列随 scale 怎么变。
  - own 随 scale 变大而下降 ⇒ 范围本身是难点，可以考虑课程（先小后大）。
  - own 一直很低 ⇒ 范围不是瓶颈。
  - full 列说明在小范围上训出来的策略能泛化多少。
- 每个 run 先核对 `eval300_*.log` 的 `[EVAL ENV]` 行（铁律 #17/#19），再核对 `train.log` 里的 `[sim2real] overrides`（E3）。

## 4. 跑完带回来

`exps/sim2real/ibrl_robotiq_*_seed*/{train.log,cfg.yaml,eval300_own.log,eval300_full.log,model0.pt}`，以及 `logs/ibrl_robotiq_*.log`。
