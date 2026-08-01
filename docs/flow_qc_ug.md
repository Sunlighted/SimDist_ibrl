# Flow-QC-UG: 流策略 + Q-chunking + Universal Guidance 的在线 model-free RL

> 目标:把 source→target 迁移的最终落点(policy 能力)做强。先在 **SOURCE** 上
> 验证这个 GPI-with-a-flow-actor 能否逼近 ~100%(RL 已知能到,是 go/no-go 门),
> 再谈 target。

---

## 1. 前情:为什么不是继续修 SimDist(关键发现汇总)

在 Threading_D0 sim2sim(§12 dynamics-only target)上,对 SimDist / MPPI 做了
一整轮消融,结论高度一致(证据见下):

**A. MPPI/SimDist 的天花板 = policy 质量,不是 value / dynamics / encoder。**
- `goodpolicy`:一个强 base policy(单独 policy-only **0.72–0.79**),套上 MPPI
  反而被拽到 **0.42–0.66**,dynamics 越 adapt 也追不上裸策略。
- MPPI 扫描(300–500 局):最好的配置(窄搜索 / trust-proposal)只能**打平**
  policy(0.72–0.74),**从不超过**;越让 value 主导选择越差
  (`H1-pureV`=0.40、`H2`=0.57、`greedyV`略降)。
- 三探针定位:`ablate_val`(把 value 换成更准的 target 专家 Q)和 `target_distill`
  的 value 在真实状态上几乎相同(Δ0.009, corr 0.998),但 MPPI 0.45 vs 0.90 →
  **value 不是区分器**;真正拉开的是 **policy**(target_distill 的 policy 克隆动作
  准 3.4×,MSE 0.094 vs 0.32)。

**B. value 在这个任务里是"净负担",不是可解除的瓶颈。**
- 在真实状态上 value 很准,但 **MPPI 查的是 dynamics 前推出的 latent**(off-manifold);
  更"尖锐/更准"的 value 反而在 off-manifold 有更多虚高峰,argmax 专挑这些 →
  更差(`ablate_val` 0.75→0.45)。
- 机制 = **optimizer's curse / model exploitation**:argmax 在**学到的 dynamics
  model** 上找"更高回报"的候选,策略越接近最优,这些候选越只可能是**模型误差**。

**C. 结论:天花板是 policy;要到 100%,必须把改进反馈进 policy;而 argmax 一个
learned dynamics model 是错误的机制(会钻误差)。**

## 2. 本方法:去掉 dynamics model,改成 model-free GPI(流 actor)

把 SimDist 的 "argmax learned dynamics" 换成 **"在 chunk-Q 上做 Universal
Guidance",而 chunk-Q 由 TD 在真环境 transition 上训练**:

- **没有 dynamics model 可钻** → 去掉了 81% 的 exploitation 天花板。这正是
  IBRL/RLPD 能到 ~100% 的同一个 model-free 理由,只是把 actor 换成更强的
  **flow 策略**、critic 换成 **chunk-Q**。
- 本质是 **generalized policy iteration**:
  - **policy evaluation**:`Q(s, a_{0:h})`,chunk-TD(`γ^h` backup),min-ensemble
    + LayerNorm(RLPD),target clamp 到 `[0,1]`(稀疏 0/1 + end_on_success ⇒ 真
    Q∈[0,1],clamp 无偏、切断负发散,见 §16/§21)。
  - **policy improvement**:(a) 用 **flow + UG** 采集(`∇_a Q` 在每个 ODE 步把
    chunk 推向高 Q,Bansal 2023 forward guidance),让 rollout 数据本身高 Q;
    **(b) advantage-weighted flow-matching**,让 flow **自己的提议分布**也移向
    高 Q chunk——不是纯 BC(纯 BC 只是摊销,会卡在 teacher)。

## 3. UG 超参 / Ta 的说明(来自已坐实的实验)

- **UG coeff ~0.3**,倒 U 型,梯度单位归一化;强 base 下"全有或全无",别用中间小 coeff。
- **Ta 不必 =1**。区分两件事:
  - **Q-chunk 长度 `h`**(`γ^h` TD backup + UG 引导信号 + 时序连贯探索):可 `h>1`。
  - **执行 `exec_ta`**(rollout/eval 每次执行几步再重查):`Ta=8` 开环在 **target
    (有 gap)** 上崩到 15%,`Ta=1`=58%——但那是 gap 造成的漂移。**source 无 gap,
    `exec_ta>1` 没那么致命**,且连贯 chunk 是 Q-chunking 的探索优点。默认 eval
    `exec_ta=1`(最稳),source 上可试 >1。二者是独立 config。

## 4. 实现

- `rl/flow_qc_ug_agent.py` — `FlowQCUGAgent`:
  - `flow`(`StateFlowPolicy`,在线权重被训练)+ `critic`(`EnsembleQ`,输入
    `concat(state, flatten(a_{0:h}))`)+ `critic_target`。
  - `_ug_chunk_raw`:flow + `∇_a Q(chunk)` UG-steered Euler ODE(复用 flow_qug 思路,
    Q 换成 chunk-Q)。`_flow_chunk_raw`:纯 flow 采样(TD next-action / AWR baseline)。
  - `update_critic`:chunk-TD(`R=Σγ^i r_i`,`boot=γ^h·1[窗口内无成功]`,clamp[0,1])。
  - `update_actor`:per-sample flow-matching × AWR 权重 `exp((Q(s,a)−V(s))/τ)`,
    `V(s)≈Q(s, flow采样)`。`actor_awr=0` 退化为纯 BC。
- `train_flow_qc_ug.py` — SOURCE 在线循环:load 预训练 flow(diverse 上训好的)→
  warm-up(flow 采集)→ 每步采集(`ug_collect_after` 之后开 UG)+ `num_critic_update`
  次 critic + 1 次 actor + soft target → 周期性 **source** eval + 存 best。
- 复用:`rl/sequence_replay.py`(存整段、sample `(Tp+1)`-段)、`rl/rebrac_agent.EnsembleQ`、
  `evaluate.run_eval_mp`(env 在子进程、agent 主进程跑 cuda;UG 的 `enable_grad`
  覆盖外层 `no_grad`)。

## 5. 运行

```bash
source set_env.sh
# 预训练 flow 已有:exps/mimicgen/threading_flow_source_expert_seed1/best.pt (Tp16/Ta8, source expert 0.38)
python train_flow_qc_ug.py \
    --flow_ckpt exps/mimicgen/threading_flow_source_expert_seed1/best.pt \
    --save_dir exps/mimicgen/threading_flow_qc_ug_seed1
# 消融:纯 BC(关 AWR)/ 关 UG 采集 / exec_ta
python train_flow_qc_ug.py ... --agent.actor_awr 0
python train_flow_qc_ug.py ... --agent.use_ug 0
python train_flow_qc_ug.py ... --agent.exec_ta 2
```

## 6. 里程碑(SOURCE 上按序过门)

1. **Gate 0**:source flow-only 基线(expert base ≈ 0.33–0.38)。
2. **Gate 1**:开在线循环,source 成功率**单调上升**、`train/q_mean` 稳步上升、
   eval 跟涨 ⇒ GPI 在工作。
3. **Gate 2(关键)**:能否逼近 **~100%(RL 天花板)**。到了 ⇒ model-free UG-flow
   成立,再上 source→target(那时 UG 用 target-adapting 的 chunk-Q,而非 dynamics)。
   到不了 ⇒ 诊断(见 §7)。

## 7. 会卡在哪(预警)

- **只做 BC 摊销** → 卡在 teacher。必须靠 UG 采集 + AWR 真把 Q 增益写回 flow
  (`actor_awr=1`)。
- **chunk-Q 过估计/发散** → min-ensemble + LN + clamp[0,1];必要时提高 pessimism。
- **UG 是局部近似 greedy**,不保证全局 Q-max;靠 online TD 让 Q 在真 reward 上持续
  变好 + 反复改进来爬。正因为**不 argmax learned model**,才没有 SimDist 的
  exploitation 天花板。
- **稀疏 reward 冷启动**:source flow base 已能偶尔成功(有 reward 信号)+ chunk-Q
  的 `γ^h` 快回传,应能起;起不来可 Q 离线 warm-start 或 demo 锚定。
