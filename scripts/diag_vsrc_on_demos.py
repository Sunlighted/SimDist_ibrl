"""V_src 在 TARGET demo 状态上到底平不平?

背景: 我从 train.log 的 `sgft/v_source` 读出 "min 0.53 / max 0.63" 就断言
"V_src 在 target 上几乎是常数"。但那个 stat 记的是 `v_s.mean().item()` ——
**每个点是一个 batch(256) 的均值**, min/max 是 25000 个 batch 均值的 min/max,
被 CLT 压得极窄。逐状态的真实分布被平均掉了。这里直接测。

测两件事:
  1. 逐状态 V_src 的分布 (std / 分位数) —— 到底有多大跨度
  2. Spearman(V_src, -steps_to_success) —— V 是否真的把"离成功近"排在前面
     (这才是 shaping 有没有信息的判据; 平不平只是表象)
  3. 沿 demo 轨迹的实际 shaping 项 gamma^n*V(s')-V(s) 的符号/量级

输出走 stderr (train_rl 的 Logger 会劫持 stdout)。
"""
import os
import sys
import argparse

import numpy as np
import torch
import pyrallis
from scipy.stats import spearmanr

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_agent",
                    default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    ap.add_argument("--demo_path",
                    default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    ap.add_argument("--num_data", type=int, default=20)
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--discount", type=float, default=0.99)
    args = ap.parse_args()

    import train_rl
    from scripts.verify_iql_critic import load_demos
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    # 用 source run 自己的 cfg 重建 agent (arch + obs space 必须与 ckpt 一致)
    src_dir = os.path.dirname(args.source_agent)
    src_cfg: train_rl.MainConfig = pyrallis.load(
        train_rl.MainConfig, open(os.path.join(src_dir, "cfg.yaml"), "r"))  # type: ignore

    # load_frozen_agent 需要一个 env 只为读 obs/action 维度; 建一个最小的
    env = PixelRobosuite(
        env_name=src_cfg.task_name, robots=src_cfg.robots,
        episode_length=src_cfg.episode_length, reward_shaping=False,
        image_size=src_cfg.image_size, rl_image_size=src_cfg.rl_image_size,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=src_cfg.state_stack, prop_stack=1,
        domain_shift=DomainShiftConfig(mode="none"),
    )
    agent = train_rl.load_frozen_agent(args.source_agent, env, src_cfg, "cuda")
    log(f"[agent] loaded {args.source_agent} (act_method={agent.cfg.act_method})")

    episodes = load_demos(args.demo_path, args.num_data)
    log(f"[demos] {len(episodes)} target demos")

    # ---- 逐状态 V_src + steps_to_go
    all_v, all_stg, per_demo_rho = [], [], []
    shaping_all = []
    with torch.no_grad():
        for e in episodes:
            st = torch.from_numpy(e["states"]).cuda()          # [T+1, obs]
            v = agent.value({"state": st}).view(-1).cpu().numpy()   # [T+1]
            T = len(e["actions"])
            stg = np.arange(T + 1)[::-1].astype(np.float32)     # steps_to_success
            all_v.append(v); all_stg.append(stg)
            r = spearmanr(v, -stg).correlation
            if not np.isnan(r):
                per_demo_rho.append(r)
            # 沿轨迹的 shaping: gamma^n * V(s_{t+n}) - V(s_t)
            disc = args.discount ** args.nstep
            n = args.nstep
            if T > n:
                sh = disc * v[n:T + 1] - v[0:T + 1 - n]
                shaping_all.append(sh)

    V = np.concatenate(all_v)
    STG = np.concatenate(all_stg)
    SH = np.concatenate(shaping_all) if shaping_all else np.array([0.0])

    log("\n===== 1) 逐状态 V_src 分布 (demo 状态) =====")
    log(f"  n_states = {len(V)}")
    log(f"  mean {V.mean():.4f}   std {V.std():.4f}")
    log(f"  min {V.min():.4f}   p5 {np.percentile(V,5):.4f}   p25 {np.percentile(V,25):.4f}   "
        f"p50 {np.percentile(V,50):.4f}   p75 {np.percentile(V,75):.4f}   "
        f"p95 {np.percentile(V,95):.4f}   max {V.max():.4f}")
    log(f"  ★ 逐状态跨度 (p95-p5) = {np.percentile(V,95)-np.percentile(V,5):.4f}")
    log(f"    [对照] train.log 里 batch 均值的跨度 ≈ 0.10 (被 CLT 压窄的那个数)")

    log("\n===== 2) V_src 是否随进度单调? (shaping 有无信息的判据) =====")
    rho_all = spearmanr(V, -STG).correlation
    log(f"  ★ 全局 Spearman(V_src, -steps_to_success) = {rho_all:+.4f}")
    log(f"  ★ 每 demo 内 Spearman 均值 = {np.mean(per_demo_rho):+.4f}  "
        f"(std {np.std(per_demo_rho):.3f}, n={len(per_demo_rho)})")
    log(f"    >0 = V 认得出进度(shaping 有方向); ~0 = 无信息; <0 = 反向")

    # 按 steps_to_go 分桶看 V 的走势
    log("\n  V_src vs steps_to_success (分桶均值):")
    for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 10000)]:
        m = (STG >= lo) & (STG < hi)
        if m.sum() > 0:
            log(f"    steps_to_go [{lo:4d},{hi:5d}): n={m.sum():5d}  V = {V[m].mean():.4f}")

    log("\n===== 3) 沿 demo 轨迹的实际 shaping 项 =====")
    log(f"  gamma^{args.nstep} * V(s') - V(s):  mean {SH.mean():+.5f}   std {SH.std():.5f}")
    log(f"    p5 {np.percentile(SH,5):+.5f}   p50 {np.percentile(SH,50):+.5f}   "
        f"p95 {np.percentile(SH,95):+.5f}")
    log(f"    正的比例 = {(SH>0).mean()*100:.1f}%")
    log(f"    [对照] 训练 replay 里实测 sgft/shaping ≈ -0.026 (batch 均值)")
    log("\n  判读: 若 demo 上 shaping 显著为正 / Spearman 强正 => V_src 有信息,")
    log("        我说的'常数'是错的; 若接近 0 => 信息确实很弱。")
    os._exit(0)


if __name__ == "__main__":
    main()
