"""阶段1 go/no-go: 用 base 的 latent value V_src 给合成 transition 做 bootstrap,
会不会毒化 residual 的 critic?

关键: 合成 target y = gamma * V_src(z')。这只在 V_src 和 residual 自己的价值
V_res 既 (1) 排序一致 又 (2) SCALE 一致 时才安全 —— TD target 吃绝对值,
排序对但 scale 错 (V_src 高估) 会把 critic 拉离真实尺度 (§前面 SGFT 崩的同款)。

在 residual_30k 的最终 ckpt + 其冻结 base 上, 对 demo 状态 s 同时算:
  V_src(s) = base.model.predict_value(base.model.encode(s))         [latent value, 想 bootstrap 用]
  V_res(s) = min_k critic(_norm(s), clip(base_action(s)+residual(s)))  [residual 自己的价值]

报: Spearman(排序) + 均值/std/回归斜率(scale) + 按 steps_to_go 分桶对照。
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
    ap.add_argument("--ckpt", default="exps/mimicgen/threading_residual_30k_seed1/model0.pt")
    ap.add_argument("--demo_path",
                    default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    ap.add_argument("--num_data", type=int, default=20)
    args = ap.parse_args()

    import train_residual_simdist
    from scripts.verify_iql_critic import load_demos

    agent, _, _ = train_residual_simdist.load_model(args.ckpt, "cuda")
    agent.eval()
    base = agent._base_policy[0]
    wm = base.model
    log(f"[agent] loaded {args.ckpt}; base wm raw_state={getattr(wm,'raw_state',False)}")

    episodes = load_demos(args.demo_path, args.num_data)
    log(f"[demos] {len(episodes)} target demos")

    vsrc, vres, stg_all = [], [], []
    with torch.no_grad():
        for e in episodes:
            st = torch.from_numpy(e["states"]).cuda().float()   # [T+1, obs]
            T = st.shape[0] - 1
            # V_src: base latent value
            z = wm.encode(st)
            v_src = wm.predict_value(z, target=False).view(-1)   # [T+1]
            # V_res: residual's own value = min_Q(_norm(s), a_combined)
            s_norm = agent._norm(st)
            base_a = agent._base_action(st)                      # [T+1, act]
            res = agent.actor(s_norm, base_a).clamp(-agent.clip_scale, agent.clip_scale)
            a_comb = (base_a + res).clamp(-1.0, 1.0)
            q_all = agent.critic(s_norm, a_comb)                 # [K, T+1]
            v_res = agent._min_q(q_all).view(-1)                 # [T+1]
            vsrc.append(v_src.cpu().numpy())
            vres.append(v_res.cpu().numpy())
            stg_all.append(np.arange(T + 1)[::-1].astype(np.float32))

    Vs = np.concatenate(vsrc); Vr = np.concatenate(vres); STG = np.concatenate(stg_all)

    log("\n===== V_src (base latent) vs V_res (residual critic) =====")
    log(f"  n = {len(Vs)}")
    log(f"  V_src : mean {Vs.mean():.4f}  std {Vs.std():.4f}  range [{Vs.min():.3f}, {Vs.max():.3f}]")
    log(f"  V_res : mean {Vr.mean():.4f}  std {Vr.std():.4f}  range [{Vr.min():.3f}, {Vr.max():.3f}]")

    log("\n----- (1) 排序一致性 -----")
    sp = spearmanr(Vs, Vr).correlation
    log(f"  ★ Spearman(V_src, V_res) = {sp:+.4f}")
    log(f"    Sp(V_src, -stg) = {spearmanr(Vs,-STG).correlation:+.3f}   "
        f"Sp(V_res, -stg) = {spearmanr(Vr,-STG).correlation:+.3f}")

    log("\n----- (2) SCALE 一致性 -----")
    # 线性回归 V_res ~ a*V_src + b
    A = np.vstack([Vs, np.ones_like(Vs)]).T
    slope, intercept = np.linalg.lstsq(A, Vr, rcond=None)[0]
    log(f"  ★ V_res ≈ {slope:.3f} * V_src + {intercept:+.3f}   (理想: slope=1, intercept=0)")
    log(f"    均值差 V_src-V_res = {Vs.mean()-Vr.mean():+.4f}")
    log(f"    std 比 V_src/V_res = {Vs.std()/max(Vr.std(),1e-9):.3f}")

    log("\n----- 分桶对照 (steps_to_success) -----")
    log(f"  {'stg':>12} {'n':>6} {'V_src':>8} {'V_res':>8} {'diff':>8}")
    for lo, hi in [(0,5),(5,15),(15,30),(30,60),(60,100),(100,10000)]:
        m = (STG>=lo)&(STG<hi)
        if m.sum()>0:
            log(f"  [{lo:4d},{hi:5d}) {m.sum():>6} {Vs[m].mean():>8.4f} {Vr[m].mean():>8.4f} "
                f"{Vs[m].mean()-Vr[m].mean():>+8.4f}")

    log("\n  判读:")
    log("   排序一致(Sp>0.8) + scale一致(slope~1,intercept~0) => (a) latent value bootstrap 安全,免decode")
    log("   排序一致 但 scale 偏 (slope!=1)                    => 需先把 V_src 线性校准到 V_res 尺度再bootstrap")
    log("   排序不一致                                          => (a) 会毒化,必须走 (b) decode+residual-Q")
    os._exit(0)


if __name__ == "__main__":
    main()
