"""能不能用 V_src(z) > c 合成想象 transition 的稀疏奖励锚点?

想法: 想象 transition 缺真实 r=1 锚点。用 base value 过阈值 r_hat(z)=1[V_src(z)>c]
造一个伪锚点。成败取决于: V_src>c 这个事件能否干净地对应"真实成功", 否则
假阳性(V_src>c 但没成功)会喂给 critic 假的 r=1 -> 高估 -> reward hack(§5c)。

在含成功+失败的真实 base rollout 上测(不是纯成功 demo, 才看得到假阳性):
  正类定义两种:
    (a) 严格 r=1 那一步(真实稀疏奖励发生的瞬间)
    (b) 成功轨迹的最后 K 步(= "接近成功", 想象奖励更想捕捉这个)
  扫阈值 c: precision(V_src>c 里有多少真是正类)/ recall / FPR(致命)
  报 ROC-AUC + 最佳 c

输出走 stderr(Workspace 的 Logger 劫持 stdout)。
"""
import os
import sys
import argparse

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", default="exps/mimicgen/threading_residual_latent_h5base_seed1")
    ap.add_argument("--rollouts", default="exps/mimicgen/threading_residual_latent_h5base_seed1/iql_critic_verify/rollouts.npz")
    ap.add_argument("--last_k", type=int, default=5, help="正类(b): 成功轨迹最后 K 步")
    args = ap.parse_args()

    import train_residual_simdist as T
    cfg = pyrallis.load(T.MainConfig, open(os.path.join(args.base_run, "cfg.yaml")))  # type: ignore
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/403b1f91-4a38-4d1d-99d2-5d860804db5f/scratchpad/vthr_ws"
    os.makedirs(cfg.save_dir, exist_ok=True)
    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))
    ws = T.Workspace(cfg, from_main=False)
    base = ws.agent._base_policy[0]
    wm = base.model
    log(f"[base] {cfg.base_adapt_ckpt}")

    eps = np.load(args.rollouts, allow_pickle=True)["episodes"]
    log(f"[rollouts] {len(eps)} episodes")

    # 逐步收集 V_src(z_t) + 标签
    V, is_rstep, is_lastk, is_succ_ep = [], [], [], []
    with torch.no_grad():
        for e in eps:
            st = torch.from_numpy(e["states"]).cuda().float()   # [T+1, obs]
            rw = e["rewards"]                                    # [T]
            succ = bool(e["success"])
            T_ = len(rw)
            v = wm.predict_value(wm.encode(st), target=False).view(-1).cpu().numpy()  # [T+1]
            # 对 s_0..s_{T_-1}(有对应 reward 的步)
            for t in range(T_):
                V.append(v[t])
                is_rstep.append(1.0 if rw[t] > 0 else 0.0)          # (a) 真实 r=1 那步
                # (b) 成功轨迹的最后 last_k 步
                is_lastk.append(1.0 if (succ and t >= T_ - args.last_k) else 0.0)
                is_succ_ep.append(1.0 if succ else 0.0)
    V = np.array(V); is_rstep = np.array(is_rstep); is_lastk = np.array(is_lastk)
    is_succ_ep = np.array(is_succ_ep)
    log(f"[data] {len(V)} steps | r=1 steps: {int(is_rstep.sum())} | "
        f"last-{args.last_k} of success: {int(is_lastk.sum())} | "
        f"steps in success eps: {int(is_succ_ep.sum())} / fail eps steps: {int((1-is_succ_ep).sum())}")

    log(f"\n[V_src 分布] all: mean {V.mean():.3f} p50 {np.median(V):.3f} "
        f"p90 {np.percentile(V,90):.3f} p99 {np.percentile(V,99):.3f} max {V.max():.3f}")

    def roc_and_sweep(name, pos):
        pos = pos.astype(bool)
        npos, nneg = pos.sum(), (~pos).sum()
        if npos == 0:
            log(f"\n=== {name}: 无正类, skip"); return
        vp, vn = V[pos], V[~pos]
        log(f"\n=== {name}  (正类 {npos}, 负类 {nneg}) ===")
        log(f"  V_src on POS: mean {vp.mean():.3f} p10 {np.percentile(vp,10):.3f} p50 {np.median(vp):.3f}")
        log(f"  V_src on NEG: mean {vn.mean():.3f} p90 {np.percentile(vn,90):.3f} p99 {np.percentile(vn,99):.3f}")
        # ROC-AUC (rank-based)
        from scipy.stats import rankdata
        r = rankdata(V)
        auc = (r[pos].sum() - npos*(npos+1)/2) / (npos*nneg)
        log(f"  ★ ROC-AUC = {auc:.4f}   (1.0=完美可分, 0.5=瞎猜)")
        # 扫阈值
        log(f"  {'c':>6} {'precision':>10} {'recall':>8} {'FPR(假阳)':>10} {'#pred+':>8}")
        for c in [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
            pred = V > c
            tp = (pred & pos).sum(); fp = (pred & ~pos).sum()
            prec = tp/max(pred.sum(),1); rec = tp/npos; fpr = fp/max(nneg,1)
            log(f"  {c:>6.3f} {prec:>10.3f} {rec:>8.3f} {fpr:>10.4f} {int(pred.sum()):>8}")

    roc_and_sweep("(a) 严格 r=1 那步", is_rstep)
    roc_and_sweep(f"(b) 成功轨迹最后 {args.last_k} 步", is_lastk)

    log("\n  判读:")
    log("   若某个 c 能 precision高(假阳低 FPR) + recall还行 => 阈值奖励可行, 造锚点不污染")
    log("   若 precision 一直低 / FPR 高 => V_src>c 分不开成功与非成功 => 会喂假 r=1 -> 毒化, 不可行")
    os._exit(0)


if __name__ == "__main__":
    main()
