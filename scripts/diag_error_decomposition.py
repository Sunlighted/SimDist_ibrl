"""Which part of the planner's scoring error actually hurts? (§27 的枢纽)

    eps(a) = V_hat(z_H^imag(a)) - y(a)          [y = V_hat(encode(s_H^real)), 已验证 100/100]
           = eps_bar(s0)  +  eps_tilde(s0, a)
             ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^
             集内常数         集内波动

argmax 对 eps_bar **完全免疫**（常数不改排序）:
    性能损失 <= 2 * sup_a |eps_tilde(a)|

⇒ 模型"准不准"(E||delta||^2) 不是重点; 重点是**误差是否依赖于动作**。
  一个系统性偏 1mm、但对所有候选偏得一样的模型, planning 用它是完美的。
  而 MSE 训练优化的恰恰是前者, 完全不管后者 —— objective mismatch 的精确形态。

判决性指标:
    SNR = Var_a[y] / Var_a[eps_tilde]     argmax 有效 <=> SNR >> 1
    harmless_frac = Var_s[eps_bar] / Var_total[eps]

  harmless_frac 高 (eps 主要是系统性偏移) => 有救: 换一个只惩罚 eps_tilde 的模型目标
        min_f E_s [ Var_a ( V_hat(f^H(s,a)) - y ) ]   (允许任意大的 bias, 只要一致)
  harmless_frac 低 (eps 主要是动作依赖)   => planning 在此任务上数学上死路一条
"""
import os
import sys
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/home/Chenyang/ibrl")

NEEDLE = slice(9, 12)
RING = slice(23, 26)
CONTACT = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/claude-1003/-home-Chenyang-ibrl/"
                                       "08677077-8099-4d03-9973-403b524caa19/scratchpad/calib_data2.pt")
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    args = ap.parse_args()

    d = torch.load(args.cache)
    Z, Y, sets, S0 = d["Z"].cuda(), d["Y"].cuda(), d["sets"], d["S0"].cuda()
    print(f"[data] {len(Y)} candidates / {len(sets)} sets")

    # V_hat on the IMAGINED latent = what the planner scores today
    import pyrallis
    from train_simdist_adapt import AdaptConfig, AdaptWorkspace
    cfg = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/edec_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    model = ws.agent.to("cuda").model
    with torch.no_grad():
        V = model.predict_value(Z, target=False).view(-1)

    EPS = (V - Y).cpu().numpy()
    y = Y.cpu().numpy()
    b = np.cumsum([0] + list(sets))
    s0 = S0.cpu().numpy()

    rows = []
    for k in range(len(sets)):
        sl = slice(b[k], b[k + 1])
        e, yy = EPS[sl], y[sl]
        nrd = np.linalg.norm(s0[b[k], NEEDLE] - s0[b[k], RING])
        rows.append(dict(
            eps_bar=e.mean(),                 # 集内常数 (无害)
            var_eps_tilde=e.var(),            # 集内波动 (唯一有害的)
            var_y=yy.var(),                   # 候选间真实价值差异 (信号)
            sp_stock=spearmanr(-e + yy, yy).correlation,   # = V 的集内排序
            sp_ideal=spearmanr(yy + e.mean(), yy).correlation,  # 只留 eps_bar => 必须 =1
            contact=nrd < CONTACT,
        ))

    def report(name, sub):
        eb = np.array([r["eps_bar"] for r in sub])
        vt = np.array([r["var_eps_tilde"] for r in sub])
        vy = np.array([r["var_y"] for r in sub])
        var_bar = eb.var()                       # 跨集: Var_s[eps_bar]
        var_tilde = vt.mean()                    # 集内: E_s[Var_a[eps_tilde]]
        harmless = var_bar / (var_bar + var_tilde)
        snr = vy.mean() / max(var_tilde, 1e-12)
        rho_pred = np.sqrt(snr / (1 + snr))      # 线性高斯下的集内 Spearman 预测
        sp = np.nanmean([r["sp_stock"] for r in sub])
        spi = np.nanmean([r["sp_ideal"] for r in sub])
        print(f"\n=== {name}  ({len(sub)} sets)")
        print(f"  eps 分解:  Var_s[eps_bar] = {var_bar:.5f}   E_s[Var_a[eps_tilde]] = {var_tilde:.5f}")
        print(f"  ★ 无害占比 (系统性偏移):  {100*harmless:.1f}%     有害占比 (动作依赖): {100*(1-harmless):.1f}%")
        print(f"  信号 E_s[Var_a[y]] = {vy.mean():.5f}")
        print(f"  ★ SNR = Var[y]/Var[eps_tilde] = {snr:.3f}   (argmax 有效需 >> 1)")
        print(f"    -> 预测集内 Spearman = sqrt(SNR/(1+SNR)) = {rho_pred:.3f}   实测 = {sp:.3f}")
        print(f"    sanity: 只留 eps_bar 时的集内 Spearman = {spi:.3f} (必须 = 1.000)")

    report("ALL", rows)
    report("free-space", [r for r in rows if not r["contact"]])
    report("CONTACT", [r for r in rows if r["contact"]])

    print("\n===== 判读 =====")
    eb = np.array([r["eps_bar"] for r in rows]); vt = np.array([r["var_eps_tilde"] for r in rows])
    h = eb.var() / (eb.var() + vt.mean())
    if h > 0.6:
        print(f"  无害占比 {100*h:.0f}% > 60% ⇒ **有救**: eps 主要是系统性偏移。换目标")
        print(f"     min_f E_s[ Var_a( V_hat(f^H(s,a)) - y ) ]  (只惩罚动作依赖的误差, 允许任意 bias)")
    else:
        print(f"  无害占比 {100*h:.0f}% < 60% ⇒ **planning 在此任务上数学上死路**: 误差主要是动作依赖的,")
        print(f"     而那正是接触动力学的本质 (不同动作 -> 不同接触模式 -> 模型对每种错得不一样)。")


if __name__ == "__main__":
    main()
