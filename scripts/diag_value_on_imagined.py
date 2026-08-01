"""GO/NO-GO: SimDist 的 value 头在"想象 latent"上还准不准?

目标 pipeline: 在 SimDist latent 里想象 rollout (z -> next(z,a) -> z'), 用 value 头
给合成 transition 提供 bootstrap 学习信号 (y = gamma*V(z')), 喂 model-free RL。
前提: V 在 IMAGINED z' 上的排序必须仍然可靠。§17 只测了 demo 真实状态 encode 的 z;
§27 证明想象 latent 的几何会漂 (var-finetune 2.15 灾难)。所以这里直接测想象 z 上的 V。

方法 (沿真实 TARGET demo 轨迹, 真实动作序列):
  z0 = encode(s_0)
  想象:   z_k^imag = next(...next(z0, a_0)..., a_{k-1})       [纯 latent rollout]
  真值:   z_k^real = encode(s_k)                              [真实状态 encode]
  对比 V(z_k^imag) vs V(z_k^real), k=1..K:
    - Spearman(V^imag, V^real)          排序是否保持 (bootstrap 只需排序对)
    - Spearman(V^imag, -steps_to_go)    V^imag 是否仍认得"离成功近" (真正的判据)
    - |V^imag - V^real| 绝对漂移         标定漂多少 (真实 reward 能锚回多少)
    - ||z^imag - z^real||               几何漂移 (旁证, §27 的量)

判读:
  Spearman(V^imag,-stg) 强正 (>0.8) 且 随 k 衰减慢  => 想象 value 可用 => 建 pipeline
  排序快速崩 (k=2,3 就 <0.5)                        => 想象 value 中毒 => 缩短 horizon 或先修几何
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
    ap.add_argument("--stage1", default="exps/mimicgen/threading_simdist_diversev2_h5_seed1",
                    help="stage-1 SimDist run dir (for cfg + world model arch)")
    ap.add_argument("--adapt_ckpt",
                    default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1_frozen094/model0.pt",
                    help="the adapted (target) world-model ckpt = the 0.94 base")
    ap.add_argument("--demo_path",
                    default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    ap.add_argument("--num_data", type=int, default=20)
    ap.add_argument("--K", type=int, default=5, help="max imagination horizon")
    args = ap.parse_args()

    import train_simdist
    from scripts.verify_iql_critic import load_demos

    cfg = pyrallis.load(train_simdist.MainConfig,  # type: ignore
                        open(os.path.join(args.stage1, "cfg.yaml")))
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/403b1f91-4a38-4d1d-99d2-5d860804db5f/scratchpad/vimag_ws"
    os.makedirs(cfg.save_dir, exist_ok=True)
    pyrallis.dump(cfg, open(os.path.join(cfg.save_dir, "cfg.yaml"), "w"))
    ws = train_simdist.Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(args.adapt_ckpt))
    model = ws.agent.model.to("cuda").eval()
    log(f"[model] loaded adapt ckpt {args.adapt_ckpt}")

    episodes = load_demos(args.demo_path, args.num_data)
    log(f"[demos] {len(episodes)} target demos, K={args.K}")

    # 沿每条 demo 轨迹, 对每个起点 t 想象 K 步, 收集 (k, V_imag, V_real, geom, stg)
    # 用真实 demo 动作序列驱动想象 (这就是"用真实策略采的合成 transition"的 value 保真度)
    rec = {k: dict(vi=[], vr=[], geom=[], stg=[]) for k in range(1, args.K + 1)}
    with torch.no_grad():
        for e in episodes:
            st = torch.from_numpy(e["states"]).cuda()      # [T+1, obs]
            ac = torch.from_numpy(e["actions"]).cuda()     # [T, act]
            T = ac.shape[0]
            z_real_all = model.encode(st)                  # [T+1, zdim]
            # 从每个起点 t0 想象 K 步
            for t0 in range(T):
                z = z_real_all[t0:t0 + 1]                  # [1, zdim]
                for k in range(1, args.K + 1):
                    if t0 + k > T:
                        break
                    z = model.next(z, ac[t0 + k - 1:t0 + k])   # imagine one step
                    z_real = z_real_all[t0 + k:t0 + k + 1]
                    vi = model.predict_value(z, target=False).item()
                    vr = model.predict_value(z_real, target=False).item()
                    geom = (z - z_real).norm().item()
                    stg = T - (t0 + k)                     # steps to success
                    rec[k]["vi"].append(vi); rec[k]["vr"].append(vr)
                    rec[k]["geom"].append(geom); rec[k]["stg"].append(stg)

    log("\n===== 想象 latent 上 value 的可靠性 (沿真实 demo 动作) =====")
    log(f"  {'k':>2} {'n':>6} {'Sp(Vi,Vr)':>10} {'Sp(Vi,-stg)':>12} {'Sp(Vr,-stg)':>12} "
        f"{'|Vi-Vr|':>9} {'geom drift':>11} {'Vi_std':>7}")
    for k in range(1, args.K + 1):
        r = rec[k]
        if len(r["vi"]) < 10:
            continue
        vi = np.array(r["vi"]); vr = np.array(r["vr"])
        stg = np.array(r["stg"], dtype=np.float32); geom = np.array(r["geom"])
        sp_ivr = spearmanr(vi, vr).correlation
        sp_istg = spearmanr(vi, -stg).correlation
        sp_rstg = spearmanr(vr, -stg).correlation
        drift = np.abs(vi - vr).mean()
        log(f"  {k:>2} {len(vi):>6} {sp_ivr:>10.3f} {sp_istg:>12.3f} {sp_rstg:>12.3f} "
            f"{drift:>9.4f} {geom.mean():>11.4f} {vi.std():>7.4f}")

    log("\n  Sp(Vi,-stg) = 想象value 认不认得进度 (bootstrap 有没有用的直接判据)")
    log("  Sp(Vr,-stg) = 真实value 的上限 (§17 说 ~0.96)")
    log("  判读: Sp(Vi,-stg) 随 k 衰减慢 & k=1 就强 => 想象 value 可 bootstrap => 建 pipeline")
    log("        快速崩 => 想象 value 中毒 => 缩 horizon / 先修几何")
    os._exit(0)


if __name__ == "__main__":
    main()
