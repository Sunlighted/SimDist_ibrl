"""判决性检验: source 域的 MPPI 能到 ~100%, 而 target 只有 0.85 —— 我的 SNR 理论解释得了吗?

理论预测 (§27):  argmax 的成败只由一个数决定
    SNR = Var_a[y] / Var_a[eps_tilde]
          候选间真实价值差异 / 模型误差中"随动作变化"的部分

source 与 target 的差异会同时推高分子、压低分母:
  * 分母↓: source 接触是"硬"的 (gripper kp 1000 vs 120, 针 7.6g vs 22.8g) => 接触模式更确定、
           更少滑移/分段切换 => 模型好学; 且 diverse_v2 就是 source 域数据, 训练分布完全匹配。
  * 分子↑: 环孔 16mm vs 13.6mm (每边余量 3mm vs 1.8mm) => 容差大 => "好动作"与"坏动作"的后果
           差别更明显, 不易被噪声淹没。

⇒ 预测: source 的 SNR 应显著 >> 1 (target 实测 1.01)。
⇒ 若成立: 理论在一个未调参的域上做出正确的定量预测, 结论升级为
     **planning 的成败不取决于"模型准不准", 而取决于 SNR = 任务容差信号 / 模型误差的动作依赖部分。**
     (这也解释了 TD-MPC2 在 DMControl 封神、在精密装配无人用。)
⇒ 若 source 的 SNR 也只有 ~1: 整套理论崩 —— 它解释不了 source 的 100%。

用的是 STAGE-1 的 world model (source dynamics + source value + source policy) 在 SOURCE env,
即"source MPPI"那个配置本身; 不是 target-adapted 的 ckpt。
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

import train_simdist  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.eval_oracle_dyn_mppi import ProbePool, restore  # noqa: E402

ERR = sys.stderr
NEEDLE = slice(9, 12)
RING = slice(23, 26)
CONTACT = 0.15


def log(*a):
    print(*a, file=ERR, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="exps/mimicgen/threading_simdist_diversev2_h5_seed1")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--probe_every", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ring_size_scale", type=float, default=1.0,
                    help="只改几何容差: 0.85 = target 的环孔(13.6mm), 其余物理仍是 source")
    ap.add_argument("--ring_friction_scale", type=float, default=1.0,
                    help="环壁摩擦: 2.0 = target 的值 (针会卡在鑫口)")
    ap.add_argument("--eval_only", type=int, default=0,
                    help="只测成绩, 跳过 SNR 分解 (用于 proposal 消融)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = pyrallis.load(train_simdist.MainConfig,  # type: ignore
                        open(os.path.join(args.stage1, "cfg.yaml")))
    cfg.use_wb = 0
    cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/ssnr_ws"
    cfg.simdist.use_planner = 1                      # stage-1 训练时是 policy-only, 这里要 MPPI
    os.makedirs(cfg.save_dir, exist_ok=True)
    pyrallis.dump(cfg, open(os.path.join(cfg.save_dir, "cfg.yaml"), "w"))  # Workspace 会读它
    ws = train_simdist.Workspace(cfg, from_main=False)
    ws.agent.load_state_dict(torch.load(os.path.join(args.stage1, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner

    # 与 target 实验完全相同的缩水配方 (唯一变量是"域")
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = args.num_samples, args.num_base, args.horizon
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    p.init_std, p.min_std, p.temperature, p.momentum, p.base_policy_std = 1.0, 0.05, 0.4, 0.0, 0.1
    planner.discounts = (p.discount ** torch.arange(args.horizon, device="cuda")).view(args.horizon, 1)
    planner.final_discount = p.discount ** args.horizon
    agent.cfg.use_planner = 1

    # ★ SOURCE env: Workspace 只暴露 eval_env_params(=TARGET), source env 只有对象。
    # 复制 eval params 再把 domain_shift 换成 train 的 (mode=none) 即得 source env。
    ep = dict(ws.eval_env_params)
    ep["domain_shift"] = cfg.train_domain_shift          # <- SOURCE (mode=none)
    ep["hand_init_perturb_joint_sigma"] = 0.1
    ep["hand_init_perturb"] = None
    # ★ 单变量: 在 SOURCE 物理上只改环的几何/摩擦 (其余 friction/mass/gripper_kp 全 =1.0)
    if args.ring_size_scale != 1.0 or args.ring_friction_scale != 1.0:
        from env.domain_shift import DomainShiftConfig
        ep["domain_shift"] = DomainShiftConfig(
            mode="fixed",
            ring_size_scale=args.ring_size_scale,
            ring_friction_scale=args.ring_friction_scale,
        )   # 其余字段默认 = 1.0 / 0 => 物理仍是 source, 只有环变了
    log(f"[setup] SOURCE env (domain_shift.mode="
        f"{getattr(ep.get('domain_shift', None), 'mode', '?')}) | H={p.horizon} N={p.num_samples}")

    main_env = PixelRobosuite(**ep)
    probe_env = PixelRobosuite(**ep); probe_env.reset()
    pool = ProbePool(ep, args.probe_workers)

    # ---- source MPPI 的成绩 (先确认它真的 ~100%)
    @torch.no_grad()
    def evaluate(n):
        e2 = PixelRobosuite(**ep)
        succ = []
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        L = int(ep.get("episode_length", 400))
        for i in range(n):
            obs, _ = e2.reset(); s = 0.0
            for t in range(L):
                # ★ BUG FIX: 必须走 agent.act (它尊重 cfg.use_planner)。之前这里硬编码
                # planner.plan, 于是 "policy 裸跑" 实际上还是 MPPI => 净贡献恒等于 0,
                # 那个"完美的零"是 bug 的签名, 不是发现。
                a = agent.act(obs, eval_mode=True)
                obs, r, term, _, _ = e2.step(a)
                if r > 0:
                    s = 1.0
                if term:
                    break
            succ.append(s)
            if (i + 1) % 10 == 0:
                log(f"    eval {i+1}/{n}: running {np.mean(succ):.3f}")
        return float(np.mean(succ))

    # ★ 必须同时测 policy 裸跑: 缩小环孔会让 policy 本身也变差,
    #   只有 (MPPI - policy) 才是 planner 的净贡献
    agent.cfg.use_planner = 0
    pol = evaluate(args.eval_episodes)
    agent.cfg.use_planner = 1
    score = evaluate(args.eval_episodes)
    log(f"\n[ring_size={args.ring_size_scale} ring_fric={args.ring_friction_scale}]"
        f"  policy 裸跑 = {pol:.3f}   MPPI = {score:.3f}   "
        f"★ planner 净贡献 = {score - pol:+.3f}   (n={args.eval_episodes})")
    if args.eval_only:
        pool.close()
        os._exit(0)

    # ---- 收集候选 + 真实 SOURCE rollout, 做 eps 分解
    rows = []
    L = int(ep.get("episode_length", 400))
    with torch.no_grad():
        for e in range(args.episodes):
            obs, _ = main_env.reset()
            for t in range(L):
                if t % args.probe_every == 0:
                    s0 = obs["state"]
                    z0 = model.encode(s0.view(1, -1))
                    base = planner._policy_rollout(z0, planner._num_base())
                    base = base + torch.randn_like(base) * p.base_policy_std
                    mean = torch.zeros(p.horizon, planner.action_dim, device="cuda")
                    std = torch.full((p.horizon, planner.action_dim), p.init_std, device="cuda")
                    nrand = p.num_samples - planner._num_base()
                    sim_state = main_env.env.sim.get_state().flatten().copy()
                    for it in range(p.iterations):          # 完整 CEM
                        noise = torch.randn(p.horizon, nrand, planner.action_dim, device="cuda")
                        cands = torch.cat([(mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1),
                                           base.clamp(-1, 1)], dim=1)
                        N = cands.shape[1]
                        z = z0.expand(N, -1).clone()
                        for h in range(p.horizon):
                            z = model.next(z, cands[h])
                        v = model.predict_value(z, target=False).view(-1)
                        real = pool.rollout(sim_state, cands)
                        y = model.predict_value(model.encode(real[-1]), target=False).view(-1)
                        eps = (v - y).cpu().numpy()
                        yy = y.cpu().numpy()
                        nrd = float(np.linalg.norm(
                            s0[NEEDLE].cpu().numpy() - s0[RING].cpu().numpy()))
                        rows.append(dict(eps_bar=eps.mean(), var_eps_tilde=eps.var(),
                                         var_y=yy.var(), contact=nrd < CONTACT,
                                         sp=spearmanr(v.cpu().numpy(), yy).correlation))
                        ret = v * planner.final_discount
                        ev, ei = torch.topk(ret, p.num_elites)
                        ea = cands[:, ei]
                        w = torch.exp(p.temperature * (ev - ev.max())); w = w / (w.sum() + 1e-9)
                        nm = (w.view(1, -1, 1) * ea).sum(1)
                        ns = torch.sqrt((w.view(1, -1, 1) * (ea - nm.unsqueeze(1)) ** 2).sum(1))
                        mean = p.momentum * mean + (1 - p.momentum) * nm
                        std = ns.clamp(min=p.min_std)
                a = planner.plan(obs["state"], eval_mode=True)
                obs, _, term, _, _ = main_env.step(a.cpu())
                if term:
                    break
            log(f"  ep{e}: {len(rows)} sets")
    pool.close()

    def report(name, sub):
        eb = np.array([r["eps_bar"] for r in sub]); vt = np.array([r["var_eps_tilde"] for r in sub])
        vy = np.array([r["var_y"] for r in sub]); sp = np.nanmean([r["sp"] for r in sub])
        vb, vtm, vym = eb.var(), vt.mean(), vy.mean()
        snr = vym / max(vtm, 1e-12)
        log(f"\n=== {name} ({len(sub)} sets)")
        log(f"  Var[eps_bar]={vb:.5f}  Var[eps_tilde]={vtm:.5f}  Var[y]={vym:.5f}")
        log(f"  ★ SNR = {snr:.3f}    预测集内 Spearman = {np.sqrt(snr/(1+snr)):.3f}   实测 = {sp:.3f}")

    report("SOURCE ALL", rows)
    report("SOURCE free-space", [r for r in rows if not r["contact"]])
    report("SOURCE CONTACT", [r for r in rows if r["contact"]])
    log(f"\n===== 对比 TARGET (同配方 H=5): SNR = 1.012, 成绩 0.850 =====")
    log(f"       SOURCE: 成绩 {score:.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
