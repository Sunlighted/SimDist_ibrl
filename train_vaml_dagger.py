"""VAML + DAgger: online decision-aware dynamics adaptation for the SimDist planner.

WHY (see CLAUDE.md §27): open-loop decision-aware finetune崩了 4 次——把 dynamics 在
【当前】候选分布上修准(SNR ×7),planner 立刻走到新分布,argmax 主动钻新模型的高估点(正反馈,
打地鼠)。本脚本用两个机制抑制正反馈,这是数学上唯一还没被堵死的路:

  1. DAgger 累积 buffer:候选/标签不清空,让 dynamics 在"所有历史上 planner 想去过的地方"都保持
     准确,逐渐堵死可钻的洞。
  2. Soft update (tau~0.1):每轮 dynamics 只朝新解走一小步 theta<-(1-tau)theta_old+tau theta_new,
     让【分布漂移速度】慢于【模型修正速度】(信赖域思想)。

Loss = decision-aware VAML(只惩罚价值误差的 SET 内方差,argmax 免疫集内常数 eps_bar,§27)
       + 物理锚 lambda*MSE(next(z,a), sg encode(s'))  —— 防止 dynamics 被扭成"value 读数对但物理
       荒谬"的东西,那会毁掉 planner 的 policy-rollout proposal(proposal 是 MPPI 的命根子)。

标签 y = V_hat(encode(s_H^real)),来自反事实 sim rollout(sim2sim;真机拿不到,写进 limitation)。

每 K 轮 300 局评测。**预期收益有限**:天花板已被 fx_C+oracle 两个正交实验锁定为
f(模型误差/容错窗口),即使完美收敛也就 0.85->0.88~0.90(residual RL 已 0.96)。价值在于回答
"在线 decision-aware model learning 能否抑制 planning 的正反馈",结果正负都记。
"""
import os
import sys
import copy
import argparse

import numpy as np
import torch
import pyrallis
from scipy.stats import spearmanr

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.eval_oracle_dyn_mppi import ProbePool, restore  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


@torch.no_grad()
def collect_round(agent, main_env, pool, args):
    """用【当前】planner 跑 rollout,复现完整 CEM 的所有轮,冻结每个候选集,反事实 rollout 打标签。
    返回 list of dict(z0[obs], a[N,H,act], snext[N,obs], y[N])。z0 存 raw s0 而非 latent——
    因为 dynamics 每轮在变,latent 必须每次用【当前】encoder 现算。"""
    model, planner = agent.model, agent.planner
    p = planner.cfg
    sets = []
    ep_len = int(main_env.episode_length) if hasattr(main_env, "episode_length") else 400
    n_ep = 0
    for e in range(args.collect_eps):
        obs, _ = main_env.reset()
        for t in range(ep_len):
            if t % args.probe_every == 0:
                s0 = obs["state"]
                z0 = model.encode(s0.view(1, -1))
                base = planner._policy_rollout(z0, planner._num_base())
                base = base + torch.randn_like(base) * p.base_policy_std
                mean = torch.zeros(p.horizon, planner.action_dim, device="cuda")
                std = torch.full((p.horizon, planner.action_dim), p.init_std, device="cuda")
                nrand = p.num_samples - planner._num_base()
                sim_state = main_env.env.sim.get_state().flatten().copy()
                for it in range(p.iterations):                     # 完整 CEM 每一轮都采
                    noise = torch.randn(p.horizon, nrand, planner.action_dim, device="cuda")
                    cands = torch.cat([(mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1),
                                       base.clamp(-1, 1)], dim=1)   # [H, N, act]
                    N = cands.shape[1]
                    z = z0.expand(N, -1).clone()
                    for h in range(p.horizon):
                        z = model.next(z, cands[h])
                    real = pool.rollout(sim_state, cands)           # [H, N, obs] 真实
                    y = model.predict_value(model.encode(real[-1]), target=False).view(-1)
                    sets.append(dict(
                        s0=s0.view(1, -1).expand(N, -1).cpu(),      # [N, obs]
                        a=cands.permute(1, 0, 2).cpu(),             # [N, H, act]
                        snext=real[-1].cpu(),                       # [N, obs] 真实 s_H(用于物理锚)
                        y=y.cpu(),
                    ))
                    # CEM 前进(用当前模型的打分,复现 planner 行为)
                    ret = model.predict_value(z, target=False).view(-1) * planner.final_discount
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
        n_ep += 1
    return sets


def drift_metric(agent, new_sets, old_sets):
    """新一轮候选集相对旧 buffer 的分布漂移:新候选的真实 s_H 到旧 s_H 最近邻的平均距离。
    收敛的签名 = 这个量逐轮缩小(planner 找不到新地方去了)。"""
    if not old_sets:
        return float("nan")
    ns = torch.cat([d["snext"] for d in new_sets[-len(new_sets)//1:]])[:2000]
    os_ = torch.cat([d["snext"] for d in old_sets])[:20000]
    d = torch.cdist(ns, os_).min(dim=1)[0]
    return d.mean().item()


def train_round(agent, buffer, opt, args):
    """在【累积】buffer 上跑 T 步 decision-aware VAML + 物理锚,只更新 model.dynamics。"""
    model = agent.model
    H = args.horizon
    for _ in range(args.train_steps):
        idx = np.random.randint(0, len(buffer), args.sets_per_batch)
        vaml_l, anchor_l = [], []
        for k in idx:
            d = buffer[k]
            s0 = d["s0"].cuda(); a = d["a"].cuda(); y = d["y"].cuda(); snext = d["snext"].cuda()
            with torch.no_grad():
                z = model.encode(s0)                    # encoder 冻结
            z_roll = z
            for h in range(H):
                z_roll = model.next(z_roll, a[:, h])     # dynamics 训练
            v = model.predict_value(z_roll, target=False).view(-1)   # value 冻结
            eps = v - y
            vaml_l.append(eps.var())                     # ★ 只罚集内方差(argmax 免疫 eps_bar)
            with torch.no_grad():
                z_true = model.encode(snext)
            anchor_l.append(((z_roll - z_true) ** 2).mean())   # 物理锚
        loss = torch.stack(vaml_l).mean() + args.anchor * torch.stack(anchor_l).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 10.0)
        opt.step()
    return loss.item()


@torch.no_grad()
def val_snr(agent, buffer, n=200):
    """held-out(buffer 尾部)上的集内 SNR = Var_a[y] / Var_a[eps_tilde]。"""
    model = agent.model
    sub = buffer[-n:]
    vt, vy = [], []
    for d in sub:
        s0 = d["s0"].cuda(); a = d["a"].cuda(); y = d["y"].cuda()
        z = model.encode(s0)
        for h in range(agent.planner.cfg.horizon):
            z = model.next(z, a[:, h])
        eps = (model.predict_value(z, target=False).view(-1) - y).cpu().numpy()
        vt.append(eps.var()); vy.append(y.cpu().numpy().var())
    vt, vy = np.mean(vt), np.mean(vy)
    return vy / max(vt, 1e-9), vt


@torch.no_grad()
def evaluate(agent, env_params, n, seed):
    env = PixelRobosuite(**env_params)
    succ = []
    torch.manual_seed(seed); np.random.seed(seed)
    L = int(env_params.get("episode_length", 400))
    for i in range(n):
        obs, _ = env.reset(); s = 0.0
        for t in range(L):
            a = agent.act(obs, eval_mode=True)           # 尊重 use_planner
            obs, r, term, _, _ = env.step(a)
            if r > 0:
                s = 1.0
            if term:
                break
        succ.append(s)
    return float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--collect_eps", type=int, default=8)
    ap.add_argument("--probe_every", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--train_steps", type=int, default=1500)
    ap.add_argument("--sets_per_batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1, help="soft update: dyn <- (1-tau)old + tau new")
    ap.add_argument("--anchor", type=float, default=0.3, help="物理锚 MSE 权重")
    ap.add_argument("--buffer_max", type=int, default=40000, help="累积候选集上限(FIFO)")
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/vaml_ws"
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = args.num_samples, args.num_base, args.horizon
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    planner.discounts = (p.discount ** torch.arange(args.horizon, device="cuda")).view(args.horizon, 1)
    planner.final_discount = p.discount ** args.horizon

    ep = dict(ws.eval_env_params)
    main_env = PixelRobosuite(**ep); main_env.reset()
    pool = ProbePool(ep, args.probe_workers)

    # 只训 dynamics
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in model.dynamics.parameters():
        prm.requires_grad_(True)
    opt = torch.optim.Adam(model.dynamics.parameters(), lr=args.lr)

    base = evaluate(agent, ep, args.eval_episodes, args.seed)
    log(f"[BASELINE] stock dynamics MPPI = {base:.3f} (n={args.eval_episodes})")

    buffer = []
    for rnd in range(args.rounds):
        agent.cfg.use_planner = 1
        new_sets = collect_round(agent, main_env, pool, args)
        drift = drift_metric(agent, new_sets, buffer)
        buffer.extend(new_sets)
        if len(buffer) > args.buffer_max:
            buffer = buffer[-args.buffer_max:]
        snr_before, vt_before = val_snr(agent, buffer)

        # soft update: 训一份 dynamics 副本, 再朝它插值一小步
        dyn_old = copy.deepcopy(model.dynamics.state_dict())
        loss = train_round(agent, buffer, opt, args)
        dyn_new = model.dynamics.state_dict()
        for key in dyn_new:
            dyn_new[key] = (1 - args.tau) * dyn_old[key] + args.tau * dyn_new[key]
        model.dynamics.load_state_dict(dyn_new)

        snr_after, vt_after = val_snr(agent, buffer)
        log(f"[R{rnd}] buffer={len(buffer)} sets | drift={drift:.4f} | loss={loss:.5f} | "
            f"Var[eps_tilde] {vt_before:.5f}->{vt_after:.5f} | SNR {snr_before:.2f}->{snr_after:.2f}")

        if (rnd + 1) % args.eval_every == 0 or rnd == args.rounds - 1:
            sc = evaluate(agent, ep, args.eval_episodes, args.seed)
            log(f"[R{rnd}] ★ MPPI = {sc:.3f}   (baseline {base:.3f})")

    pool.close()
    log(f"\n===== DONE: baseline {base:.3f} =====")
    os._exit(0)


if __name__ == "__main__":
    main()
