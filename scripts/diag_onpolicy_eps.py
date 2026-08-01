"""候选一的判决实验: var-finetune 在固定 cache 上让 Var[eps_tilde] 降 7x,
成绩却掉 —— 因为 val 指标量在 STOCK planner 的固定分布上, 而成绩量在
每个模型自己 CEM 搜出的 ON-POLICY 分布上。dynamics 一改, 搜的分布就漂走。

对 {stock, var} 两个 dynamics, 在各自 planner 实际搜出的候选上重算 eps, 报:
    Var[eps_tilde]   集内波动 (var-loss 直接压的; cache 上它赢)
    sup|eps_tilde|   argmax 的真正约束 (性能损失 <= 2*sup); var-loss 完全不管
    p99|eps_tilde|
    Var_s[eps_bar]   bias 的逐状态漂移 (var-loss 放开 bias => 绝对水平失锚?)
    SNR = Var[y]/Var[eps_tilde]

锐化的候选一预测: var 在自己的 on-policy 分布上, Var[eps_tilde] 的优势缩水甚至反转,
且 sup|eps_tilde| / Var_s[eps_bar] 不降反升 —— "无锚绝对水平" 的签名。

eps(a) = V_hat(f^H(phi(s0), a)) - V_hat(encode(s_H^real(a)))   [同一个 value head]
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

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.eval_oracle_dyn_mppi import ProbePool  # noqa: E402

ERR = sys.stderr
NEEDLE = slice(9, 12)
RING = slice(23, 26)
CONTACT = 0.15
SCR = "/tmp/claude-1003/-home-Chenyang-ibrl/403b1f91-4a38-4d1d-99d2-5d860804db5f/scratchpad"


def log(*a):
    print(*a, file=ERR, flush=True)


def set_metrics(eps_list, vy_list):
    """eps_list: 每组的 eps (np array). 返回该分布的判决指标。"""
    vt = np.mean([e.var() for e in eps_list])                       # 集内波动 (mean)
    sup = np.mean([np.abs(e - e.mean()).max() for e in eps_list])   # 每组 sup, 取均值
    p99 = np.mean([np.percentile(np.abs(e - e.mean()), 99) for e in eps_list])
    eb = np.array([e.mean() for e in eps_list])
    var_bar = eb.var()                                              # bias 逐状态漂移
    vy = np.mean(vy_list)
    snr = vy / max(vt, 1e-12)
    return dict(vt=vt, sup=sup, p99=p99, var_bar=var_bar, vy=vy, snr=snr)


def report(name, m):
    log(f"  [{name:22s}] Var[eps~]={m['vt']:.5f}  sup|eps~|={m['sup']:.4f}  "
        f"p99={m['p99']:.4f}  Var_s[eps_bar]={m['var_bar']:.5f}  "
        f"Var[y]={m['vy']:.5f}  SNR={m['snr']:.3f}")


def var_finetune(model, cache, tsets, H, steps, lr, sets_per_batch, seed):
    """在固定 cache 上跑 var-loss finetune dynamics, 返回 finetuned dynamics state_dict。"""
    torch.manual_seed(seed); np.random.seed(seed)
    Y, S0, A, bounds = cache["Y"], cache["S0"], cache["A"], cache["bounds"]
    act_dim = A.shape[1] // H
    Av = A.view(-1, H, act_dim)
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in model.dynamics.parameters():
        prm.requires_grad_(True)
    opt = torch.optim.Adam(model.dynamics.parameters(), lr=lr)

    def eps_of(ks):
        out = []
        for k in ks:
            sl = slice(bounds[k], bounds[k + 1])
            with torch.no_grad():
                z = model.encode(S0[sl])
            a = Av[sl]
            for h in range(H):
                z = model.next(z, a[:, h])
            v = model.predict_value(z, target=False).view(-1)
            out.append(v - Y[sl])
        return out

    for i in range(steps):
        ks = np.random.choice(tsets, sets_per_batch, replace=False)
        es = eps_of(ks)
        loss = torch.stack([e.var() for e in es]).mean()   # decision-aware var loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 10.0)
        opt.step()
        if (i + 1) % 2000 == 0:
            log(f"    var-finetune step {i+1}: loss {loss.item():.5f}")
    for prm in model.parameters():
        prm.requires_grad_(False)
    return {k: v.detach().clone() for k, v in model.dynamics.state_dict().items()}


@torch.no_grad()
def cache_metrics(model, cache, vsets, H):
    """固定 cache val split 上的 eps 分解 (planner-independent, STOCK 分布)。"""
    Y, S0, A, bounds = cache["Y"], cache["S0"], cache["A"], cache["bounds"]
    act_dim = A.shape[1] // H
    Av = A.view(-1, H, act_dim)
    eps_list, vy_list = [], []
    for k in vsets:
        sl = slice(bounds[k], bounds[k + 1])
        z = model.encode(S0[sl])
        a = Av[sl]
        for h in range(H):
            z = model.next(z, a[:, h])
        v = model.predict_value(z, target=False).view(-1)
        eps_list.append((v - Y[sl]).cpu().numpy())
        vy_list.append(Y[sl].var().item())
    return set_metrics(eps_list, vy_list)


@torch.no_grad()
def onpolicy_collect(agent, main_env, pool, episodes, probe_every, p, H, seed):
    """跑真实 MPPI on-policy; 在每个 probe 点跑完整 CEM, 记录 FINAL 迭代候选集的 eps。
    这就是 argmax 实际操作的分布 —— 由当前 dynamics 决定。"""
    model, planner = agent.model, agent.planner
    torch.manual_seed(seed); np.random.seed(seed)
    L = int(main_env_ep_len)
    eps_list, vy_list, contact_list = [], [], []
    for e in range(episodes):
        obs, _ = main_env.reset()
        for t in range(L):
            if t % probe_every == 0:
                s0 = obs["state"]
                z0 = model.encode(s0.view(1, -1))
                base = planner._policy_rollout(z0, planner._num_base())
                base = base + torch.randn_like(base) * p.base_policy_std
                mean = torch.zeros(H, planner.action_dim, device="cuda")
                std = torch.full((H, planner.action_dim), p.init_std, device="cuda")
                nrand = p.num_samples - planner._num_base()
                sim_state = main_env.env.sim.get_state().flatten().copy()
                final_v = final_y = None
                for it in range(p.iterations):
                    noise = torch.randn(H, nrand, planner.action_dim, device="cuda")
                    cands = torch.cat(
                        [(mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1),
                         base.clamp(-1, 1)], dim=1)
                    N = cands.shape[1]
                    z = z0.expand(N, -1).clone()
                    for h in range(H):
                        z = model.next(z, cands[h])
                    v = model.predict_value(z, target=False).view(-1)
                    real = pool.rollout(sim_state, cands)          # [H, N, obs]
                    y = model.predict_value(model.encode(real[-1]), target=False).view(-1)
                    final_v, final_y = v, y                          # 保留最后一轮
                    # CEM 更新
                    ret = v * planner.final_discount
                    ev, ei = torch.topk(ret, p.num_elites)
                    ea = cands[:, ei]
                    w = torch.exp(p.temperature * (ev - ev.max())); w = w / (w.sum() + 1e-9)
                    nm = (w.view(1, -1, 1) * ea).sum(1)
                    ns = torch.sqrt((w.view(1, -1, 1) * (ea - nm.unsqueeze(1)) ** 2).sum(1))
                    mean = p.momentum * mean + (1 - p.momentum) * nm
                    std = ns.clamp(min=p.min_std)
                eps = (final_v - final_y).cpu().numpy()
                eps_list.append(eps)
                vy_list.append(final_y.var().item())
                nrd = float(np.linalg.norm(s0[NEEDLE].cpu().numpy() - s0[RING].cpu().numpy()))
                contact_list.append(nrd < CONTACT)
            a = planner.plan(obs["state"], eval_mode=True)
            obs, _, term, _, _ = main_env.step(a.cpu())
            if term:
                break
        log(f"    ep{e}: {len(eps_list)} probe-sets")
    return eps_list, vy_list, np.array(contact_list)


@torch.no_grad()
def evaluate(agent, env_params, episodes, seed):
    env = PixelRobosuite(**env_params)
    succ = []
    torch.manual_seed(seed); np.random.seed(seed)
    L = int(env_params.get("episode_length", 400))
    for i in range(episodes):
        obs, _ = env.reset(); s = 0.0
        for t in range(L):
            a = agent.act(obs, eval_mode=True)
            obs, r, term, _, _ = env.step(a)
            if r > 0:
                s = 1.0
            if term:
                break
        succ.append(s)
    return float(np.mean(succ))


main_env_ep_len = 400


def main():
    global main_env_ep_len
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--cache", default="/tmp/claude-1003/-home-Chenyang-ibrl/"
                                       "08677077-8099-4d03-9973-403b524caa19/scratchpad/calib_data2.pt")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--ft_steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--probe_every", type=int, default=6)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    H = args.horizon

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = os.path.join(SCR, "onpolicy_eps_ws")
    os.makedirs(cfg.save_dir, exist_ok=True)
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner

    # 与 decision-aware / cache 采集完全同配方
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = 32, 10, H
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    planner.discounts = (p.discount ** torch.arange(H, device="cuda")).view(H, 1)
    planner.final_discount = p.discount ** H
    agent.cfg.use_planner = 1

    ep = dict(ws.eval_env_params)          # TARGET env
    main_env_ep_len = int(ep.get("episode_length", 400))

    # ---- cache
    d = torch.load(args.cache)
    cache = dict(Y=d["Y"].cuda(), S0=d["S0"].cuda(), A=d["A"].cuda(), sets=d["sets"])
    cache["bounds"] = np.cumsum([0] + list(cache["sets"]))
    nsets = len(cache["sets"])
    perm = np.random.RandomState(0).permutation(nsets)
    n_val = int(0.15 * nsets)
    vsets, tsets = perm[:n_val], perm[n_val:]
    log(f"[cache] {len(d['Y'])} cands / {nsets} sets (train {len(tsets)} / val {len(vsets)}), H={H}")

    stock_dyn = {k: v.detach().clone() for k, v in model.dynamics.state_dict().items()}

    # ---- var-finetune (在 cache train split 上), 得到 var dynamics
    log("\n===== var-finetune dynamics on fixed cache =====")
    var_dyn = var_finetune(model, cache, tsets, H, args.ft_steps, args.lr, 16, args.seed)

    # ---- 固定 cache val 上的指标 (STOCK 分布): stock vs var
    log("\n===== [A] 固定 cache val split (= STOCK planner 分布) =====")
    model.dynamics.load_state_dict(stock_dyn)
    m_cache_stock = cache_metrics(model, cache, vsets, H)
    report("stock @ cache-val", m_cache_stock)
    model.dynamics.load_state_dict(var_dyn)
    m_cache_var = cache_metrics(model, cache, vsets, H)
    report("var   @ cache-val", m_cache_var)
    log(f"  => cache 上 var 把 Var[eps~] {m_cache_stock['vt']:.5f} -> {m_cache_var['vt']:.5f} "
        f"({m_cache_var['vt']/m_cache_stock['vt']:.2f}x), SNR {m_cache_stock['snr']:.2f} -> {m_cache_var['snr']:.2f}")

    # ---- on-policy: 每个模型在自己 CEM 搜出的分布上
    pool = ProbePool(ep, args.probe_workers)
    main_env = PixelRobosuite(**ep)

    log("\n===== [B] STOCK on its OWN on-policy searched distribution =====")
    model.dynamics.load_state_dict(stock_dyn)
    e_s, vy_s, c_s = onpolicy_collect(agent, main_env, pool, args.episodes,
                                      args.probe_every, p, H, args.seed)
    m_op_stock = set_metrics(e_s, vy_s)
    report("stock @ stock-onpol", m_op_stock)
    if c_s.any() and (~c_s).any():
        report("  stock free", set_metrics([e for e, c in zip(e_s, c_s) if not c],
                                           [v for v, c in zip(vy_s, c_s) if not c]))
        report("  stock contact", set_metrics([e for e, c in zip(e_s, c_s) if c],
                                              [v for v, c in zip(vy_s, c_s) if c]))

    log("\n===== [C] VAR on its OWN on-policy searched distribution =====")
    model.dynamics.load_state_dict(var_dyn)
    e_v, vy_v, c_v = onpolicy_collect(agent, main_env, pool, args.episodes,
                                      args.probe_every, p, H, args.seed)
    m_op_var = set_metrics(e_v, vy_v)
    report("var   @ var-onpol", m_op_var)
    if c_v.any() and (~c_v).any():
        report("  var free", set_metrics([e for e, c in zip(e_v, c_v) if not c],
                                         [v for v, c in zip(vy_v, c_v) if not c]))
        report("  var contact", set_metrics([e for e, c in zip(e_v, c_v) if c],
                                            [v for v, c in zip(vy_v, c_v) if c]))
    pool.close()

    # ---- 成绩 (确认悖论复现)
    log("\n===== [D] scores (paradox check) =====")
    model.dynamics.load_state_dict(stock_dyn)
    sc_stock = evaluate(agent, ep, args.eval_episodes, args.seed)
    model.dynamics.load_state_dict(var_dyn)
    sc_var = evaluate(agent, ep, args.eval_episodes, args.seed)
    log(f"  stock score = {sc_stock:.3f}   var score = {sc_var:.3f}   (n={args.eval_episodes})")

    log("\n===== 判读 =====")
    log(f"  cache-val:   Var[eps~]  stock {m_cache_stock['vt']:.5f} -> var {m_cache_var['vt']:.5f}  "
        f"({m_cache_var['vt']/m_cache_stock['vt']:.2f}x)   [var 赢的地方]")
    log(f"  on-policy:   Var[eps~]  stock {m_op_stock['vt']:.5f} -> var {m_op_var['vt']:.5f}  "
        f"({m_op_var['vt']/max(m_op_stock['vt'],1e-9):.2f}x)")
    log(f"  on-policy:   sup|eps~|  stock {m_op_stock['sup']:.4f} -> var {m_op_var['sup']:.4f}  "
        f"({m_op_var['sup']/max(m_op_stock['sup'],1e-9):.2f}x)   [argmax 真正在意的]")
    log(f"  on-policy:   Var_s[bar] stock {m_op_stock['var_bar']:.5f} -> var {m_op_var['var_bar']:.5f}  "
        f"({m_op_var['var_bar']/max(m_op_stock['var_bar'],1e-9):.2f}x)   [绝对水平失锚?]")
    log(f"  score: {sc_stock:.3f} -> {sc_var:.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
