"""机制钉死: Var_s[eps_bar] 20x 怎么伤到分数的?

上一个脚本 (diag_onpolicy_eps) 只记了 CEM 最后一轮 (窄, std 夹到 0.05) 的 eps,
那里 stock/var 的 Var[eps~] 都极小 (0.00022 vs 0.00024) —— 看不出差别。

假设: 伤害在 CEM **第一轮的宽候选** (init_std=1.0)。宽候选到达差异巨大的终点,
value 曲面一旦全局失标定 (Var_s[eps_bar] 爆炸), 这些不同终点的 bias 不再相消,
直接变成宽候选集内的 eps~ —— 粗选就错, 窄轮精修围着错的 elite。

⇒ 分 it0(宽) / it_last(窄) 两轮各记 eps, stock vs var。
   预测: var 在 it0 的 Var[eps~]/sup 爆炸, it_last 两者都良性。
   若成立 => 解法必须改善 value 曲面的**全局标定**, 而非动 return 的零点 (B1 那类全废,
   因 CEM topk+softmax 对 per-call 常数平移不变)。
"""
import os
import sys
import argparse

import numpy as np
import torch
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from train_simdist_adapt import AdaptConfig, AdaptWorkspace  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.eval_oracle_dyn_mppi import ProbePool  # noqa: E402
from scripts.diag_onpolicy_eps import var_finetune, set_metrics  # noqa: E402

ERR = sys.stderr
NEEDLE = slice(9, 12)
RING = slice(23, 26)
CONTACT = 0.15
SCR = "/tmp/claude-1003/-home-Chenyang-ibrl/403b1f91-4a38-4d1d-99d2-5d860804db5f/scratchpad"


def log(*a):
    print(*a, file=ERR, flush=True)


def report(name, m):
    log(f"  [{name:20s}] Var[eps~]={m['vt']:.5f}  sup|eps~|={m['sup']:.4f}  "
        f"p99={m['p99']:.4f}  Var_s[eps_bar]={m['var_bar']:.5f}  "
        f"Var[y]={m['vy']:.5f}  SNR={m['snr']:.3f}")


@torch.no_grad()
def collect_per_iter(agent, main_env, pool, episodes, probe_every, p, H, seed):
    """每个 probe 点记录 CEM it0(宽) 和 it_last(窄) 各自的 eps。"""
    model, planner = agent.model, agent.planner
    torch.manual_seed(seed); np.random.seed(seed)
    L = int(main_env.env.horizon) if hasattr(main_env.env, "horizon") else 400
    L = int(getattr(main_env, "_episode_length", L) or L)
    first_e, first_vy, last_e, last_vy = [], [], [], []
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
                    real = pool.rollout(sim_state, cands)
                    y = model.predict_value(model.encode(real[-1]), target=False).view(-1)
                    eps = (v - y).cpu().numpy()
                    if it == 0:                                  # 宽轮
                        first_e.append(eps); first_vy.append(y.var().item())
                    if it == p.iterations - 1:                   # 窄轮
                        last_e.append(eps); last_vy.append(y.var().item())
                    ret = v * planner.final_discount
                    ev, ei = torch.topk(ret, p.num_elites)
                    ea = cands[:, ei]
                    w = torch.exp(p.temperature * (ev - ev.max())); w = w / (w.sum() + 1e-9)
                    nm = (w.view(1, -1, 1) * ea).sum(1)
                    ns = torch.sqrt((w.view(1, -1, 1) * (ea - nm.unsqueeze(1)) ** 2).sum(1))
                    mean = p.momentum * mean + (1 - p.momentum) * nm
                    std = ns.clamp(min=p.min_std)
                a = planner.plan(obs["state"], eval_mode=True)
            else:
                a = planner.plan(obs["state"], eval_mode=True)
            obs, _, term, _, _ = main_env.step(a.cpu())
            if term:
                break
        log(f"    ep{e}: {len(first_e)} probe-sets")
    return (first_e, first_vy), (last_e, last_vy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--cache", default="/tmp/claude-1003/-home-Chenyang-ibrl/"
                                       "08677077-8099-4d03-9973-403b524caa19/scratchpad/calib_data2.pt")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--ft_steps", type=int, default=6000)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--probe_every", type=int, default=6)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    H = args.horizon

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = os.path.join(SCR, "widenar_ws")
    os.makedirs(cfg.save_dir, exist_ok=True)
    ws = AdaptWorkspace(cfg)
    ws.agent.load_state_dict(torch.load(os.path.join(args.run_dir, "latest.pt")))
    agent = ws.agent.to("cuda")
    model, planner = agent.model, agent.planner
    p = planner.cfg
    p.num_samples, p.num_base_override, p.horizon = 32, 10, H
    p.iterations, p.num_elites, p.use_reward = 2, 8, 0
    planner.discounts = (p.discount ** torch.arange(H, device="cuda")).view(H, 1)
    planner.final_discount = p.discount ** H
    agent.cfg.use_planner = 1
    ep = dict(ws.eval_env_params)
    epl = int(ep.get("episode_length", 400))

    d = torch.load(args.cache)
    cache = dict(Y=d["Y"].cuda(), S0=d["S0"].cuda(), A=d["A"].cuda(), sets=d["sets"])
    cache["bounds"] = np.cumsum([0] + list(cache["sets"]))
    nsets = len(cache["sets"])
    tsets = np.random.RandomState(0).permutation(nsets)[int(0.15 * nsets):]

    stock_dyn = {k: v.detach().clone() for k, v in model.dynamics.state_dict().items()}
    var_path = os.path.join(SCR, "var_dyn_H%d.pt" % H)
    if os.path.exists(var_path):
        log(f"[var] load cached {var_path}")
        var_dyn = torch.load(var_path)
    else:
        log("[var] finetune ...")
        var_dyn = var_finetune(model, cache, tsets, H, args.ft_steps, 1e-4, 16, args.seed)
        torch.save(var_dyn, var_path)
    model.dynamics.load_state_dict(stock_dyn)

    pool = ProbePool(ep, args.probe_workers)
    main_env = PixelRobosuite(**ep)
    main_env._episode_length = epl  # for L

    log("\n===== STOCK on-policy: it0(宽) vs it_last(窄) =====")
    model.dynamics.load_state_dict(stock_dyn)
    (sf_e, sf_v), (sl_e, sl_v) = collect_per_iter(
        agent, main_env, pool, args.episodes, args.probe_every, p, H, args.seed)
    report("stock it0 宽", set_metrics(sf_e, sf_v))
    report("stock it_last 窄", set_metrics(sl_e, sl_v))

    log("\n===== VAR on-policy: it0(宽) vs it_last(窄) =====")
    model.dynamics.load_state_dict(var_dyn)
    (vf_e, vf_v), (vl_e, vl_v) = collect_per_iter(
        agent, main_env, pool, args.episodes, args.probe_every, p, H, args.seed)
    report("var   it0 宽", set_metrics(vf_e, vf_v))
    report("var   it_last 窄", set_metrics(vl_e, vl_v))
    pool.close()

    ms_f, mv_f = set_metrics(sf_e, sf_v), set_metrics(vf_e, vf_v)
    ms_l, mv_l = set_metrics(sl_e, sl_v), set_metrics(vl_e, vl_v)
    log("\n===== 判读: 伤害是否在宽轮? =====")
    log(f"  it0 宽   Var[eps~]  stock {ms_f['vt']:.5f} -> var {mv_f['vt']:.5f}  "
        f"({mv_f['vt']/max(ms_f['vt'],1e-9):.2f}x)   sup {ms_f['sup']:.4f} -> {mv_f['sup']:.4f}  "
        f"({mv_f['sup']/max(ms_f['sup'],1e-9):.2f}x)")
    log(f"  it_last 窄 Var[eps~]  stock {ms_l['vt']:.5f} -> var {mv_l['vt']:.5f}  "
        f"({mv_l['vt']/max(ms_l['vt'],1e-9):.2f}x)   sup {ms_l['sup']:.4f} -> {mv_l['sup']:.4f}  "
        f"({mv_l['sup']/max(ms_l['sup'],1e-9):.2f}x)")
    log("  预测: 宽轮 var 爆炸 (>>1x), 窄轮两者相当 (~1x) => 病在宽轮的全局失标定。")
    os._exit(0)


if __name__ == "__main__":
    main()
