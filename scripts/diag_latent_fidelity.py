"""根因判决: var-finetune 掉 27.5 点, 是不是 "value-mimicking 坍缩"?

假设: var-loss 只压 Var_a[ V_hat(f^H) - V_hat(encode(s_real)) ], 它可以靠把 f^H 的
VALUE 对齐到真实终点的 value 来降 eps —— 而不需要 f^H 这个 STATE 本身接近 s_real。
两个不同的终点可以有相同的 value。⇒ f 退化成 "value 模仿器": value-eps 降, 但 latent
几何 ||f^H - encode(s_real)|| 反而升。planner 于是在几何失真的 f 上 argmax, 被带偏。

每个候选同时记:
    value_err = | V_hat(f^H(s0,a)) - V_hat(encode(s_real(a))) |     (价值一致性)
    state_err = || f^H(s0,a) - encode(s_real(a)) ||_2               (几何保真度)
encoder 冻结且 stock/var 共用 => 两者的 state_err 直接可比。

判读:
  var state_err >> stock  且  var value_err << stock   => value-mimicking 坍缩 (根因)
       => B2 该惩罚 f^H 的 ensemble 几何 disagreement
  state_err 两者相当                                    => 纯 bias 失标定
       => B2 惩罚 value 不确定性, 或走 A (正则 Var_s[eps_bar])
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
from scripts.diag_onpolicy_eps import var_finetune  # noqa: E402

ERR = sys.stderr
SCR = "/tmp/claude-1003/-home-Chenyang-ibrl/403b1f91-4a38-4d1d-99d2-5d860804db5f/scratchpad"


def log(*a):
    print(*a, file=ERR, flush=True)


@torch.no_grad()
def collect(agent, main_env, pool, episodes, probe_every, p, H, seed, epl):
    """每个 probe 点, 记宽(it0)/窄(it_last)轮每个候选的 value_err 与 state_err。"""
    model, planner = agent.model, agent.planner
    torch.manual_seed(seed); np.random.seed(seed)
    # rec[(round, metric)] = list of per-candidate arrays
    rec = {("wide", "verr"): [], ("wide", "serr"): [],
           ("narrow", "verr"): [], ("narrow", "serr"): []}
    for e in range(episodes):
        obs, _ = main_env.reset()
        for t in range(epl):
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
                        z = model.next(z, cands[h])            # imagined latent f^H
                    real = pool.rollout(sim_state, cands)      # [H, N, obs]
                    zreal = model.encode(real[-1])             # encode(s_real)
                    v_im = model.predict_value(z, target=False).view(-1)
                    v_re = model.predict_value(zreal, target=False).view(-1)
                    verr = (v_im - v_re).abs().cpu().numpy()
                    serr = (z - zreal).norm(dim=-1).cpu().numpy()   # latent L2 per cand
                    key = "wide" if it == 0 else ("narrow" if it == p.iterations - 1 else None)
                    if key:
                        rec[(key, "verr")].append(verr)
                        rec[(key, "serr")].append(serr)
                    ret = v_im * planner.final_discount
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
        log(f"    ep{e}: {len(rec[('wide','verr')])} probe-sets")
    return rec


def summ(rec, rnd):
    verr = np.concatenate(rec[(rnd, "verr")])
    serr = np.concatenate(rec[(rnd, "serr")])
    return dict(verr=verr.mean(), serr=serr.mean(), serr_med=np.median(serr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--cache", default="/tmp/claude-1003/-home-Chenyang-ibrl/"
                                       "08677077-8099-4d03-9973-403b524caa19/scratchpad/calib_data2.pt")
    ap.add_argument("--horizon", type=int, default=3)
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
    cfg.save_dir = os.path.join(SCR, "latfid_ws")
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

    stock_dyn = {k: v.detach().clone() for k, v in model.dynamics.state_dict().items()}
    var_path = os.path.join(SCR, "var_dyn_H%d.pt" % H)
    if os.path.exists(var_path):
        log(f"[var] load cached {var_path}")
        var_dyn = torch.load(var_path)
    else:
        d = torch.load(args.cache)
        cache = dict(Y=d["Y"].cuda(), S0=d["S0"].cuda(), A=d["A"].cuda(), sets=d["sets"])
        cache["bounds"] = np.cumsum([0] + list(cache["sets"]))
        tsets = np.random.RandomState(0).permutation(len(cache["sets"]))[int(0.15 * len(cache["sets"])):]
        log("[var] finetune ...")
        var_dyn = var_finetune(model, cache, tsets, H, 6000, 1e-4, 16, args.seed)
        torch.save(var_dyn, var_path)

    pool = ProbePool(ep, args.probe_workers)
    main_env = PixelRobosuite(**ep)

    log("\n===== STOCK =====")
    model.dynamics.load_state_dict(stock_dyn)
    rec_s = collect(agent, main_env, pool, args.episodes, args.probe_every, p, H, args.seed, epl)
    log("\n===== VAR =====")
    model.dynamics.load_state_dict(var_dyn)
    rec_v = collect(agent, main_env, pool, args.episodes, args.probe_every, p, H, args.seed, epl)
    pool.close()

    log("\n===== 根因判读: value 一致性 vs latent 几何保真度 =====")
    log(f"  {'':16s}  {'|value_err|':>12s}  {'||f^H-real||_2':>14s}  (latent median)")
    for rnd in ["wide", "narrow"]:
        s, v = summ(rec_s, rnd), summ(rec_v, rnd)
        log(f"  [{rnd:6s}] stock    {s['verr']:12.5f}  {s['serr']:14.4f}  ({s['serr_med']:.4f})")
        log(f"  [{rnd:6s}] var      {v['verr']:12.5f}  {v['serr']:14.4f}  ({v['serr_med']:.4f})")
        log(f"           ratio(v/s) value {v['verr']/max(s['verr'],1e-9):.2f}x   "
            f"state {v['serr']/max(s['serr'],1e-9):.2f}x")
    log("\n  判读: var 若 value↓ 而 state↑ => value-mimicking 坍缩 (根因, B2 惩罚几何 disagreement)")
    log("        state 两者相当           => 纯 bias 失标定 (B2 惩罚 value 不确定 / 走 A)")
    os._exit(0)


if __name__ == "__main__":
    main()
