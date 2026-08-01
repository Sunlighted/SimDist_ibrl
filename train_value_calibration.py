"""Calibrate the value head onto the IMAGINED latent, then plan with it.

§27 established: the distilled V_hat is fine (feed it the REAL s_H and MPPI scores
100/100), but the planner feeds it z_H^imagined, which the learned dynamics puts in
the wrong place -- so the within-candidate-set ranking collapses to Spearman +0.52
and MPPI stalls at 0.825.

`scripts/diag_latent_bottleneck.py` then showed the dynamics DISTORTS but does not
DESTROY: a regression z_H^imagined -> V_hat(encode(s_H^real)) reaches held-out
R2=0.898 (vs 0.911 from (s_0,a) directly, i.e. the latent bottleneck costs nothing),
lifting the within-set ranking to +0.90.

So: don't fix the model, learn to READ it. Train a calibration head
    J(z_H^imagined) ~= V_hat(encode(s_H^real))
on exactly the candidate distribution the planner queries (MPPI's own candidates,
22/32 of which are random actions), then let MPPI score with J instead of V_hat.

The label needs the REAL s_H, which here comes from a counterfactual sim rollout.
NOTE for a real robot: no counterfactual is needed -- any ordinary trajectory gives
(s_t, a_t..a_{t+H-1}, s_{t+H}), and rolling the same actions through the model from
s_t yields z_H^imagined. Only the ACTION COVERAGE matters (the planner samples random
actions, so the data must contain them too).

CAVEAT: the calibrated planner visits a different state distribution than the one the
head was trained on (DAgger-style shift). `--rounds > 1` re-collects with the current
head and re-trains.
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
from scripts.eval_oracle_dyn_mppi import ProbePool, restore  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


class CalibHead(torch.nn.Module):
    """z_H^imagined -> the value the REAL terminal state would have had."""

    def __init__(self, zdim, hidden=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(zdim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, z):
        return self.net(z).squeeze(-1) * self.y_std + self.y_mean


@torch.no_grad()
def collect(agent, main_env, pool, planner, model, args, stock_predict_value=None):
    """Roll MPPI and label EVERY candidate the planner actually scores.

    IMPORTANT: the planner runs `iterations` CEM rounds. Round 1 samples wide
    (mean=0, std=init_std); round 2+ samples NARROW, centred on round-1's elites --
    and it is the LAST round's mean that gets executed. An earlier version of this
    collector only reproduced round 1, so the calibration head was never trained on
    the distribution that actually decides the action (and those late candidates,
    all crowded near the elites, are the hardest to tell apart). We now replay the
    full CEM and harvest every round.

    Labels use the STOCK V_hat (passed in explicitly) -- never the calibrated head,
    which model.predict_value may have been monkey-patched to in a previous round.
    """
    p = planner.cfg
    pv = stock_predict_value if stock_predict_value is not None else model.predict_value
    S0, A, ZI, Y = [], [], [], []
    sets = []
    ep_len = int(main_env.episode_length) if hasattr(main_env, "episode_length") else 400
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

                for it in range(p.iterations):          # ★ 复现完整 CEM, 每轮都采
                    noise = torch.randn(p.horizon, nrand, planner.action_dim, device="cuda")
                    cands = torch.cat([(mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1),
                                       base.clamp(-1, 1)], dim=1)
                    N = cands.shape[1]

                    z = z0.expand(N, -1).clone()
                    for h in range(p.horizon):
                        z = model.next(z, cands[h])

                    real = pool.rollout(sim_state, cands)
                    y = pv(model.encode(real[-1]), target=False).view(-1)

                    S0.append(s0.view(1, -1).expand(N, -1).cpu())
                    A.append(cands.permute(1, 0, 2).reshape(N, -1).cpu())
                    ZI.append(z.cpu()); Y.append(y.cpu()); sets.append(N)

                    # advance the CEM exactly as the planner would (using the STOCK
                    # score, since that is the planner that generated this trajectory)
                    ret = pv(z, target=False).view(-1) * planner.final_discount
                    ev, ei = torch.topk(ret, p.num_elites)
                    ea = cands[:, ei]
                    w = torch.exp(p.temperature * (ev - ev.max()))
                    w = w / (w.sum() + 1e-9)
                    new_mean = (w.view(1, -1, 1) * ea).sum(1)
                    new_std = torch.sqrt((w.view(1, -1, 1) * (ea - new_mean.unsqueeze(1)) ** 2).sum(1))
                    mean = p.momentum * mean + (1 - p.momentum) * new_mean
                    std = new_std.clamp(min=p.min_std)
                # NOTE: do NOT restore(main_env) here. The candidates are rolled out in
                # the ProbePool's OWN envs; main_env was never touched (get_state() only
                # reads). Calling restore() on it would (a) zero qacc_warmstart -> the
                # next step is a cold-start solve, so the collected trajectory is NOT the
                # one a normal MPPI rollout would take, (b) reset timestep -> break the
                # horizon check, (c) reset the strict-success validator -> wipe the dwell
                # streak. That silently biases the whole training distribution.
            a = planner.plan(obs["state"], eval_mode=True)
            obs, _, term, _, _ = main_env.step(a.cpu())
            if term:
                break
        log(f"    collect ep{e}: {len(sets)} sets")
    return (torch.cat(ZI).cuda(), torch.cat(Y).cuda(), sets,
            torch.cat(S0).cuda(), torch.cat(A).cuda())


def _set_bounds(sets):
    return np.cumsum([0] + list(sets))


def within_set_spearman_idx(pred, Y, sets, which):
    """集内 Spearman, 但只在指定的候选集上 (用于 held-out)。"""
    b = _set_bounds(sets)
    pr, yy = pred.cpu().numpy(), Y.cpu().numpy()
    out = []
    for k in which:
        a, c = pr[b[k]:b[k + 1]], yy[b[k]:b[k + 1]]
        if c.std() > 0:
            r = spearmanr(a, c).correlation
            if not np.isnan(r):
                out.append(r)
    return float(np.mean(out)) if out else float("nan")


def train_head(Z, Y, zdim, steps, lr, sets, val_frac=0.15, seed=0, loss_kind="mse"):
    """Split BY CANDIDATE SET, not by candidate. All 32 candidates of one s_0 share
    almost the same z_H; splitting them across train/val leaks (the earlier version did
    this and reported an inflated held-out R2=0.850)."""
    torch.manual_seed(seed)
    bounds = np.cumsum([0] + list(sets))
    nsets = len(sets)
    sperm = torch.randperm(nsets)
    nv_sets = max(1, int(nsets * val_frac))
    vsets, tsets = sperm[:nv_sets].tolist(), sperm[nv_sets:].tolist()
    idx_of = lambda ss: torch.cat([torch.arange(bounds[k], bounds[k + 1]) for k in ss]).to(Z.device)
    vi, ti = idx_of(vsets), idx_of(tsets)
    head = CalibHead(zdim).cuda()
    head.y_mean.fill_(Y[ti].mean().item())
    head.y_std.fill_(max(Y[ti].std().item(), 1e-6))
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    if loss_kind == "mse":
        for i in range(steps):
            idx = ti[torch.randint(0, len(ti), (512,), device=Z.device)]
            loss = torch.nn.functional.mse_loss(head(Z[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    else:
        # LISTWISE: the planner only ever ranks the 32 candidates of ONE s_0. The MSE
        # objective is dominated by ACROSS-state variance (y std 0.175 global vs 0.095
        # within-set), so the head spends its capacity predicting "what is this state
        # worth" -- which the planner never uses -- and only interpolates the weak
        # within-set signal. Here we optimise EXACTLY the within-set ranking:
        # softmax CE between the head's scores and the true y, inside each set.
        tau = 0.05
        for i in range(steps):
            ks = [tsets[j] for j in torch.randint(0, len(tsets), (16,)).tolist()]
            ls = []
            for k in ks:
                sl = slice(bounds[k], bounds[k + 1])
                logit = head(Z[sl])
                tgt = torch.softmax(Y[sl] / tau, dim=0)
                ls.append(-(tgt * torch.log_softmax(logit / tau, dim=0)).sum())
            loss = torch.stack(ls).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pv = head(Z[vi])
    ss_res = ((pv - Y[vi]) ** 2).sum().item()
    ss_tot = ((Y[vi] - Y[vi].mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    sp = spearmanr(pv.cpu().numpy(), Y[vi].cpu().numpy()).correlation
    with torch.no_grad():
        pall = head(Z)
    # ★ 判决性指标: held-out 候选集上的集内 Spearman (planner 面对的永远是新候选集)
    ho = within_set_spearman_idx(pall, Y, sets, vsets)
    tr = within_set_spearman_idx(pall, Y, sets, tsets)
    return head, r2, sp, ho, tr


def within_set_spearman(pred, Y, sets):
    out, i = [], 0
    pr, yy = pred.cpu().numpy(), Y.cpu().numpy()
    for n in sets:
        a, b = pr[i:i + n], yy[i:i + n]
        i += n
        if b.std() > 0:
            r = spearmanr(a, b).correlation
            if not np.isnan(r):
                out.append(r)
    return float(np.mean(out))


@torch.no_grad()
def evaluate(agent, planner, env_params, episodes, seed):
    env = PixelRobosuite(**env_params)
    succ = []
    torch.manual_seed(seed); np.random.seed(seed)
    ep_len = int(env_params.get("episode_length", 400))
    for e in range(episodes):
        obs, _ = env.reset()
        s = 0.0
        for t in range(ep_len):
            a = planner.plan(obs["state"], eval_mode=True)
            obs, r, term, _, _ = env.step(a.cpu())
            if r > 0:
                s = 1.0
            if term:
                break
        succ.append(s)
        if (e + 1) % 10 == 0:
            log(f"    eval {e+1}/{episodes}: running {np.mean(succ):.3f}")
    return float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="exps/mimicgen/threading_simdist_adapt_long_h5_seed1")
    ap.add_argument("--rounds", type=int, default=1, help="DAgger rounds (re-collect with the current head)")
    ap.add_argument("--collect_eps", type=int, default=20)
    ap.add_argument("--probe_every", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_base", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--train_steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default="")
    ap.add_argument("--data_cache", default="",
                    help="save/load the collected (z_imag, y, sets) so loss/arch can be "
                         "iterated without re-collecting (collection is the slow part)")
    ap.add_argument("--loss", default="mse", choices=["mse", "listwise"],
                    help="mse = regress y (dominated by ACROSS-state variance, which the "
                         "planner never uses). listwise = softmax cross-entropy WITHIN each "
                         "candidate set -> all capacity goes to the only thing that matters: "
                         "ranking the 32 candidates of the SAME s_0.")
    ap.add_argument("--collect_only", type=int, default=0,
                    help="只收集数据到 --data_cache 就退出 (给 train_decision_aware_dyn.py 用)")
    ap.add_argument("--match_scale", type=int, default=0,
                    help="linearly rescale the head's output onto V_hat's global mean/std "
                         "(ranking unchanged). Isolates 'bad ranking' from 'wrong scale "
                         "broke the CEM softmax weighting'.")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg: AdaptConfig = pyrallis.load(AdaptConfig, open(os.path.join(args.run_dir, "cfg.yaml")))  # type: ignore
    cfg.eval_hand_init_perturb_joint_sigma = 0.1
    cfg.use_wb = 0; cfg.mp_eval = 0
    cfg.save_dir = "/tmp/claude-1003/-home-Chenyang-ibrl/08677077-8099-4d03-9973-403b524caa19/scratchpad/calib_ws"
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
    main_env = PixelRobosuite(**ep)
    main_env.reset()
    pool = ProbePool(ep, args.probe_workers)
    zdim = model.encode(torch.zeros(1, main_env.observation_shape[0], device="cuda")).shape[1]
    log(f"[setup] latent dim {zdim} | H={p.horizon} N={p.num_samples} (base {args.num_base})")

    # baseline: the stock planner (V_hat on the imagined latent)
    base_score = evaluate(agent, planner, ep, args.eval_episodes, args.seed)
    log(f"\n[BASELINE] stock V_hat(z_imag): TARGET strict = {base_score:.3f} (n={args.eval_episodes})")

    Zall, Yall, sets_all = None, None, []
    stock_predict_value = model.predict_value
    for rnd in range(args.rounds):
        if args.data_cache and os.path.exists(args.data_cache) and rnd == 0:
            d = torch.load(args.data_cache)
            Z, Y, sets = d["Z"].cuda(), d["Y"].cuda(), d["sets"]
            S0, A = d["S0"].cuda(), d["A"].cuda()
            log(f"\n===== ROUND {rnd} : 复用缓存 ({len(Y)} cands / {len(sets)} sets)")
        else:
            log(f"\n===== ROUND {rnd} : collecting (完整 CEM {p.iterations} 轮) =====")
            Z, Y, sets, S0, A = collect(agent, main_env, pool, planner, model, args,
                                        stock_predict_value=stock_predict_value)
            if args.data_cache and rnd == 0:
                torch.save({"Z": Z.cpu(), "Y": Y.cpu(), "sets": sets,
                            "S0": S0.cpu(), "A": A.cpu()}, args.data_cache)
                log(f"  [cache] saved -> {args.data_cache}")

        if args.collect_only:
            log(f"  [collect_only] 数据已存, 退出")
            pool.close()
            os._exit(0)

        # ★★ 判决性对照: (s_0, a) -> y 的集内排序 (chunk-Q 的生死)
        # z_H 的集内排序上限 ~0.55。若 (s_0,a) 显著更高 => 信息还在, 只是被 dynamics 毁了
        # => chunk-Q 有希望(直接学 Q(s,a_chunk), 不要 dynamics)。若也 ~0.55 => 整条 planning
        # 路线(含 chunk-Q)都死: "从(s,a)预测 H 步后的价值"这件事本身就学不出集内区分度。
        SA = torch.cat([S0, A], dim=1)
        _, r2_sa, sp_sa, ho_sa, tr_sa = train_head(SA, Y, SA.shape[1], args.train_steps,
                                                   args.lr, sets, loss_kind="listwise")
        log(f"  [★chunk-Q 判据] (s_0,a) -> y  listwise:  集内 Spearman "
            f"held-out {ho_sa:+.3f} (训练集 {tr_sa:+.3f})")
        Zall = Z if Zall is None else torch.cat([Zall, Z])
        Yall = Y if Yall is None else torch.cat([Yall, Y])
        sets_all += sets
        log(f"  [data] {len(Yall)} candidates / {len(sets_all)} sets | y mean {Yall.mean():.3f} std {Yall.std():.3f}")

        with torch.no_grad():
            v_stock = stock_predict_value(Z, target=False).view(-1)
        sp_stock = within_set_spearman(v_stock, Y, sets)

        head, r2, sp, ho_sp, tr_sp = train_head(Zall, Yall, zdim, args.train_steps, args.lr,
                                                sets_all, loss_kind=args.loss)
        with torch.no_grad():
            v_cal = head(Z)
        sp_cal = within_set_spearman(v_cal, Y, sets)
        log(f"  [head] loss={args.loss}  held-out(按候选集) R2 = {r2:.3f}  全局 Spearman = {sp:+.3f}")
        log(f"  [rank] 集内 Spearman (训练集):  stock {sp_stock:+.3f}  ->  calibrated {tr_sp:+.3f}")
        log(f"  [rank] ★ 集内 Spearman (HELD-OUT 候选集):  calibrated {ho_sp:+.3f}   "
            f"<- planner 面对的就是这个; 必须 > stock {sp_stock:+.3f} 才可能涨分")

        # ---- SCALE DIAGNOSTIC: CEM weights elites by softmax(temp * (R - R_max)) with
        # temp=0.4, so it is sensitive to the ABSOLUTE spread of the score. If the head's
        # WITHIN-SET spread is much smaller than V_hat's, the softmax flattens, elite
        # weighting degenerates toward a plain average -- which is what random-J does
        # (and random-J scores 0.000 in the mixed pool). Then a drop in score would be a
        # SCALE artifact, not a ranking failure.
        def within_set_std(v):
            out, i = [], 0
            vv = v.cpu().numpy()
            for n in sets:
                out.append(vv[i:i + n].std()); i += n
            return float(np.mean(out))
        wss_stock, wss_cal = within_set_std(v_stock), within_set_std(v_cal)
        log(f"  [scale] 全局 std:  V_hat {v_stock.std():.4f}  J {v_cal.std():.4f}")
        log(f"  [scale] **集内** std: V_hat {wss_stock:.4f}  J {wss_cal:.4f}   "
            f"(比值 J/V_hat = {wss_cal / max(wss_stock, 1e-9):.2f}x)")
        log(f"  [scale] CEM softmax 有效锐度 temp*std: V_hat {p.temperature * wss_stock:.4f}  "
            f"J {p.temperature * wss_cal:.4f}")

        # let MPPI score with the calibrated head (same signature as predict_value)
        head.eval()
        if args.match_scale:
            # rank-preserving affine map onto V_hat's global distribution -> isolates
            # "ranking quality" from "scale broke the CEM weighting"
            with torch.no_grad():
                jm, js = v_cal.mean().item(), v_cal.std().clamp(min=1e-6).item()
                vm, vs = v_stock.mean().item(), v_stock.std().clamp(min=1e-6).item()
            log(f"  [scale] match_scale ON: J -> (J-{jm:.3f})/{js:.3f}*{vs:.3f}+{vm:.3f}")
            def calibrated_predict_value(z, target=False, _h=head, _jm=jm, _js=js, _vm=vm, _vs=vs):
                return (_h(z) - _jm) / _js * _vs + _vm
        else:
            def calibrated_predict_value(z, target=False, _h=head):
                return _h(z)
        model.predict_value = calibrated_predict_value

        score = evaluate(agent, planner, ep, args.eval_episodes, args.seed)
        log(f"\n[ROUND {rnd}] CALIBRATED planner: TARGET strict = {score:.3f} "
            f"(baseline {base_score:.3f}, oracle-dyn upper bound 1.000)")

        if args.save:
            torch.save({"head": head.state_dict(), "zdim": zdim, "round": rnd,
                        "score": score, "r2": r2}, args.save)
            log(f"  saved -> {args.save}")

    pool.close()
    log("\n===== SUMMARY =====")
    log(f"  stock V_hat(z_imag)      : {base_score:.3f}")
    log(f"  calibrated J(z_imag)     : {score:.3f}")
    log(f"  oracle dyn + V_hat (上界) : 1.000")
    os._exit(0)


if __name__ == "__main__":
    main()
