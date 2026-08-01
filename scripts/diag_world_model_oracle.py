"""Diagnose a frozen SimDist world model using a strong RL policy as an ORACLE.

Idea (user's plan): we have a near-perfect target policy (an IBRL QAgent that
solves the TARGET task at 95%+). Roll it out in the TARGET env to get REAL,
KNOWN-GOOD trajectories, then feed them through the FROZEN world model that the
SimDist-adapt MPPI planner uses. Three localizing tests:

  (1) RETURN RANKING (reward/value head vs planner search).
      For each state on an oracle trajectory, compare the model-predicted return
      of the ORACLE's own H-step action sequence R_model(oracle) against the
      return the MPPI PLANNER actually achieves from that state R_model(planner).
        * R_model(oracle) < R_model(planner)  -> the value/reward model RANKS the
          good actions BELOW what the planner found. The planner would REJECT the
          oracle action even if it sampled it. Bottleneck = reward/value model.
        * R_model(oracle) > R_model(planner)  -> the model PREFERS the oracle
          sequence but MPPI never found it. Bottleneck = planner optimization
          (sampling / proposal coverage).

  (2) DYNAMICS CONSISTENCY (coverage).
      Latent 1-step and H-step prediction error ||next(z,a) - sg(encode(s'))||^2
      on the ORACLE trajectories vs on the PLANNER's own trajectories. If the
      dynamics error on oracle states is much higher, the dynamics model does not
      COVER the states the good policy visits -> coverage problem.

  (3) VALUE CALIBRATION vs the RL critic.
      The world-model value V_wm(s) vs the oracle's true value V_rl(s) =
      min-ensemble Q(s, pi(s)) along trajectories: correlation + spread. A flat /
      mis-scaled V_wm (it was distilled on SOURCE and FROZEN through adapt) can
      make the MPPI return gamma^H*V(z_H) uninformative.

Usage:
  python scripts/diag_world_model_oracle.py \
      --stage1_ckpt exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
      --adapt_ckpt  exps/mimicgen/threading_simdist_adapt_seed1/model0.pt \
      --rl_agent    exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt \
      --num_rl_episode 30 --num_mppi_episode 10 --gpu 0
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import pyrallis


def build_world_model_agent(stage1_ckpt, adapt_ckpt, original_mppi=True):
    """Rebuild the stage-1 SimDist agent arch + TARGET env, load the ADAPT weights,
    force MPPI on with the original-recipe planner (exactly what train_simdist_adapt
    does)."""
    from train_simdist import Workspace, MainConfig
    from rl.simdist_planner import apply_original_recipe

    stage1_dir = os.path.dirname(stage1_ckpt)
    base_cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(stage1_dir, "cfg.yaml")))  # type: ignore
    base_cfg.simdist.use_planner = 1
    if original_mppi:
        apply_original_recipe(base_cfg.simdist.planner)
    ws = Workspace(base_cfg, from_main=False)
    agent = ws.agent
    agent.load_state_dict(torch.load(adapt_ckpt))
    agent = agent.to("cuda")
    agent.eval()
    return agent, ws.eval_env_params, base_cfg


@torch.no_grad()
def rollout(env, agent, num_episode, seed0, is_world_model, label):
    """Roll `agent` in `env` for num_episode episodes. Returns a list of episodes;
    each episode is a dict of stacked cpu tensors: states[T+1,obs], actions[T,act],
    rewards[T], success(bool). Stores states BEFORE each act (act may mutate obs)."""
    from common_utils import ibrl_utils as utils

    episodes = []
    with utils.eval_mode(agent):
        for ep in range(num_episode):
            np.random.seed(seed0 + ep)
            obs, _ = env.reset()
            states, actions, rewards = [obs["state"].clone()], [], []
            terminal = False
            while not terminal:
                action = agent.act(obs, eval_mode=True, cpu=True)
                obs, reward, terminal, _, _ = env.step(action)
                states.append(obs["state"].clone())
                actions.append(action.clone().float())
                rewards.append(float(reward))
            ep_reward = float(np.sum(rewards))
            episodes.append(dict(
                states=torch.stack(states),                       # [T+1, obs]
                actions=torch.stack(actions),                     # [T, act]
                rewards=torch.tensor(rewards, dtype=torch.float32),  # [T]
                success=ep_reward > 0,
            ))
            print(f"  [{label}] ep {ep:3d}  len {len(actions):3d}  "
                  f"reward {ep_reward:.0f}  {'OK' if ep_reward > 0 else '..'}")
    sr = np.mean([e["success"] for e in episodes])
    print(f"  [{label}] success rate = {100*sr:.1f}%  ({sum(e['success'] for e in episodes)}/{num_episode})")
    return episodes


@torch.no_grad()
def estimate_return(planner, z0, action_seq):
    """Model-predicted return of ONE action sequence. z0:[1,z], action_seq:[H,act].
    Returns (total, disc_reward, final_value) as floats."""
    H = action_seq.shape[0]
    cfg = planner.cfg
    z = z0.clone()
    rew = 0.0
    for t in range(H):
        a = action_seq[t:t+1]                       # [1,act]
        if cfg.use_reward:
            r = planner.model.predict_reward(z, a)  # [1]
            rew += (cfg.discount ** t) * float(r)
        z = planner.model.next(z, a)
    v = float(planner.model.predict_value(z))       # V(z_H)
    fv = (cfg.discount ** H) * v
    total = (rew + fv) if cfg.use_reward else fv
    return total, rew, fv


@torch.no_grad()
def plan_full(planner, state):
    """Run the MPPI plan from `state` and return the executed plan's return plus the
    BEST candidate return found in the final iteration.

    Returns dict: mean_return (return of the optimized mean sequence, what MPPI
    would execute), best_return (max candidate return in the last iteration)."""
    cfg = planner.cfg
    H = cfg.horizon
    device = planner.device
    z0 = planner.model.encode(state.view(1, -1))                # [1,z]

    num_base = planner._num_base()
    if num_base > 0:
        base_actions = planner._policy_rollout(z0, num_base)    # [H,num_base,act]
        if cfg.base_policy_std > 0:
            base_actions = base_actions + torch.randn_like(base_actions) * cfg.base_policy_std
    else:
        base_actions = None

    mean = torch.zeros(H, planner.action_dim, device=device)
    std = torch.full((H, planner.action_dim), cfg.init_std, device=device)
    num_rand = cfg.num_samples - num_base
    best_return = -1e9
    for _ in range(cfg.iterations):
        noise = torch.randn(H, num_rand, planner.action_dim, device=device)
        rand_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)
        if base_actions is not None:
            actions = torch.cat([rand_actions, base_actions.clamp(-1, 1)], dim=1)
        else:
            actions = rand_actions
        returns = planner._estimate_returns(z0, actions)        # [N]
        best_return = max(best_return, float(returns.max()))
        elite_val, elite_idx = torch.topk(returns, cfg.num_elites)
        elite_acts = actions[:, elite_idx]
        score = torch.exp(cfg.temperature * (elite_val - elite_val.max()))
        score = score / (score.sum() + 1e-9)
        new_mean = (score.view(1, -1, 1) * elite_acts).sum(1)
        new_std = torch.sqrt((score.view(1, -1, 1) * (elite_acts - new_mean.unsqueeze(1)) ** 2).sum(1))
        mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
        std = new_std.clamp(min=cfg.min_std)
    mean_total, mean_rew, mean_fv = estimate_return(planner, z0, mean.clamp(-1, 1))
    return dict(mean_return=mean_total, mean_rew=mean_rew, mean_fv=mean_fv,
                best_return=best_return)


@torch.no_grad()
def dynamics_errors(model, states, actions, H):
    """states[T+1,obs], actions[T,act] on cuda. Returns lists of 1-step and H-step
    latent MSE per valid start index."""
    T = actions.shape[0]
    z_all = model.encode(states)                                # [T+1,z]
    err1, errH = [], []
    for t in range(T):
        z1 = model.next(z_all[t:t+1], actions[t:t+1])
        err1.append(float(F_mse(z1, z_all[t+1:t+2])))
    for t in range(T - H + 1):
        z = z_all[t:t+1]
        for k in range(H):
            z = model.next(z, actions[t+k:t+k+1])
        errH.append(float(F_mse(z, z_all[t+H:t+H+1])))
    return err1, errH


def F_mse(a, b):
    return ((a - b) ** 2).mean()


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def summ(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        print(f"  {name:32s}  (empty)")
        return
    print(f"  {name:32s}  mean {arr.mean():+.4f}  median {np.median(arr):+.4f}  "
          f"p10 {np.percentile(arr,10):+.4f}  p90 {np.percentile(arr,90):+.4f}  n={arr.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["target", "source"], default="target",
                    help="target = adapt WM in target env w/ target RL oracle; "
                         "source = stage-1 WM in nominal source env w/ source RL oracle")
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--adapt_ckpt", default="", help="required for --mode target")
    ap.add_argument("--rl_agent", default="", help="target RL oracle (--mode target)")
    ap.add_argument("--rl_source_agent",
                    default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt",
                    help="source RL oracle (--mode source)")
    ap.add_argument("--num_rl_episode", type=int, default=30)
    ap.add_argument("--num_mppi_episode", type=int, default=10)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import train_rl
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    DOM = args.mode.upper()   # "TARGET" or "SOURCE"
    print("=" * 78)
    if args.mode == "target":
        # TARGET: adapt-ckpt world model (value/reward FROZEN from stage-1) probed
        # by the target-trained RL oracle in the TARGET env.
        print("MODE = TARGET : adapt world model + TARGET env + target RL oracle")
        wm_agent, env_params, base_cfg = build_world_model_agent(args.stage1_ckpt, args.adapt_ckpt)
        rl_agent, _, _ = train_rl.load_model(args.rl_agent, "cuda")
    else:
        # SOURCE CONTROL: the SAME (stage-1) value/reward heads, but probed on their
        # HOME domain — the source RL oracle rolled in the nominal SOURCE env. The
        # value head was distilled to exactly V_src(s)=minQ_src(s,pi_src(s)) here,
        # so this is the "should-work" reference for the target numbers.
        print("MODE = SOURCE : stage-1 world model + SOURCE (nominal) env + source RL oracle")
        wm_agent, _, base_cfg = build_world_model_agent(args.stage1_ckpt, args.stage1_ckpt)
        rl_agent, _, env_params = train_rl.load_model(args.rl_source_agent, "cuda")
        # force nominal SOURCE dynamics (the source agent's own eval env is target)
        env_params = dict(env_params)
        env_params["domain_shift"] = DomainShiftConfig(mode="none")
        print("  overrode env domain_shift -> mode=none (SOURCE / nominal)")
    rl_agent.eval()
    planner = wm_agent.planner
    model = wm_agent.model
    H = planner.cfg.horizon
    print(f"  planner: H={H} num_samples={planner.cfg.num_samples} "
          f"num_base={planner._num_base()} iters={planner.cfg.iterations} "
          f"use_reward={planner.cfg.use_reward} discount={planner.cfg.discount}")
    print(f"  world model: latent_dim={model.cfg.latent_dim} discrete_heads="
          f"{model.cfg.discrete_heads} shared_encoder={model.cfg.shared_encoder}")

    # ONE shared env so both agents see identical dynamics/init distribution.
    env = PixelRobosuite(**env_params)

    print("=" * 78)
    print(f"Rolling out ORACLE (RL) policy in {DOM} env ...")
    rl_eps = rollout(env, rl_agent, args.num_rl_episode, 10_000, False, "RL")
    print(f"Rolling out MPPI PLANNER (world model) in {DOM} env ...")
    mppi_eps = rollout(env, wm_agent, args.num_mppi_episode, 20_000, True, "MPPI")

    dev = "cuda"
    rl_success = [e for e in rl_eps if e["success"]]
    print("=" * 78)
    print(f"ORACLE success episodes used for probing: {len(rl_success)}")

    # ---------------- (2) dynamics consistency ----------------
    print("\n" + "-" * 78)
    print("(2) DYNAMICS CONSISTENCY  ||next(z,a) - encode(s')||^2   (coverage test)")
    rl_e1, rl_eH = [], []
    for e in rl_success:
        a, b = dynamics_errors(model, e["states"].to(dev), e["actions"].to(dev), H)
        rl_e1 += a; rl_eH += b
    mp_e1, mp_eH = [], []
    for e in mppi_eps:
        a, b = dynamics_errors(model, e["states"].to(dev), e["actions"].to(dev), H)
        mp_e1 += a; mp_eH += b
    summ("oracle  1-step latent MSE", rl_e1)
    summ("planner 1-step latent MSE", mp_e1)
    summ(f"oracle  {H}-step latent MSE", rl_eH)
    summ(f"planner {H}-step latent MSE", mp_eH)
    if rl_e1 and mp_e1:
        ratio = np.median(rl_e1) / max(np.median(mp_e1), 1e-9)
        abs_oracle = np.median(rl_eH) if rl_eH else float("nan")
        # A coverage gap needs BOTH a large ratio AND a non-negligible ABSOLUTE
        # error (a 3x ratio on ~1e-5 latent MSE is just noise, not a real gap).
        coverage = ratio > 2 and abs_oracle > 1e-2
        print(f"  -> oracle/planner 1-step median ratio = {ratio:.2f}x ; "
              f"oracle {H}-step abs MSE = {abs_oracle:.2e}")
        print(f"     ({'COVERAGE GAP (dynamics inaccurate on oracle states)' if coverage else 'ok — dynamics is ACCURATE on oracle states (abs err negligible)'})")

    # ---------------- (1) return ranking ----------------
    print("\n" + "-" * 78)
    print("(1) RETURN RANKING  R_model(oracle H-seq)  vs  R_model(MPPI plan)")
    print("    (probed on ORACLE success states; MPPI re-planned from each)")
    diff_total, R_oracle, R_plan, R_best = [], [], [], []
    orc_rew, orc_fv, pl_rew, pl_fv = [], [], [], []
    # teacher-forced: oracle terminal value from the REAL reached state E(o_{t+H})
    # (NO dynamics rollout), and its roll-vs-real gap.
    o_fv_tf, R_oracle_tf, diff_tf = [], [], []
    gamma_H = planner.cfg.discount ** H
    # subsample (episode, start-t) pairs to bound MPPI re-plan cost
    max_states = 400
    pool = []
    for ei, e in enumerate(rl_success):
        T = e["actions"].shape[0]
        for t in range(T - H + 1):
            pool.append((ei, t))
    if len(pool) > max_states:
        idx = np.random.RandomState(0).choice(len(pool), max_states, replace=False)
        pool = [pool[i] for i in idx]
    for ei, t in pool:
        e = rl_success[ei]
        s_t = e["states"][t].to(dev)
        seq = e["actions"][t:t+H].to(dev)
        z0 = model.encode(s_t.view(1, -1))
        o_tot, o_rew, o_fv = estimate_return(planner, z0, seq)
        pf = plan_full(planner, s_t)
        # teacher-forced oracle terminal value: encode the REAL state H steps ahead
        s_tH = e["states"][t + H].to(dev)
        v_tf = float(model.predict_value(model.encode(s_tH.view(1, -1))))
        fv_tf = gamma_H * v_tf
        r_tf = o_rew + fv_tf if planner.cfg.use_reward else fv_tf
        o_fv_tf.append(fv_tf); R_oracle_tf.append(r_tf)
        diff_tf.append(r_tf - pf["mean_return"])
        R_oracle.append(o_tot); R_plan.append(pf["mean_return"]); R_best.append(pf["best_return"])
        diff_total.append(o_tot - pf["mean_return"])
        orc_rew.append(o_rew); orc_fv.append(o_fv)
        pl_rew.append(pf["mean_rew"]); pl_fv.append(pf["mean_fv"])
    summ("R_model(oracle)", R_oracle)
    summ("R_model(planner mean)", R_plan)
    summ("R_model(planner best cand)", R_best)
    summ("R_oracle - R_planner_mean", diff_total)
    print("  --- decomposition (reward-term / value-term) ---")
    summ("oracle  disc-reward term", orc_rew)
    summ("planner disc-reward term", pl_rew)
    summ("oracle  gamma^H V(z_H)", orc_fv)
    summ("planner gamma^H V(z_H)", pl_fv)
    d = np.asarray(diff_total)
    frac_oracle_worse = float((d < 0).mean())
    print(f"\n  -> fraction where model RANKS ORACLE BELOW its own plan: "
          f"{100*frac_oracle_worse:.1f}%")
    # also vs best candidate (did the planner's SEARCH miss the oracle?)
    d_best = np.asarray(R_oracle) - np.asarray(R_best)
    print(f"  -> fraction where R_model(oracle) > planner BEST candidate found: "
          f"{100*float((d_best>0).mean()):.1f}%  (search-miss signal)")
    if frac_oracle_worse > 0.5:
        print("  ==> VERDICT leans: REWARD/VALUE MODEL bottleneck (model misranks good actions).")
    else:
        print("  ==> VERDICT leans: PLANNER OPTIMIZATION bottleneck (model likes oracle, search misses it).")

    # ---------------- (1b) TEACHER-FORCED split: dynamics-OOD vs pure value ----
    print("\n" + "-" * 78)
    print("(1b) TEACHER-FORCED SPLIT  (is the misrank from dynamics rolling oracle")
    print("     actions OOD, or from V itself?).  Replace the oracle's dynamics-")
    print("     rolled terminal value gamma^H*V(z_H) with the REAL reached state")
    print("     gamma^H*V(E(o_{t+H})).  Planner side unchanged (dynamics-rolled).")
    summ("oracle gamma^H V(z_H)  [rolled]", orc_fv)
    summ("oracle gamma^H V(E(o_tH)) [real]", o_fv_tf)
    roll_gap = np.asarray(orc_fv) - np.asarray(o_fv_tf)
    summ("dyn-roll gap  (rolled - real)", roll_gap)
    frac_tf_worse = float((np.asarray(diff_tf) < 0).mean())
    print(f"\n  -> DYNAMICS-ROLLED misrank (from test 1) : {100*frac_oracle_worse:.1f}%")
    print(f"  -> TEACHER-FORCED  misrank (real state)  : {100*frac_tf_worse:.1f}%")
    drop = frac_oracle_worse - frac_tf_worse
    print(f"     (teacher-forcing the oracle terminal state removed "
          f"{100*drop:+.1f} pts of misrank)")
    if frac_tf_worse < 0.5 and drop > 0.15:
        print("  ==> Most of the misrank is DYNAMICS-OOD on oracle actions: V ranks the")
        print("      REAL reached state correctly; the dynamics rollout deflates it.")
        print("      FIX: inject off-policy (RL-rollout / elite+noise) data into D_real,")
        print("      finetune dynamics; the frozen-source VALUE head is mostly fine.")
    elif frac_tf_worse >= 0.5:
        print("  ==> Even on the REAL reached state, V still ranks the oracle below the")
        print("      planner => PURE VALUE pathology (last-mile near-success ranking).")
        print("      FIX: adapt/re-distill the VALUE head on target (e.g. oracle critic).")
    else:
        print("  ==> Mixed: both dynamics-OOD and value contribute.")

    # planner-side roll-vs-real gap: does dynamics mis-estimate terminal value MORE
    # on oracle actions than on the planner's OWN (in-distribution) actions?
    pl_roll_gap = []
    with torch.no_grad():
        for e in mppi_eps:
            S = e["states"].to(dev); A = e["actions"].to(dev)
            Tn = A.shape[0]
            zc = model.encode(S)
            for t in range(Tn - H + 1):
                z = zc[t:t+1]
                for k in range(H):
                    z = model.next(z, A[t+k:t+k+1])
                vr = float(model.predict_value(z))
                vreal = float(model.predict_value(zc[t+H:t+H+1]))
                pl_roll_gap.append(gamma_H * (vr - vreal))
    summ("planner dyn-roll gap (in-distrib)", pl_roll_gap)
    if len(roll_gap) and len(pl_roll_gap):
        print(f"  -> |roll gap| oracle {np.abs(roll_gap).mean():.4f}  vs  "
              f"planner {np.abs(np.asarray(pl_roll_gap)).mean():.4f}  "
              f"({'oracle actions are more OOD for dynamics' if np.abs(roll_gap).mean() > 1.5*np.abs(np.asarray(pl_roll_gap)).mean() else 'similar — dynamics not oracle-specifically OOD'})")

    # ---------------- (3) value calibration ----------------
    print("\n" + "-" * 78)
    print("(3) VALUE CALIBRATION  V_wm(encode(s))  vs  V_rl(s)=minQ(s,pi(s))")
    vwm, vrl, rew_pred_succ, rew_pred_nonsucc = [], [], [], []
    near_wm, near_rl = [], []   # last-K (near-success) states — the "last mile"
    NEAR_K = 10
    with torch.no_grad():
        for e in rl_success:
            S = e["states"].to(dev)                       # [T+1,obs]
            z = model.encode(S)
            vw = model.predict_value(z).detach().cpu().numpy()      # [T+1]
            vr = rl_agent.value({"state": S}).detach().cpu().numpy()  # [T+1]
            vwm += list(vw); vrl += list(vr)
            near_wm += list(vw[-NEAR_K:]); near_rl += list(vr[-NEAR_K:])
            # reward head at each transition, split by whether it was the success step
            A = e["actions"].to(dev); r = e["rewards"].numpy()
            rp = model.predict_reward(z[:-1], A).detach().cpu().numpy()
            rew_pred_succ += list(rp[r > 0.5]); rew_pred_nonsucc += list(rp[r <= 0.5])
    vwm = np.asarray(vwm); vrl = np.asarray(vrl)
    corr = float(np.corrcoef(vwm, vrl)[0, 1]) if vwm.size > 1 else float("nan")
    summ("V_wm  (world model value)", vwm)
    summ("V_rl  (oracle critic value)", vrl)
    print(f"  Pearson(V_wm, V_rl)  global    = {corr:+.3f}")
    print(f"  Spearman(V_wm,V_rl)  global    = {spearman(vwm, vrl):+.3f}")
    print(f"  Spearman(V_wm,V_rl)  last-{NEAR_K} steps = {spearman(np.asarray(near_wm), np.asarray(near_rl)):+.3f}"
          f"   <-- the near-success 'last mile' the global corr hides")
    summ("reward-head @ success steps", rew_pred_succ)
    summ("reward-head @ non-success steps", rew_pred_nonsucc)
    print("  (success reward-head should be ~1.0 and clearly > non-success)")

    print("\n" + "=" * 78)
    print("DONE.")
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
