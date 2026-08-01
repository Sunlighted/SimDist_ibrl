"""V-gated DAgger-style intervention eval.

Roll out a BC policy (default: threading_target_mlp_bc_n20_seed1) in the SOURCE
domain, and at every step estimate the state value V(s) with a source RL agent
(default: threading_ibrl_source_strict_seed1):

    V(s) = E_a~pi[ Q(s, a) ]  ≈  min-ensemble Q_src(s, pi_src(s))   (QAgent.value)

Idea: V(s) measures "can this still be saved" under the source RL policy. When
V drops below a threshold (the trajectory is "about to crash"), the source RL
agent TAKES OVER control (DAgger-style) until V recovers above a (possibly
higher, hysteresis) threshold, then control switches back to the BC policy.

The script is essentially an eval loop with controller switching + logging, plus
per-episode visualization of V(t) with the takeover regions shaded.

Default config = the validated TARGET-domain "early intervention" setup:
target BC (threading_target_mlp_bc_n20) rolled out in the §12 target domain, a
target RL agent (threading_ibrl_newtgt) for V + takeover, and EARLY takeover
(rel_drop catches the V dip onset, while the state is still in the RL's
distribution). On Threading this lifts target success 50% -> 90% with ~12%
takeover. Late takeover (waiting for V to crash) does NOT rescue in target —
the RL can't recover from the OOD death states BC drives into.

Outputs (plots + videos + traces.npz) go to <rl_run_dir>/intervention_eval/
by default, i.e. exps/mimicgen/threading_ibrl_newtgt_seed1/intervention_eval/.

Examples
--------
    source set_env.sh
    # default: target BC + target RL + target domain, early intervention
    python scripts/intervention_eval.py

    # SOURCE domain instead (source RL agent, no shift):
    python scripts/intervention_eval.py \
        --rl_weight exps/mimicgen/threading_ibrl_source_strict_seed1/model0.pt \
        --shift_from_rl_cfg 0

    # ablations: pure BC (never take over) vs always-RL-from-reset
    python scripts/intervention_eval.py --threshold -1 --rel_drop 0 --detect_stuck 0
    python scripts/intervention_eval.py --threshold 2.0 --recover_threshold 3.0
"""

import argparse
import os
from collections import deque

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("MUJOCO_GL", "egl")

import common_utils
from common_utils import ibrl_utils as utils
from common_utils import Recorder
from env.robosuite_wrapper import PixelRobosuite
from env.strict_success import StrictSuccessConfig
from env.domain_shift import DomainShiftConfig
import train_bc
import train_rl


DEFAULT_BC = "exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt"
# target RL agent (98% on target) for V + takeover; with --shift_from_rl_cfg the
# rollout domain follows its eval_domain_shift (= the §12 target).
DEFAULT_RL = "exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt"


def domain_shift_from_cfg(weight_path: str) -> DomainShiftConfig:
    """Build a DomainShiftConfig from <weight_dir>/cfg.yaml's eval_domain_shift.

    Mirrors evaluate/eval.py --shift_from_cfg: lets the rollout domain follow the
    RL agent's own eval domain (source agent -> mode=none -> source; a *target*
    RL agent like threading_ibrl_newtgt -> the §12 dynamics-only target).
    """
    import dataclasses
    import yaml

    cfg_path = os.path.join(os.path.dirname(weight_path), "cfg.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}
    cfg_shift = raw.get("eval_domain_shift", {}) or {}
    kwargs = {}
    for k, v in cfg_shift.items():
        kwargs[k] = tuple(v) if isinstance(v, list) else v
    valid = {fld.name for fld in dataclasses.fields(DomainShiftConfig)}
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return DomainShiftConfig(**kwargs)


def estimate_value(rl_agent, obs) -> float:
    """V(s) = QAgent.value on a single (un-batched) env obs dict."""
    batched = {k: v.unsqueeze(0) for k, v in obs.items() if torch.is_tensor(v)}
    v = rl_agent.value(batched)
    return float(v.reshape(-1)[0].item())


def bc_action(bc_agent, obs, queue: deque):
    """Pop the next BC action, refilling the chunk queue when empty."""
    if not queue:
        if hasattr(bc_agent, "act_chunk"):
            chunk = bc_agent.act_chunk(obs, eval_mode=True)  # (Ta, D)
        else:
            chunk = bc_agent.act(obs, eval_mode=True).unsqueeze(0)
        for a in chunk:
            queue.append(a)
    return queue.popleft()


def _state_vec(obs):
    """Flat CPU numpy state for stagnation measurement."""
    return obs["state"].detach().reshape(-1).float().cpu().numpy()


def rollout_episode(
    env: PixelRobosuite,
    bc_agent,
    rl_agent,
    *,
    threshold: float,
    recover_threshold: float,
    min_takeover_steps: int,
    rel_drop: float,
    rel_window: int,
    detect_stuck: bool,
    stuck_window: int,
    stuck_eps: float,
    recorder: Recorder | None,
):
    """One episode. Returns a dict of per-step traces + episode summary.

    Two ORTHOGONAL takeover triggers (BC -> RL):
      * low-V : V(s) < threshold (or a relative drop)        — "about to crash"
      * stuck : net state displacement over the last
                `stuck_window` steps < `stuck_eps`           — "frozen / hovering,
                never finishing" (the high-flat-V failure mode)
    """
    obs, image_obs = env.reset()

    values: list[float] = []
    controllers: list[str] = []          # "bc" / "rl" per step
    reasons: list[str] = []              # "none"/"low_v"/"stuck" (why RL is driving)
    movements: list[float] = []          # net state displacement over the window
    rewards: list[float] = []

    controller = "bc"
    active_reason = "none"
    bc_queue: deque = deque()
    steps_in_takeover = 0
    recent_v: deque = deque(maxlen=rel_window)
    state_hist: deque = deque(maxlen=stuck_window + 1)
    terminal = False

    while not terminal:
        if recorder is not None:
            recorder.add(image_obs)

        v = estimate_value(rl_agent, obs)
        values.append(v)
        recent_v.append(v)

        s = _state_vec(obs)
        state_hist.append(s)
        # net displacement vs `stuck_window` steps ago (jitter-robust: net, not
        # path length — hovering in place reads as ~0 even with small wiggles).
        if len(state_hist) > stuck_window:
            move = float(np.linalg.norm(s - state_hist[0]))
        else:
            move = float("nan")  # not enough history yet
        movements.append(move)
        is_stuck = detect_stuck and (not np.isnan(move)) and (move < stuck_eps)

        # ---- decide controller (two triggers + hysteresis) -------------
        v_ref = max(recent_v) if len(recent_v) > 0 else v
        drop_trigger = rel_drop > 0 and (v < v_ref - rel_drop)
        low_v = (v < threshold) or drop_trigger
        if controller == "bc":
            if low_v:
                controller, active_reason = "rl", "low_v"
                steps_in_takeover = 0
                bc_queue.clear()  # stale chunk; refill when we return to BC
            elif is_stuck:
                controller, active_reason = "rl", "stuck"
                steps_in_takeover = 0
                bc_queue.clear()
        else:  # RL is driving — return to BC only once recovered AND moving
            recovered = v >= recover_threshold and not is_stuck
            if recovered and steps_in_takeover >= min_takeover_steps:
                controller, active_reason = "bc", "none"

        # ---- act -------------------------------------------------------
        if controller == "rl":
            action = rl_agent.act(obs, eval_mode=True)
            steps_in_takeover += 1
        else:
            action = bc_action(bc_agent, obs, bc_queue)

        controllers.append(controller)
        reasons.append(active_reason if controller == "rl" else "none")
        obs, reward, terminal, _, image_obs = env.step(action)
        rewards.append(float(reward))

    score = float(np.sum(rewards))
    ctrl_arr = np.array([1 if c == "rl" else 0 for c in controllers])
    n_takeover = int(np.sum(ctrl_arr))
    # count distinct takeover bursts (rising edges)
    n_interventions = int(np.sum((ctrl_arr[1:] == 1) & (ctrl_arr[:-1] == 0)))
    if len(ctrl_arr) and ctrl_arr[0] == 1:
        n_interventions += 1
    n_lowv = sum(r == "low_v" for r in reasons)
    n_stuck = sum(r == "stuck" for r in reasons)

    return {
        "values": np.array(values),
        "movements": np.array(movements),
        "controllers": controllers,
        "reasons": reasons,
        "score": score,
        "success": score > 0,
        "ep_len": len(values),
        "n_takeover_steps": n_takeover,
        "n_interventions": n_interventions,
        "n_lowv_steps": n_lowv,
        "n_stuck_steps": n_stuck,
        "takeover_frac": n_takeover / max(1, len(ctrl_arr)),
    }


def plot_episode(trace, threshold, recover_threshold, stuck_eps, out_path, title):
    values = trace["values"]
    moves = trace["movements"]
    reasons = trace["reasons"]
    t = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(t, values, color="C0", lw=1.5, label="V(s) (source RL)")

    # shade RL-takeover regions, colored by trigger reason
    reason_color = {"low_v": "C3", "stuck": "C1"}
    labeled = set()
    i = 0
    n = len(reasons)
    while i < n:
        if reasons[i] in reason_color:
            r = reasons[i]
            j = i
            while j < n and reasons[j] == r:
                j += 1
            lbl = f"takeover: {r}" if r not in labeled else None
            labeled.add(r)
            ax.axvspan(i, j, color=reason_color[r], alpha=0.20, label=lbl)
            i = j
        else:
            i += 1

    ax.axhline(threshold, color="C3", ls="--", lw=1, label=f"V thr={threshold:g}")
    if recover_threshold != threshold:
        ax.axhline(recover_threshold, color="C2", ls=":", lw=1,
                   label=f"recover thr={recover_threshold:g}")

    # second axis: net state displacement over the stuck window + stuck floor
    ax2 = ax.twinx()
    ax2.plot(t, moves, color="0.5", lw=0.9, alpha=0.7, label="state move (window)")
    ax2.axhline(stuck_eps, color="C1", ls="--", lw=1, label=f"stuck eps={stuck_eps:g}")
    ax2.set_ylabel("net state displacement", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")
    ax2.set_ylim(bottom=0)

    ax.set_xlabel("env step")
    ax.set_ylabel("V(s)")
    ax.set_title(title)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc_weight", type=str, default=DEFAULT_BC)
    p.add_argument("--rl_weight", type=str, default=DEFAULT_RL)
    p.add_argument("--num_game", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    # V-gating knobs (defaults = the validated TARGET early-intervention config)
    p.add_argument("--threshold", type=float, default=0.32,
                   help="Take over when V(s) drops below this (absolute).")
    p.add_argument("--recover_threshold", type=float, default=0.5,
                   help="Return to BC when V >= this (hysteresis). "
                        "Must be >= --threshold.")
    p.add_argument("--min_takeover_steps", type=int, default=10,
                   help="Min consecutive RL steps before switching back to BC.")
    p.add_argument("--rel_drop", type=float, default=0.08,
                   help="EARLY trigger: take over when V drops > this below the "
                        "recent max (catches the dip onset before V crashes — "
                        "the key lever for target-domain rescue). 0 = disabled.")
    p.add_argument("--rel_window", type=int, default=8,
                   help="Window (steps) for the --rel_drop recent-max.")
    # stuck (stagnation) trigger — catches the high-flat-V "hovering, never
    # finishes" failure that V-thresholding alone misses.
    p.add_argument("--detect_stuck", type=int, default=1,
                   help="1 = also take over when the state stops moving "
                        "(net displacement over --stuck_window < --stuck_eps).")
    p.add_argument("--stuck_window", type=int, default=15,
                   help="Window (steps) for the net-displacement stuck check.")
    p.add_argument("--stuck_eps", type=float, default=0.02,
                   help="Net state displacement below this over the window "
                        "= stuck. Tune vs the printed move distribution.")
    p.add_argument("--takeover_act_method", type=str, default=None,
                   help="Override RL act_method while it drives "
                        "(rl/ibrl/ibrl_soft). Default = the agent's own.")
    p.add_argument("--shift_from_rl_cfg", type=int, default=1,
                   help="1 = roll out in the RL agent's OWN eval domain (read "
                        "eval_domain_shift from its cfg.yaml). With a source "
                        "agent this is mode=none (source); with a target RL "
                        "agent (e.g. threading_ibrl_newtgt_seed1) this is the "
                        "§12 target. Keeps V + takeover in-domain. Default 0 = "
                        "source (BC cfg env, shift stripped).")
    # strict-success gating (anti reward-hack). Default ON to match the
    # source_strict RL agent's training regime + honest scoring; the env_params
    # from train_bc.load_model carry NO strict gate, so we add it here.
    p.add_argument("--strict_success", type=int, default=1,
                   help="1 = strict success gate at rollout (dwell + ring-disp); "
                        "0 = raw upstream instantaneous position test.")
    p.add_argument("--strict_dwell_steps", type=int, default=3)
    p.add_argument("--strict_max_tripod_disp", type=float, default=0.04)
    p.add_argument("--strict_max_needle_speed", type=float, default=0.0)
    # output
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write plots/videos/npz. Default: "
                        "<rl_run_dir>/intervention_eval.")
    p.add_argument("--no_video", action="store_true", help="Skip video recording.")
    p.add_argument("--no_plot", action="store_true", help="Skip per-episode plots.")
    args = p.parse_args()

    if args.recover_threshold is None:
        args.recover_threshold = args.threshold
    assert args.recover_threshold >= args.threshold, \
        "recover_threshold must be >= threshold for sane hysteresis"

    common_utils.set_all_seeds(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.rl_weight), "intervention_eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[intervention] output dir: {out_dir}")

    # ---- load both agents -------------------------------------------------
    print(common_utils.wrap_ruler("loading BC policy"))
    bc_agent, _, env_params = train_bc.load_model(args.bc_weight, args.device, verbose=False)
    print(common_utils.wrap_ruler("loading source RL agent (value + takeover)"))
    rl_agent, _, _ = train_rl.load_model(args.rl_weight, args.device)

    if args.takeover_act_method is not None and hasattr(rl_agent, "cfg"):
        print(f"[intervention] RL act_method {rl_agent.cfg.act_method} "
              f"-> {args.takeover_act_method}")
        rl_agent.cfg.act_method = args.takeover_act_method

    # rollout env: SOURCE by default (train_bc.load_model strips the BC cfg's
    # eval_domain_shift). With --shift_from_rl_cfg, follow the RL agent's own
    # eval domain (target RL agent -> target rollout). Build our own env so we
    # control the domain + video + strict gate.
    env_params = dict(env_params)
    if args.shift_from_rl_cfg:
        env_params["domain_shift"] = domain_shift_from_cfg(args.rl_weight)
        print(f"[intervention] domain_shift (from RL cfg) = {env_params['domain_shift']}")
    else:
        print("[intervention] domain = SOURCE (no shift)")
    if args.strict_success:
        env_params["strict_success"] = StrictSuccessConfig(
            enabled=True,
            dwell_steps=args.strict_dwell_steps,
            max_tripod_disp=args.strict_max_tripod_disp,
            max_needle_speed=args.strict_max_needle_speed,
        )
        print(f"[intervention] strict_success = {env_params['strict_success']}")
    else:
        print("[intervention] strict_success OFF (raw upstream success)")
    record = not args.no_video
    if record:
        env_params["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env = PixelRobosuite(**env_params)

    # ---- rollout ----------------------------------------------------------
    traces = []
    with torch.no_grad(), utils.eval_mode(rl_agent), utils.eval_mode(bc_agent):
        for ep in range(args.num_game):
            np.random.seed(args.seed + ep)
            recorder = Recorder(out_dir) if record else None
            trace = rollout_episode(
                env, bc_agent, rl_agent,
                threshold=args.threshold,
                recover_threshold=args.recover_threshold,
                min_takeover_steps=args.min_takeover_steps,
                rel_drop=args.rel_drop,
                rel_window=args.rel_window,
                detect_stuck=bool(args.detect_stuck),
                stuck_window=args.stuck_window,
                stuck_eps=args.stuck_eps,
                recorder=recorder,
            )
            traces.append(trace)

            tag = "SUCCESS" if trace["success"] else "fail   "
            print(f"ep {ep:2d} | {tag} | len {trace['ep_len']:3d} | "
                  f"takeover {trace['n_takeover_steps']:3d} steps "
                  f"({100*trace['takeover_frac']:4.1f}%) in "
                  f"{trace['n_interventions']} bursts "
                  f"[low_v {trace['n_lowv_steps']}, stuck {trace['n_stuck_steps']}] | "
                  f"V min {trace['values'].min():.3f}")

            if recorder is not None:
                recorder.save(f"ep{ep:02d}_seed{args.seed+ep}_{tag.strip()}")
            if not args.no_plot:
                plot_episode(
                    trace, args.threshold, args.recover_threshold, args.stuck_eps,
                    os.path.join(out_dir, f"ep{ep:02d}_value.png"),
                    title=(f"ep {ep} ({tag.strip()}) | takeover "
                           f"{100*trace['takeover_frac']:.0f}% "
                           f"(low_v {trace['n_lowv_steps']}, stuck {trace['n_stuck_steps']})"),
                )

    # ---- summary ----------------------------------------------------------
    succ = np.array([t["success"] for t in traces])
    scores = np.array([t["score"] for t in traces])
    to_frac = np.array([t["takeover_frac"] for t in traces])
    n_int = np.array([t["n_interventions"] for t in traces])
    all_v = np.concatenate([t["values"] for t in traces])
    all_move = np.concatenate([t["movements"] for t in traces])
    all_move = all_move[~np.isnan(all_move)]
    n_lowv = np.array([t["n_lowv_steps"] for t in traces])
    n_stuck = np.array([t["n_stuck_steps"] for t in traces])

    print(common_utils.wrap_ruler("summary"))
    print(f"episodes:            {len(traces)}")
    print(f"success rate:        {100*succ.mean():.1f}%  ({int(succ.sum())}/{len(traces)})")
    print(f"mean score:          {scores.mean():.3f}")
    print(f"mean takeover frac:  {100*to_frac.mean():.1f}% of steps "
          f"(low_v {n_lowv.sum()} steps, stuck {n_stuck.sum()} steps total)")
    print(f"mean interventions:  {n_int.mean():.2f} bursts/episode")
    print(f"V over all steps:    mean {all_v.mean():.3f}  "
          f"p10 {np.percentile(all_v,10):.3f}  p50 {np.percentile(all_v,50):.3f}  "
          f"p90 {np.percentile(all_v,90):.3f}  min {all_v.min():.3f}  max {all_v.max():.3f}")
    print(f"  -> tune --threshold relative to this V distribution")
    if len(all_move):
        print(f"state move (window): p5 {np.percentile(all_move,5):.3f}  "
              f"p10 {np.percentile(all_move,10):.3f}  p50 {np.percentile(all_move,50):.3f}  "
              f"p90 {np.percentile(all_move,90):.3f}  min {all_move.min():.3f}")
        print(f"  -> tune --stuck_eps relative to this displacement distribution")

    # persist raw traces for later analysis
    npz_path = os.path.join(out_dir, "traces.npz")
    np.savez(
        npz_path,
        values=np.array([t["values"] for t in traces], dtype=object),
        movements=np.array([t["movements"] for t in traces], dtype=object),
        controllers=np.array([t["controllers"] for t in traces], dtype=object),
        reasons=np.array([t["reasons"] for t in traces], dtype=object),
        success=succ, scores=scores, takeover_frac=to_frac, n_interventions=n_int,
        n_lowv_steps=n_lowv, n_stuck_steps=n_stuck,
        threshold=args.threshold, recover_threshold=args.recover_threshold,
        stuck_eps=args.stuck_eps, stuck_window=args.stuck_window,
    )
    print(f"saved traces -> {npz_path}")
    if not args.no_plot:
        print(f"saved per-episode V plots -> {out_dir}/epNN_value.png")
    if record:
        print(f"saved videos -> {out_dir}/epNN_*.mp4")

    os._exit(0)  # EGL teardown can hang


if __name__ == "__main__":
    main()
