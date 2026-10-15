"""Tracking-lag analysis of `policy_server.py --scripted track` runs (2026-10-15).

For each run (one --save_dir with steps.jsonl, optionally the client's --joint_log .npz):

  per (axis, amplitude):
    cmd speed    commanded grip-site displacement per step / dt  (eef_target(t) - eef(t))
    real speed   measured displacement per step / dt over the steady part of the move
    done/step    real displacement over the next step / commanded  (1.0 = the sim's assumption:
                 ReferenceArm already models the OSC's own lag, so the target IS what the sim
                 arm reaches in one step)
    overshoot    how far the measured grip site keeps going after the command stops (mm)
                 -- the quantity that put the gripper into the table at 20 Hz
  image age      camera frame age at the moment the client sent the step (ROS receive time:
                 a LOWER bound, exposure + USB transfer come on top)
  joint delay    (joint log only) time shift between the 100 Hz command and the measured joints

    python analyze_tracking.py <run_dir> [<run_dir> ...] [--fig out.jpg]
      a run dir may also hold joints.npz (robot_client.py --joint_log <run_dir>/joints.npz)
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--fig", default="")
args = ap.parse_args()


def load(run):
    rows = [json.loads(ln) for ln in open(os.path.join(run, "steps.jsonl")) if ln.strip()]
    return [r for r in rows if r.get("script")]


def segments(rows):
    """-> {(axis, amp): [rows]} in order."""
    out = {}
    for r in rows:
        s = r["script"]
        if s.get("phase") in (None, "done"):
            continue
        out.setdefault((s["axis"], s["amp"]), []).append(r)
    return out


def joint_delay(npz):
    d = np.load(npz)
    jt, jq, ct, cq = d["joint_t"], d["joint_q"], d["cmd_t"], d["cmd_q"]
    if len(ct) < 10 or len(jt) < 10:
        return None
    j = int(np.argmax(np.ptp(cq, 0)))                      # the joint that moved most
    t = np.arange(max(jt[0], ct[0]), min(jt[-1], ct[-1]), 0.001)
    real = np.interp(t, jt, jq[:, j])
    best = (None, np.inf)
    for tau in np.arange(0.0, 0.4, 0.001):
        cmd = np.interp(t - tau, ct, cq[:, j])
        e = float(np.mean((real - cmd) ** 2))
        if e < best[1]:
            best = (tau, e)
    return dict(joint=j + 1, delay_ms=1e3 * best[0], rms_mrad=1e3 * np.sqrt(best[1]),
                state_hz=(len(jt) - 1) / (jt[-1] - jt[0]))


report = {}
curves = []
for run in args.runs:
    rows = load(run)
    if not rows:
        print(f"{run}: no scripted steps")
        continue
    hz = float(rows[0].get("hz", 20.0))
    dt = 1.0 / hz
    print(f"\n=== {run}  ({hz:g} Hz, {len(rows)} steps)")
    print(f"{'axis':4s} {'amp':>4s} | {'cmd cm/s':>8s} {'real cm/s':>9s} {'done/step':>9s} | "
          f"{'overshoot mm':>12s} {'settle s':>8s}")
    rep = {}
    for (ax, amp), seg in segments(rows).items():
        j = "xyz".index(ax)
        e = np.array([r["eef_full"] for r in seg])[:, j]
        tg = np.array([r["eef_target_full"] for r in seg])[:, j]
        ph = [r["script"]["phase"] for r in seg]
        mv = np.array([r["script"]["moved"] for r in seg])
        away = [i for i, p in enumerate(ph) if p == "away"]
        steady = away[len(away) // 3:]                    # skip the acceleration
        cmd = np.abs(tg[steady] - e[steady]) / dt if steady else np.array([np.nan])
        nxt = [i for i in steady if i + 1 < len(e)]
        real = np.abs(np.diff(e))[nxt] / dt if nxt else np.array([np.nan])
        ratio = (np.abs(e[[i + 1 for i in nxt]] - e[nxt]) / np.maximum(np.abs(tg[nxt] - e[nxt]), 1e-6)
                 if nxt else np.array([np.nan]))
        h1 = [i for i, p in enumerate(ph) if p == "hold1"]
        if h1:
            over = 1e3 * (mv[h1].max() - mv[h1[0]])     # beyond where it was when the command stopped
            fin = mv[h1[-1]]
            settle = next((k * dt for k, i in enumerate(h1) if abs(mv[i] - fin) < 5e-4), np.nan)
        else:
            over, settle = np.nan, np.nan
        rep[f"{ax}{amp:g}"] = dict(cmd_cm_s=float(100 * np.median(cmd)), real_cm_s=float(100 * np.median(real)),
                                   done_per_step=float(np.median(ratio)), overshoot_mm=float(over),
                                   settle_s=float(settle))
        print(f"{ax:4s} {amp:4.1f} | {100 * np.median(cmd):8.1f} {100 * np.median(real):9.1f} "
              f"{np.median(ratio):9.2f} | {over:12.1f} {settle:8.2f}")
        curves.append((f"{os.path.basename(run.rstrip('/'))} {ax}{amp:g}", dt, e - e[0], tg - e[0], ph))
    ages = {}
    for r in rows:
        for k, v in (r.get("ages") or {}).items():
            ages.setdefault(k, []).append(v)
    for k, v in ages.items():
        print(f"  age {k:8s} p50 {1e3 * np.median(v):5.0f} ms  p90 {1e3 * np.percentile(v, 90):5.0f} ms")
    npz = os.path.join(run, "joints.npz")
    jd = joint_delay(npz) if os.path.exists(npz) else None
    if jd:
        print(f"  joint {jd['joint']}: measured lags the 100 Hz command by {jd['delay_ms']:.0f} ms "
              f"(residual {jd['rms_mrad']:.2f} mrad), joint states at {jd['state_hz']:.0f} Hz")
    report[run] = dict(hz=hz, segments=rep, joint_delay=jd,
                       ages={k: float(np.median(v)) for k, v in ages.items()})
    json.dump(report[run], open(os.path.join(run, "tracking_report.json"), "w"), indent=1)

if args.fig and curves:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(curves)
    fig, axs = plt.subplots((n + 2) // 3, 3, figsize=(15, 3.2 * ((n + 2) // 3)), squeeze=False)
    for a, (name, dt, e, tg, ph) in zip(axs.ravel(), curves):
        t = np.arange(len(e)) * dt
        a.plot(t, 1e3 * tg, "r--", lw=1, label="target (sim OSC, 1 step)")
        a.plot(t, 1e3 * e, "k", lw=1.5, label="measured")
        a.set_title(name, fontsize=9)
        a.set_xlabel("s")
        a.set_ylabel("mm")
    axs.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=90)
    print("wrote", args.fig)
