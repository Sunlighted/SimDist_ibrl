"""Online DAgger adaptation loop: few-shot improve a co-trained BC student on
the TARGET domain using a strong TARGET RL expert (chat 2026-06-13).

Setup
-----
  student : a co-trained MLP-BC policy (~50% TARGET success), e.g.
            exps/mimicgen/threading_cotrain_unweighted_n1000_seed1/model0.pt
  expert  : a strong TARGET RL agent (~98%), e.g.
            exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt
  goal    : push the student toward the expert on TARGET, PURE IL (no RL on the
            student), few-shot. Upper bound = the expert's success rate.

Each round (default 5 trajectories collected, 10 rounds => 50 traj budget):
  1. COLLECT (subprocess scripts/collect_target_recovery.py): roll the CURRENT
     student in the TARGET sim; the expert's V(s) gates a DAgger takeover when
     the student is about to crash; keep `--n_per_round` type-C
     (took_over AND strict-success) episodes and APPEND them to one cumulative
     recovery pool (DAgger Dataset Aggregation).
  2. TRAIN (subprocess train_weighted_bc.py --init_policy_ckpt <seed>): continue
     cotraining (source 1000 + target 20 + the growing recovery pool, the
     recovery channel up-weighted) from the previous round's best ckpt. Eval
     every `--epoch_len` steps on `--num_eval_episode` games; TopkSaver keeps
     the round's best (model0.pt).
  3. SELECT: the trainer's final 500-game TARGET eval picks the round's best
     ckpt; we parse it, record the score, and seed the NEXT round's collection
     with that ckpt.

Records <out_root>/results.json + prints a round-vs-best-target table so you can
see whether (and how fast) success improves with trajectory budget.

NOTE on cost: "eval every 1000 steps on 200 games" is expensive. With
--inner_epochs E and --num_eval_episode 200 that is E*200 eval episodes per
round PLUS the trainer's final 3*500 topk eval. Dial --inner_epochs /
--num_eval_episode to trade resolution for wall-clock.

Example
-------
    source set_env.sh
    python scripts/online_dagger_adapt.py \
        --student exps/mimicgen/threading_cotrain_unweighted_n1000_seed1/model0.pt \
        --expert  exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt \
        --out_root exps/mimicgen/threading_online_dagger_seed1 \
        --n_per_round 5 --max_rounds 10 --inner_epochs 20
"""

import argparse
import json
import os
import re
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BEST_RE = re.compile(r"\[final eval \| BEST.*?\]\s*target\s*=\s*([0-9.]+)\s*\((.*?)\)")
_SRC_RE = re.compile(r"\[final eval \| source\]\s*score\s*=\s*([0-9.]+)")


def run_subprocess(cmd, log_path):
    """Run cmd, tee combined stdout/stderr to log_path AND the console."""
    print(f"\n>>> {' '.join(cmd)}\n    (log -> {log_path})", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess failed (rc={proc.returncode}): {' '.join(cmd)}")


def parse_best(train_log_path):
    """Return (best_target_500, best_ckpt_path, source_500) from a train.log,
    or (None, None, None) if the final-eval line is absent (round incomplete)."""
    if not os.path.exists(train_log_path):
        return None, None, None
    tgt, ckpt, src = None, None, None
    with open(train_log_path, "r") as f:
        txt = f.read()
    m = _BEST_RE.search(txt)
    if m:
        tgt = float(m.group(1))
        ckpt = m.group(2).strip()
    ms = _SRC_RE.search(txt)
    if ms:
        src = float(ms.group(1))
    return tgt, ckpt, src


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", required=True, help="initial seed BC ckpt.")
    p.add_argument("--expert", required=True, help="TARGET RL expert ckpt.")
    p.add_argument("--out_root", required=True, help="run dir for all artifacts.")
    p.add_argument("--train_cfg", default=None,
                   help="config_path for continue-training (default: the "
                        "student's sibling cfg.yaml).")
    p.add_argument("--shift_cfg", default=None,
                   help="cfg.yaml whose eval_domain_shift defines the TARGET "
                        "(default: --train_cfg).")
    p.add_argument("--n_per_round", type=int, default=5)
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--inner_epochs", type=int, default=20,
                   help="num_epoch per round; total steps/round = this*epoch_len.")
    p.add_argument("--epoch_len", type=int, default=1000,
                   help="grad steps between in-training evals (the '1000 steps').")
    p.add_argument("--eval_period", type=int, default=1)
    p.add_argument("--num_eval_episode", type=int, default=200)
    p.add_argument("--recovery_batch_ratio", type=float, default=0.25)
    p.add_argument("--recovery_weight", type=float, default=1.0)
    p.add_argument("--hand_init_perturb_joint_sigma", type=float, default=0.1)
    p.add_argument("--takeover_act_method", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--use_wb", type=int, default=0)
    p.add_argument("--resume", type=int, default=1,
                   help="1: skip rounds whose train.log already has a final eval.")
    args = p.parse_args()

    train_cfg = args.train_cfg or os.path.join(os.path.dirname(args.student), "cfg.yaml")
    shift_cfg = args.shift_cfg or train_cfg
    assert os.path.exists(train_cfg), f"train_cfg not found: {train_cfg}"
    assert os.path.exists(shift_cfg), f"shift_cfg not found: {shift_cfg}"

    os.makedirs(args.out_root, exist_ok=True)
    recovery_dir = os.path.join(args.out_root, "recovery")
    recovery_h5 = os.path.join(recovery_dir, "dataset.hdf5")
    results_path = os.path.join(args.out_root, "results.json")

    results = []
    if os.path.exists(results_path):
        results = json.load(open(results_path))

    # Each round continues from the GLOBAL-BEST ckpt so far (NOT the latest
    # round's). Continue-training is high-variance: a round can regress below a
    # previous one (observed 2026-06-13: round 3 hit 0.58, round 4 fell back to
    # 0.496). Seeding the next round from the latest (possibly regressed) ckpt
    # propagates the regression and restarts the climb from a worse point.
    # Seeding from the best-ever ckpt makes progress monotone-ish and faithful
    # to "pick the best checkpoint for the next round".
    best_score = -1.0
    best_ckpt = args.student  # round 0 seeds from the original student
    for d in results:
        if d.get("best_target_500") is not None and d["best_target_500"] > best_score:
            best_score = d["best_target_500"]
            best_ckpt = d["best_ckpt"]

    print("=" * 70)
    print("ONLINE DAgger ADAPTATION")
    print(f"  student (seed) : {args.student}")
    print(f"  expert         : {args.expert}")
    print(f"  train_cfg      : {train_cfg}")
    print(f"  shift_cfg      : {shift_cfg}")
    print(f"  out_root       : {args.out_root}")
    print(f"  {args.n_per_round} traj/round x {args.max_rounds} rounds = "
          f"{args.n_per_round * args.max_rounds} traj budget")
    print(f"  inner: {args.inner_epochs} epoch x {args.epoch_len} steps, "
          f"eval/{args.eval_period}ep x {args.num_eval_episode} games")
    print("=" * 70, flush=True)

    for r in range(args.max_rounds):
        round_dir = os.path.join(args.out_root, f"round_{r}")
        round_log = os.path.join(round_dir, "train.log")

        # ---- resume: skip a fully-completed round ----
        if args.resume:
            tgt, ckpt, src = parse_best(round_log)
            if tgt is not None and ckpt and os.path.exists(ckpt):
                if tgt > best_score:
                    best_score, best_ckpt = tgt, ckpt
                print(f"[resume] round {r} already done: best_target={tgt:.4f} "
                      f"({ckpt}); skipping. global-best so far={best_score:.4f}")
                # ensure it is recorded
                if not any(d["round"] == r for d in results):
                    pool = _pool_size(recovery_h5)
                    results.append(dict(round=r, pool_demos=pool, best_target_500=tgt,
                                        source_500=src, best_ckpt=ckpt))
                    json.dump(results, open(results_path, "w"), indent=2)
                continue

        os.makedirs(round_dir, exist_ok=True)
        # seed THIS round's collection + continue-training from the global best.
        seed_ckpt = best_ckpt
        print(f"[round {r}] seeding from global-best ({best_score:.4f}): {seed_ckpt}",
              flush=True)

        # ---- 1) COLLECT n_per_round type-C episodes (append to cumulative pool) ----
        collect_log = os.path.join(round_dir, "collect.log")
        collect_cmd = [
            sys.executable, "scripts/collect_target_recovery.py",
            "--student", seed_ckpt, "--expert", args.expert,
            "--shift_cfg", shift_cfg, "--out_dir", recovery_dir,
            "--n_collect", str(args.n_per_round),
            "--seed", str(args.seed), "--ep_offset", str(r * 100000),
            "--hand_init_perturb_joint_sigma", str(args.hand_init_perturb_joint_sigma),
        ]
        if args.takeover_act_method:
            collect_cmd += ["--takeover_act_method", args.takeover_act_method]
        run_subprocess(collect_cmd, collect_log)
        pool = _pool_size(recovery_h5)
        print(f"[round {r}] recovery pool now {pool} demos", flush=True)

        # ---- 2) TRAIN: continue cotraining from seed_ckpt with the recovery pool ----
        train_cmd = [
            sys.executable, "train_weighted_bc.py",
            "--config_path", train_cfg,
            "--init_policy_ckpt", seed_ckpt,
            "--recovery_path", recovery_h5,
            "--recovery_ep_types", "C",
            "--recovery_batch_ratio", str(args.recovery_batch_ratio),
            "--recovery_weight", str(args.recovery_weight),
            "--num_epoch", str(args.inner_epochs),
            "--epoch_len", str(args.epoch_len),
            "--eval_period", str(args.eval_period),
            "--num_eval_episode", str(args.num_eval_episode),
            "--save_dir", round_dir,
            "--use_wb", str(args.use_wb),
            "--seed", str(args.seed),
        ]
        run_subprocess(train_cmd, round_log)

        # ---- 3) SELECT round-best (parsed from the trainer's final 500-eval) ----
        tgt, ckpt, src = parse_best(round_log)
        if tgt is None or not ckpt:
            # fallback to model0.pt (TopkSaver best by in-training eval)
            ckpt = os.path.join(round_dir, "model0.pt")
            print(f"[round {r}] WARN: could not parse final eval; "
                  f"falling back to {ckpt}")
        assert os.path.exists(ckpt), f"round-best ckpt missing: {ckpt}"
        if tgt is not None and tgt > best_score:
            best_score, best_ckpt = tgt, ckpt

        results.append(dict(round=r, pool_demos=pool,
                            best_target_500=tgt, source_500=src, best_ckpt=ckpt))
        json.dump(results, open(results_path, "w"), indent=2)
        print(f"[round {r}] best_target_500={tgt}  source_500={src}  | "
              f"global-best={best_score:.4f} ({best_ckpt})", flush=True)
        _print_table(results)

    print("\n" + "=" * 70)
    print("ONLINE DAgger DONE")
    _print_table(results)
    print(f"results -> {results_path}")


def _pool_size(h5_path):
    if not os.path.exists(h5_path):
        return 0
    try:
        import h5py
        with h5py.File(h5_path, "r") as hf:
            return int(hf["data"].attrs["num_demos"])
    except Exception:
        return -1


def _print_table(results):
    print("\n  round | pool | best_target(500) | source(500)")
    print("  ------+------+------------------+------------")
    base = None
    for d in results:
        if base is None and d["best_target_500"] is not None:
            base = d["best_target_500"]
        t = d["best_target_500"]
        s = d["source_500"]
        ts = f"{t:.4f}" if t is not None else "  n/a "
        ss = f"{s:.4f}" if s is not None else "  n/a "
        delta = f" (+{t - base:.3f})" if (t is not None and base is not None) else ""
        print(f"  {d['round']:5d} | {d['pool_demos']:4d} | {ts:>16}{delta} | {ss:>10}")
    print(flush=True)


if __name__ == "__main__":
    main()
