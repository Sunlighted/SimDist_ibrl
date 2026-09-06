#!/usr/bin/env python3
"""Audit finished runs against the authoritative target definitions (docs/infra.md 13).

Checks every run directory's cfg.yaml for the four things that failed silently in
the 2026-09-06 debugging round:

  1. domain shift        -- must equal the task's stage-1 eval_domain_shift.
                            train_rl.py's default is mode="none" (SOURCE physics).
  2. act_method          -- must be "ibrl" when a bc_policy is set, else the prior
                            never enters the max-Q argmax and prior_margin is dead.
  3. critic architecture -- every IBRL run in this project is 5x1024 + layer_norm;
                            train_rl.py's dataclass default is 10x512 without.
  4. sigma               -- 0.1 train and eval for all three tasks.

    python3 scripts/audit_runs.py exps/simdist_threading_3arm/*/
    python3 scripts/audit_runs.py exps/coffee_g5x_3arm/*/ exps/square_rotw1_6arm/*/
"""
import argparse, os, sys, yaml

# docs/infra.md 13 -- the stage-1 whose cfg DEFINES each task's target.
TRUTH = {
    "Threading_D0": "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1",
    "Square_D0":    "exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1",
    "Coffee_D0":    "exps/mimicgen/coffee_s1g5x_dynens",
}
SHIFT_KEYS = ("mode", "friction_scale", "gripper_kp_scale", "mass_scale",
              "ring_friction_scale", "ring_size_scale", "peg_friction_scale",
              "peg_size_scale", "lid_damping_scale", "pod_holder_friction_scale",
              "solref_scale", "solimp_scale", "hole_offset")

def sig(d):
    """Semantic signature: only the knobs that actually DEVIATE from the identity.

    A literal key-by-key compare false-positives on cosmetic differences between
    cfgs written by different code versions -- e.g. one file has
    `hole_offset: [0,0,0]` and another omits the field entirely, which are the
    same physics. Scales of 1.0 and all-zero offsets are the identity.
    """
    d = d or {}
    out = {}
    for k in SHIFT_KEYS:
        v = d.get(k)
        if v is None:
            continue
        if k == "mode":
            out[k] = v
            continue
        if isinstance(v, (list, tuple)):
            if all(float(x) == 0.0 for x in v):
                continue
            out[k] = tuple(float(x) for x in v)
            continue
        if float(v) == 1.0:
            continue
        out[k] = float(v)
    return tuple(sorted(out.items()))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()

    truth = {}
    for task, p in TRUTH.items():
        f = os.path.join(p, "cfg.yaml")
        if os.path.isfile(f):
            truth[task] = yaml.safe_load(open(f)).get("eval_domain_shift")
        else:
            print(f"WARN: no stage-1 cfg for {task} at {f}", file=sys.stderr)

    bad = ok = 0
    for d in sorted(a.dirs):
        d = d.rstrip("/")
        f = os.path.join(d, "cfg.yaml")
        if not os.path.isfile(f):
            continue
        c = yaml.safe_load(open(f))
        task = c.get("task_name")
        # adapt/dyna runs carry no shift of their own -- they inherit stage-1's
        shift = c.get("eval_domain_shift")
        inherited = False
        if shift is None and c.get("stage1_ckpt"):
            sp = os.path.join(os.path.dirname(c["stage1_ckpt"]), "cfg.yaml")
            if os.path.isfile(sp):
                s1 = yaml.safe_load(open(sp))
                shift, task, inherited = s1.get("eval_domain_shift"), s1.get("task_name"), True

        problems = []
        if task in truth and sig(shift) != sig(truth[task]):
            got = {k: v for k, v in (shift or {}).items() if k in SHIFT_KEYS and v not in (None, 1.0, 0.0)}
            problems.append(f"shift != target ({got})")
        q = c.get("q_agent") or {}
        if c.get("bc_policy") and q.get("act_method") == "rl":
            problems.append("act_method=rl with a bc_policy -> not IBRL")
        sc = q.get("state_critic") or {}
        if sc and (sc.get("num_q"), sc.get("hidden_dim"), sc.get("layer_norm")) != (5, 1024, 1):
            problems.append(f"critic {sc.get('num_q')}x{sc.get('hidden_dim')} ln={sc.get('layer_norm')}")
        for k in ("train_hand_init_perturb_joint_sigma", "eval_hand_init_perturb_joint_sigma"):
            if k in c and c[k] not in (0.1, None):
                problems.append(f"{k}={c[k]}")

        if problems:
            bad += 1
            print(f"RERUN  {os.path.basename(d)}")
            for p in problems:
                print(f"         - {p}")
        else:
            ok += 1
    print(f"\n== {ok} clean, {bad} need a rerun"
          + ("  (shift shown is the INHERITED stage-1 one where applicable)" if bad else ""))
    return 0

if __name__ == "__main__":
    sys.exit(main())
