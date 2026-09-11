"""Stage an ADAPTED SimDist checkpoint so `evaluate/eval.py --mode simdist` can read it.

WHY THIS EXISTS
    `evaluate/eval.py --mode simdist` calls `train_simdist.load_model`, which does
    `pyrallis.load(train_simdist.MainConfig, <ckpt dir>/cfg.yaml)`. But
    `train_simdist_adapt.py` writes an *AdaptConfig* there (stage1_ckpt,
    bc_anchor_coef, max_transitions, ...), so the load raises
        Exception: The fields {...} do not belong to the class
    for EVERY adapt run -- state and pixel alike. `docs/simdist.md:425-428`
    records the intended workaround: put the adapted weights next to a STAGE-1
    cfg.yaml. This script does that, and derives the planner flags from the adapt
    cfg instead of leaving them to be re-guessed.

WHY IT MATTERS
    Iron law #14: the headline number must come from an independent re-eval, not
    from model0.pt's in-training best (systematically optimistic by 0.06-0.19).
    That re-eval goes through this path, so this path has to work.

WHY THE PLANNER FLAGS ARE NOT OPTIONAL
    The stage-1 cfg has `use_planner: 0` (stage 1 evaluates the policy head
    directly), while adaptation is meaningless without planning. And iron law #9:
    planner hyperparameters do not transfer across tasks OR across
    oracle-vs-learned dynamics. So the emitted command copies `original_mppi` and
    `planner_set` out of the ADAPT cfg verbatim -- a standalone eval that omits
    them silently measures a different planner than the run did.
"""
import argparse, hashlib, json, os, shutil, sys
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapt_dir", required=True, help="a train_simdist_adapt.py save_dir")
    p.add_argument("--which", default="model0.pt", help="model0.pt (topk best) or latest.pt")
    p.add_argument("--out_dir", required=True, help="staging dir to create")
    p.add_argument("--num_game", type=int, default=500)
    p.add_argument("--mp", type=int, default=10)
    args = p.parse_args()

    acfg_path = os.path.join(args.adapt_dir, "cfg.yaml")
    ckpt = os.path.join(args.adapt_dir, args.which)
    for f in (acfg_path, ckpt):
        if not os.path.exists(f):
            sys.exit(f"FATAL: missing {f}")

    acfg = yaml.safe_load(open(acfg_path))
    stage1_ckpt = acfg.get("stage1_ckpt")
    if not stage1_ckpt:
        sys.exit("FATAL: adapt cfg has no stage1_ckpt -- cannot find the stage-1 cfg.yaml")
    s1_cfg = os.path.join(os.path.dirname(stage1_ckpt), "cfg.yaml")
    if not os.path.exists(s1_cfg):
        sys.exit(f"FATAL: missing stage-1 cfg {s1_cfg}")

    # iron law #12: num_eval_proc must divide num_eval_episode.
    if args.num_game % args.mp != 0:
        sys.exit(f"FATAL: --mp {args.mp} does not divide --num_game {args.num_game} "
                 f"(mp eval asserts this)")

    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(s1_cfg, os.path.join(args.out_dir, "cfg.yaml"))
    shutil.copy(ckpt, os.path.join(args.out_dir, "model0.pt"))

    sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    prov = {"adapt_dir": args.adapt_dir, "source_ckpt": ckpt, "sha256": sha,
            "stage1_cfg": s1_cfg, "original_mppi": acfg.get("original_mppi", 0),
            "planner_set": acfg.get("planner_set", "")}
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as g:
        json.dump(prov, g, indent=2)

    # planner flags copied out of the ADAPT cfg, not defaulted
    flags = []
    if int(acfg.get("original_mppi", 0)):
        flags.append("--original_mppi 1")     # implies --use_planner 1
    else:
        flags.append("--use_planner 1")       # stage-1 cfg says 0; adaptation needs 1
    if acfg.get("planner_set", ""):
        flags.append(f"--planner_set '{acfg['planner_set']}'")

    print(f"staged {ckpt}\n    -> {args.out_dir}/model0.pt")
    print(f"    cfg from stage 1: {s1_cfg}")
    print(f"    sha256 {sha[:16]}...  (provenance.json written)")
    print("\n# run this (planner flags come FROM THE ADAPT CFG -- do not drop them):")
    print(f"python evaluate/eval.py --weight {args.out_dir}/model0.pt --mode simdist \\\n"
          f"    --num_game {args.num_game} --mp {args.mp} {' '.join(flags)} \\\n"
          f"    --shift_from_cfg 1 --strict_success 1 \\\n"
          f"    --hand_init_perturb_joint_sigma 0.1")


if __name__ == "__main__":
    main()
