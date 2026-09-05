#!/usr/bin/env python3
"""Repair the `state_policy` block of a fine-tuned BC checkpoint's cfg.yaml.

The bug: train_bc.py's `--load_model` rebuilds the policy from the SOURCE
checkpoint's own cfg (train_bc.py:537 -> load_model() -> _load_model()), so the
fine-tuned run's `state_policy` block is never used -- it is written to cfg.yaml
as whatever the defaults were (hidden_dim 256, dropout 0.5) while the weights are
the source architecture's (1024, 0.0).

Nothing notices until something reloads the checkpoint FROM its cfg:

    train_rl.py --bc_policy <prior>/model0.pt
    RuntimeError: size mismatch for net.9.weight: copying a param with shape
    torch.Size([7, 1024]) ... the shape in current model is torch.Size([7, 256])

which is a queue wait and a scheduled job wasted, per seed. This copies the
`state_policy` block from the source cfg the run recorded in `load_model`, so the
config matches the weights that are actually in the file. It rewrites nothing
else and is idempotent.

    python3 scripts/fix_bc_prior_cfg.py exps/mimicgen/il_ft_target_s160_seed*/
    python3 scripts/fix_bc_prior_cfg.py --check exps/mimicgen/*_il_ft_*/
"""
import argparse, os, sys, yaml

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="run directories holding cfg.yaml")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    a = ap.parse_args()

    bad = fixed = ok = 0
    for d in a.dirs:
        d = d.rstrip("/")
        p = os.path.join(d, "cfg.yaml")
        if not os.path.isfile(p):
            print(f"  no cfg.yaml: {d}"); continue
        cfg = yaml.safe_load(open(p))
        src = cfg.get("load_model")
        if not src or src == "none":
            print(f"  not a fine-tune (load_model={src}): {d}"); continue
        src_cfg_path = os.path.join(os.path.dirname(src), "cfg.yaml")
        if not os.path.isfile(src_cfg_path):
            print(f"  source cfg missing ({src_cfg_path}): {d}"); continue
        want = yaml.safe_load(open(src_cfg_path))["state_policy"]
        have = cfg.get("state_policy")
        if have == want:
            ok += 1
            continue
        bad += 1
        print(f"  {os.path.basename(d)}: {have} -> {want}")
        if not a.check:
            cfg["state_policy"] = want
            with open(p, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
            fixed += 1
    print(f"== {ok} already correct, {bad} mismatched"
          + (f", {fixed} rewritten" if not a.check else " (check only, nothing written)"))
    return 1 if (a.check and bad) else 0

if __name__ == "__main__":
    sys.exit(main())
