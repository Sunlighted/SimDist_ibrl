"""Remap an offline-IQL checkpoint into a QAgent-shaped flat state_dict.

`scripts/pretrain_iql_qagent.py` saves `{"critic": {...}, "actor": {...}, ...}`
(separate nets), while `train_rl.py --load_pretrained_agent` expects the FLAT
state_dict of a whole `QAgent` and silently keeps only the keys that already
match (`compatible = {k: v for k, v in pretrained.items() if k in cur ...}`).
Feeding it the nested dict therefore matches ZERO keys and warm-starts nothing —
the run would look like a cold start with no error at all. This script does the
prefixing so the offline phase actually transfers.

Also copies the critic into `critic_target` (they start identical in QAgent), so
the target net does not stay random while the online net is warm.

Usage:
    python scripts/iql_ckpt_to_agent.py \
        --iql exps/mimicgen/iql_offline_source/iql_qagent.pt \
        --out exps/mimicgen/iql_offline_source/agent_warmstart.pt
"""
import argparse
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iql", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ck = torch.load(args.iql, map_location="cpu")
    out = {}
    for src, dsts in [("critic", ["critic", "critic_target"]), ("actor", ["actor"])]:
        sd = ck.get(src)
        if sd is None:
            print(f"[warn] no '{src}' in the IQL ckpt", file=sys.stderr)
            continue
        for dst in dsts:
            for k, v in sd.items():
                out[f"{dst}.{k}"] = v
    torch.save(out, args.out)
    print(f"[remap] {len(out)} keys -> {args.out}", file=sys.stderr)
    print(f"        sample: {list(out)[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
