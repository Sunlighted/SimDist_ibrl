"""Directly maximise demo-level contact_frac under a diversity (stratification) constraint.

CLAUDE.md 37: demo-level `contact_frac` is the strongest WITHIN-POOL predictor of
stage-3 adapt (n=28 rho=+0.61 p=0.0006; within the 10-arm repro952_pool500 group
rho=+0.64 p=0.044), and the 2x2 intervention showed any selection criterion must
be applied UNDER a stratification constraint -- greedy top-k collapses diversity
and the gain vanishes.

Every selector used so far optimised contact_frac only INDIRECTLY:
`select_matched_demos.py` minimises distance to a 3-D profile target, which on
`expert_n500` happened to pull contact_frac up (+0.025) but on
`repro952_pool500` does nothing (+0.007) because that pool already sits on the
target in the other two dimensions. This one optimises it DIRECTLY.

Stratify on eef-init |r| (preserves the pool's initial-state marginal by
construction), then take the highest-contact_frac demo in each stratum.
`--mode min` builds the matched low-contact_frac control.

Usage:
    python scripts/select_by_contact_frac.py --pool <dir> --n 20 50 \
        --out_fmt "release/data/mimicgen/<name>_n{n}"
"""
import argparse, json, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
import h5py, numpy as np
from scripts.select_matched_demos import features, describe, write_subset

p = argparse.ArgumentParser()
p.add_argument("--pool", required=True)
p.add_argument("--out_fmt", required=True)
p.add_argument("--n", type=int, nargs="+", default=[20, 50])
p.add_argument("--mode", choices=["max", "min"], default="max")
p.add_argument("--strat", choices=["eefr", "len"], default="eefr")
p.add_argument("--dry_run", action="store_true")
a = p.parse_args()

pool = features(a.pool)
print("pool")
describe(f"ALL ({os.path.basename(a.pool)})", pool)
key = pool["eefr"] if a.strat == "eefr" else pool["len"]
for n in a.n:
    strata = np.array_split(np.argsort(key), n)
    pick = [int(st[np.argmin(pool["contact_frac"][st]) if a.mode == "min"
                  else np.argmax(pool["contact_frac"][st])]) for st in strata]
    idx = np.array(sorted(pick))
    describe(f"SELECTED n={n} ({a.mode} contact_frac, strat={a.strat})", pool, idx)
    if not a.dry_run:
        write_subset(a.pool, pool["keys"], idx, os.path.join(_REPO, a.out_fmt.format(n=n)),
                     f"{a.mode}imise demo contact_frac, stratified on {a.strat}")
    print()
