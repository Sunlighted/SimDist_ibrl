#!/usr/bin/env python3
"""Read the anchor calibration cells and pick, per form, the lambda closest to F1.

Parses `[D_SRC] <d>  (L2mean ..)  bc_loss <b>` out of each cell's train.log and
groups by form (the tag prefix before the last underscore). F1 defines the target
working point; the winner per form is the cell whose D_SRC is nearest to it.

    python scripts/report_anchor_cal.py exps/anchor_cal
"""
import os, re, sys, glob

PAT = re.compile(r"\[D_SRC\]\s+([0-9.]+)\s+\(L2mean\s+([0-9.]+)\)\s+bc_loss\s+([0-9.]+)")

def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "exps/anchor_cal"
    cells = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "train.log"))):
        tag = os.path.basename(os.path.dirname(d))
        m = None
        for line in open(d, errors="replace"):
            mm = PAT.search(line)
            if mm:
                m = mm
        if m:
            cells[tag] = (float(m.group(1)), float(m.group(3)))
    if not cells:
        print(f"no cells with [D_SRC] under {root}", file=sys.stderr)
        return 1

    f1 = [t for t in cells if t.startswith("f1_")]
    if not f1:
        print("f1_* cell missing -- it defines the target D_SRC", file=sys.stderr)
        return 1
    target = cells[f1[0]][0]
    print(f"target D_SRC (F1) = {target:.4f}   bc_loss {cells[f1[0]][1]:.5f}\n")

    forms = {}
    for tag, v in cells.items():
        form = tag.rsplit("_", 1)[0]
        forms.setdefault(form, []).append((tag, *v))

    print(f"{'form':10s} {'cell':14s} {'D_SRC':>8s} {'bc_loss':>9s} {'|dD|':>7s}  pick")
    picks = {}
    for form in sorted(forms):
        rows = sorted(forms[form], key=lambda r: abs(r[1] - target))
        for k, (tag, d, b) in enumerate(sorted(forms[form], key=lambda r: r[1])):
            best = (tag == rows[0][0])
            if best:
                picks[form] = (tag, d, b)
            print(f"{form if k == 0 else '':10s} {tag:14s} {d:8.4f} {b:9.5f} "
                  f"{abs(d - target):7.4f}  {'<= pick' if best else ''}")
    print("\n== picks (lambda per form, matched to F1's working point) ==")
    for form, (tag, d, b) in sorted(picks.items()):
        lam = tag.rsplit("_", 1)[1]
        # bracketing check: a pick at the edge of the swept range is not a match,
        # it is the best of a range that does not contain the target.
        ds = sorted(x[1] for x in forms[form])
        edge = "" if ds[0] <= target <= ds[-1] else "   <-- TARGET OUTSIDE SWEPT RANGE"
        print(f"  {form:10s} lambda={lam:8s} D_SRC={d:.4f} (target {target:.4f}) "
              f"bc_loss={b:.5f}{edge}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
