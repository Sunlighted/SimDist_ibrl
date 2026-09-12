#!/usr/bin/env python3
"""Per task, pick the F5 lambda whose HELD-OUT D_SRC matches that task's F1 cell.

    python scripts/report_f5_cal.py exps/anchform_cal
"""
import os, re, sys, glob

PAT = re.compile(r"\[D_SRC\]\s+([0-9.]+)\s+\(L2mean\s+[0-9.]+\)"
                 r"\s+heldout_D_SRC\s+([0-9.]+)[^b]*bc_loss\s+([0-9.]+)")

def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "exps/anchform_cal"
    cells = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "train.log"))):
        tag = os.path.basename(os.path.dirname(f))
        m = None
        for line in open(f, errors="replace"):
            mm = PAT.search(line)
            if mm:
                m = mm
        if m:
            cells[tag] = (float(m.group(2)), float(m.group(3)), float(m.group(1)))
    if not cells:
        print(f"no cells with heldout_D_SRC under {root}", file=sys.stderr)
        return 1

    for task in sorted({t.split("_")[0] for t in cells}):
        f1 = f"{task}_f1"
        if f1 not in cells:
            print(f"[{task}] F1 cell missing -- it defines the target"); continue
        tgt, tgt_bc, _ = cells[f1]
        print(f"\n== {task}:  target held-out D_SRC = {tgt:.4f}  (F1, bc_loss {tgt_bc:.5f})")
        rows = [(t, *v) for t, v in cells.items()
                if t.startswith(f"{task}_f5_")]
        if not rows:
            print("   no F5 cells"); continue
        rows.sort(key=lambda r: r[1])
        best = min(rows, key=lambda r: abs(r[1] - tgt))
        print(f"   {'cell':18s} {'heldout':>8s} {'(pool)':>8s} {'bc_loss':>9s} "
              f"{'|dD|':>7s} {'vs F1 bc':>9s}")
        for t, h, bc, pl in rows:
            print(f"   {t:18s} {h:8.4f} {pl:8.4f} {bc:9.5f} {abs(h-tgt):7.4f} "
                  f"{100*(bc/tgt_bc-1):+8.1f}%  {'<= pick' if t == best[0] else ''}")
        hs = [r[1] for r in rows]
        if not (min(hs) <= tgt <= max(hs)):
            print(f"   !! target OUTSIDE the swept range [{min(hs):.4f}, {max(hs):.4f}] "
                  f"-- extend lambda {'down' if tgt > max(hs) else 'up'}")
        else:
            print(f"   => ANCH_F5_{'SQ' if task=='square' else 'COF'}="
                  f"{best[0].rsplit('_',1)[1]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
