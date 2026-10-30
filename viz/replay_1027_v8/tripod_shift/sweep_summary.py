import glob, os, re, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insert_geom import analyse
S = os.path.dirname(os.path.abspath(__file__))
real_move = json.load(open(os.path.join(S, "tripod_first_last.json")))
rows = []
for dd in sorted(glob.glob(os.path.join(S, "sweep", "dx*_dy*"))):
    mt = re.search(r"dx(-?\d+)_dy(-?\d+)", dd); dx, dy = int(mt.group(1)), int(mt.group(2))
    if not os.path.exists(os.path.join(dd, "demo_5_log.npz")): continue
    for i in range(6):
        L = np.load(os.path.join(dd, f"demo_{i}_log.npz")); r = analyse(L)
        trip = L["tripod"]; push = 1e3 * np.linalg.norm(trip[-1, :2] - trip[0, :2])
        rm = real_move[f"episode_000{i+1}"]["moved_mm_deg"]
        rows.append(dict(dx=dx, dy=dy, demo=i, succ=bool(L["ok"].any()), depth=1e3 * r["depth"].max(),
                         lat_end=1e3 * r["lat"][-1], push=push, push_err=abs(push - np.hypot(rm[0], rm[1])),
                         iou_scene=float(L["final_iou"][0]), iou_wrist=float(L["final_iou"][1])))
json.dump(rows, open(os.path.join(S, "sweep_rows.json"), "w"))
import collections
G = collections.defaultdict(list)
for r in rows: G[(r["dx"], r["dy"])].append(r)
print(f"{'dx':>4} {'dy':>4} | succ all | succ ep3-6 | depth ep3-6 mean | push ep3-6 mean | IoU scene/wrist (end, mean)")
out = []
for (dx, dy), rs in sorted(G.items()):
    t = [r for r in rs if r["demo"] >= 2]
    out.append((dx, dy, sum(r["succ"] for r in rs), sum(r["succ"] for r in t), np.mean([r["depth"] for r in t]),
                np.mean([r["push"] for r in t]), np.mean([r["iou_scene"] for r in rs]), np.mean([r["iou_wrist"] for r in rs])))
for o in out:
    print(f"{o[0]:4d} {o[1]:4d} |   {o[2]}/6    |    {o[3]}/4     |   {o[4]:6.1f} mm      |   {o[5]:5.1f} mm      | {o[6]:.3f} / {o[7]:.3f}")
