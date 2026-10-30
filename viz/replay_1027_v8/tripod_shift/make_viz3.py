import sys, json, os
sys.path.insert(0, "/home/Chenyang/ibrl"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from insert_geom import analyse
S = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
A, B = os.path.join(S, "base_0_0"), os.path.join(S, "best_10_0")
seg = json.load(open(os.path.join(S, "segments.json")))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
def phase(ep, t):
    s = seg.get(ep)
    if s is None: return "HUMAN (GELLO)", (80, 160, 255)
    a, b = s["policy"]
    return ("idle", (180, 180, 180)) if t < a else ("POLICY (stage-1)", (255, 170, 40)) if t <= b else ("HUMAN takeover", (80, 160, 255))
bar = lambda w: np.full((22, w, 3), 30, np.uint8)
fig, axes = plt.subplots(2, 6, figsize=(26, 7))
for i in range(6):
    ep = f"episode_000{i+1}"
    La, Lb = np.load(f"{A}/demo_{i}_log.npz"), np.load(f"{B}/demo_{i}_log.npz")
    ra, rb = analyse(La), analyse(Lb)
    fa = [f for f in imageio.get_reader(f"{A}/demo_{i}_track.mp4")]; fb = [f for f in imageio.get_reader(f"{B}/demo_{i}_track.mp4")]
    n = min(len(fa), len(fb)); W = []
    for k in range(n):
        im = Image.fromarray(np.concatenate([bar(448), fa[k][:224], bar(448), fb[k][:224], bar(448), fa[k][224:]], 0)); dr = ImageDraw.Draw(im)
        kk = min(k, len(ra["depth"]) - 1)
        for y, lab, L, r in ((4, "SIM  tripod as fitted (scene cam)", La, ra), (250, "SIM  tripod +10 mm x", Lb, rb)):
            ok = L["ok"][:kk + 1].any()
            dr.text((6, y), f"{lab}   t={k:3d}  depth {1e3*r['depth'][kk]:+5.0f} mm" + ("  IN RING" if ok else ""), fill=(120, 255, 120) if ok else (255, 255, 255), font=font)
        ph, col = phase(ep, k); dr.text((6, 496), f"REAL 10-27 {ep}", fill=(255, 255, 255), font=font); dr.text((150, 496), ph, fill=col, font=font)
        W.append(np.array(im))
    imageio.mimsave(os.path.join(OUT, f"{ep}_sim_vs_shifted_vs_real.mp4"), W, fps=10, macro_block_size=2)
    for row, (L, r, lab) in enumerate(((La, ra, "as fitted"), (Lb, rb, "tripod +10 mm x"))):
        ax = axes[row, i]; t = np.arange(len(r["depth"]))
        s = seg.get(ep)
        if s: ax.axvspan(s["policy"][0], s["policy"][1], color="orange", alpha=0.15)
        ax.plot(t, 1e3 * r["depth"], label="tip past ring plane"); ax.plot(t, 1e3 * r["lat"], label="lateral offset at ring")
        ax.plot(t, 1e3 * np.linalg.norm(L["tripod"][:, :2] - L["tripod"][0, :2], axis=1), label="tripod pushed")
        ax.axhline(12, color="r", ls=":", lw=1); ax.set_ylim(-60, 120); ax.set_xlim(max(0, len(t) - 300), len(t))
        if L["ok"].any(): ax.axvline(int(np.argmax(L["ok"])), color="g", ls="--")
        ax.set_title(f"{ep} | {lab} | {'IN RING' if L['ok'].any() else 'not in'}", fontsize=9)
        if i == 0: ax.set_ylabel("mm"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "insertion_last300steps.png"), dpi=65)
