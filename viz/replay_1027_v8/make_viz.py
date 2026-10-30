import sys, json, glob, os
sys.path.insert(0, "/home/Chenyang/ibrl")
import numpy as np, imageio, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import env.threading_variants as tv
S = sys.argv[1]; V = sys.argv[2]; OUT = os.path.join(S, "viz"); tag = sys.argv[3]
seg = json.load(open(os.path.join(S, "segments.json")))
ds = h5py.File("/user_data/Chenyang/real_stage2/threading_robotiq_1027_c1030/dataset.hdf5")
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
except Exception: font = ImageFont.load_default()
def phase(ep, t):
    s = seg.get(ep)
    if s is None: return "HUMAN (GELLO)", (80, 160, 255)
    a, b = s["policy"]
    return ("idle", (180, 180, 180)) if t < a else ("POLICY (stage-1)", (255, 170, 40)) if t <= b else ("HUMAN takeover", (80, 160, 255))
fig, axes = plt.subplots(6, 4, figsize=(20, 22))
summary = []
for i in range(6):
    ep = f"episode_000{i+1}"; L = np.load(os.path.join(V, f"demo_{i}_log.npz"))
    T = ds[f"data/demo_{i}"]["actions"].shape[0]; n = T - 1
    base = L["base"]
    dring = 1e3 * np.linalg.norm(L["needle_tip"] - L["ring"], axis=1)
    trip = 1e3 * np.linalg.norm(L["tripod"][:, :2] - L["tripod"][0, :2], axis=1)
    tz_r = 1e3 * (L["tcp_real"][:, 2] - base[2]); tz_s = 1e3 * (L["tcp_sim"][:, 2] - base[2])
    kn_s = L["grip_sim"]; kn_r = np.array([tv.robotiq_real_to_sim(k)[0] for k in L["kn_real"]])
    ok = L["ok"]; t = np.arange(len(dring))
    # needle shaft pitch in world (tip-down +): shaft axis = body -y? use tip - body vector
    v = L["needle_tip"] - L["needle"]; pitch = np.degrees(np.arctan2(-v[:, 2], np.linalg.norm(v[:, :2], axis=1)))
    summary.append(dict(ep=ep, pitch_rest=float(pitch[0]), succ=bool(ok.any()), dmin=float(dring.min()), trip_max=float(trip.max())))
    ax = axes[i]
    for a_ in ax:
        s = seg.get(ep)
        if s: a_.axvspan(s["policy"][0], s["policy"][1], color="orange", alpha=0.15); a_.axvspan(s["policy"][1], n, color="tab:blue", alpha=0.07)
        else: a_.axvspan(0, n, color="tab:blue", alpha=0.07)
        if ok.any(): a_.axvline(int(np.argmax(ok)), color="g", ls="--")
    ax[0].plot(t, tz_r, label="real (measured q, FK)"); ax[0].plot(t, tz_s, "--", label="sim"); ax[0].set_ylabel(f"{ep}\nTCP z above base [mm]")
    ax[1].plot(t, kn_r, label="real knuckle -> sim units"); ax[1].plot(t, kn_s, "--", label="sim driver"); ax[1].set_ylabel("gripper")
    c = np.argmax(kn_r > 0.1) if i != 2 else 100 + np.argmax(kn_r[100:] > 0.1); ax[1].set_xlim(max(0, c - 15), c + 20)
    ax[2].plot(t, dring, color="k"); ax[2].axhline(12, color="r", ls=":", label="ring radius 12 mm"); ax[2].set_ylim(0, 150); ax[2].set_ylabel("sim needle tip - ring [mm]")
    ax[3].plot(t, trip, color="purple"); ax[3].set_ylabel("sim tripod pushed [mm]")
    if i == 0:
        ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); ax[2].legend(fontsize=8)
        for a_, ti in zip(ax, ["TCP height (track servo)", "gripper close (zoom)", "needle tip to ring centre", "tripod displacement"]): a_.set_title(ti)
    # annotated video
    rd = imageio.get_reader(os.path.join(V, f"demo_{i}_track.mp4")); W = []
    for k, fr in enumerate(rd):
        im = Image.fromarray(np.concatenate([np.full((26, fr.shape[1], 3), 30, np.uint8), fr[:224], np.full((26, fr.shape[1], 3), 30, np.uint8), fr[224:]], 0))
        d = ImageDraw.Draw(im); ph, col = phase(ep, k)
        d.text((6, 4), f"SIM v8 cell 10-30 [{tag}]   t={k:3d}  tip-ring {dring[min(k,len(dring)-1)]:5.1f} mm  tripod +{trip[min(k,len(trip)-1)]:4.1f} mm" + ("  SUCCESS" if ok[:k+1].any() else ""), fill=(255, 255, 255), font=font)
        d.text((6, 254), f"REAL 10-27 {ep}   ", fill=(255, 255, 255), font=font); d.text((10 + d.textlength(f"REAL 10-27 {ep}   ", font=font), 254), ph, fill=col, font=font)
        W.append(np.array(im))
    imageio.mimsave(os.path.join(OUT, f"{ep}_{tag}.mp4"), W, fps=10, macro_block_size=2)
plt.tight_layout(); plt.savefig(os.path.join(OUT, f"timeseries_{tag}.png"), dpi=70)
print(json.dumps(summary, indent=0))
