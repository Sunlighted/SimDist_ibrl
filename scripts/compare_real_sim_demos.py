"""Real GELLO demos vs our sim: pixel-aligned side-by-sides + trajectory stats.

Three outputs, all under viz/real_vs_sim/ (view locally):

  replay_<ep>_t<k>.png   REAL frame vs SIM rendered at the REAL robot's joint
                         angles (same timestep), for both policy cameras.
                         Columns: real raw | real undistorted | sim | 50/50 blend.
                         The blend is the calibration check: anything that
                         doesn't line up is a sim/real mismatch, not motion.
  replay_<ep>.mp4        the same (undistorted | sim), every frame, 20 fps.
  phase_<ep>.png         REAL demo vs SIM EXPERT demos at matched episode phase.
  traj_stats.png/.txt    EE path, speed, gripper, episode length: real vs sim.

The sim replay moves the peg out of view: its real pose isn't recorded, so any
sim placement would be a guess that reads as a calibration error.

Real frames are cropped exactly as the policy's sim cameras are defined
(env/peg_insertion.py): a square about the principal point, side = 2*min
distance to the edge, then resized. The recorded K is the hd720 factory K the
sim FOVs were derived from; D is nonzero, so both raw and undistorted versions
are shown -- whichever one lines up with the sim is what the publisher must send.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import real2sim.splat_obs  # noqa: F401,E402  FIRST: fixes the gs_mujoco import order
import cv2  # noqa: E402
import h5py  # noqa: E402

REAL_DIR = "/scratch/yifanr/simdist/gello_software/demos/peg"
SIM_DS = "/scratch/yifanr/simdist/data/peg_d0_src_splat224/shard_0/dataset.hdf5"
OUT = "viz/real_vs_sim"
CAMS = {"scene": "zedview", "wrist": "robot0_zedm_wrist"}
HZ = 20.0
C_SIM, C_REAL = "#2a78d6", "#eb6834"      # categorical slots 1, 2


def square_crop(img, K):
    """Principal-point-centred square crop (matches the sim camera's FOV)."""
    h, w = img.shape[:2]
    cx, cy = K[0, 2], K[1, 2]
    half = int(np.floor(min(cx, w - cx, cy, h - cy)))
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half]


def decode(b):
    return cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


def label(im, txt, scale=0.55):
    im = np.ascontiguousarray(im.copy())
    for col, th in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
        cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    return im


class Real:
    def __init__(self, path):
        f = h5py.File(path, "r")
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.attrs = {k: f.attrs[k] for k in f.attrs}
        intr = json.loads(self.attrs["camera_intrinsics"])
        self.K = {c: np.array(intr[c]["K"]).reshape(3, 3) for c in CAMS}
        self.D = {c: np.array(intr[c]["D"]) for c in CAMS}
        self.q = f["observation/joint_positions"][()]
        self.fingers = f["observation/gripper_finger_positions"][()]
        self.width = f["observation/gripper_width"][()]
        self.act_q = f["action/joint_positions"][()]
        self.act_grip = f["action/gripper_width_percent"][()]
        self.imgs = {c: f[f"observation/images/{c}"] for c in CAMS}
        self.T = len(self.q)

    def frame(self, cam, t, S, undistort):
        im = decode(self.imgs[cam][t])
        if undistort:
            im = cv2.undistort(im, self.K[cam], self.D[cam], None, self.K[cam])
        return cv2.resize(square_crop(im, self.K[cam]), (S, S), interpolation=cv2.INTER_AREA)


class SimReplay:
    """PegInsertion_D0 + splat compositor; renders the arm at given joint angles."""

    def __init__(self, S):
        import env.peg_insertion  # noqa: F401
        from env.robosuite_wrapper import PixelRobosuite
        cams = list(CAMS.values())
        self.e = PixelRobosuite(env_name="PegInsertion_D0", robots="Panda", episode_length=10,
                                camera_names=cams, rl_cameras=cams, image_size=S,
                                rl_image_size=S, use_state=True, device="cpu", splat_obs={})
        np.random.seed(0)
        self.e.reset()
        self.sim = self.e.env.sim
        m = self.sim.model
        rob = self.e.env.robots[0]
        self.jidx = rob._ref_joint_pos_indexes
        names = [m.joint_id2name(i) for i in range(m.njnt)]
        self.gidx = [m.get_joint_qpos_addr(n) for n in names if n and "finger_joint" in n]
        peg = [n for n in names if n and "peg" in n]
        self.peg_adr = m.get_joint_qpos_addr(peg[0]) if peg else None
        self.eef_site = rob.eef_site_id
        self.S = S

    def set(self, q, fingers):
        d = self.sim.data
        d.qpos[self.jidx] = q
        if len(self.gidx) == 2:
            d.qpos[self.gidx[0]], d.qpos[self.gidx[1]] = fingers[0], -fingers[1]
        if self.peg_adr is not None:                       # park the peg out of view
            a = self.peg_adr[0] if isinstance(self.peg_adr, tuple) else self.peg_adr
            d.qpos[a:a + 3] = [0.0, 0.0, -5.0]
        d.qvel[:] = 0
        self.sim.forward()

    def eef(self):
        return self.sim.data.site_xpos[self.eef_site].copy()

    def render(self, cam):
        return self.e._splat.composite(self.sim, CAMS[cam], self.S, self.S)[::-1]


def sim_expert_demos(n):
    f = h5py.File(SIM_DS, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda k: int(k.split("_")[1]))
    out, lens_all = [], []
    for k in keys:
        g = d[k]
        ok = bool(g.attrs.get("success", False))
        # expert rollouts carry injected-noise steps flagged 0, so "expert-only"
        # never matches: the expert episodes are the ones with frac in [0.5, 0.9)
        # and snapshot episodes have frac 0.
        exp = float(g["expert_flags"][()].mean()) >= 0.5
        if ok:
            lens_all.append(g["actions"].shape[0])
        if ok and exp and len(out) < n:
            out.append(dict(key=k, state=g["obs/state"][()], act=g["actions"][()],
                            imgs={c: g[f"obs/{CAMS[c]}_image"] for c in CAMS}))
    return out, np.array(lens_all), f


def sim_frame(demo, cam, t, S):
    b = demo["imgs"][cam][t]
    im = decode(b) if not isinstance(b, np.ndarray) or b.ndim == 1 else b
    return cv2.resize(im, (S, S), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=448)
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--n_sim", type=int, default=60)
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    S = a.S

    reals = [Real(p) for p in sorted(glob.glob(f"{REAL_DIR}/*.h5"))]
    print(f"[real] {len(reals)} episodes: " +
          ", ".join(f"{r.name} {r.T} frames" for r in reals), flush=True)
    rp = SimReplay(S)

    # ---- 1+2. same-joints replay: stills + video; also FK the real EE path ----
    for r in reals:
        r.eef = np.zeros((r.T, 3))
        for t in range(r.T):
            rp.set(r.q[t], r.fingers[t])
            r.eef[t] = rp.eef()
        # key frames: evenly spaced + the moment the gripper first closes
        close = np.argmax(r.act_grip < 0.5) if (r.act_grip < 0.5).any() else r.T // 2
        ts = sorted(set(np.linspace(0, r.T - 1, a.n_frames).astype(int)) | {int(close)})
        for t in ts:
            rp.set(r.q[t], r.fingers[t])
            rows = []
            for cam in CAMS:
                raw = r.frame(cam, t, S, undistort=False)
                und = r.frame(cam, t, S, undistort=True)
                sim = rp.render(cam)
                blend = (0.5 * und.astype(np.float32) + 0.5 * sim).astype(np.uint8)
                rows.append(np.concatenate([
                    label(raw, f"REAL raw  {cam}  t={t} ({t / HZ:.1f}s)"),
                    label(und, "REAL undistorted"),
                    label(sim, "SIM @ same joints (peg hidden)"),
                    label(blend, "50/50 blend (undist + sim)")], 1))
            tag = "_GRASP" if t == close else ""
            cv2.imwrite(f"{OUT}/replay_{r.name}_t{t:04d}{tag}.png",
                        np.concatenate(rows, 0)[..., ::-1])
        print(f"[write] {r.name}: {len(ts)} replay stills (grasp at t={close})", flush=True)

        if not a.no_video:
            V = S // 2
            vw = cv2.VideoWriter(f"{OUT}/replay_{r.name}.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), HZ, (4 * V, V))
            for t in range(r.T):
                rp.set(r.q[t], r.fingers[t])
                tiles = []
                for cam in CAMS:
                    und = cv2.resize(r.frame(cam, t, S, True), (V, V))
                    sim = cv2.resize(rp.render(cam), (V, V))
                    tiles += [label(und, f"REAL {cam} {t / HZ:.1f}s", 0.45),
                              label(sim, f"SIM {cam}", 0.45)]
                vw.write(np.concatenate(tiles, 1)[..., ::-1])
            vw.release()
            print(f"[write] {OUT}/replay_{r.name}.mp4", flush=True)

    # ---- 3. phase-matched: real demo vs sim expert demos ----
    sims, sim_lens, _f = sim_expert_demos(a.n_sim)
    print(f"[sim] {len(sims)} expert-only successful demos; all successful: "
          f"n={len(sim_lens)} len mean {sim_lens.mean():.1f} "
          f"p5/p50/p95 {np.percentile(sim_lens, [5, 50, 95]).round(0)}", flush=True)
    phases = np.linspace(0, 1, 6)
    V = 256
    for r in reals:
        rows = []
        for cam in CAMS:
            rr = [label(cv2.resize(r.frame(cam, int(p * (r.T - 1)), S, True), (V, V)),
                        f"REAL {cam} {int(p * 100)}% ({p * (r.T - 1) / HZ:.1f}s)", 0.4)
                  for p in phases]
            for sd in sims[:2]:
                Ts = len(sd["state"])
                ss = [label(sim_frame(sd, cam, int(p * (Ts - 1)), V),
                            f"SIM {sd['key']} {int(p * 100)}% ({p * (Ts - 1) / HZ:.1f}s)", 0.4)
                      for p in phases]
                rows.append(np.concatenate(ss, 1))
            rows.insert(len(rows) - 2, np.concatenate(rr, 1))
        cv2.imwrite(f"{OUT}/phase_{r.name}.png", np.concatenate(rows, 0)[..., ::-1])
        print(f"[write] {OUT}/phase_{r.name}.png", flush=True)

    # ---- 4. trajectory statistics ----
    stats_plot(reals, sims, sim_lens)


def summarize(eef, grip_closed_t, T):
    v = np.linalg.norm(np.diff(eef, axis=0), axis=1) * HZ
    return dict(len=T, dur_s=T / HZ, path_m=float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()),
                v_mean=float(v.mean()), v_p95=float(np.percentile(v, 95)), v_max=float(v.max()),
                grasp_s=grip_closed_t / HZ if grip_closed_t is not None else float("nan"),
                start=eef[0], z_min=float(eef[:, 2].min()))


def stats_plot(reals, sims, sim_lens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = []
    for r in reals:
        close = int(np.argmax(r.act_grip < 0.5)) if (r.act_grip < 0.5).any() else None
        R.append((r.name, r.eef, r.width, summarize(r.eef, close, r.T)))
    Sd = []
    for sd in sims:
        st = sd["state"]
        eef, g = st[:, :3], st[:, 7] - st[:, 8]           # finger qpos -> width
        closing = np.where(sd["act"][:, 6] > 0)[0]        # robosuite: +1 = close
        Sd.append((sd["key"], eef, g, summarize(eef, int(closing[0]) if len(closing) else None, len(st))))

    lines = ["REAL vs SIM demo statistics (EE = grip site, sim world frame via MuJoCo FK)", ""]
    hdr = f"{'demo':14s} {'steps':>6s} {'dur s':>6s} {'path m':>7s} {'v mean':>7s} {'v p95':>6s} {'v max':>6s} {'grasp s':>7s} {'z_min':>6s}  start xyz"
    lines.append(hdr)
    for n, _, _, s in R:
        lines.append(f"{n:14s} {s['len']:6d} {s['dur_s']:6.1f} {s['path_m']:7.3f} {s['v_mean']:7.3f} {s['v_p95']:6.3f} {s['v_max']:6.3f} {s['grasp_s']:7.1f} {s['z_min']:6.3f}  {np.round(s['start'], 3)}")
    agg = {k: np.array([s[k] for *_, s in Sd]) for k in ["len", "dur_s", "path_m", "v_mean", "v_p95", "v_max", "grasp_s", "z_min"]}
    lines.append(f"{'SIM expert med':14s} {np.median(agg['len']):6.0f} {np.median(agg['dur_s']):6.1f} {np.median(agg['path_m']):7.3f} {np.median(agg['v_mean']):7.3f} {np.median(agg['v_p95']):6.3f} {np.median(agg['v_max']):6.3f} {np.nanmedian(agg['grasp_s']):7.1f} {np.median(agg['z_min']):6.3f}  {np.round(np.median([s['start'] for *_, s in Sd], 0), 3)}")
    lines.append(f"  (sim: n={len(Sd)} expert-only successful demos from {SIM_DS})")
    lines.append(f"  all successful sim demos: n={len(sim_lens)}, length p5/p50/p95 = {np.percentile(sim_lens, [5, 50, 95]).round(0)} steps")
    lines.append(f"  sim episode_length cap = 300 steps (15.0 s)")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open(f"{OUT}/traj_stats.txt", "w").write(txt + "\n")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axx in ax.ravel():
        axx.grid(True, color="#e6e5e1", lw=0.6); axx.set_axisbelow(True)
        for sp in ["top", "right"]:
            axx.spines[sp].set_visible(False)
    ls = ["-", "--", ":", "-."]

    def both(fn):
        for i, (n, eef, g, s) in enumerate(Sd):
            fn(eef, g, C_SIM, 0.18, 1.0, "_" if i else f"sim expert (n={len(Sd)})", "-")
        for i, (n, eef, g, s) in enumerate(R):
            fn(eef, g, C_REAL, 1.0, 2.0, f"real {n}", ls[i % 4])

    t = lambda x: np.arange(len(x)) / HZ
    both(lambda e, g, c, al, lw, lb, st: ax[0, 0].plot(t(e), e[:, 2], color=c, alpha=al, lw=lw, label=lb, ls=st))
    ax[0, 0].set(title="EE height z vs time", xlabel="s", ylabel="z (m, sim world)")
    ax[0, 0].axvline(15.0, color="#52514e", lw=1, ls=":"); ax[0, 0].text(15.1, ax[0, 0].get_ylim()[1], "sim cap 300 steps", va="top", fontsize=8, color="#52514e")

    both(lambda e, g, c, al, lw, lb, st: ax[0, 1].plot(e[:, 0], e[:, 1], color=c, alpha=al, lw=lw, label=lb, ls=st))
    ax[0, 1].set(title="EE path, top-down", xlabel="x (m)", ylabel="y (m)"); ax[0, 1].set_aspect("equal", "datalim")

    def spd(e, g, c, al, lw, lb, st):
        v = np.linalg.norm(np.diff(e, axis=0), axis=1) * HZ
        ax[0, 2].plot(t(v), v, color=c, alpha=al, lw=lw, label=lb, ls=st)
    both(spd)
    ax[0, 2].set(title="EE speed", xlabel="s", ylabel="m/s")

    both(lambda e, g, c, al, lw, lb, st: ax[1, 0].plot(t(g), g, color=c, alpha=al, lw=lw, label=lb, ls=st))
    ax[1, 0].set(title="gripper width", xlabel="s", ylabel="m")

    both(lambda e, g, c, al, lw, lb, st: ax[1, 1].plot(np.linspace(0, 1, len(e)), e[:, 2], color=c, alpha=al, lw=lw, label=lb, ls=st))
    ax[1, 1].set(title="EE height vs NORMALISED phase", xlabel="episode fraction", ylabel="z (m)")

    ax[1, 2].hist(sim_lens / HZ, bins=40, color=C_SIM, alpha=0.85, label=f"sim successful (n={len(sim_lens)})")
    for i, (n, _, _, s) in enumerate(R):
        ax[1, 2].axvline(s["dur_s"], color=C_REAL, lw=2, ls=ls[i % 4], label=f"real {n}")
    ax[1, 2].axvline(15.0, color="#52514e", lw=1, ls=":", label="sim cap (15 s)")
    ax[1, 2].set(title="episode duration", xlabel="s", ylabel="count")
    for axx in ax.ravel():
        axx.legend(fontsize=8, frameon=False)
    fig.suptitle("Real GELLO demos vs sim expert demos (PegInsertion_D0, 20 Hz)", fontsize=13)
    fig.savefig(f"{OUT}/traj_stats.png", dpi=110, facecolor="#fcfcfb")
    print(f"[write] {OUT}/traj_stats.png", flush=True)


if __name__ == "__main__":
    main()
