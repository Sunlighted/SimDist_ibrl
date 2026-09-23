# Real-robot teleop data collection — requirements for the robot side

**Task:** Threading (needle → tripod ring), Franka Panda + GELLO teleop.
**Purpose:** these demos are consumed by a policy trained entirely in simulation. Anything that differs
from the simulator's conventions — frame, frequency, units, field of view — silently shifts the data
out of the distribution the policy was trained on. Nothing throws an error; the policy just fails.

Everything below is either (a) a measured property of the simulator, or (b) something we verified
against the four episodes you already sent. Numbers are not guesses.

---

## 0. Status of the 4 episodes received (2026-09-22)

| file | verdict | reason |
|---|---|---|
| `episode_0002.h5` | **usable** | all topics live (ee_pose age 0.3 ms, robot 1.1 ms, cameras 3–30 ms) |
| `episode_0004.h5` | **usable** | same |
| `episode_0001.h5` | **unusable** | follower-robot topics were **stale for the whole episode**: `age/ee_pose` and `age/robot` median **13 s**, p95 **25 s**. `observation/ee_pose` has 13 unique rows in 802 frames — the arm state never updated. Only the GELLO leader, the gripper command and the cameras were live |
| `episode_0003.h5` | **unusable** | **truncated file**: the HDF5 superblock says the file should be 312,042,935 bytes, the file is 247,955,456 bytes. Not recoverable — the root group metadata (written at close) is missing |

**Two recorder bugs to fix before the next session:**

1. **Refuse to start recording if any topic is stale.** Check every subscribed topic's age at episode start
   (and abort mid-episode if it exceeds, say, 100 ms). Episode 0001 looks perfectly normal until you plot
   the `age` fields. This exact failure has happened before on this setup (an earlier `scan_run` had 14
   pose files with byte-identical robot state).
2. **Close the HDF5 file properly** (context manager / explicit `close()`, and `flush()` periodically).
   A killed writer leaves a file that looks complete by size but cannot be opened.

Your `age/*` fields are what made the first diagnosis possible in one minute — please keep them.

---

## 1. Control frequency: we need **20 Hz** ⚠️

The simulator runs the policy at **20 Hz** (one action every 50 ms). This is baked into what an action
*means*: each action is a displacement that the controller has 50 ms to realise. A policy trained at
20 Hz and run at 15 or 30 Hz moves at the wrong speed.

- **Recording:** your current recorder samples at 30 Hz. That is acceptable — we can resample to 20 Hz
  offline because you log exact timestamps. If you can record natively at 20 Hz, prefer that.
- **Deployment (later, when the policy runs on the robot):** the control loop **must be 20 Hz**.
  If your stack cannot do 20 Hz, tell us **before** collecting a large batch — we would have to retrain
  the whole pipeline at your frequency (≈2 days of compute) and re-collect.
- For reference, our planner's inference cost on one RTX PRO 6000 (batch 1, including host↔device):
  **58.8 ms** with the current search settings, **36.5 ms** with a cheaper setting that still fits 20 Hz.
  We will pick settings that fit 50 ms on *your* inference machine, but we need to know its GPU.

**Camera frame rate is a separate issue.** The cameras free-run; the control loop must **not block** on
them. At each 20 Hz tick, take the most recent frame and log its capture timestamp (see §4).

---

## 2. Action convention

The policy outputs a 7-vector, every element in `[-1, 1]`:

```
a = [dx, dy, dz, ax, ay, az, g]
```

| element | meaning |
|---|---|
| `dx, dy, dz` | translation, scaled to **±0.05 m per step**, in the **robot base frame** |
| `ax, ay, az` | rotation as an **axis-angle** vector, scaled to **±0.5 rad per step**, in the **base frame** |
| `g` | gripper: **only the sign matters** (`g > 0` close, `g ≤ 0` open). Magnitude is ignored |

How the target is formed (this is the part that differs between stacks):

```
target_position    = CURRENT MEASURED TCP position + 0.05 * a[0:3]
target_orientation = R(axis_angle(0.5 * a[3:6])) @ CURRENT MEASURED TCP orientation     # LEFT-multiplied
```

- The target is relative to the **current measured pose**, **not** to the previous commanded target.
  (Integrating on the previous target accumulates drift — a different system.)
- The rotation is applied by **left**-multiplication in the base frame, not in the tool frame.
- **TCP definition:** the simulator's control point is 97 mm from the flange along the tool axis.
  Franka's default `O_T_EE` with the hand is 103.4 mm. Either set `F_T_EE`/`NE_T_EE` to match 97 mm,
  or tell us the offset you use and we will convert. A 6 mm error in the control point turns every
  rotation command into an unintended translation.

**For the demos you record:** you do not have to emit actions in this format. Log the GELLO leader
joint targets (you already do) plus the measured state, and we convert. But if you write your own
deployment controller later, it must implement exactly the above.

---

## 3. Robot state to log (and a frame-convention warning)

Per control tick:

| field | note |
|---|---|
| `joint_positions` (7) | **follower**, measured. This is the most valuable field — we can reconstruct everything else from it with forward kinematics |
| `joint_velocities`, `joint_efforts` | keep |
| `ee_pose` (7) | position + quaternion. **Document the frame and the quaternion order explicitly** |
| `gripper_width` (m) and/or finger positions | keep both |
| GELLO leader `joint_positions` (7), gripper command | the action source |
| per-topic `age` | keep — this is what caught the bad episode |
| timestamps | one monotonic clock for everything (see §4) |

### ⚠️ Two of your systems report different EE frames

We checked both against the simulator's forward kinematics:

| source | position origin | orientation |
|---|---|---|
| `scan_run2/pose_*/state.json` (the hand-eye calibration set) | the **flange** (matches sim FK to **0.5 mm**) | **45.03°** rotated from the sim hand frame |
| the demo recorder (`/franka_robot_state_broadcaster/robot_state`) | the **TCP** (matches sim TCP to **6.9 mm**, = the 103.4 vs 97 mm offset) | **identical** to the sim hand frame (agreement to 1e-4 in xyzw), i.e. **no 45° offset** |

Both are self-consistent; they are simply different conventions. This matters because the wrist-camera
extrinsic was solved in the *first* convention. **Please state, in writing, which frame each logged pose
field uses.** If you change stacks or firmware, re-check it.

Quaternion order in the demo recorder is **xyzw** (verified: at the ready pose it is ≈(1,0,0,0), a 180°
rotation about x, which is the gripper pointing down).

---

## 4. Cameras

Two cameras, both already correct in your recorder: `scene` (ZED 2i, static) and `wrist` (ZED Mini).

**Store the raw frames at native resolution.** Do not crop or resize on the robot side. We do the crop
offline, and the crop depends on calibration you have not finished yet — if you store only cropped
images and the calibration changes, the data is lost.

| item | requirement |
|---|---|
| format | rectified **left** image, RGB (the ZED SDK's rectified output is undistorted — keep it that way) |
| resolution | native, uncropped (your current 1280×720 is fine) |
| **`camera_info` per camera per session** | **missing today.** Log `fx, fy, cx, cy`, distortion, resolution, and the SDK resolution mode into every episode file |
| capture timestamp | log each frame's own capture timestamp, on the **same clock** as the control tick, so we can measure how stale each frame is |
| exposure / white balance | **lock them.** Auto-exposure drift has already destroyed one scan on this setup |
| camera pose | **do not move either camera** between calibration and collection. If a camera is re-mounted, the extrinsic is void and everything collected after it is unusable until re-calibrated |

**Resolution modes matter.** ZED intrinsics are per resolution mode and are **not** related by a simple
scale factor — on this setup, rescaling HD2K intrinsics to 720p produced a 17% error. Measured values
we already have (from `scan_run2/intrinsics.json`, 2208×1242):

| camera | fx = fy | vertical FOV |
|---|---:|---:|
| ZED 2i (`scene`) | 1057.55 | **60.845°** |
| ZED Mini (`wrist`) | 1475.29 | **45.656°** |

The simulator renders the two cameras at exactly those vertical FOVs. If you record at 1280×720, we crop
a centred square of side `2 · fy_720 · tan(FOV/2)` around the principal point and resize to 224×224 —
geometrically identical, provided the images are rectified and we know `fy_720`, `cx_720`, `cy_720`.

**What we need from tomorrow's calibration:** the **native intrinsics of the exact resolution mode you
record in** (both cameras). Please do not derive them by scaling the HD2K numbers.

**Pick ONE resolution mode and never change it.** 720p is a good choice: the policy input is 224×224, so
HD2K's extra pixels are thrown away anyway, and 1280×720 at 20 Hz is far cheaper to record. The only
requirements are that the *same* mode is used for stage-2 demos and for deployment, and that we get that
mode's own `camera_info`. Two things do not survive a mode change: `fy` (the sim's vertical FOV is set
from it) and `cx, cy` — MuJoCo's camera is a **symmetric** pinhole with no principal-point offset, so the
square crop has to be centred on `(cx, cy)`, not on the image centre. An off-centre principal point that
we crop around the image centre instead is a silent, constant sim↔real image shift.

**How well does the HD2K number already do at 720p?** 2026-09-23, measured on `episode_0001` frame 0:
the sim robot was driven to the **real joint angles** and its link origins projected into the real frame
with the calibrated extrinsics. At the HD2K-derived 60.84° the skeleton lands on the real arm
(`figs/sim2real/skeleton_on_real.png`); at 51.7° it is grossly wrong, at 70° still plausible by eye
(`figs/sim2real/fov_probe_on_real.png`). So the current assumption is in the right ballpark but pinned
only to roughly ±8°, and an 8% focal error is an 8% scale error in the image — **not good enough for a
~1.8 mm insertion**. A single-view focal fit from the ChArUco in the scene was attempted and failed: the
target is too occluded/oblique to detect. Hence the ask above stands, and it is now the largest
un-measured quantity in the sim↔real chain.

---

## 4b. Scene appearance — measured gap, and what to change on your side

We processed your `episode_0002` frames exactly as the policy will see them (crop to the training field
of view, resize to 224) and compared against the simulator frames the policy is being trained on
(`figs/sim2real/bg_real_vs_sim_224.png`). The viewpoint and the arm's scale/pose match well — the
extrinsics are fine. The **appearance** does not:

| | mean brightness (0–255) | per-frame range in training data |
|---|---:|---|
| simulator, `agentview` (40 episodes) | 205.5 | **158 – 239** |
| **your `scene` camera** | **130.0** | — |
| simulator, wrist | 212.6 | |
| **your wrist** | **133.2** | |

**Your frames are darker than every frame in the training set** (28 grey levels below the darkest one).
That alone can break a vision policy.

**Please do (cheap, same-day):**
1. **Lock exposure and white balance, and raise the exposure** so that the white work surface averages
   roughly **180–220** grey (0–255) in the recorded image. Do not leave auto-exposure on: it drifts
   between episodes and has already destroyed one dataset on this setup.
2. **Enlarge the white work surface** (white board / white cloth) so the table fills the camera frame the
   way it does in simulation, covering the green mat and the wooden desk edge that are currently visible.
3. **Route the cables out of the camera views.**
4. **Nobody in the camera view during recording.** In `episode_0002` an operator reaches in at t≈550 and
   repositions the needle; the simulator has no humans, so those frames are pure noise to the policy.

**We will handle on our side:** the poster / wall region that you cannot remove — we are adding
image-based background randomisation to the simulator so the policy stops relying on a blank background.
No action needed from you for that, but **do not move the poster or the furniture between sessions**
either; a fixed background is easier to cover than a changing one.

---

## 5. Episode protocol

- **Reset pose:** Franka "ready" `[0, −π/4, 0, −3π/4, 0, π/2, π/4]`, gripper half open (~41.7 mm).
  Your recorder already enforces this (`start_pose_forced: true`) — good. Residual joint error of a few
  degrees is fine.
- **Object placement:** see the companion placement map (`figs/sim2real/placement_map.png`) and the
  per-episode schedule CSV. All distances are measured from the **near table edge** (the one closest to
  the robot) and the **table centreline**, and all left/right are **from the robot's point of view**.

  | object | reference point | position | orientation |
  |---|---|---|---|
  | needle (lying flat) | midpoint of its full 16 cm length | 24.0–39.1 cm from the near edge; 21.6–31.6 cm to the robot's **left** of the centreline | handle end pointing **away** from the robot, within **±30°** |
  | tripod (**fixed every episode**) | the point on the table directly below the ring centre | 44.1 cm from the near edge; 8.4 cm to the robot's **right** of the centreline | ring hole axis **parallel to the long table edge** |

  Tape the needle region (15.1 × 10.0 cm) and the tripod feet. **Follow the schedule CSV rather than
  placing by hand** — hand placement clusters in the comfortable part of the region, and the policy is
  evaluated over the whole region.
- **Duration:** aim to finish within **20 s**. Avoid long pauses: a pause is recorded as a long run of
  "do nothing" actions and the policy learns to stand still.
- **Success labelling — required.** We need the frame index at which the needle is through the ring and
  stable. Either log a key-press marker during teleop, or fill in a per-episode success frame afterwards.
  Episodes without it are dropped by our loader.
- **Keep failed episodes too**, flagged as failures. They are useless for imitation but useful for the
  later dynamics-adaptation stage.
- Nothing on the table that the simulator does not have: remove the ChArUco board, tools and loose
  cables from the camera views during collection. (The wall poster can stay — see §4b.)
- **Grasp check:** in `episode_0004` the gripper closes all the way to 0 mm, which means it closed on
  nothing — the 40 mm handle would hold it at ~39 mm, as it does in `episode_0002`. Please mark such
  episodes as failures, or discard them.
- **Object placement must follow §5's table:** in the two good episodes the tripod was about 0.3 m away
  (in y) from where the simulator puts it, and the needle 0.10 m beyond the sampled region. The ring
  HEIGHT matched the simulator to within 1 cm, so the hardware is right — only the layout was off.
  A policy trained on the simulator's layout has never seen the objects where they were.

---

## 6. How many

- **≥ 50 episodes**, 100–200 preferred. Demo count is the dominant factor for this stage in our
  experiments, and it has not saturated at 200.
- Single-draw variance across a 20-demo set is larger than most of the effects we try to measure, so
  more episodes is the cheapest thing you can do for us.

---

## 7. Checklist per session

Before:
- [ ] cameras and table have not moved since calibration; exposure and white balance locked
- [ ] all topics live — check `age` for every subscribed topic, abort if stale
- [ ] `camera_info` for the exact resolution mode is logged into the file
- [ ] robot returns to the Franka ready pose; gripper half open
- [ ] table clear of anything not in the simulator

After:
- [ ] every file opens (`python -c "import h5py; h5py.File(p).keys()"`) — catches truncation at the source
- [ ] `age` medians are < 100 ms for all topics
- [ ] success frame recorded for each successful episode
- [ ] copy with checksum verification (`sha256`), into a **directory that already exists**
      (`scp` to a non-existent path silently creates a *file* and each subsequent copy overwrites it —
      this is how two of the first four episodes were lost)

---

## 8. Open questions for you

1. Can the deployment control loop run at **20 Hz**? If not, what is the maximum, and what GPU is in the
   inference machine?
2. Which frame does each pose field in the demo recorder use — please confirm in writing
   (our measurement says TCP position + hand-frame orientation, xyzw).
3. `F_T_EE` / `NE_T_EE` currently in use, so we can reconcile the 97 mm vs 103.4 mm control point.
4. Native intrinsics for the recording resolution mode of both cameras (after tomorrow's calibration).
5. Has either camera been re-mounted since the `scan_run2` calibration session?
6. Is the gripper the Panda hand? The simulator assumes the Panda hand everywhere; an earlier photo of
   this cell showed a Robotiq 2F-85.
