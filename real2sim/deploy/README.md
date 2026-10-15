# Real-robot runner for the SimDist pixel policy (FR3 cell)

Two processes, because the ROS 2 container has no torch/GPU and the host has no ROS:

| process | where | does |
|---|---|---|
| `policy_server.py` | robot **host**, conda env `simdist`, RTX 3090 | JPEG → undistort (factory K,D) → principal-point square crop → 224 → ZED→splat colour map; MuJoCo FK for `prop`; policy; OSC action → joint target via a reference copy of the sim arm |
| `robot_client.py` | inside `ros2_devcontainer-gello-ros2-1` | subscribes cameras / joints / gripper, queries the server at 20 Hz, publishes `/gello/joint_states` at 100 Hz (interpolated) with `gello_publisher` paused — the same path as `reset_arm.py` |

Deployed copy on the host: `~/simdist/deploy_simdist` (code subset + `exps/.../model0.pt`),
env created by `setup_env.sh` there. `robot_client.py` is copied to
`~/simdist/gello_software/simdist_runner/` = `/workspace/simdist_runner/` in the container.

## Which policy

`exps/mimicgen/peg_pixel_stage1_splat_seed1/model0.pt` — arm 1 stage-1 (fixed board,
photoreal splat obs, policy head, no planner; 0.960 in sim on splat obs).
⚠️ It was trained on the **OVAL** pocket (the pre-2026-09-23 sim target) with the
rectangular 33×21 mm peg rendered in MuJoCo. Test it with the peg that fits the oval.
The board must sit at the nominal pose the replay stills show aligned
(`viz/real_vs_sim/replay_*_t0000.png`); arm 1 never saw a moved board.
Place the peg upright inside the sim reset range (robot base frame x 0.39–0.58 m,
y 0.14–0.23 m, yaw within ±30°), e.g. base (0.49, 0.18).

## Threading Robotiq cell (2026-10-15)

`ReferenceArm` also builds `Threading_RealCellRobotiq` from the ckpt's `sim2real_cell` (all DR off).
prop = the training env's own observables + `prop_frame: base` ⇒ FK(q) in the base frame, identical to
training (`scripts/test_sim2real.py` M). Client: `--gripper-state-topic /gripper/joint_states` (it sends the
one `robotiq_85_left_knuckle_joint` value; the server maps it). Server: pass `--z_min` (measure it with a
closed-fingertip touch-off, `eef[2]` + 2 mm; sim estimate ≈ 0.950). Full real-side list:
`docs/real-teleop-stage2.md` §2b. ⚠️ No policy exists for this cell yet (control rate undecided).

Tracking-lag test (no policy): `policy_server.py --scripted track --task ... --cell ...` + client `--joint-log`; analysis `real2sim/zed_capture/calib_20261014/tools/analyze_tracking.py`; procedure in `calib_20261014/NEXT_VISIT.md` §1.

## Run

Start the usual teleop session (cameras via `publish_zed_uvc.py`, robot, gripper,
GELLO) and `reset_arm.py` to the ready pose. Then:

```bash
# 1) host terminal
cd ~/simdist/deploy_simdist
~/anaconda3/envs/simdist/bin/python real2sim/deploy/policy_server.py \
    --ckpt exps/mimicgen/peg_pixel_stage1_splat_seed1/model0.pt --save_dir runs/$(date +%m%d_%H%M)

# 2) container terminal -- DRY RUN first: prints actions, publishes nothing
docker exec -it ros2_devcontainer-gello-ros2-1 bash
source /opt/ros/humble/setup.bash && source /workspace/ros2/install/setup.bash
python3 /workspace/simdist_runner/robot_client.py --steps 60

# 3) look at the server's runs/<ts>/obs_*.png -- the policy's actual input. It should
#    look like the splat renders (viz/real_vs_sim/color_correction_fit.png, right column).

# 4) live (hand on the e-stop)
python3 /workspace/simdist_runner/robot_client.py --live --steps 300
# afterwards: realign the GELLO / run reset_arm.py (the client leaves the controller off)
```

Useful flags: server `--no_color` (raw ZED colours), `--z_min` (sim-world grip-site z; PegInsertion
default 0.995, the real demos bottom out at 1.008; **required** for any other task); client `--hz 10` (each target gets
100 ms — slower, gentler), `--max-step-rad` (default 0.03 rad per step).

## Safety behaviour

- server refuses a step (target = current joints) if the grip-site target is below
  `--z_min` or outside the table box, and clips joints to FR3∩Panda limits − 0.05 rad
- client clips every joint to ±`max_step_rad` per step, stops on any input older than
  `--max_age` (0.25 s), on a server error, or if the robot stops publishing (reflex)
- every exit (done, Ctrl-C, error): hold the measured pose → deactivate
  `joint_impedance_controller` (the arm holds) → SIGCONT `gello_publisher`

## Verified offline

`replay_test.py` plays the recorded GELLO episodes through the real wire protocol
(`viz/real_vs_sim/replay_test/`); `fit_color_correction.py` fits and validates the
colour map on the same episodes (episode 2 held out).
