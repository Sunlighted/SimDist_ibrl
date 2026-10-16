#!/usr/bin/env bash
# NEXT_VISIT.md §1, one command: tracking-lag / overshoot test at 20 Hz then 10 Hz, no policy.
#
#   bash ~/simdist/deploy_simdist_1015/run_tracking_test.sh          # 20 then 10 Hz
#   bash ~/simdist/deploy_simdist_1015/run_tracking_test.sh 10       # only 10 Hz
#   ACTION_MAP=reseed bash ...                                       # the 10-15 baseline map
#
# Does by itself: bring up cameras + robot + Robotiq + GELLO publisher in the container (unless
# already up; no matching gate, the controller stays off), wrist camera_info stop-gap if the
# ZED-M serial is unreadable, the scripted server per rate, the arm back to DROID home between
# rates (reset_arm, controller left off), the live client (--activate: holds the measured pose,
# so a misaligned GELLO cannot pull the arm), the analysis + figure.
# Asks ONCE before the arm moves. Stops before the next rate if any overshoot > 10 mm.
# Leaves the stack up at the end (controller off); stop it with: bash $0 --down
set -euo pipefail

C=ros2_devcontainer-gello-ros2-1
ROSENV='export CYCLONEDDS_URI=file:///workspace/ros2/cyclonedds_ws.xml; source /opt/ros/humble/setup.bash; source /workspace/ros2/install/setup.bash'
D="$HOME/simdist/deploy_simdist_1015"
PY="$HOME/anaconda3/envs/simdist/bin/python"
RUNNER_HOST="$HOME/simdist/gello_software/simdist_runner"
CELL=real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml
SCRIPTS=/workspace/ros2/src/franka_gello_state_publisher/scripts
MAX_OVERSHOOT_MM=10
ACTION_MAP=${ACTION_MAP:-sim_ref}
cd "$D"

dex() { docker exec "$C" bash -lc "$ROSENV; $1"; }

if [[ "${1:-}" == "--down" ]]; then
    docker exec "$C" pkill -INT -f bringup_stack.py || true
    docker exec "$C" pkill -f wrist_camera_info.py || true
    echo "stack stopping (logs in demos/.session_logs)"; exit 0
fi
RATES=("$@"); [[ ${#RATES[@]} -eq 0 ]] && RATES=(20 10)

need=(/franka/joint_states /gripper/joint_states /cameras/scene/left/image/compressed
      /cameras/wrist/left/image/compressed /cameras/scene/left/camera_info)
stack_up() {
    local t; t=$(dex "timeout 10 ros2 topic list" 2>/dev/null) || return 1
    for x in "${need[@]}"; do grep -qx "$x" <<<"$t" || return 1; done
}

# ---- 1. stack
if stack_up && docker exec "$C" pgrep -f "publish_zed_uvc" >/dev/null; then
    echo "[stack] already up"
else
    echo "[stack] bringing up cameras + robot + Robotiq + GELLO (no matching gate, controller off) ..."
    docker exec "$C" pkill -f wrist_camera_info.py || true
    docker exec -d "$C" bash -lc "$ROSENV; cd /workspace/simdist_runner; exec python3 bringup_stack.py \
        --kill-stale --no-record --gripper robotiq --gripper-config example_fr3_config_robotiq.yaml \
        > /workspace/simdist_runner/bringup.log 2>&1"
    for i in $(seq 180); do
        grep -q STACK_UP "$RUNNER_HOST/bringup.log" 2>/dev/null && break
        if grep -qE "exited|Error|error:|interrupted" "$RUNNER_HOST/bringup.log" 2>/dev/null; then
            tail -20 "$RUNNER_HOST/bringup.log"; echo "[stack] FAILED"; exit 1; fi
        sleep 1
    done
    grep -q STACK_UP "$RUNNER_HOST/bringup.log" || { tail -20 "$RUNNER_HOST/bringup.log"; echo "[stack] timeout"; exit 1; }
    for i in $(seq 30); do stack_up && break; sleep 1; done
    stack_up || { echo "[stack] topics missing"; dex "ros2 topic list"; exit 1; }
    echo "[stack] up"
fi
if ! dex "timeout 8 ros2 topic list" | grep -qx /cameras/wrist/left/camera_info; then
    echo "[stack] wrist camera_info missing (ZED-M serial unreadable): starting the stop-gap"
    docker exec -d "$C" bash -lc "$ROSENV; exec python3 /workspace/simdist_runner/wrist_camera_info.py \
        > /workspace/simdist_runner/wrist_camera_info.log 2>&1"
    sleep 4
fi

# ---- 2. one confirmation
echo
echo "======================================================================"
echo " The arm will move: DROID home, then ~6-8 cm in z and x at 3 speeds, rates: ${RATES[*]} Hz,"
echo " action map ${ACTION_MAP} (commanded TCP capped at 8 cm/s, <= 3 cm ahead of the arm)"
echo " (>= 15 cm above the table). Do NOT touch the GELLO. Hand on the e-stop."
echo "======================================================================"
read -r -p "Type g then Enter to start, anything else aborts: " go
[[ "$go" == "g" ]] || { echo "aborted; nothing moved."; exit 0; }

DIRS=()
for HZ in "${RATES[@]}"; do
    STAMP=$(date +%m%d_%H%M%S)
    RUN="runs/track${HZ}_${ACTION_MAP}_${STAMP}"
    mkdir -p "$RUN"
    echo; echo "[$HZ Hz] arm -> DROID home"
    dex "python3 $SCRIPTS/reset_arm.py --yes --no-resume --gripper-move-action ''" 2>&1 | tail -4

    echo "[$HZ Hz] server -> $RUN"
    "$PY" real2sim/deploy/policy_server.py --scripted track --task Threading_RealCellRobotiq \
        --cell "$CELL" --action_map "$ACTION_MAP" --save_dir "$RUN" --save_every 1 > "$RUN/server.log" 2>&1 &
    SRV=$!
    trap 'kill $SRV 2>/dev/null || true' EXIT
    for i in $(seq 60); do grep -q "listening" "$RUN/server.log" && break; sleep 1; done
    grep -q "listening" "$RUN/server.log" || { tail "$RUN/server.log"; echo "server did not start"; exit 1; }

    # the robot comes up with spawn_controller:=false and teleop only loads the controller at its
    # hand-over, which bringup skips: load it INACTIVE (no motion), as reset_arm.py does
    if ! dex "timeout 10 ros2 control list_controllers" 2>/dev/null | grep -q joint_impedance_controller; then
        echo "[$HZ Hz] loading joint_impedance_controller (inactive)"
        dex "ros2 run controller_manager spawner joint_impedance_controller --inactive \
            --controller-manager-timeout 30 \
            --param-file /workspace/ros2/src/franka_fr3_arm_controllers/config/controllers.yaml" 2>&1 | tail -2
    fi
    dex "timeout 10 ros2 control list_controllers" 2>/dev/null | grep joint_impedance_controller || true

    echo "[$HZ Hz] live"
    set +e
    dex "cd /workspace/simdist_runner; python3 robot_client_1015.py --gripper-state-topic /gripper/joint_states \
        --hz $HZ --steps 3000 --joint-log /workspace/simdist_runner/track${HZ}.npz --live --activate --yes" \
        2>&1 | tee "$RUN/client.log" | grep -vE "^\[ *[0-9]+\]"
    rc=${PIPESTATUS[0]}
    set -e
    kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
    cp "$RUNNER_HOST/track${HZ}.npz" "$RUN/joints.npz" 2>/dev/null || echo "(no joint log)"
    DIRS+=("$RUN")
    "$PY" real2sim/zed_capture/calib_20261014/tools/analyze_tracking.py "$RUN" || true
    [[ $rc -eq 0 ]] || { echo "[$HZ Hz] client exited with $rc; stopping."; break; }
    worst=$("$PY" -c "import json,math;r=json.load(open('$RUN/tracking_report.json'));k='past_target_mm' if '$ACTION_MAP'=='sim_ref' else 'overshoot_mm';v=[s[k] for s in r['segments'].values() if not math.isnan(s[k])];print(max(v) if v else 0)" 2>/dev/null || echo 0)
    echo "[$HZ Hz] worst overshoot ${worst} mm"
    if "$PY" -c "import sys; sys.exit(0 if float('$worst') > $MAX_OVERSHOOT_MM else 1)"; then
        echo "[$HZ Hz] overshoot > ${MAX_OVERSHOOT_MM} mm: stopping before the next rate."; break
    fi
done

echo; echo "[arm] -> DROID home, controller left off"
dex "python3 $SCRIPTS/reset_arm.py --yes --no-resume --gripper-move-action ''" 2>&1 | tail -3 || true
if [[ ${#DIRS[@]} -gt 0 ]]; then
    "$PY" real2sim/zed_capture/calib_20261014/tools/analyze_tracking.py "${DIRS[@]}" \
        --fig "${DIRS[-1]}/track.jpg" > "${DIRS[-1]}/summary.txt" 2>&1 || true
    cat "${DIRS[-1]}/summary.txt"
fi
echo "runs: ${DIRS[*]}"
echo "stack still up (controller off). To teleop again: bash $0 --down, then run_teleop_session.py as usual."
