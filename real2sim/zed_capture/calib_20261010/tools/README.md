# Camera calibration tools (2026-10-10)

Copies of the scripts used on the robot workstation (`~/simdist/calib/`, `~/simdist/gello_software/simdist_runner/`).
Paths inside expect that layout. Usage and results: `../README.md`.
`overlay_real_scene.py` (gate C) runs from the SimDist_ibrl repo root with conda `simdist`.

| script | runs where | what |
|---|---|---|
| `joint_tap.py` | ROS container | mirrors `/franka/joint_states` into a JSON file the host can read |
| `record_calib.py` | host, `~/anaconda3/bin/python3` | records raw HD720 left eyes (+ joints); `--intrinsics` = one camera, hand-held board; live coverage grid and tilt counter |
| `solve_intrinsics.py` | host, conda `simdist` | K, D from one or more runs; coverage, tilt and model-swap checks |
| `solve_calib.py` | host, conda `simdist` | joint refinement of T_hand_wrist, T_base_board, T_scene_board and the square size; FK = robosuite Panda (`robot0_base` / `robot0_right_hand`) |
