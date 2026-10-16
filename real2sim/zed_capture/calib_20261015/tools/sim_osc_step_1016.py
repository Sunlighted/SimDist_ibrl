"""Sim OSC per-step displacement of the Robotiq cell at |a| = 0.2 / 0.5 / 1.0 (SIM_TODO §1).

The rate is the cell's control_freq; run once per cell to compare (THREADING_ROBOTIQ_CELL_YAML picks it):
  python sim_osc_step_1016.py                                     # 10-15 cell, 10 Hz
  THREADING_ROBOTIQ_CELL_YAML=.../threading_real_cell_20261014.yaml python sim_osc_step_1016.py   # 20 Hz
Same script as the real tracking test: from DROID home, 15 steps... of a constant action along x / z,
grip-site speed after the first step and over the steady part."""
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config
import env.threading_variants  # noqa: F401  registers Threading_RealCellRobotiq

cfg = load_controller_config(default_controller="OSC_POSE")
e = robosuite.make("Threading_RealCellRobotiq", robots="Panda", controller_configs=cfg,
                   has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                   initialization_noise=None)
hz = e.control_freq
print(f"control_freq {hz:g} Hz | OSC kp {cfg['kp']} | output_max {cfg['output_max'][:3]} m/step")
print(f"{'axis':4s} {'|a|':>4s} | {'step1 mm':>8s} {'steady mm/step':>14s} {'steady cm/s':>11s} | real sim_ref 10 Hz cm/s (README)")
real = {("x", 0.2): 2.7, ("x", 0.5): 5.8, ("x", 1.0): 5.3, ("z", 0.2): 2.5, ("z", 0.5): 6.6, ("z", 1.0): 7.5}
for ax in "xz":
    for amp in (0.2, 0.5, 1.0):
        np.random.seed(0); e.reset()
        a = np.zeros(7); a[6] = -1; a["xyz".index(ax)] = amp * (-1 if ax == "z" else 1)
        p = [e.sim.data.site_xpos[e.robots[0].eef_site_id].copy()]
        n = int(round(0.6 * hz))      # 0.6 s of motion, as many steps as fit
        for _ in range(n):
            e.step(a); p.append(e.sim.data.site_xpos[e.robots[0].eef_site_id].copy())
        d = np.linalg.norm(np.diff(np.array(p), axis=0), axis=1)
        st = d[len(d) // 2:].mean()
        print(f"{ax:4s} {amp:4.1f} | {1e3 * d[0]:8.1f} {1e3 * st:14.1f} {100 * st * hz:11.1f} | {real[(ax, amp)]}")
