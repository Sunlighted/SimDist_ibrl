"""Strict success validation for MimicGen Threading_D0.

The raw task success (`Threading._check_success`) is an *unguarded,
instantaneous* 3D position test:

    ||geom_xpos["needle_obj_needle"] - mean(ring geom centers)|| < 12 mm

with no check that the needle actually threaded the hole cleanly. Two facts let
a learned policy reward-hack it:

  1. It fires on a SINGLE control step (and with end_on_success the episode ends
     there), so a transient clip-through or a fast fly-through that momentarily
     lands the needle's *center* inside the 12 mm ball counts as success. At
     20 Hz a needle moving 0.5 m/s travels 25 mm/step ≈ the ball diameter, so a
     wild swing can cross the ball between two checks.
  2. The tripod (ring) is a FREE body of only ~18 g. Because success is the
     *relative* distance between the two centers, ramming the lightweight ring
     toward the needle satisfies the test just as well as inserting the needle —
     which is how "硬捅 / 捅出屏幕 but success" episodes happen.

This module layers validity gates on top of the raw position test, computed in
the ibrl wrapper (we do NOT edit the vendored mimicgen submodule). A frame only
counts when the raw test passes AND the enabled gates pass; success is declared
only after the valid condition holds for `dwell_steps` consecutive control
steps. When `enabled`, the wrapper uses the validated signal for BOTH the
returned `success` flag and the per-step reward, so RL is trained against the
honest objective rather than the hackable one.

Gates (per the chosen design):
  - dwell:        position test must hold for `dwell_steps` consecutive steps
                  (kills transient clip-through / fly-through).
  - tripod:       ring center must stay within `max_tripod_disp` of its
                  post-reset position (kills "shove the ring" + "off-screen").
  - needle speed: OPTIONAL (off when `max_needle_speed <= 0`); needle linear
                  speed must be below threshold (kills high-momentum ramming).

Threading-only: relies on `ring_geom_ids` / `needle_joint` from
`resolve_threading_bindings`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StrictSuccessConfig:
    # Master switch. Default off so existing configs / trained-checkpoint evals
    # are byte-for-byte unchanged; flip to true in BOTH train and eval configs
    # to train + measure against the honest objective.
    enabled: bool = False

    # Position test must hold for this many CONSECUTIVE control steps (20 Hz,
    # so 3 = 0.15 s). Kills the one-frame transient / fly-through successes.
    dwell_steps: int = 3

    # Ring center may drift at most this far (meters) from its post-reset
    # snapshot. Kills the "ram the free 18 g tripod toward the needle" hack and
    # off-screen successes. Reference 0.04 m — deliberately generous so small
    # legitimate contact nudges during a real insertion aren't rejected; lower
    # it toward ~0.02 m to be stricter. WARNING: too high re-admits the shove
    # hack (the cheat only needs to move the ring as far as the needle is away).
    max_tripod_disp: float = 0.04

    # OPTIONAL needle linear-speed gate (m/s). <= 0 disables it (default). Set
    # e.g. 0.5 to also reject high-momentum ramming where the needle is still
    # moving fast on the success frame.
    max_needle_speed: float = 0.0


class SuccessValidator:
    """Stateful per-episode validator. Holds the consecutive-valid streak and
    the post-reset ring-center snapshot. Construct once; `reset()` at every
    episode start; `step()` once per control step with the raw success bit.

    Bindings are passed in on each call (rather than cached) because robosuite
    hard_reset rebuilds the sim — the wrapper re-resolves bindings every reset.
    """

    def __init__(self, cfg: StrictSuccessConfig):
        self.cfg = cfg
        self._streak = 0
        self._ring_center0: Optional[np.ndarray] = None

    def reset(self, env, bindings: dict) -> None:
        """Snapshot the ring center AFTER env.reset() + any domain-shift forward
        so the tripod-displacement gate measures drift from the true start pose
        (a shifted hole_offset is part of that start pose, not drift)."""
        self._streak = 0
        self._ring_center0 = self._ring_center(env, bindings)

    @staticmethod
    def _ring_center(env, bindings: dict) -> np.ndarray:
        return np.mean(env.sim.data.geom_xpos[bindings["ring_geom_ids"]], axis=0)

    def _frame_valid(self, env, bindings: dict, raw_success: bool) -> bool:
        if not raw_success:
            return False
        # Gate: the ring hasn't been shoved away from its reset pose.
        if self.cfg.max_tripod_disp > 0 and self._ring_center0 is not None:
            disp = np.linalg.norm(self._ring_center(env, bindings) - self._ring_center0)
            if disp > self.cfg.max_tripod_disp:
                return False
        # Gate (optional): the needle isn't still moving fast (no ramming).
        if self.cfg.max_needle_speed > 0:
            qvel = np.asarray(env.sim.data.get_joint_qvel(bindings["needle_joint"]))
            if np.linalg.norm(qvel[:3]) > self.cfg.max_needle_speed:
                return False
        return True

    def step(self, env, bindings: dict, raw_success: bool) -> bool:
        """Update the streak with this frame's validity and return whether a
        validated success is now in effect (streak >= dwell_steps)."""
        if self._frame_valid(env, bindings, raw_success):
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self.cfg.dwell_steps
