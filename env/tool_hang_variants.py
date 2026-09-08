"""ToolHang stage-1 ("insert the frame") task variants.

Importing this module registers the env(s) with robosuite -- robosuite's
`EnvMeta` metaclass auto-registers any `MujocoEnv` subclass on definition, so no
edit to the vendored submodule is needed (same trick as `env/threading_variants`).

Why: the full `ToolHang` is a two-stage task (insert the L-shaped frame into the
stand, THEN hang the wrench on the frame's hook). robomimic's state-based
BC-RNN gets ~14% on it, which makes it useless as a source task. `ToolHangInsert`
keeps stage 1 only -- exactly robosuite's own `_check_frame_assembled()` -- and
declares success once the frame is assembled AND the gripper has let go of it.

Why also require the release:
  `_check_frame_assembled()` is a pure geometry check on the frame/base poses.
  It fires while the gripper is still holding the frame in place, i.e. before we
  know the insertion is actually stable. Since the eval wrapper ends the episode
  on success (`end_on_success=True`), a held-in-place success would be a
  free pass for a policy that jams the frame against the stand and freezes.
  Requiring "not touching the frame" makes success mean "inserted and it stands
  on its own", and it is exactly where the human demos let go anyway.

Everything else (objects, obs, action space, reset distribution, 53-dim state)
is byte-for-byte `ToolHang`, so the robomimic ph demos replay unchanged.
"""

from robosuite.environments.manipulation.tool_hang import ToolHang
from robosuite.utils.sim_utils import check_contact


class ToolHangInsert(ToolHang):
    """ToolHang truncated to stage 1: insert the frame into the stand and let go.

    success = ToolHang._check_frame_assembled()  and  gripper not touching frame
    """

    # Flip to False for the "insertion geometry only" variant (success may fire
    # while the frame is still grasped).
    REQUIRE_RELEASE = True

    def _gripper_touching_frame(self):
        """True if either fingerpad is in contact with any frame geom."""
        for pad in ("left_fingerpad", "right_fingerpad"):
            geoms = self.robots[0].gripper.important_geoms[pad]
            if check_contact(self.sim, geoms, self.frame.contact_geoms):
                return True
        return False

    def _check_success(self):
        if not self._check_frame_assembled():
            return False
        if self.REQUIRE_RELEASE and self._gripper_touching_frame():
            return False
        return True


class ToolHangInsertHold(ToolHangInsert):
    """Geometry-only variant: success as soon as the frame is assembled, even if
    the gripper is still holding it. Kept as an ablation handle."""

    REQUIRE_RELEASE = False
