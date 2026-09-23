"""MimicGen env interface for PegInsertion_D0 (env/peg_insertion.py).

Same two-subtask structure as MG_Square: grasp the peg (motion relative to the
peg), then insert it (motion relative to the board). Importing this module
registers `MG_PegInsertion` with MimicGen's interface registry (metaclass
registration, same as the built-in MG_* classes).
"""

from mimicgen.env_interfaces.robosuite import RobosuiteInterface

import env.peg_insertion  # noqa: F401  (registers PegInsertion_D0 with robosuite)


class MG_PegInsertion(RobosuiteInterface):
    """Corresponds to env.peg_insertion.PegInsertion and its _D0 variant."""

    def get_object_poses(self):
        # the board is a static fixture (no joint), but its body pose is still
        # read from sim, so board_pos_noise / board_yaw_noise are handled
        return dict(
            peg=self.get_object_pose(obj_name=self.env.peg.root_body, obj_type="body"),
            board=self.get_object_pose(obj_name=self.env.board.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        # first subtask is grasping the peg (motion relative to the peg)
        signals["grasp"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.peg.contact_geoms])
        )
        # final subtask is inserting the peg into the board pocket (motion
        # relative to the board) - final subtask signal is not needed
        return signals
