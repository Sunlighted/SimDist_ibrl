"""MimicGen + the Robotiq Threading cell's execution layer (env/exec_layer.py).

MimicGen turns each target end-effector pose into a delta action relative to `get_robot_eef_pose()`, which the
stock MG_Threading reads off the MEASURED grip site. With the execution layer an action is a delta on the
REFERENCE arm (as on the robot, and as the converter's `ref_inverse` labels); the measured arm trails the
reference by ~1 cm, so labels relative to it would overshoot by that lag at deploy.

`patch()` makes MG_Threading (the name every prepared Threading source dataset records) return the reference
pose whenever the env has an execution layer, and the measured one otherwise (byte-for-byte stock).
Threading_RealCellRobotiq calls it when it installs a layer, so no generation path can miss it.
"""
_PATCHED = False


def patch():
    global _PATCHED
    if _PATCHED:
        return
    try:
        from mimicgen.env_interfaces.robosuite import MG_Threading
    except ImportError:          # no MimicGen in this process: nothing to patch
        return
    stock = MG_Threading.get_robot_eef_pose

    def get_robot_eef_pose(self):
        layer = getattr(self.env, "exec_layer", None)
        if layer is not None:
            return layer.ref_site_pose()
        return stock(self)

    MG_Threading.get_robot_eef_pose = get_robot_eef_pose
    _PATCHED = True
