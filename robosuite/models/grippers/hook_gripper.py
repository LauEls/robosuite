"""
Null Gripper (if we don't want to attach gripper to robot eef).
"""
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class HookGripper(GripperModel):
    """
    Endeffector that is a hook without actuation

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/hook_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None
    
    @property
    def _important_geoms(self):
        return {
            "left_finger": ["hook_1_col", "inner_pad"],
            "right_finger": ["hook_3_col", "outer_pad"],
            "left_fingerpad": ["inner_pad"],
            "right_fingerpad": ["outer_pad"],
        }
