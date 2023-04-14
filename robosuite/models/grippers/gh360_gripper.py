"""
Universal Gummi Gripper v2
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel

class GH360GripperBase(GripperModel):
    """
    Universal Gummi Gripper

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/gh360_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_finger_tip_col", "left_finger_base_col", "left_finger_pad_left"],
            "right_finger": ["right_finger_tip_col", "right_finger_base_col", "right_finger_pad_right"],
            "left_fingerpad": ["gripper_finger_pad_left"],
            "right_fingerpad": ["gripper_finger_pad_right"],
        }


class GH360Gripper(GH360GripperBase):
    """
    Modifies GummiGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1