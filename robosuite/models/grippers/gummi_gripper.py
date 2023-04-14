"""
Universal Gummi Gripper
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel

class GummiGripperBase(GripperModel):
    """
    Universal Gummi Gripper

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/gummi_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["gripper_lf_vis", "gripper_finger_pad_2"],
            "right_finger": ["gripper_mf_vis", "gripper_finger_pad_1"],
            "left_fingerpad": "gripper_finger_pad_2",
            "right_fingerpad": "gripper_finger_pad_1",
        }


class GummiGripper(GummiGripperBase):
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
        print(self.init_qpos)
        print(action)
        assert len(action) == self.dof
        
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        # self.current_action = np.clip(self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1