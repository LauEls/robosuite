import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class GH360(ManipulatorModel):
    """
    GH360 is a articulated soft robot arm implemented without tendons inspired by the GH2 arm from Fielwork Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/gh360/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)))
    
    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "Robotiq140Gripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        # return np.array([0.0, 0.0, 0.0, np.pi/2, np.pi/2, 0.0, 0.0])
        return np.array([0.0204, -0.1854, 0.1467, 1.5825, 1.8675, 0.0, 0.0])
        # return np.array([4.31648455e-03, -1.26763655e-01,  1.46990937e-01,  1.39553796e+00, 
                        #  1.76128936e+00,  9.81432871e-04, -1.32544425e-02])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
