import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class GH360T(ManipulatorModel):
    """
    GH360T is a articulated soft robot arm implemented with tendons inspired by the GH2 arm from Fielwork Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/gh360t/robot.xml"), idn=idn)

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
        return np.array([-4.15279534e-03, -1.26763878e-01, -4.83506730e-04,  1.37694984e+00,
        1.75309334e+00, -6.82966358e-05, -8.73310244e-03])
        #return np.array([0.0, 0.0, 0.1066, 1.1496, 1.3725,  0.0, 0.0])
        #return np.array([0.000, 0.000, 0.000, np.pi/2, np.pi/2, 0.000, 0.000])

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
