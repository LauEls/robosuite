from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .osc import OperationalSpaceController
from .joint_pos import JointPositionController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController
from .gh2_joint_pos import GH2JointPositionController
from .gh2_joint_tor import GH2JointTorqueController
from .gh2_osc import GH2OperationalSpaceController
from .gh360t_equilibrium_point import GH360TEquilibriumPointController
from .gh360t_joint_pos import GH360TJointPositionController
from .gh360t_motor_pos import GH360TMotorPositionController


CONTROLLER_INFO = {
    "JOINT_VELOCITY": "Joint Velocity",
    "JOINT_TORQUE": "Joint Torque",
    "JOINT_POSITION": "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE": "Operational Space Control (Position + Orientation)",
    "OSC_POSE_VI": "Operational Space Control (Position + Orientation) with variable impedance",
    "IK_POSE": "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
    "GH2_JOINT_POSITION": "Joint Position Controller for GH2 arm",
    "GH2_JOINT_TORQUE": "Joint Torque Controller for GH2 arm",
    "GH2_OSC_POSITION": "Operational Space Control (Position Only) for GH2 arm",
    "GH2_OSC_POSE":     "Operational Space Control (Position + Orientation) for GH2 arm",
    "GH360T_EQUILIBRIUM_POINT":     "Equilibrium Point Controller for GH360 arm",
    "GH360T_EQUILIBRIUM_POINT_SOFT":     "Equilibrium Point Controller with no stiffness variation for GH360 arm",
    "GH360T_EQUILIBRIUM_POINT_FIXED":     "Equilibrium Point Controller with a fixed stiffness value for GH360 arm",
    "GH360T_JOINT_POSITION":    "Joint Position Controller for GH360 arm",
    "GH360T_MOTOR_POSITION":    "Motor Position Controller for GH360 arm",
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
