from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .gripper_tester import GripperTester

from .panda_gripper import PandaGripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_three_finger_gripper import RobotiqThreeFingerGripper, RobotiqThreeFingerDexterousGripper
from .panda_gripper import PandaGripper
from .jaco_three_finger_gripper import JacoThreeFingerGripper, JacoThreeFingerDexterousGripper
from .robotiq_140_gripper import Robotiq140Gripper
from .wiping_gripper import WipingGripper
from .wiping_gripper_gh360 import WipingGripperGH360
from .null_gripper import NullGripper
from .gummi_gripper import GummiGripper
from .hook_gripper import HookGripper
from .gh360_gripper import GH360Gripper


GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "PandaGripper": PandaGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "JacoThreeFingerDexterousGripper": JacoThreeFingerDexterousGripper,
    "WipingGripper": WipingGripper,
    "WipingGripperGH360": WipingGripperGH360,
    "Robotiq85Gripper": Robotiq85Gripper,
    "Robotiq140Gripper": Robotiq140Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "RobotiqThreeFingerDexterousGripper": RobotiqThreeFingerDexterousGripper,
    "GummiGripper": GummiGripper,
    "HookGripper": HookGripper,
    "GH360Gripper": GH360Gripper,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()
