from robosuite.controllers.base_controller import Controller
import numpy as np


class GH2JointTorqueController(Controller):
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 policy_freq=20,
                 torque_limits=None,
                 interpolator=None,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Control dimension
        #self.control_dim = len(joint_indexes["joints"])
        #print("control_dim: ", self.control_dim)
        self.control_dim = 25

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits (if not specified, set them to actuator limits by default)
        self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None                           # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None                               # Torques returned every time run_controller is called

    def set_goal(self, torques):
        """
        Sets goal based on input @torques.

        Args:
            torques (Iterable): Desired joint torques

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Check to make sure torques is size self.joint_dim
        assert len(torques) == self.control_dim, "Delta torque must be equal to the robot's joint dimension space!"

        self.goal_torque = np.clip(self.scale_action(torques), self.torque_limits[0], self.torque_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_torque is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                self.current_torque = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_torque = np.array(self.goal_torque)

        # Add gravity compensation
        # self.torques = self.current_torque + self.torque_compensation
        self.torques = self.current_torque

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint torque goal to be all zeros (pre-compensation)
        """
        self.goal_torque = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    @property
    def name(self):
        return 'JOINT_TORQUE'
