from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import numpy as np


# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}


class GH360TJointPositionController(Controller):
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 kp=50,
                 damping_ratio=1,
                 impedance_mode="fixed",
                 kp_limits=(0, 300),
                 damping_ratio_limits=(0, 100),
                 policy_freq=20,
                 qpos_limits=None,
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
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits
        self.position_limits = np.array(qpos_limits) if qpos_limits is not None else qpos_limits

        # kp kd
        self.kp = self.nums2array(kp, self.control_dim)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], self.control_dim)
        self.kp_max = self.nums2array(kp_limits[1], self.control_dim)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], self.control_dim)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], self.control_dim)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, "Error: Tried to instantiate OSC controller for unsupported " \
                                                  "impedance mode! Inputted impedance mode: {}, Supported modes: {}". \
            format(impedance_mode, IMPEDANCE_MODES)

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim *= 3
        elif self.impedance_mode == "variable_kp":
            self.control_dim *= 2

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize
        self.goal_qpos = None

    def set_goal(self, action, set_qpos=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_qpos (Iterable): If set, overrides @action and sets the desired absolute joint position goal state

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary
        jnt_dim = len(self.qpos_index)
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = action[:jnt_dim], action[jnt_dim:2*jnt_dim], action[2*jnt_dim:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:jnt_dim], action[jnt_dim:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        else:  # This is case "fixed"
            delta = action

        # Check to make sure delta is size self.joint_dim
        assert len(delta) == jnt_dim, "Delta qpos must be equal to the robot's joint dimension space!"

        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

        # self.goal_qpos = action
        self.goal_qpos = set_goal_position(scaled_delta,
                                           self.joint_pos,
                                           position_limit=self.position_limits,
                                           set_pos=set_qpos)

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        desired_qpos = None

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                desired_qpos = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_qpos = np.array(self.goal_qpos)

        # torques = pos_err * kp + vel_err * kd
        # print("goal: ",self.goal_qpos)
        print("desired: ",desired_qpos)
        print("current: ", self.joint_pos)
        position_error = desired_qpos - self.joint_pos
        vel_pos_error = -self.joint_vel
        desired_torque = (np.multiply(np.array(position_error), np.array(self.kp))
                          + np.multiply(vel_pos_error, self.kd))

        # Return desired torques plus gravity compensations
        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation
        #print(self.torques)
        self.torques = self.jointToMotorTorques(self.torques)
        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets joint position goal to be current position
        """
        self.goal_qpos = self.joint_pos

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    def jointToMotorTorques(self, qtorques):
        motor_torques = np.zeros(25)
        scalar = 1
        shoulder_yaw_scalar = 0.0745
        shoulder_roll_scalar = 0.0505
        shoulder_pitch_scalar = 0.05
        upperarm_roll_scalar = 0.043
        elbow_scalar = 0.04
        wrist_pitch_scalar = 0.03#0.03

        if qtorques[0] < 0:
            motor_torques[2] =  self.torqueClamp(-qtorques[0]/shoulder_yaw_scalar/2)
            motor_torques[4] =  self.torqueClamp(-qtorques[0]/shoulder_yaw_scalar/2)
        else:
            motor_torques[1] =  self.torqueClamp(qtorques[0]/shoulder_yaw_scalar/2)
            motor_torques[3] =  self.torqueClamp(qtorques[0]/shoulder_yaw_scalar/2)

        if qtorques[1] < 0.0:
            motor_torques[6] =  self.torqueClamp(-qtorques[1]/shoulder_roll_scalar/2)
            motor_torques[8] =  self.torqueClamp(-qtorques[1]/shoulder_roll_scalar/2)
        else:
            motor_torques[5] =  self.torqueClamp(qtorques[1]/shoulder_roll_scalar/2)
            motor_torques[7] =  self.torqueClamp(qtorques[1]/shoulder_roll_scalar/2)

        if qtorques[2] < 0:
            motor_torques[10] =  self.torqueClamp(-qtorques[2]/shoulder_pitch_scalar/2)
            motor_torques[12] =  self.torqueClamp(-qtorques[2]/shoulder_pitch_scalar/2)
        else:
            motor_torques[9] =  self.torqueClamp(qtorques[2]/shoulder_pitch_scalar/2)
            motor_torques[11] =  self.torqueClamp(qtorques[2]/shoulder_pitch_scalar/2)

        if qtorques[3] < 0.0:
            motor_torques[14] =  self.torqueClamp(-qtorques[3]/upperarm_roll_scalar/2)
            motor_torques[16] =  self.torqueClamp(-qtorques[3]/upperarm_roll_scalar/2)
        else:
            motor_torques[13] =  self.torqueClamp(qtorques[3]/upperarm_roll_scalar/2)
            motor_torques[15] =  self.torqueClamp(qtorques[3]/upperarm_roll_scalar/2)

        if qtorques[4] < 0:
            motor_torques[18] =  self.torqueClamp(-qtorques[4]/elbow_scalar/2)
            motor_torques[20] =  self.torqueClamp(-qtorques[4]/elbow_scalar/2)
        else:
            motor_torques[17] =  self.torqueClamp(qtorques[4]/elbow_scalar/2)
            motor_torques[19] =  self.torqueClamp(qtorques[4]/elbow_scalar/2)

        motor_torques[0] = self.torqueClamp(qtorques[5])

        if qtorques[6] < 0:
            motor_torques[22] =  self.torqueClamp(-qtorques[6]/wrist_pitch_scalar/2)
            motor_torques[23] =  self.torqueClamp(-qtorques[6]/wrist_pitch_scalar/2)
        else:
            motor_torques[21] =  self.torqueClamp(qtorques[6]/wrist_pitch_scalar/2)
            motor_torques[24] =  self.torqueClamp(qtorques[6]/wrist_pitch_scalar/2)
            pass

        return motor_torques

    def torqueClamp(self, torque):
        max = 350
        min = -max
        if torque > max:
            new_torque = max
        elif torque < min:
            new_torque = min
        else:
            new_torque = torque
        return new_torque

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return 'JOINT_POSITION'
