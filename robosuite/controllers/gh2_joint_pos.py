from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import numpy as np

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}

class GH2JointPositionController(Controller):
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
                 policy_freq=20, #20
                 qpos_limits=None,
                 interpolator=None,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):
        #print(eef_name)
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
        #kp = pow((2*np.pi*5),2)
        kp = 50
        #print("kp: ", kp)
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
        #self.goal_qpos = np.array([0,0,0,0,-np.pi/2,0,0])

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
        #print("delta: ",delta)
        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None
        # print("action: ", action)
        # print("scaled delta: ", scaled_delta)
        # print("goal 1: ", self.goal_qpos)
        self.goal_qpos = action
        # self.goal_qpos = set_goal_position(scaled_delta,
        #                                    #self.joint_pos,
        #                                    self.goal_qpos,
        #                                    position_limit=self.position_limits,
        #                                    set_pos=set_qpos)
        #print("goal 2: ", self.goal_qpos)
        #self.goal_qpos = np.array([1,1,-1,1,-1,-1,-1])
        #self.goal_qpos = np.array([0,0,0,0,0,0,0])
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        #print("Goal qpos: ",self.goal_qpos)
        if self.goal_qpos is None:
            print("goal_qpos is None!!!")
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

        #torques = pos_err * kp + vel_err * kd
        #print("goal: ",self.goal_qpos)
        #print("desired: ",desired_qpos)
        #print("current: ", self.joint_pos)
        position_error = desired_qpos - self.joint_pos
        vel_pos_error = -self.joint_vel
        # desired_torque = position_error * self.kp + vel_pos_error * self.kd
        desired_torque = (np.multiply(np.array(position_error), np.array(self.kp))
                          + np.multiply(vel_pos_error, self.kd))

        # self.torques = desired_torque #+ self.torque_compensation
        # Return desired torques plus gravity compensations
        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation
        # print("desired pos: ", desired_qpos[0])
        # print("current pos: ", self.joint_pos[0])
        # print("pos error: ",position_error[0])
        # print("torque: ",self.torques[0])
        self.torques = jointToMotorTorques(self.torques)
        #print(self.torques)

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets joint position goal to be current position
        """
        print("RESET")
        self.goal_qpos = self.joint_pos

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)
    
def jointToMotorTorques(qtorques):
    motor_torques = np.zeros(25)
    scalar = 0.1 #100

    # motor_torques[20] = 1
    # return motor_torques

    #shoulder1 joint
    if qtorques[0] < 0:
        motor_torques[1] =  torqueClamp(-qtorques[0]/scalar/2)
        motor_torques[4] =  torqueClamp(-qtorques[0]/scalar/2)
        
        # motor_torques[2] =  torqueClamp(-qtorques[0]/scalar/2+0.4)
        # motor_torques[3] =  torqueClamp(-qtorques[0]/scalar/2+0.4)
        # motor_torques[1] =  torqueClamp(0.4)
        # motor_torques[4] =  torqueClamp(0.4)
    else:
        motor_torques[2] =  torqueClamp(qtorques[0]/scalar/2)
        motor_torques[3] =  torqueClamp(qtorques[0]/scalar/2)
        
        # motor_torques[1] =  torqueClamp(qtorques[0]/scalar/2+0.4)
        # motor_torques[4] =  torqueClamp(qtorques[0]/scalar/2+0.4)
        # motor_torques[2] =  torqueClamp(0.4)
        # motor_torques[3] =  torqueClamp(0.4)

    #shoulder2 joint
    if qtorques[1] < 0.0:
        motor_torques[6] =  torqueClamp(-qtorques[1]/scalar/2)
        motor_torques[7] =  torqueClamp(-qtorques[1]/scalar/2)
        # motor_torques[5] =  torqueClamp(-qtorques[1]/scalar/2+0.4)
        # motor_torques[8] =  torqueClamp(-qtorques[1]/scalar/2+0.4)
        # motor_torques[6] =  torqueClamp(0.4)
        # motor_torques[7] =  torqueClamp(0.4)
    else:
        motor_torques[5] =  torqueClamp(qtorques[1]/scalar/2)
        motor_torques[8] =  torqueClamp(qtorques[1]/scalar/2)
        # motor_torques[6] =  torqueClamp(qtorques[1]/scalar/2+0.4)
        # motor_torques[7] =  torqueClamp(qtorques[1]/scalar/2+0.4)
        # motor_torques[5] =  torqueClamp(0.4)
        # motor_torques[8] =  torqueClamp(0.4)

    #shoulder3 joint
    if qtorques[2] < 0:
        motor_torques[10] =  torqueClamp(-qtorques[2]/scalar/2)
        motor_torques[12] =  torqueClamp(-qtorques[2]/scalar/2)
        # motor_torques[9] =  torqueClamp(-qtorques[2]/scalar/2+0.4)
        # motor_torques[11] =  torqueClamp(-qtorques[2]/scalar/2+0.4)
        # motor_torques[10] =  torqueClamp(0.4)
        # motor_torques[12] =  torqueClamp(0.4)
    else:
        motor_torques[9] =  torqueClamp(qtorques[2]/scalar/2)
        motor_torques[11] =  torqueClamp(qtorques[2]/scalar/2)
        # motor_torques[10] =  torqueClamp(qtorques[2]/scalar/2+0.4)
        # motor_torques[12] =  torqueClamp(qtorques[2]/scalar/2+0.4)
        # motor_torques[9] =  torqueClamp(0.4)
        # motor_torques[11] =  torqueClamp(0.4)

    #upper_arm1 joint
    if qtorques[3] < 0.0:
        motor_torques[13] =  torqueClamp(-qtorques[3]/scalar/2)
        motor_torques[15] =  torqueClamp(-qtorques[3]/scalar/2)
        # motor_torques[14] =  torqueClamp(-qtorques[3]/scalar/2+0.4)
        # motor_torques[16] =  torqueClamp(-qtorques[3]/scalar/2+0.4)
        # motor_torques[13] =  torqueClamp(0.4)
        # motor_torques[15] =  torqueClamp(0.4)
    else:
        motor_torques[14] =  torqueClamp(qtorques[3]/scalar/2)
        motor_torques[16] =  torqueClamp(qtorques[3]/scalar/2)
        # motor_torques[13] =  torqueClamp(qtorques[3]/scalar/2+0.4)
        # motor_torques[15] =  torqueClamp(qtorques[3]/scalar/2+0.4)
        # motor_torques[14] =  torqueClamp(0.4)
        # motor_torques[16] =  torqueClamp(0.4)

    #upper_arm2 joint
    if qtorques[4] < 0:
        motor_torques[18] =  torqueClamp(-qtorques[4]/scalar/2)
        motor_torques[20] =  torqueClamp(-qtorques[4]/scalar/2)
        # motor_torques[17] =  torqueClamp(-qtorques[4]/scalar/2+0.4)
        # motor_torques[19] =  torqueClamp(-qtorques[4]/scalar/2+0.4)
        # motor_torques[18] =  torqueClamp(0.4)
        # motor_torques[20] =  torqueClamp(0.4)
    else:
        motor_torques[17] =  torqueClamp(qtorques[4]/scalar/2)
        motor_torques[19] =  torqueClamp(qtorques[4]/scalar/2)
        # motor_torques[18] =  torqueClamp(qtorques[4]/scalar/2+0.4)
        # motor_torques[20] =  torqueClamp(qtorques[4]/scalar/2+0.4)
        # motor_torques[17] =  torqueClamp(0.4)
        # motor_torques[19] =  torqueClamp(0.4)

    #print(motor_torques[17:21])
    #lower_arm1 joint
    # if qtorques[5]/100 > 280:
    #     motor_torques[0] = 280
    # elif qtorques[5]/100 < -280:
    #     motor_torques[0] = -280
    # else:
    #     motor_torques[0] = qtorques[5]/100
    motor_torques[0] = torqueClamp(qtorques[5]/scalar)

    #lower_arm2 joint
    if qtorques[6] < 0:
        motor_torques[22] =  torqueClamp(-qtorques[6]/scalar/2)
        motor_torques[24] =  torqueClamp(-qtorques[6]/scalar/2)
        # motor_torques[21] =  torqueClamp(-qtorques[6]/scalar/2+0.4)
        # motor_torques[23] =  torqueClamp(-qtorques[6]/scalar/2+0.4)
        # motor_torques[22] =  torqueClamp(0.4)
        # motor_torques[24] =  torqueClamp(0.4)
    else:
        motor_torques[21] =  torqueClamp(qtorques[6]/scalar/2)
        motor_torques[23] =  torqueClamp(qtorques[6]/scalar/2)
        # motor_torques[22] =  torqueClamp(qtorques[6]/scalar/2+0.4)
        # motor_torques[24] =  torqueClamp(qtorques[6]/scalar/2+0.4)
        # motor_torques[21] =  torqueClamp(0.4)
        # motor_torques[23] =  torqueClamp(0.4)
    # print("motor torques: "+str(motor_torques))

    #motor_torques = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    return motor_torques

def torqueClamp(torque):
    max = 280
    min = -max
    if torque > max:
        new_torque = max
    elif torque < min:
        new_torque = min
    else:
        new_torque = torque
    return new_torque
