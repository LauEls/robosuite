from robosuite.controllers.base_controller import Controller
from robosuite.utils.tendon import Tendon
import numpy as np
import os
import json
from robosuite.utils.motor_joint import MotorJoint
from robosuite.utils.soft_joint import SoftJoint
from robosuite.utils.mjcf_utils import xml_path_completion
import csv
import time
import copy


class GH360TEquilibriumPointController(Controller):
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
                 #torque_limits=None,
                 force_limits=300,
                 interpolator=None,
                 joints = None,
                 tendon_model_file = None,
                 motor_max_pos = 31.42,
                 stiffness_mode = "variable", # if variable_stiffness is False, this will swap between the robot joints have no stiffness and having a fixed stiffness
                 tendon_noise = False,
                 tendon_noise_level = 0.0,
                 motor_move_sim = False,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Control dimension
        self.stiffness_mode = stiffness_mode

        if self.stiffness_mode == "variable":
            self.control_dim = 13#MAYBE READ THAT OUT OF A CONFIG FILE -> should be 13 at the end
        elif self.stiffness_mode == "fixed" or self.stiffness_mode == "no_stiffness":
            self.control_dim = 7

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits (if not specified, set them to actuator limits by default)
        #self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits
        self.force_limits = np.array(force_limits) #SHOULD COME FROM CONFIG FILE
        self.current_stiffness = np.zeros(6)
        self.min_stiffness = 0.0
        # self.variable_stiffness = variable_stiffness
        # self.soft = soft


        # control frequency
        self.control_freq = policy_freq

        # # interpolator
        # self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None                           # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None                               # Torques returned every time run_controller is called
        
        # filepath = os.path.join(os.path.dirname(__file__), "config/gh360t.json")
        self.motor_count = 0
        self.arm = []
        for joint_name in joints:
            filepath = os.path.join(os.path.dirname(__file__), "config/gh360/"+joint_name+".json")
            try:
                with open(filepath) as f:
                    variant = json.load(f)
            except FileNotFoundError:
                print("Error opening default controller filepath at: {}. "
                    "Please check filepath and try again.".format(filepath))
            #print(joint_name)
            # print(variant["motor_init_pos"])
            if variant["actuators"] == 4:
                if self.stiffness_mode == "no_stiffness" or self.stiffness_mode == "variable":
                    joint_fixed_stiffness = 0.0
                elif self.stiffness_mode == "fixed":
                    joint_fixed_stiffness = variant["fixed_stiffness"]
                self.motor_count += 2
                self.arm.append(SoftJoint(
                    joint_name=joint_name,
                    motor_max_pos=motor_max_pos,
                    left_positive_tendon_kwargs=variant["left"]["pos_tendon"],
                    left_negative_tendon_kwargs=variant["left"]["neg_tendon"],
                    right_positive_tendon_kwargs=variant["right"]["pos_tendon"],
                    right_negative_tendon_kwargs=variant["right"]["neg_tendon"],
                    file_name=tendon_model_file,
                    motor_init_pos=variant["motor_init_pos"],
                    fixed_stiffness=joint_fixed_stiffness,
                    tendon_noise=tendon_noise,
                    tendon_noise_level = tendon_noise_level,
                    motor_move_sim=motor_move_sim,
                    # fixed_stiffness_switch=not self.variable_stiffness
                ))
            elif variant["actuators"] == 1:
                self.motor_count += 1
                self.arm.append(MotorJoint(
                    joint_name=joint_name,
                    id=variant["id"],
                    motor_max_pos=variant["motor_max_pos"],
                    motor_init_pos=variant["motor_init_pos"]
                ))
        #TODO: Make a soft joint class and a nomal joint class
        # self.shoulder_yaw = SoftJoint()
        # self.arm.append(self.shoulder_yaw)
        # self.shoulder_roll = SoftJoint()
        # self.arm.append(self.shoulder_roll)
        # self.shoulder_pitch = SoftJoint()
        # self.arm.append(self.shoulder_pitch)
        # self.upperarm_roll = SoftJoint()
        # self.arm.append(self.upperarm_roll)
        # self.elbow = SoftJoint()
        # self.arm.append(self.elbow)
        # self.lowerarm_roll = MotorJoint()
        # self.arm.append(self.lowerarm_roll)
        # self.wrist_pitch = SoftJoint()
        # self.arm.append(self.wrist_pitch)
        self.write_data = False
        file_base_dir = '/home/laurenz/phd_project/sac/scripts/test_data/v6'
        self.motor_pos_file = os.path.join(file_base_dir, 'motor_pos.csv')
        self.delta_action_file = os.path.join(file_base_dir, 'delta_action.csv')
        self.joint_pos_file = os.path.join(file_base_dir, 'joint_pos.csv')
        self.eef_pos_file = os.path.join(file_base_dir, 'eef_pos.csv')
        # f = open(self.data_file, 'w')
        # data_writer = csv.writer(f)
        # data_writer.writerow(["rewards"])
        # f.close()

        self.last_time = 0.0

    def get_motor_pos(self):
        motor_pos = []
        for joint in self.arm:
            if type(joint) == SoftJoint:
                motor_pos.append(joint.motor_pos_right)
                motor_pos.append(joint.motor_pos_left)
                
        return motor_pos
    
    def get_motor_vel(self):
        motor_vel = []
        for joint in self.arm:
            if type(joint) == SoftJoint:
                motor_vel.append(joint.motor_vel_right)
                motor_vel.append(joint.motor_vel_left)
                
        return motor_vel

    def set_goal(self, delta_action):
        """
        Sets goal based on input @torques.

        Args:
            torques (Iterable): Desired joint torques

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # if self.last_time == 0.0:
        #     self.last_time = time.time()
        # else:
        #     new_time = time.time()
        #     time_step = new_time - self.last_time
        #     print("control frequency: ", time_step)
        #     self.last_time = copy.copy(new_time)
        
        # Update state
        self.update()


        # Check to make sure motor_pos is size self.joint_dim
        assert len(delta_action) == self.control_dim, "Delta torque must be equal to the robot's joint dimension space!"
       
        # print("min: ", self.input_min)
        # print("max: ", self.input_max)
        delta_action = np.clip(delta_action, self.input_min, self.input_max)
        delta_action = delta_action/10
        i_joint = 0
        i_motor = 0
        # print("delta_action: ",delta_action)

        # f = open(self.data_file, 'a')
        # data_writer = csv.writer(f)
        # data_writer.writerow(delta_action)
        # f.close()
        current_motor_pos = []
        # print("-------------------------------------------")
        # np.set_printoptions(suppress=True)
        # print("Joint Pos: ",np.around(self.joint_pos,4))
        while i_motor < self.control_dim:#self.control_dim:
            joint_motor_count = self.arm[i_joint].motor_count
            motor_inc = 1
            if joint_motor_count == 2:
                delta_eq_point = delta_action[i_motor]
                if self.stiffness_mode == "variable":
                    delta_stiffness = delta_action[i_motor+1]
                    motor_inc += 1
                else: 
                    delta_stiffness = 0

                # if self.variable_stiffness:
                delta_motor_pos = [delta_eq_point+delta_stiffness, delta_eq_point-delta_stiffness]
                # else:   
                    # delta_motor_pos = [delta_eq_point, delta_eq_point]
                # elif self.soft:
                #     delta_motor_pos = [delta_eq_point, delta_eq_point]
                # else:
                #     if self.arm[i_joint].current_stiffness < self.arm[i_joint].fixed_stiffness:
                #         delta_motor_pos = [delta_eq_point+self.arm[i_joint].fixed_stiffness, delta_eq_point-self.arm[i_joint].fixed_stiffness]
                #     else:
                #         delta_motor_pos = [delta_eq_point, delta_eq_point]
            else:
                delta_motor_pos = delta_action[i_motor]


            # if i_joint == 6:
            #     print(delta_motor_pos)
            #     print(delta_eq_point)
            #     print(delta_stiffness)
            #print(self.arm[i_joint].joint_name)
            # if self.arm[i_joint].joint_name == "elbow":
            #     print("delta_motor_pos", delta_motor_pos)
            self.arm[i_joint].update_goal_pos(delta_motor_pos)
            # print("Motor Position: ", delta_motor_pos)
            # self.arm[i_joint].update_goal_pos(delta_motor_pos, absolute_pos=True)

            # if motor_count == 2:
            #     print("Motor Positions: ", self.arm[i_joint].motor_pos_right,", ",self.arm[i_joint].motor_pos_left)

            if self.write_data:                                                                             
                if joint_motor_count == 2:
                    current_motor_pos.append(self.arm[i_joint].motor_pos_right)
                    current_motor_pos.append(self.arm[i_joint].motor_pos_left)
                else:
                    current_motor_pos.append(self.arm[i_joint].goal_motor_pos)
                    current_motor_pos.append(delta_motor_pos)

            i_motor += motor_inc
            i_joint += 1

            
        if self.write_data:
            timestamp = time.time()

            current_motor_pos.insert(0, timestamp)
            delta_action_data = np.insert(delta_action, 0, timestamp)

            f = open(self.motor_pos_file, 'a')
            data_writer = csv.writer(f)
            data_writer.writerow(current_motor_pos)
            f.close()

            # f = open(self.delta_action_file, 'a')
            # data_writer = csv.writer(f)
            # data_writer.writerow(delta_action_data)
            # f.close()

            

            # motor_count = self.arm[i_joint].motor_count
            # self.arm[i_joint].update_goal_pos(delta_motor_pos[i_motor:i_motor+motor_count])
            # i_motor += motor_count
            # i_joint += 1

        # # self.goal_torque = np.clip(self.scale_action(torques), self.torque_limits[0], self.torque_limits[1])

        # if self.interpolator is not None:
        #     self.interpolator.set_goal(self.goal_torque)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        # if self.goal_torque is None:
        #     self.set_goal(np.zeros(self.control_dim))

        # if self.last_time == 0.0:
        #     self.last_time = time.time()
        # else:
        #     new_time = time.time()
        #     time_step = new_time - self.last_time
        #     print("control frequency: ", time_step)
        #     self.last_time = copy.copy(new_time)

        # Update state
        self.update()

        # # Only linear interpolator is currently supported
        # if self.interpolator is not None:
        #     # Linear case
        #     if self.interpolator.order == 1:
        #         self.current_torque = self.interpolator.get_interpolated_goal()
        #     else:
        #         # Nonlinear case not currently supported
        #         pass
        # else:
        #     self.current_torque = np.array(self.goal_torque)

        self.torques = np.zeros(25)
        joint_torques = np.ndarray([7,4])
        # print(self.joint_pos)
        for i_joint in range(len(self.arm)):
            if self.arm[i_joint].motor_count == 2:
                current_joint_torques = self.arm[i_joint].get_torques(self.joint_pos[i_joint])
                
                self.torques[self.arm[i_joint].tendon_right_pos.id] = current_joint_torques[0]
                self.torques[self.arm[i_joint].tendon_right_neg.id] = current_joint_torques[1]
                self.torques[self.arm[i_joint].tendon_left_pos.id] = current_joint_torques[2]
                self.torques[self.arm[i_joint].tendon_left_neg.id] = current_joint_torques[3]

                # if self.arm[i_joint].joint_name == "shoulder_pitch":
                #     print("shoulder_pitch torques: ",current_joint_torques)
                #     print("tendon_ids: ", [self.arm[i_joint].tendon_right_pos.id, self.arm[i_joint].tendon_right_neg.id, self.arm[i_joint].tendon_left_pos.id, self.arm[i_joint].tendon_left_neg.id])
                
                # print("torques"+self.arm[i_joint].joint_name+": ",self.torques)
            else:
                self.torques[self.arm[i_joint].id] = self.arm[i_joint].get_torques(self.joint_pos[i_joint],self.joint_vel[i_joint])


        if self.write_data:
            timestamp = time.time()

            robot_state = np.concatenate((timestamp, self.joint_pos, self.ee_pos, self.torques), axis=None)
            # robot_state = [timestamp, self.joint_pos, self.ee_pos]
            # robot_state = np.insert(robot_state, 0, timestamp)

            f = open(self.joint_pos_file, 'a')
            data_writer = csv.writer(f)
            data_writer.writerow(robot_state)
            f.close()

        # t = self.arm[0].get_torques(self.joint_pos[6])
        # # print(self.joint_pos)
        # # print(t)
        # joint_torques[6] = t

        # self.torques[21] = joint_torques[6][1]
        # self.torques[22] = joint_torques[6][0]
        # self.torques[23] = joint_torques[6][2]
        # self.torques[24] = joint_torques[6][3]
        # # Add gravity compensation
        # self.torques = self.current_torque + self.torque_compensation
        # print(self.torques[21:25])

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        # self.torques = np.zeros(25)
        # self.torques[12] = -100.0
        # print("shoulder_pitch: ",self.torques[9:13])
        # print("torques: ",self.torques)
        return self.torques

    def reset_goal(self):
        """
        Resets joint torque goal to be all zeros (pre-compensation)
        """
        #TODO-THINK ABOUT WHAT SHOULD HAPPEN DURING RESET

        # for j in self.arm:
        #     j.reset()
        
        # self.goal_torque = np.zeros(self.control_dim)

        # # Reset interpolator if required
        # if self.interpolator is not None:
        #     self.interpolator.set_goal(self.goal_torque)
        

    # def calculateTendonTorque(self):
    #     tendon_length = self.calculateTendonLength()

    # def calculateTendonLength(self):
    #     """
    #     Calculates the length of the tendons according to the current motor and joint positions
    #     """

        



    # @property
    # def name(self):
    #     return 'JOINT_TORQUE'
