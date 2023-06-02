import numpy as np
import math
import os
import json
from robosuite.utils.tendon import Tendon
from robosuite.utils.joint import Joint

class SoftJoint(Joint):
    def __init__(
        self,
        left_positive_tendon_kwargs,
        left_negative_tendon_kwargs,
        right_positive_tendon_kwargs,
        right_negative_tendon_kwargs,
        motor_max_pos,
        joint_name,
        file_name,
        motor_init_pos = 0,
        fixed_stiffness = 0
        # config_file,
    ):
        # config_dir = "../controllers/config/gh2"
        # file_path = os.path.join(config_dir, config_file)
        # try:
        #     with open(file_path) as f:
        #         config = json.load(f)
        # except FileNotFoundError:
        #     print("Error opening config filepath at: {}. "
        #         "Please check filepath and try again.".format(file_path))
        # #name for identification
        # self.name = config['name']
        # #minimum and maximum angle of the passive joint
        # self.minAngle = config['minAngle']
        # self.maxAngle = config['maxAngle']

        # self.left_side = Tendon(config['left'])
        # self.right_side = Tendon(config['right'])
 
        self.joint_name = joint_name
        self.motor_count = 2
        self.motor_max = motor_max_pos
        self.motor_min = -motor_max_pos
        self.right_max_pos = False
        self.right_min_pos = False
        self.left_max_pos = False
        self.left_min_pos = False
        self.right_max_tendon = False
        self.right_min_tendon = False
        self.left_max_tendon = False
        self.left_min_tendon = False
        self.fixed_stiffness = fixed_stiffness

        self.motor_pos_left = 0
        self.motor_pos_right = 0
        self.tendon_left_pos = Tendon(**left_positive_tendon_kwargs, file_name=file_name)
        self.tendon_left_neg = Tendon(**left_negative_tendon_kwargs, file_name=file_name)
        self.tendon_right_pos = Tendon(**right_positive_tendon_kwargs, file_name=file_name)
        self.tendon_right_neg = Tendon(**right_negative_tendon_kwargs, file_name=file_name)

        self.current_joint_pos = 0
        self.current_stiffness = 0

        self.update_goal_pos([motor_init_pos, motor_init_pos])
        # self.current_left_positive_length = self.left_positive_tendon.zero_active_length + self.left_positive_tendon.free_length + self.left_positive_tendon.zero_passive_length
        # self.current_left_negative_length = self.left_negative_tendon.zero_active_length + self.left_negative_tendon.free_length + self.left_negative_tendon.zero_passive_length
        # self.current_right_positive_length = self.right_positive_tendon.zero_active_length + self.right_positive_tendon.free_length + self.right_positive_tendon.zero_passive_length
        # self.current_right_negative_length = self.right_negative_tendon.zero_active_length + self.right_negative_tendon.free_length + self.right_negative_tendon.zero_passive_length


    # def get_tendon_lengths(self, delta_motor_pos_left, delta_motor_pos_right, joint_pos):
    #     delta_joint_pos = joint_pos - self.current_joint_pos

    #     self.current_left_positive_length = self.left_positive_tendon.get_tendon_length(delta_motor_pos_left, delta_joint_pos)
    #     self.current_left_negative_length = self.left_negative_tendon.get_tendon_length(-delta_motor_pos_left, delta_joint_pos)
    #     self.current_right_positive_length = self.right_positive_tendon.get_tendon_length(delta_motor_pos_right, delta_joint_pos)
    #     self.current_right_negative_length = self.right_negative_tendon.get_tendon_length(-delta_motor_pos_right, delta_joint_pos)

    #     self.current_joint_pos = joint_pos
    #     self.motor_pos_left += delta_motor_pos_left
    #     self.motor_pos_right += delta_motor_pos_right

    #     return [self.current_left_positive_length, self.current_left_negative_length, self.current_right_positive_length, self.current_right_negative_length]

    def update_goal_pos(self, delta_motor_pos):
        # if delta_motor_pos[0] > delta_motor_pos[1]:
        #     self.current_stiffness
        
        #TODO: Check if max or min angle of the motor would be exceeded
        if (self.right_max_pos or self.right_max_tendon) and delta_motor_pos[0] > 0.0:
            # print("Can't exceed right max pos!")
            delta_right = 0.0
        elif (self.right_min_pos or self.right_min_tendon) and delta_motor_pos[0] < 0.0:
            # print("Can't exceed right min pos!")
            delta_right = 0.0
        else:
            self.motor_pos_right += delta_motor_pos[0]

            if self.motor_pos_right > self.motor_max:
                delta_right = delta_motor_pos[0] - (self.motor_pos_right - self.motor_max)
                self.motor_pos_right = self.motor_max
            elif self.motor_pos_right < self.motor_min:
                delta_right = delta_motor_pos[0] + (np.abs(self.motor_pos_right)-np.abs(self.motor_min))
                self.motor_pos_right = self.motor_min
            else:
                delta_right = delta_motor_pos[0]

        if (self.left_max_pos or self.left_max_tendon) and delta_motor_pos[1] > 0.0:
            # print("Can't exceed left max pos!")
            delta_left = 0.0
        elif (self.left_min_pos or self.left_min_tendon) and delta_motor_pos[1] < 0.0:
            # print("Can't exceed left min pos!")
            delta_left = 0.0
        else:
            self.motor_pos_left += delta_motor_pos[1]

            if self.motor_pos_left > self.motor_max:
                delta_left = delta_motor_pos[1] - (self.motor_pos_left - self.motor_max)
                self.motor_pos_left = self.motor_max
            elif self.motor_pos_left < self.motor_min:
                delta_left = delta_motor_pos[1] + (np.abs(self.motor_pos_left)-np.abs(self.motor_min))
                self.motor_pos_left = self.motor_min
            else:
                delta_left = delta_motor_pos[1]

        

        if delta_right < delta_left:
            neg_delta_stiffness = (delta_left - delta_right)/2
            if self.current_stiffness - neg_delta_stiffness < 0.0:
                new_neg_delta_stiffness = abs(self.current_stiffness - neg_delta_stiffness)
                self.current_stiffness = 0.0
                delta_right += new_neg_delta_stiffness
                delta_left -= new_neg_delta_stiffness
                self.motor_pos_right += new_neg_delta_stiffness
                self.motor_pos_left -= new_neg_delta_stiffness
            else:
                self.current_stiffness -= neg_delta_stiffness
        else:
            self.current_stiffness += (delta_right - delta_left)/2
        
        # print("Current_Stiffness: ", self.current_stiffness)

        #TODO: CHECK RETURN VALUE AND DO SOMETHING ABOUT IT
        self.right_min_pos = not self.tendon_right_pos.update_active_pulley(delta_right)
        self.right_max_pos = not self.tendon_right_neg.update_active_pulley(-delta_right)
        self.left_min_pos = not self.tendon_left_pos.update_active_pulley(delta_left)
        self.left_max_pos = not self.tendon_left_neg.update_active_pulley(-delta_left)

        # if self.joint_name == "wrist_pitch":
        #     print("Joint Name: ", self.joint_name)
        #     print("Right Max: ", self.right_max_pos)
        #     print("Right Min: ", self.right_min_pos)
        #     print("Left Max: ", self.left_max_pos)
        #     print("Left Min: ", self.left_min_pos)
        #     print(delta_right)
        #     print(delta_left)
        # print("Right Motor Pos: ",self.motor_pos_right)
        # print("Left Motor Pos: ", self.motor_pos_left)

    #     self.left_positive_tendon.update_active_angle(delta_goal_pos[0])
    #     self.left_negative_tendon.update_active_angle(-delta_goal_pos[0])
    #     self.right_positive_tendon.update_active_angle(delta_goal_pos[1])
    #     self.right_negative_tendon.update_active_angle(-delta_goal_pos[1])

    #     self.motor_pos_left += delta_goal_pos[0]
    #     self.motor_pos_right += delta_goal_pos[1]

    
    def get_torques(self, joint_pos): #delta_motor_pos,
        self.current_joint_pos = joint_pos

        self.f_left_pos = self.tendon_left_pos.update_tendon(-joint_pos)
        self.f_left_neg = self.tendon_left_neg.update_tendon(joint_pos)
        self.f_right_pos = self.tendon_right_pos.update_tendon(-joint_pos)
        self.f_right_neg = self.tendon_right_neg.update_tendon(joint_pos)

        self.right_max_tendon = self.tendon_right_pos.check_max()
        self.right_min_tendon = self.tendon_right_neg.check_max()
        self.left_max_tendon = self.tendon_left_pos.check_max()
        self.left_min_tendon = self.tendon_left_neg.check_max()

        self.current_joint_pos = joint_pos

        return [self.f_right_pos, self.f_right_neg, self.f_left_pos, self.f_left_neg]


    
