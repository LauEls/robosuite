import numpy as np
import math
import os
import json
from robosuite.utils.tendon import Tendon
from robosuite.utils.joint import Joint

class MotorJoint(Joint):
    def __init__(
        self,
        joint_name,
        motor_max_pos,
        input_max=1,
        input_min=-1,
        output_max=0.05,
        output_min=-0.05,
        kp=50,
        damping_ratio = 1,
    ):
        self.goal_motor_pos = 0

        self.kp = kp
        self.kd = 2 * np.sqrt(kp) * damping_ratio

        self.motor_max_pos = motor_max_pos
        self.motor_min_pos = -motor_max_pos
        self.motor_count = 1

    def update_goal_pos(self, delta_goal_pos):
        self.goal_motor_pos += delta_goal_pos
        if self.goal_motor_pos > self.motor_max_pos:
            self.goal_motor_pos = self.motor_max_pos
        elif self.goal_motor_pos < self.motor_min_pos:
            self.goal_motor_pos = self.motor_min_pos

    def get_torques(self, current_joint_pos, current_joint_vel, gravity_compensation):
        torque = 0

        position_error = self.goal_motor_pos - current_joint_pos
        vel_pos_error = -current_joint_vel
        desired_torque = position_error * self.kp + vel_pos_error*self.kd

        torque = desired_torque + gravity_compensation

        return torque


    
