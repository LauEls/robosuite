import numpy as np
import math
import os
import json

class Tendon:
    def __init__(
        self,
        l_free,
        r_active,
        r_passive,
        alpha_active_zero,
        alpha_passive_zero,
        r_tendon,
        l_relaxed,
        file_name,
        id = 0, #TODO-ADD ACTUATOR ID'S TO EACH TENDON
    ):

        self.l_free = l_free
        self.r_active = r_active
        self.r_passive = r_passive
        self.r_tendon = r_tendon
        self.alpha_active = alpha_active_zero
        self.alpha_passive_zero = alpha_passive_zero
        self.alpha_passive = alpha_passive_zero
        self.l_relaxed = l_relaxed
        self.id = id
        self.tendon_max = False

        # TODO: Add retrieving a force-strain curve from file and save it in 2D array
        filepath = os.path.join(os.path.dirname(__file__), "config/tendon_models/"+file_name)
        self.tendon_model = np.loadtxt(filepath, skiprows=1, delimiter=',')

        # l_total = self.calc_tendon_length()
        # self.calc_tendon_force(l_total)

    def calc_tendon_length(self):
        # Calculate Length on Active Pulley
        l_active = 0
        new_r_active = self.r_active+self.r_tendon
        if self.alpha_active < 2*math.pi:
            l_active = self.alpha_active*(new_r_active)
        else:
            rest_angle = self.alpha_active
            rot = 1
            while rest_angle > 2*math.pi:
                l_active += 2*math.pi*(new_r_active+self.r_tendon*2*rot)
                rest_angle -= 2*math.pi
                rot += 1
            l_active += rest_angle*(new_r_active+self.r_tendon*2*rot)

        # Calculate Length on Passive Pulley
        l_passive = self.alpha_passive*(self.r_passive+self.r_tendon)

        # Total Tendon Length
        l_total = l_active + self.l_free + l_passive

        return l_total

    def calc_tendon_force(self, l_total):
        # if self.id > 20 and self.id < 25:
        #     print(self.id,": ",self.alpha_active)
        strain = (l_total-self.l_relaxed)/self.l_relaxed*100
        #print("Strain: ", strain)

        difference_array = np.absolute(self.tendon_model[:,1]-strain)
        index = difference_array.argmin()
        if index == len(self.tendon_model[:,1])-1:
            self.tendon_max = True
        else:
            self.tendon_max = False

        f_tendon = self.tendon_model[index,0]
        # if self.id > 20 and self.id < 25:
        #     print("Force: ",f_tendon)
        return f_tendon

    def update_active_pulley(self, delta_active):
            self.alpha_active += delta_active
            
            if self.alpha_active < 1.5708:
                return False
            else:
                return True
        
        #print("Alpha Active: ", self.alpha_active)

    def update_tendon(self, alpha_passive): #delta_active, 
    
        self.alpha_passive = self.alpha_passive_zero + alpha_passive

        l_total = self.calc_tendon_length()
        f_tendon = self.calc_tendon_force(l_total)

        return f_tendon

    def check_max(self):
        if self.tendon_max:
            return True
        else:
            return False
        
        


    # def calc_tendon_lenght_on_pulley(self, center_pos, pulley_pos_1, pulley_pos_2, pulley_radius):
    #     angle_on_pulley = 0
    #     length_on_pulley = 0

    #     dist_c_p1 = math.dist(center_pos, pulley_pos_1)
    #     dist_c_p2 = math.dist(center_pos, pulley_pos_2)
    #     dist_p1_p2 = math.dist(pulley_pos_1, pulley_pos_2)
    #     angle_on_pulley = math.acos((dist_c_p1^2 + dist_c_p2^2 - dist_p1_p2^2)/(2*dist_c_p1*dist_c_p2))
    #     length_on_pulley = angle_on_pulley * pulley_radius

    #     return [length_on_pulley, angle_on_pulley]

    # def update_tendong_length(self, angle, pulley_radius):
    #     return angle * pulley_radius

    # def get_tendon_length(self, delta_passive_angle):
    #     # if delta_active_angle != 0:
    #     #     self.current_active_angle += delta_active_angle
    #     #     self.current_active_length = self.update_tendong_length(self.current_active_angle, self.active_radius)

    #     if delta_passive_angle != 0:
    #         self.current_passive_angle += delta_passive_angle
    #         self.current_passive_length = self.update_tendong_length(self.current_passive_length, self.passive_radius)

    #     return self.current_active_length + self.free_length + self.current_passive_length

    # def update_active_angle(self, delta_active_angle):
    #     self.current_active_angle += delta_active_angle
    #     self.current_active_length = self.update_tendong_length(self.current_active_angle, self.active_radius)

    # def get_tendon_torque(self, delta_passive_angle):
    #     length = self.get_tendon_length(delta_passive_angle)

    #     #TODO: function that translates length into force and return force to joint class

    

        
