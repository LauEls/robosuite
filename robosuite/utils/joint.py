import numpy as np
import math
import os
import json
from abc import ABC, abstractmethod

from robosuite.utils.tendon import Tendon

class Joint:
    @abstractmethod
    def get_torques(self):
        # return a list of torques
        pass
    
    @abstractmethod
    def update_goal_pos(self):
        pass
