import numpy as np
import copy
import json

class Camera:
    def __init__(self, d, camera_index, T):
        self.d = d  
        self.index = camera_index  
        self.t = 0  
        self.b = np.zeros(self.d)
        self.V = np.zeros((self.d, self.d))
        self.payoffs = np.zeros(T)  
        self.best_payoffs = np.zeros(T)
        self.theta = np.zeros(d)

    def store_info(self, x, y, t, r, br):
        self.t += 1
        self.V = self.V + np.outer(x, x)
        self.b = self.b + r * x
        self.payoffs[t] += r
        self.best_payoffs[t] += br
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        return self.V, self.b, self.t



class Group(Camera):
    def __init__(self, b, V, cameras_begin, d, camera_num, rounds, payoffs, best_payoffs, cameras={}, t=0):
        self.d = d
        if not cameras:  
            self.cameras = dict()
            for i in range(cameras_begin, cameras_begin + camera_num):
                self.cameras[i] = Camera(self.d, i, rounds)  
        else:
            self.cameras = copy.deepcopy(cameras)
        self.cameras_begin = cameras_begin
        self.camera_num = camera_num
        self.b = b
        self.t = t  
        self.V = V
        self.payoffs = payoffs
        self.best_payoffs = best_payoffs
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)  

    def get_camera(self, camera_index):
        return self.cameras[camera_index]

    
    def store_info(self, x, y, t, r, br):
        
        self.V = self.V + np.outer(x, x)
        
        self.b = self.b + r * x
        self.t += 1
        self.best_payoffs[t] += br
        self.payoffs[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        V_t = self.V
        b_t = self.b
        t = self.t
        return V_t, b_t, t




class Set_based_Group(Group):
    def __init__(self, b, V, cameras_begin, d, camera_num, rounds, payoffs, best_payoffs, theta_phase, cameras={}, t=0, T_phase=0):
        super(Set_based_Group, self).__init__(b, V, cameras_begin, d, camera_num, rounds, payoffs, best_payoffs, cameras, t)
        self.T_phase = T_phase
        self.theta_phase = theta_phase
        self.checks = {i: False for i in self.cameras}
        self.checked = len(self.cameras) == sum(self.checks.values())

    def phase_update(self):
        self.T_phase = self.t
        self.theta_phase = self.theta

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.cameras) == sum(self.checks.values())


def load_config(config_path='Experiment_config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)
    
    