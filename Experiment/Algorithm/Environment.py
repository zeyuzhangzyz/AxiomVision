import cmath
import json
import numpy as np
import sys
from paras import *

def get_payoff(index,performance_index,n, version = ''):
    global real_payoff
    global svd_payoff
    global svd_result
    real_payoff = np.load(f'raw_data/real_payoff_{index}_{performance_index}_{n}{version}.npy')
    svd_payoff = np.load(f'raw_data/svd_payoff_{index}_{performance_index}_{n}{version}.npy')
    svd_result = np.load(f'raw_data/svd_results_{index}_{performance_index}_{n}{version}.npz')
def change_real(value):
    global change_real
    change_real = value

def change_threshold(value):
    global acc_threshold
    acc_threshold = value

def gamma(t, d, beta):
    # regularization parameter for matrix
    # tmp = 4 * cmath.sqrt(d) + 2 * np.log(2 * t / beta)
    return 1

def alpha(beta, d, t, g = 4):
    # the g is group number got from the svd decomposition
    # a = np.sqrt(beta**2/4/d + d * np.log(t / d) + 2 * np.log(4*t*g))
    return 0.25


class Environment:
    def __init__(self, d, num_cameras, theta, L, config_index):
        self.L = L
        self.d = d
        self.camera_num = num_cameras
        self.theta = theta
        self.config_index = config_index
        self.best_fix = 0
        load_config(config_path=f'Algorithm/experiment_config{config_index}.json')

    def get_visual_models(self,is_final):
        global svd_result
        data = svd_result
        self.visual_models = data['Tmp_lowdim'].T
        return self.visual_models[:visual_models_end[is_final]]

    def change_global_threshold(self, value):
        global acc_threshold
        acc_threshold = value

    def best_fix(self):
        return best_fix

    def feedback_Local(self, visual_models, i, k, is_final = 0, real = 0, threshold = 0.7):
        # For the general bandit problem (sigmoid and logistics), we can use the Newton gradient method instead of directly solving the closed-form solution. In the specific implementation, we used a linear method and accelerated video processing.
        global change_real, acc_threshold
        real = change_real
        threshold = acc_threshold
        x = visual_models[k, :]

        if real == 0:
            payoff = svd_payoff[i, k]
            best_payoff = np.max(svd_payoff[i,visual_models_begin[is_final]:visual_models_end[is_final]])
        else:
            payoff = real_payoff[i,k]
            best_payoff = np.max(real_payoff[i, visual_models_begin[is_final]:visual_models_end[is_final]])
        if payoff < 0 :
            y = 0
        elif payoff > 1 :
            y = 1
        else:
            y = np.random.binomial(1, payoff)
        if payoff < threshold:
            is_final = 1
        else:
            is_final = 0
        return payoff, x, y, best_payoff, is_final


    def random_sample_camera(self):
        return np.random.randint(0, self.camera_num)
