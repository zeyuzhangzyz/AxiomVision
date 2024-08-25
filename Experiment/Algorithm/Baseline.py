import numpy as np
import Base
import Environment as Envi
from paras import *


class Cloud_server:
    def __init__(self, L, n, cameraList, d, T, config_index):
        self.edge_server_list = []
        self.cameranum = n
        self.rounds = T
        self.edge_server_num = L
        self.d = d
        self.regret = np.zeros(self.rounds)
        self.payoff = np.zeros(self.rounds)
        self.best_payoff = np.zeros(self.rounds)
        self.model_selection = np.zeros(self.rounds)
        self.edge_server_inds = np.zeros(n, np.int64)
        self.config_index = config_index
        load_config(config_path=f'Algorithm/experiment_config{config_index}.json')


    def run(self, envir, T):
        y_list = list()
        x_list = list()
        final_list = list()
        for i in range(1, T + 1):
            camera_index = envir.random_sample_camera()
            last_element = 0
            visual_models = envir.get_visual_models(last_element)
            r_visual_model_index = 5
            self.payoff[i - 1], x, y, self.best_payoff[i - 1], is_final = envir.feedback_Local(visual_models=visual_models, i=camera_index,
                                                                                            k=r_visual_model_index, is_final = last_element)
            x_list.append(x)
            y_list.append(y)
            final_list.append(0)
            self.regret[i - 1] = self.best_payoff[i - 1] - self.payoff[i - 1]
            self.model_selection[i-1] = r_visual_model_index
        return self.regret, self.payoff, x_list, y_list, final_list,self.model_selection
