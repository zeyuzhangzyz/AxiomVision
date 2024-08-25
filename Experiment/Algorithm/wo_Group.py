import numpy as np
import Base
import Environment as Envi
import time
from paras import *

class Edge_server:
    def __init__(self, nl, d, begin_num, T):
        self.nl = nl
        self.d = d
        self.rounds = T
        self.begin_num = begin_num
        self.cameras = dict()
        for i in range(self.nl):
            self.cameras[i] = Base.Camera(self.d, i, self.rounds)

    def select(self, camera_index, visual_models, is_final):
        # For the general bandit problem (sigmoid and logistics), we can use the Newton gradient method instead of directly solving the closed-form solution. In the specific implementation, we used a linear method and accelerated video processing.
        visual_models = visual_models[visual_models_begin[is_final]:visual_models_end[is_final]]
        V_t, b_t, T_t = self.cameras[camera_index].get_info()
        gamma_t = Envi.gamma(T_t, self.d, beta)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        alpha_t = Envi.alpha(beta, self.d, T_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        r_visual_model_index = np.argmax(
            np.dot(visual_models, theta) +  alpha_t * np.sqrt((np.matmul(visual_models, Minv) * visual_models).sum(axis=1)))
        return r_visual_model_index + visual_models_begin[is_final]


class Cloud_server:
    def __init__(self, L, n, cameraList, d, T, config_index, txt_name = ""):
        self.cameranum = n
        self.edge_server_list = []
        self.edge_server_num = L
        self.rounds = T
        self.d = d
        self.regret = np.zeros(self.rounds)
        self.payoff = np.zeros(self.rounds)
        self.best_payoff = np.zeros(self.rounds)
        self.model_selection = np.zeros(self.rounds)
        self.edge_server_inds = np.zeros(n, np.int64)
        self.txt_name = txt_name
        self.config_index = config_index
        load_config(config_path=f'Algorithm/experiment_config{config_index}.json')
        camera_index = 0
        j = 0
        for i in cameraList:
            self.edge_server_list.append(Edge_server(i, d, camera_index, self.rounds))
            self.edge_server_inds[camera_index:camera_index + i] = j
            camera_index = camera_index + i
            j = j + 1

    def locate_camera_index(self, camera_index):
        edge_server_index = self.edge_server_inds[camera_index]
        return edge_server_index

    def select(self, camera_index, visual_models):
        is_final = 0
        visual_models = visual_models[visual_models_begin[is_final]:visual_models_end[is_final]]
        camera = self.cameras[camera_index]
        V_t, b_t, T_t = camera.get_info()
        gamma_t = Envi.gamma(T_t, self.d, beta)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        alpha_t = Envi.alpha(beta, self.d, T_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        r_visual_model_index = np.argmax(np.dot(visual_models, theta) +  alpha_t * np.sqrt((np.matmul(visual_models, Minv) * visual_models).sum(axis=1)))
        return r_visual_model_index

    def run(self, envir, T):
        y_list = list()
        x_list = list()
        final_list = list()
        start_time = time.time()
        # with open(f"time{self.txt_name}_wo_Group_{T}.txt", "w") as file:
        for i in range(1, T + 1):
            if i % 5000 == 0:
                print(i)
            # if i % 1000 == 0 :
            #     run_time = time.time() - start_time
            #     data_to_write = f"{i},{run_time}\n"
            #     file.write(data_to_write)
            last_element = 0 if not final_list else final_list[-1]
            if last_element == 0:
                camera_index = envir.random_sample_camera()
                edge_server_index = self.locate_camera_index(camera_index)
                edge_server = self.edge_server_list[edge_server_index]
            visual_models = envir.get_visual_models(last_element)
            edge_server_camera_index = camera_index- edge_server.begin_num
            r_visual_model_index = edge_server.select(edge_server_camera_index, visual_models, last_element)
            self.payoff[i - 1], x, y, self.best_payoff[i - 1], is_final = envir.feedback_Local(visual_models=visual_models,
             i=camera_index, k=r_visual_model_index, is_final = last_element)
            if is_final == 0:
                final_list.append(is_final)
            else:
                if not final_list:
                    final_list.append(1)
                else:
                    if final_list[-1] == 2:
                        final_list.append(0)

                    else:
                        final_list.append(final_list[-1] + 1)
            x_list.append(x)
            y_list.append(y)
            edge_server.cameras[edge_server_camera_index].store_info(x, y, i - 1, self.payoff[i - 1], self.best_payoff[i - 1], )
            self.regret[i - 1] = self.best_payoff[i - 1] - self.payoff[i - 1]

        return self.regret, self.payoff, x_list, y_list, final_list, self.model_selection
