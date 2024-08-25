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
        self.groups = {
            0: Base.Set_based_Group(b=np.zeros(d), t=0, V=np.zeros((d, d)), cameras_begin=begin_num, d=d, camera_num=nl,
                                  rounds=self.rounds, payoffs=np.zeros(self.rounds), T_phase=0,
                                  theta_phase=np.zeros(self.d),
                                  best_payoffs=np.zeros(self.rounds))}
        self.T_phase = 0
        self.theta = np.zeros(self.d)
        self.init_each_stage()
        self.group_inds = dict()
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.group_inds[i] = 0
        self.num_groups = np.zeros(self.rounds, np.int64)
        self.num_groups[0] = 1


    def init_each_stage(self):
        for i in self.groups:
            group = self.groups[i]
            group.checks = {j: False for j in group.cameras}
            group.checked = False
            group.phase_update()


    def group_aver_freq(self, c, t):
        if len(self.groups[c].cameras) == 0:
            return 0
        return self.groups[c].t / (len(self.groups[c].cameras) * t)


    def locate_camera_index(self, camera_index):
        l_group_index = self.group_inds[camera_index]
        return l_group_index


    def select(self, l_group_index, visual_models, is_final):
        # For the general bandit problem (sigmoid and logistics), we can use the Newton gradient method instead of directly solving the closed-form solution. In the specific implementation, we used a linear method and accelerated video processing.
        visual_models = visual_models[visual_models_begin[is_final]:visual_models_end[is_final]]
        group = self.groups[l_group_index]
        V_t, b_t, T_t = group.get_info()
        gamma_t = Envi.gamma(T_t, self.d, beta)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        alpha_t = Envi.alpha(beta, self.d, T_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        r_visual_model_index = np.argmax(np.dot(visual_models, theta) +  alpha_t * (np.matmul(visual_models, Minv) * visual_models).sum(axis=1))
        return r_visual_model_index + visual_models_begin[is_final]

    def if_split(self, camera_index1, group, t):
        t1 = group.cameras[camera_index1].t
        t2 = group.T_phase
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        theta1 = group.cameras[camera_index1].theta
        theta2 = group.theta_phase
        if np.linalg.norm(theta1 - theta2) > beta * (fact_T1 + fact_T2):
            return True

        p1 = t1 / t
        for camera_index2 in group.cameras:
            if camera_index2 == camera_index1:
                continue
            p2 = group.cameras[camera_index2].t / t
            if np.abs(p1 - p2) > beta_p * 2 * fact_t:
                return True

        return False

    def if_merge(self, c1, c2, t):
        t1 = self.groups[c1].t
        t2 = self.groups[c2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        theta1 = self.groups[c1].theta
        theta2 = self.groups[c2].theta
        p1 = self.group_aver_freq(c1, t)
        p2 = self.group_aver_freq(c2, t)

        if np.linalg.norm(theta1 - theta2) >= (beta / 2) * (fact_T1 + fact_T2):
            return False
        if np.abs(p1 - p2) >= beta_p * fact_t:
            return False

        return True

    def find_available_index(self):
        cmax = max(self.groups)
        for c1 in range(cmax + 1):
            if c1 not in self.groups:
                return c1
        return cmax + 1


    def update(self, camera_index, t):
        c = self.group_inds[camera_index]
        group = self.groups[c]
        group.update_check(camera_index)
        now_camera = group.cameras[camera_index]
        if self.if_split(camera_index, group, t):

            cnew = self.find_available_index()
            tmp_group = Base.Set_based_Group(b=now_camera.b, t=now_camera.t, V=now_camera.V, cameras_begin=camera_index, d=self.d,
                                             camera_num=1,
                                             rounds=self.rounds, cameras={camera_index: now_camera}, payoffs=now_camera.payoffs,
                                             best_payoffs=now_camera.best_payoffs, T_phase=group.T_phase,
                                             theta_phase=group.theta_phase)
            self.groups[cnew] = tmp_group
            self.group_inds[camera_index] = cnew

            del group.cameras[camera_index]
            group.V = group.V - now_camera.V
            group.b = group.b - now_camera.b
            group.t = group.t - now_camera.t
            del group.checks[camera_index]

        self.num_groups[t - 1] = len(self.groups)


    def merge(self, t):
        cmax = max(self.groups)
        for c1 in range(cmax - 1):
            if c1 not in self.groups or self.groups[c1].checked == False:
                continue
            for c2 in range(c1 + 1, cmax):
                if c2 not in self.groups or self.groups[c2].checked == False:
                    continue
                if not self.if_merge(c1, c2, t):
                    continue
                else:

                    for i in self.groups[c2].cameras:
                        self.group_inds[i] = c1
                    self.groups[c1].V = self.groups[c1].V + self.groups[c2].V
                    self.groups[c1].b = self.groups[c1].b + self.groups[c2].b
                    self.groups[c1].t = self.groups[c1].t + self.groups[c2].t
                    self.groups[c1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.groups[c1].V),
                                                        self.groups[c1].b)
                    for camera in self.groups[c2].cameras:
                        self.groups[c1].cameras.setdefault(camera, self.groups[c2].cameras[camera])
                    self.groups[c1].checks = {**self.groups[c1].checks, **self.groups[c2].checks}
                    del self.groups[c2]

        self.num_groups[t - 1] = len(self.groups)


class Cloud_server:
    def __init__(self, L, n, cameraList, d, T, txt_name, config_index):
        self.edge_server_list = []
        self.cameranum = n
        self.rounds = T
        self.edge_server_num = L
        self.d = d
        self.regret = np.zeros(self.rounds)
        self.payoff = np.zeros(self.rounds)
        self.best_payoff = np.zeros(self.rounds)
        self.model_selection = np.zeros(self.rounds)
        self.txt_name = txt_name
        self.edge_server_inds = np.zeros(n, np.int64)
        camera_index = 0
        self.config_index = config_index
        load_config(config_path=f'Algorithm/experiment_config{config_index}.json')
        j = 0
        for i in cameraList:
            self.edge_server_list.append(Edge_server(i, d, camera_index, self.rounds))
            self.edge_server_inds[camera_index:camera_index + i] = j
            camera_index = camera_index + i
            j = j + 1


    def locate_camera_index(self, camera_index):
        edge_server_index = self.edge_server_inds[camera_index]
        return edge_server_index


    def run(self, envir, phase):
        y_list = list()
        x_list = list()
        final_list = list()
        start_time = time.time()
        # with open(f"time{self.txt_name}_Set_based_{phase}.txt", "w") as file:
        for s in range(1, phase + 1):
            for edge_server in self.edge_server_list:
                edge_server.init_each_stage()
            for i in range(1, phase_cardinality ** s + 1):
                t = (phase_cardinality ** s - 1) // (phase_cardinality - 1) + i - 1
                # if t % 1000 == 0:
                #     run_time = time.time() - start_time
                #     data_to_write = f"{t},{run_time}\n"
                #     file.write(data_to_write)
                camera_index = envir.random_sample_camera()
                edge_server_index = self.locate_camera_index(camera_index)
                edge_server = self.edge_server_list[edge_server_index]
                l_group_index = edge_server.locate_camera_index(camera_index)
                l_group = edge_server.groups[l_group_index]
                last_element = 0
                visual_models = envir.get_visual_models(last_element)
                r_visual_model_index = edge_server.select(l_group_index=l_group_index, visual_models=visual_models,is_final = last_element)
                self.model_selection[t - 1] = r_visual_model_index
                self.payoff[t - 1], x, y, self.best_payoff[t - 1], is_final = envir.feedback_Local(visual_models=visual_models,
                  i=camera_index, k=r_visual_model_index, is_final=last_element)
                x_list.append(x)
                y_list.append(y)
                final_list.append(last_element)
                l_group.cameras[camera_index].store_info(x, y, t - 1, self.payoff[t - 1], self.best_payoff[t - 1])
                l_group.store_info(x, y, t - 1, self.payoff[t - 1], self.best_payoff[t - 1])
                if is_final == 0:
                    edge_server.update(camera_index, t)
                    edge_server.merge(t)
                self.regret[t - 1] = self.best_payoff[t - 1] - self.payoff[t - 1]


        return self.regret, self.payoff, x_list, y_list, final_list, self.model_selection
