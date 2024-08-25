import networkx as nx
import numpy as np
import Base
import Environment as Envi
import copy
import time
from paras import *


class Edge_server:
    def __init__(self, nl, d, begin_num, T):
        self.nl = nl
        self.d = d
        self.rounds = T
        camera_index_list = list(range(begin_num, begin_num + nl))
        self.G = nx.generators.classic.complete_graph(
            camera_index_list)
        self.groups = {
            0: Base.Group(b=np.zeros(d), t=0, V=np.zeros((d, d)), cameras_begin=begin_num, d=d, camera_num=nl,
                            rounds=self.rounds, payoffs=np.zeros(self.rounds),
                            best_payoffs=np.zeros(
                                self.rounds))}
        self.group_inds = dict()
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.group_inds[i] = 0
        self.num_groups = np.zeros(self.rounds, np.int64)
        self.num_groups[0] = 1

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
        r_visual_model_index = np.argmax(
            np.dot(visual_models, theta) + alpha_t * np.sqrt((np.matmul(visual_models, Minv) * visual_models).sum(axis=1)))
        return r_visual_model_index + visual_models_begin[is_final]

    def if_delete(self, camera_index1, camera_index2, group):
        t1 = group.cameras[camera_index1].t
        t2 = group.cameras[camera_index2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        theta1 = group.cameras[camera_index1].theta
        theta2 = group.cameras[camera_index2].theta

        return np.linalg.norm(theta1 - theta2) > beta * (fact_T1 + fact_T2)

    def update(self, camera_index, t):
        update_group = False
        c = self.group_inds[camera_index]
        i = camera_index
        origin_group = self.groups[c]
        A = [a for a in self.G.neighbors(i)]
        for j in A:
            camera2_index = j
            c2 = self.group_inds[camera2_index]
            camera1 = self.groups[c].cameras[i]
            camera2 = self.groups[c2].cameras[camera2_index]
            if camera1.t != 0 and camera2.t != 0 and self.if_delete(i, camera2_index, self.groups[c]):
                self.G.remove_edge(i, j)
                update_group = True

        if update_group:
            C = nx.node_connected_component(self.G, i)
            remain_cameras = dict()
            for m in C:
                remain_cameras[m] = self.groups[c].get_camera(m)

            if len(C) < len(self.groups[c].cameras):
                all_cameras_index = set(self.groups[c].cameras)
                all_cameras = dict()
                for camera_index_all in all_cameras_index:
                    all_cameras[camera_index_all] = self.groups[c].get_camera(camera_index_all)

                tmp_group = Base.Group(b=sum([remain_cameras[k].b for k in remain_cameras]),
                                           t=sum([remain_cameras[k].t for k in remain_cameras]),
                                           V=sum([remain_cameras[k].V for k in remain_cameras]),
                                           cameras_begin=min(remain_cameras), d=self.d, camera_num=len(remain_cameras),
                                           rounds=self.rounds,
                                           cameras=copy.deepcopy(remain_cameras),
                                           payoffs=sum([remain_cameras[k].payoffs for k in remain_cameras]),
                                           best_payoffs=sum([remain_cameras[k].best_payoffs for k in remain_cameras]))
                self.groups[c] = tmp_group

                for camera_index_used in all_cameras_index:
                    if remain_cameras.__contains__(camera_index_used):
                        all_cameras.pop(camera_index_used)

                c = max(self.groups) + 1
                while len(all_cameras) > 0:
                    j = np.random.choice(list(all_cameras))
                    C = nx.node_connected_component(self.G, j)
                    new_group_cameras = dict()
                    for k in C:
                        new_group_cameras[k] = origin_group.get_camera(k)
                    self.groups[c] = Base.Group(b=sum([new_group_cameras[n].b for n in new_group_cameras]),
                                                    t=sum([new_group_cameras[n].t for n in new_group_cameras]),
                                                    V=sum([new_group_cameras[n].V for n in new_group_cameras]),
                                                    cameras_begin=min(new_group_cameras), d=self.d,
                                                    camera_num=len(new_group_cameras),
                                                    rounds=self.rounds, cameras=copy.deepcopy(new_group_cameras),
                                                    payoffs=sum(
                                                        [new_group_cameras[k].payoffs for k in new_group_cameras]),
                                                    best_payoffs=sum(
                                                        [new_group_cameras[k].best_payoffs for k in new_group_cameras]))
                    for k in C:
                        self.group_inds[k] = c

                    c += 1
                    for camera_index in all_cameras_index:
                        if new_group_cameras.__contains__(camera_index):
                            all_cameras.pop(camera_index)

        self.num_groups[t] = len(self.groups)


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

    def run(self, envir, T):
        y_list = list()
        x_list = list()
        final_list = list()
        start_time = time.time()
        # with open(f"time{self.txt_name}_wo_Combinatorial_{T}.txt", "w") as file:
        for i in range(1, T + 1):
            if i % 5000 == 0:
                print(i, len(edge_server.groups))
            # if i % 1000 == 0 :
            #     run_time = time.time() - start_time
            #     data_to_write = f"{i},{run_time}\n"
            #     file.write(data_to_write)
            last_element = 0
            camera_index = envir.random_sample_camera()
            edge_server_index = self.locate_camera_index(camera_index)
            edge_server = self.edge_server_list[edge_server_index]
            group_index = edge_server.locate_camera_index(camera_index)
            group = edge_server.groups[group_index]
            visual_models = envir.get_visual_models(last_element)
            r_visual_model_index = edge_server.select(group_index, visual_models, last_element)
            self.model_selection[i - 1] = r_visual_model_index
            self.payoff[i - 1], x, y, self.best_payoff[i - 1], is_final = envir.feedback_Local(
                visual_models=visual_models, i=camera_index, k=r_visual_model_index, is_final=last_element)
            final_list.append(is_final)

            x_list.append(x)
            y_list.append(y)

            group.cameras[camera_index].store_info(x, y, i - 1, self.payoff[i - 1], self.best_payoff[i - 1])
            group.store_info(x, y, i - 1, self.payoff[i - 1], self.best_payoff[i - 1])
            edge_server.update(camera_index, i - 1)
            self.regret[i - 1] = self.best_payoff[i - 1] - self.payoff[i - 1]

        return self.regret, self.payoff, x_list, y_list, final_list, self.model_selection
