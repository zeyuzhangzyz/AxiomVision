import os
import numpy as np

from multiprocessing import Process
import matplotlib.pyplot as plt
from Algorithm.performance import performance_save
import Algorithm.Axiomvision as Axiomvision
import Algorithm.wo_Perspective as wo_Perspective
import Algorithm.Greedy as Greedy
import Algorithm.Dual_MS as Dual_MS
import Algorithm.wo_Combinatorial as wo_Combinatorial
import Algorithm.Set_based as Set_based
import Algorithm.Baseline as Baseline
import Algorithm.wo_Group as wo_Group
import Algorithm.Environment as Envi
import time
import random


def main_Axiomvision(num_cameras, d, L, edge_server_num, T, theta, seed, cameraList, config_index, npzname='', txt_name=''):
    CLOUD_server = Axiomvision.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, txt_name=txt_name, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time

    np.savez("npz_data/Axiomvision_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time, theta=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)



def main_Dual_MS(num_cameras, d, L, edge_server_num, T, theta, seed, cameraList, config_index = '', npzname='', txt_name=''):
    CLOUD_server = Dual_MS.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, txt_name=txt_name, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time
    np.savez("npz_data/Dual_MS_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)


def main_wo_Combinatorial(num_cameras, d, L, edge_server_num, T, theta, seed, cameraList,config_index = '', npzname='', txt_name=''):
    CLOUD_server = wo_Combinatorial.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, txt_name=txt_name, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time

    np.savez("npz_data/wo_Combinatorial_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)




def main_wo_Perspective(num_cameras, d, L, edge_server_num, T, theta, seed, cameraList, config_index='', npzname='', txt_name=''):
    CLOUD_server = wo_Perspective.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, txt_name=txt_name, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time

    np.savez("npz_data/wo_Perspective_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)

def main_Greedy(num_cameras, d, L, edge_server_num, seed, cameraList, theta, T, config_index='', npzname=''):
    CLOUD_server = Greedy.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time
    np.savez("npz_data/Greedy_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)


def main_wo_Group(edge_server_num, num_cameras, d, theta, T, L, seed, cameraList, config_index='', npzname='', txt_name=""):
    CLOUD_server = wo_Group.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, txt_name=txt_name, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time
    np.savez("npz_data/wo_Group_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)

def main_Set_based(num_cameras, d, L, edge_server_num, T, theta, seed, cameraList, config_index ='', npzname='', txt_name='', phase_cardinality=2):
    phase = (np.log(T) / np.log(phase_cardinality)).astype(np.int64)
    round = (phase_cardinality ** (phase + 1) - 1) // (phase_cardinality - 1)
    CLOUD_server = Set_based.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=round, txt_name=txt_name, config_index = config_index)
    print("round:", (phase_cardinality ** (phase + 1) - 1) // (phase_cardinality - 1))
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, phase=phase)
    run_time = time.time() - start_time
    np.savez("npz_data/Set_based_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)

def main_Baseline(num_cameras, d, theta, T, edge_server_num, L, seed,cameraList, config_index ='', npzname=''):
    CLOUD_server = Baseline.Cloud_server(edge_server_num, num_cameras, cameraList, d=d, T=T, config_index = config_index)
    envi = Envi.Environment(d=d, num_cameras=num_cameras, L=L, theta=theta, config_index = config_index)
    start_time = time.time()
    regret, payoff, x_list, y_list, final_list, model_selection = CLOUD_server.run(envi, T)
    run_time = time.time() - start_time
    np.savez("npz_data/Baseline_" + npzname, nu=num_cameras, d=d, L=L, T=T, seed=seed, regret=regret,
             run_time=run_time,  theta_theo=theta, payoff=payoff, x=x_list,
             y=y_list, final=final_list, model_selection=model_selection)



def  main_run(num_cameras, d, L, edge_server_num, T, theta, config_index='', npzname=''):
    seed = int(time.time() * 100) % 399
    np.random.seed(seed)
    random.seed(seed)
    if num_cameras % edge_server_num == 0:
        cameraList = [num_cameras // edge_server_num] * edge_server_num
    else:
        cameraList = [num_cameras // edge_server_num] * (edge_server_num - 1)
        cameraList[edge_server_num - 1] = num_cameras - (num_cameras // edge_server_num) * (edge_server_num - 1)

    main_Greedy(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
                config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)
    main_Axiomvision(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
              config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)
    main_Baseline(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
                config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)
    main_wo_Group(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
              config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)
    main_wo_Combinatorial(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
                  config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)
    main_Dual_MS(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
              config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)
    main_Set_based(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
                  config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)

def main_other_run(num_cameras, d, L, edge_server_num, T, theta, config_index='', npzname=''):
    seed = int(time.time() * 100) % 399
    # print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)
    if num_cameras % edge_server_num == 0:
        cameraList = [num_cameras // edge_server_num] * edge_server_num
    else:
        cameraList = [num_cameras // edge_server_num] * (edge_server_num - 1)
        cameraList[edge_server_num - 1] = num_cameras - (num_cameras // edge_server_num) * (edge_server_num - 1)

    main_wo_Perspective(num_cameras=num_cameras, d=d,  L=L, edge_server_num=edge_server_num, theta=theta, T=T,
                  config_index=config_index, npzname=npzname, seed=seed, cameraList=cameraList)



def main():
    is_run = True # whether to run the experiment
    is_save_data = True # whether to save the data
    is_plot = True  # whether to plot the figure

    T = 500               # the rounds of the experiment
    n = 17                  # the number of selected models
    config_index = ""       # to save different configuration and paras
    index = 1               # 0 people  2  car
    performance_index = 1   #  0 precision, 1 recall, 2 F1, 3 cum_recall, 4 cum_acc
    seed = int(time.time() * 100) % 399
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)
    folder_name = "npz_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if is_run:
        print("Running")
        num_cameras = len(np.load(f'raw_data/svd_payoff_{index}_{performance_index}_{n}.npy'))
        version = f''
        # version = f'_wo_Perspective'
        Envi.get_payoff(index, performance_index, n, version)
        filename = f'raw_data/svd_results_{index}_{performance_index}_{n}.npz'
        theta = np.load(filename)['U_lowdim']

        main_run(num_cameras=num_cameras, d=17, L=17, edge_server_num=1, T=T,
                     theta=theta, config_index = config_index,
                     npzname=f'svd_{index}_{performance_index}_{n}_{T}{version}.npz')
        version = "_wo_Perspective"
        Envi.get_payoff(index, performance_index, n, version)
        main_other_run(num_cameras=num_cameras, d=17, L=17, edge_server_num=1, T=T,
                     theta=theta, config_index = config_index,
                     npzname=f'svd_{index}_{performance_index}_{n}_{T}{version}.npz')

    if is_save_data:
        print("Saving")
        window_size = 500
        Algorithms = ['Greedy', 'Axiomvision', 'Baseline', 'wo_Group', 'wo_Combinatorial', 'Dual_MS', 'Set_based']
        rounds = [10000, 20000, 30000, 40000, 50000]
        matrix_name = f"figure_data_{index}"
        performance_save(Algorithms, window_size, rounds, T, n, index, performance_index, matrix_name)

    if is_plot:
        print("Plotting")
        matrix_name = f"figure_data_{index}"
        matrix_data = np.load(f'raw_data/{matrix_name}.npy')
        print(matrix_data)
        plt.figure(figsize=(10, 6))
        for i, Algorithm in enumerate(Algorithms):
            plt.plot(rounds, matrix_data[i], label=Algorithm)

        plt.xlabel('Rounds')
        plt.ylabel('Performance Metric')
        plt.title('Performance of Different Algorithms Over Rounds')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    main()
