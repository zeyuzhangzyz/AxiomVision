from Algorithm.performance import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def find_optimal_k(data, max_k=10):
    # find the group number g
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.show()


def SVD_calculate(index, n, performance_index,gap_length, yolov5_numbers, version, need_test=0):
    index1 = [0, 0, 2]
    index2 = [0, 2, 2]
    data_label1 = np.load(f'data_label_{index1[index]}.npy')
    data_label2 = np.load(f'data_label_{index2[index]}.npy')
    data_label1[...,3] = data_label1[...,3]*(data_label1[...,0]+data_label1[...,1])
    data_label1[..., 4] = data_label1[..., 4] * (data_label1[..., 0])
    data_label2[..., 3] = data_label2[..., 3] * (data_label2[..., 0] + data_label2[..., 1])
    data_label2[..., 4] = data_label2[..., 4] * (data_label2[..., 0])
    if need_test:
        svd_part, predict_part = merge_and_calculate_metrics(data_label1 + data_label2, gap_length, performance_index, need_test)
    else:
        svd_part = merge_and_calculate_metrics(data_label1 + data_label2, gap_length, performance_index, need_test)
    matrix = svd_part
    row_means = np.mean(matrix, axis=0)
    best_fix = np.argmax(row_means[:yolov5_numbers])
    rows_to_delete = []
    for i in range(matrix.shape[0]):
        if np.sum(matrix[i] < 0.1) > 2:
            rows_to_delete.append(i)
    matrix = np.delete(matrix, rows_to_delete, axis=0)
    # find_optimal_k(matrix) # find the group number
    if need_test:
        predict_part = np.delete(predict_part, rows_to_delete, axis=0)
    else:
        predict_part = svd_part
    U, Sigma, Vt = np.linalg.svd(matrix)
    U_lowdim = U[:, :n]
    Sigma_lowdim_vector = np.diag(Sigma[:n])
    Vt_lowdim = Vt[:n, :]
    svd_payoff = np.dot(U_lowdim, np.dot(Sigma_lowdim_vector, Vt_lowdim))
    np.save(f"raw_data/real_payoff_{index}_{performance_index}_{n}{version}.npy", predict_part)
    np.save(f"raw_data/svd_payoff_{index}_{performance_index}_{n}{version}.npy", svd_payoff)
    np.savez(f"raw_data/svd_results_{index}_{performance_index}_{n}{version}.npz", U_lowdim=U_lowdim,
             Sigma_lowdim=Sigma_lowdim_vector, Vt_lowdim=Vt_lowdim, Tmp_lowdim=np.dot(Sigma_lowdim_vector, Vt_lowdim), best_fix = best_fix)
    return rows_to_delete

def SVD_calculate_wo_Perspective(index, n, performance_index, gap_length, yolov5_numbers, version, need_test=0):

    index1 = [0, 0, 2]
    index2 = [0, 2, 2]
    data_label1 = np.load(f'data_label_{index1[index]}.npy')
    data_label2 = np.load(f'data_label_{index2[index]}.npy')
    data_label1[...,3] = data_label1[...,3]*(data_label1[...,0]+data_label1[...,1])
    data_label1[..., 4] = data_label1[..., 4] * (data_label1[..., 0])
    data_label2[..., 3] = data_label2[..., 3] * (data_label2[..., 0] + data_label2[..., 1])
    data_label2[..., 4] = data_label2[..., 4] * (data_label2[..., 0])
    if need_test:
        svd_part, predict_part = merge_and_calculate_metrics(data_label1 + data_label2, gap_length, performance_index, need_test)
    else:
        svd_part = merge_and_calculate_metrics(data_label1 + data_label2, gap_length, performance_index, need_test)
    matrix = svd_part
    row_means = np.mean(matrix, axis=0)
    best_fix = np.argmax(row_means[:yolov5_numbers])
    rows_to_delete = []
    for i in range(matrix.shape[0]):
        if np.sum(matrix[i] < 0.1) > 2:
            rows_to_delete.append(i)
    matrix = np.delete(matrix, rows_to_delete, axis=0)
    if need_test:
        predict_part = np.delete(predict_part, rows_to_delete, axis=0)
    else:
        predict_part = svd_part
    U, Sigma, Vt = np.linalg.svd(matrix)
    U_lowdim = U[:, :n]
    Sigma_lowdim_vector = np.diag(Sigma[:n])
    Vt_lowdim = Vt[:n, :]
    svd_payoff = np.dot(U_lowdim, np.dot(Sigma_lowdim_vector, Vt_lowdim))
    KMeans(U_lowdim)
    np.save(f"raw_data/real_payoff_{index}_{performance_index}_{n}_wo_Perspective{version}.npy", predict_part)
    np.save(f"raw_data/svd_payoff_{index}_{performance_index}_{n}_wo_Perspective{version}.npy", svd_payoff)
    np.savez(f"raw_data/svd_results_{index}_{performance_index}_{n}_wo_Perspective{version}.npz", U_lowdim=U_lowdim,
             Sigma_lowdim=Sigma_lowdim_vector, Vt_lowdim=Vt_lowdim, Tmp_lowdim=np.dot(Sigma_lowdim_vector, Vt_lowdim),
             best_fix=best_fix)


def main(config_path='Algorithm/experiment_config.json'):
    config = load_config(config_path)
    brightness_descriptions = config["brightness_descriptions"]
    n = 17
    folder_name = "raw_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # if you want to test the performance of the time-varying model, set parameter as follows:
    # need_test = 1
    # version = "_time_varying"
    # gap_length = 50
    need_test = 0
    gap_length = 100
    version = ""
    for index in range(3):
        for performance_index in range(3):
            SVD_calculate(index,n,performance_index,gap_length,len(brightness_descriptions), version, need_test)
            SVD_calculate_wo_Perspective(index,n,performance_index,gap_length,len(brightness_descriptions), version, need_test)


if __name__ == "__main__":
    main()
