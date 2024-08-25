import os
import numpy as np
import json
from Algorithm.video import get_txt_path

def get_file_num(folder_path, prefix, suffix):
    return sum(1 for filename in os.listdir(folder_path) if filename.startswith(prefix) and filename.endswith(suffix))


def compute_iou(rec1, rec2):
    ans1 = [(rec1[0] - rec1[2] / 2), (rec1[1] - rec1[3] / 2), (rec1[0] + rec1[2] / 2), (rec1[1] + rec1[3] / 2)]
    ans2 = [(rec2[0] - rec2[2] / 2), (rec2[1] - rec2[3] / 2), (rec2[0] + rec2[2] / 2), (rec2[1] + rec2[3] / 2)]

    left_column_max = max(ans1[0], ans2[0])
    right_column_min = min(ans1[2], ans2[2])
    up_row_max = max(ans1[1], ans2[1])
    down_row_min = min(ans1[3], ans2[3])
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    else:
        S1 = (ans1[2] - ans1[0]) * (ans1[3] - ans1[1])
        S2 = (ans2[2] - ans2[0]) * (ans2[3] - ans2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    return S_cross / (S1 + S2 - S_cross)


def check_file_status(filename):
    if not os.path.exists(filename):
        return 0
    elif os.path.getsize(filename) == 0:
        return 0
    else:
        return 1


def performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold, confidence=0.5):
    path1 = os.path.join(stdpath, f'{stdtxt}.txt')
    path2 = os.path.join(testpath, f'{testtxt}.txt')

    if not check_file_status(path1):
        if not check_file_status(path2):
            return 0, 0, 0, 0, 0
        else:
            testfile = np.loadtxt(path2).reshape(-1, 6)
            testfile = testfile[testfile[:, 5] >= confidence]
            testfile = testfile[testfile[:, 0] == label]
            return 0, 0, len(testfile), 0, 0
    elif not check_file_status(path2):
        stdfile = np.loadtxt(path1).reshape(-1, 6)
        stdfile = stdfile[stdfile[:, 5] >= confidence]
        stdfile = stdfile[stdfile[:, 0] == label]
        FN = len(stdfile)
        return 0, FN, 0, 0, 0

    else:
        stdfile = np.loadtxt(path1).reshape(-1, 6)
        stdfile = stdfile[stdfile[:, 5] >= confidence]
        stdfile = stdfile[stdfile[:, 0] == label]
        testfile = np.loadtxt(path2).reshape(-1, 6)
        testfile = testfile[testfile[:, 5] >= confidence]
        testfile = testfile[testfile[:, 0] == label]

        TP, FN, iou_cum_recall, iou_cum_acc, = 0, 0, 0, 0
        matched = np.ones(len(testfile))

        for line in stdfile:
            iou_list = [compute_iou(line[1:5], tline[1:5]) for tline in testfile]
            if iou_list:
                iou_list = np.array(iou_list)
                avi_iou_list = iou_list * matched
                result = np.max(avi_iou_list)
                max_index = np.argmax(avi_iou_list)
                iou_cum_recall += result
                if result >= threshold:
                    iou_cum_acc += result
                    TP += 1
                    matched[max_index] = 0
                else:
                    FN += 1
            else:
                FN += 1

        if TP != 0:
            return TP, FN, len(testfile) - TP, iou_cum_recall / (TP + FN), iou_cum_acc / TP
        elif FN != 0:
            return 0, FN, len(testfile), iou_cum_recall / FN, 0
        else:
            return TP, FN, len(testfile) - TP, 0, 0


def element2performance(TP, FN, FP):
    if TP != 0:
        recall = TP / (TP + FN)
        F1 = 2 * TP / (2 * TP + FP + FN)
        precision = TP / (TP + FP)
    else:
        precision = 0
        recall = 0
        F1 = 0
        if FP == 0 and FN == 0:
            F1 = 1
        if FP == 0:
            precision = 1
        if FN == 0:
            recall = 1
    return precision, recall, F1


def performance(stdpath: str, testpath: str, stdtxt: str, testtxt: str, label: int, threshold: float):
    TP, FN, FP, cumR, cumA = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
    precision, recall, F1 = element2performance(TP, FN, FP)
    return precision, recall, F1, cumR, cumA


def performance_accumulate(stdpath: str, testpath: str, src_name: str, output_name: str, label: int, threshold: float,
                           frame_begin=1, frame_end=0):
    if frame_end == 0:
        txt_files = [f for f in os.listdir(stdpath) if f.startswith(src_name) and f.endswith('.txt')]
        TP, FN, FP, cumR, cumA = 0, 0, 0, 0, 0
        for count in range(frame_begin, len(txt_files) + 1):
            stdtxt = f'{src_name}_{count}'
            testtxt = f'{output_name}_{count}'
            TP_, FN_, FP_, cumR_, cumA_ = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
            TP += TP_
            FN += FN_
            FP += FP_
            cumR += cumR_ * (TP_ + FN_)
            cumA += cumA_ * TP_
        precision, recall, F1 = element2performance(TP, FN, FP)

        if TP != 0:
            return precision, recall, F1, cumR / (TP + FN), cumA / TP
        elif FN != 0:
            return precision, recall, F1, cumR / FN, 0
        else:
            return precision, recall, F1, 0, 0
    else:
        TP, FN, FP, cumR, cumA = 0, 0, 0, 0, 0
        for count in range(frame_begin, frame_end + 1):
            stdtxt = f'{src_name}_{count}'
            testtxt = f'{output_name}_{count}'
            TP_, FN_, FP_, cumR_, cumA_ = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
            TP += TP_
            FN += FN_
            FP += FP_
            cumR += cumR_ * (TP_ + FN_)
            cumA += cumA_ * TP_
        precision, recall, F1 = element2performance(TP, FN, FP)
        if TP != 0:
            return precision, recall, F1, cumR / (TP + FN), cumA / TP
        elif FN != 0:
            return precision, recall, F1, cumR / FN, 0
        else:
            return precision, recall, F1, 0, 0

def element_accumulate(stdpath: str, testpath: str, src_name: str, output_name: str, label: int, threshold: float,
                           frame_begin=1, frame_end=0):
    if frame_end == 0:
        txt_files = [f for f in os.listdir(stdpath) if f.startswith(src_name) and f.endswith('.txt')]
        TP, FN, FP, cumR, cumA = 0, 0, 0, 0, 0
        for count in range(frame_begin, len(txt_files) + 1):
            stdtxt = f'{src_name}_{count}'
            testtxt = f'{output_name}_{count}'
            TP_, FN_, FP_, cumR_, cumA_ = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
            TP += TP_
            FN += FN_
            FP += FP_
            cumR += cumR_ * (TP_ + FN_)
            cumA += cumA_ * TP_

    else:
        TP, FN, FP, cumR, cumA = 0, 0, 0, 0, 0
        for count in range(frame_begin, frame_end + 1):
            stdtxt = f'{src_name}_{count}'
            testtxt = f'{output_name}_{count}'
            TP_, FN_, FP_, cumR_, cumA_ = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
            TP += TP_
            FN += FN_
            FP += FP_
            cumR += cumR_ * (TP_ + FN_)
            cumA += cumA_ * TP_

    return TP, FN, FP, cumR, cumA


def videos_performance_accumulate(video_names, dnn, version, label, confidence_threshold, is_free_viewpoint=0,
          gt='yolov5', src_additional_tag = '', out_additional_tag = ''):
    tmp_data = np.zeros((len(video_names), 5))

    if dnn == 'mmdet':
        out_additional_tag = '_frame'
    if gt == 'mmdet':
        src_additional_tag = '_frame'
    for video_index, video_name in enumerate(video_names):
        src_name = video_name + src_additional_tag
        output_name = video_name + out_additional_tag
        stdpath = get_txt_path(video_name, gt, 'x', is_free_viewpoint=is_free_viewpoint)
        testpath = get_txt_path(video_name, dnn, version, is_free_viewpoint=is_free_viewpoint)
        tmp_data[video_index, :] = performance_accumulate(stdpath, testpath, src_name, output_name, label=label,
                                                          threshold=confidence_threshold)
    return tmp_data

def element2result(new_array):
    TP = new_array[..., :, 0]
    FN = new_array[..., :, 1]
    FP = new_array[..., :, 2]
    iou_cum_recall = new_array[..., :, 3]
    iou_cum_acc =  new_array[..., :, 4]

    epsilon = 1e-7
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    F1 = 2 * precision * recall / (precision + recall + epsilon)
    iou_cum_recall = iou_cum_recall / (TP + FP + epsilon)
    iou_cum_acc = iou_cum_acc / (TP + epsilon)

    precision[(TP == 0) & (FP != 0)] = 0
    recall[(TP == 0) & (FN != 0)] = 0
    F1[TP == 0] = 0
    precision[(TP == 0) & (FP == 0)] = 1
    recall[(TP == 0) & (FN == 0)] = 1
    F1[(TP == 0) & (FP == 0) & (FN == 0)] = 1
    iou_cum_recall[(TP == 0) & (FN == 0)] = 1
    iou_cum_acc[(TP == 0)] = 1

    new_array[..., 0] = precision
    new_array[..., 1] = recall
    new_array[..., 2] = F1
    new_array[..., 3] = iou_cum_recall
    new_array[..., 4] = iou_cum_acc
    return new_array



def videos_element_accumulate(video_names, dnn, version, label, confidence_threshold, is_free_viewpoint=0,
          gt='yolov5', src_additional_tag = '', out_additional_tag = ''):
    tmp_data = np.zeros(5)
    brightness_descriptions = ["Significantly_Darker", "Moderately_Darker", "Slightly_Darker",
                               "Very_Slightly_Darker", "Almost_Natural_Light", "Natural_Light", "Slightly_Brighter",
                               "Moderately_Brighter", "Significantly_Brighter", "Very_Bright", "Extremely_Bright"]

    if dnn == 'mmdet':
        out_additional_tag = '_frame'
    if gt == 'mmdet':
        src_additional_tag = '_frame'
    for video_index, video_name in enumerate(video_names):


        std_video_name = video_name
        brightness_descriptions.sort(key=len, reverse=True)
        for description in brightness_descriptions:
            std_video_name = std_video_name.replace(description + '_', '')
        src_name = std_video_name + src_additional_tag
        stdpath = get_txt_path(std_video_name, gt, 'x', is_free_viewpoint=is_free_viewpoint)
        testpath = get_txt_path(video_name, dnn, version, is_free_viewpoint=is_free_viewpoint)
        tmp_data[:] += element_accumulate(stdpath, testpath, src_name, src_name, label=label,
                                                          threshold=confidence_threshold)

    return tmp_data


def segment_performance(video_names, dnn, version, label, confidence_threshold, is_free_viewpoint=0, gt='yolov5', gap=15, segments=20):
    segment_data = np.zeros((len(video_names), segments, 5))

    for video_index, video_name in enumerate(video_names):
        src_name = video_name
        output_name = video_name
        stdpath = get_txt_path(video_name, gt, 'x', is_free_viewpoint=is_free_viewpoint)
        testpath = get_txt_path(video_name, dnn, version, is_free_viewpoint=is_free_viewpoint)
        for segment in range(segments):
            segment_data[video_index, segment, :] = performance_accumulate(stdpath, testpath, src_name, output_name,
                                                                           label=label, threshold=confidence_threshold,
                                                                           frame_begin=gap * segment + 1,
                                                                           frame_end=gap * (1 + segment) + 1)

    return segment_data


def new_element2result(new_array,performance_index):
    TP = new_array[..., :, 0]
    FN = new_array[..., :, 1]
    FP = new_array[..., :, 2]
    iou_cum_recall = new_array[..., :, 3]
    iou_cum_acc =  new_array[..., :, 4]

    epsilon = 1e-7
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    F1 = 2 * precision * recall / (precision + recall + epsilon)
    iou_cum_recall = iou_cum_recall / (TP + FP + epsilon)
    iou_cum_acc = iou_cum_acc / (TP + epsilon)

    precision[(TP == 0) & (FP != 0)] = 0
    recall[(TP == 0) & (FN != 0)] = 0
    F1[TP == 0] = 0
    precision[(TP == 0) & (FP == 0)] = 1
    recall[(TP == 0) & (FN == 0)] = 1
    F1[(TP == 0) & (FP == 0) & (FN == 0)] = 1
    iou_cum_recall[(TP == 0) & (FN == 0)] = 1
    iou_cum_acc[(TP == 0)] = 1

    new_array[..., 0] = precision
    new_array[..., 1] = recall
    new_array[..., 2] = F1
    new_array[..., 3] = iou_cum_recall
    new_array[..., 4] = iou_cum_acc
    reshaped_array = new_array[..., performance_index].reshape(-1, *new_array[..., performance_index].shape[3:])
    reshaped_array = np.squeeze(reshaped_array)

    return reshaped_array


def merge_and_calculate_metrics(array, merge_length, performance_index, need_test=0):

    shape = array.shape
    new_dim_length = shape[-2] // merge_length
    if need_test == 1:
        new_dim_length = 1
    new_shape = list(shape[:-2]) + [new_dim_length, 5]
    new_array = np.zeros(new_shape)
    for i in range(new_dim_length):
        start_idx = i * merge_length
        end_idx = start_idx + merge_length
        new_array[..., i, :] = np.sum(array[..., start_idx:end_idx, :], axis=-2)
    reshaped_array = new_element2result(new_array, performance_index)
    if need_test == 1:
        test_array = np.zeros(new_shape)
        test_array[..., i, :] = np.sum(array[..., end_idx:, :], axis=-2)
        reshaped_test_array = new_element2result(test_array, performance_index)
    if need_test == 1:
        return reshaped_array, reshaped_test_array
    else:
        return reshaped_array

def merge_and_calculate_metrics_without_theta(array, merge_length, performance_index, need_test=0):

    shape = array.shape
    new_dim_length = shape[-2] // merge_length
    if need_test == 1:
        new_dim_length = 1
    new_shape = list(shape[:-2]) + [new_dim_length, 5]
    new_array = np.zeros(new_shape)
    newnew_shape = list(shape[:1]) + list(new_shape[2:])
    new_array_without_theta = np.zeros(newnew_shape)
    for i in range(new_dim_length):
        start_idx = i * merge_length
        end_idx = start_idx + merge_length
        new_array[..., i, :] = np.sum(array[..., start_idx:end_idx, :], axis=-2)
        new_array_without_theta = np.sum(new_array, axis=1)
    for i in range(7):
        new_array[:,i,...] = new_array_without_theta
    reshaped_array = new_element2result(new_array, performance_index)
    if need_test == 1:
        test_array = np.zeros(new_shape)
        test_array[..., 0, :] = np.sum(array[..., end_idx:, :], axis=-2)
        reshaped_test_array = new_element2result(test_array, performance_index)
    if need_test == 1:
        return reshaped_array, reshaped_test_array
    else:
        return reshaped_array



def handle_regret(regret, final, window_size):
    indices = np.where(np.array(final) == 0)[0]
    selected_regrets = np.array(regret)[indices]
    cumulative_regret_list = np.zeros(len(indices))
    cumulative_sum = np.cumsum(selected_regrets)
    for i in range(len(selected_regrets)):
        start = max(0, i - window_size + 1)
        window_sum = cumulative_sum[i] - cumulative_sum[start] + selected_regrets[start]
        cumulative_regret_list[i] = window_sum / min(window_size, i + 1)
    cumulative_regret = cumulative_sum[-1]
    return cumulative_regret_list, cumulative_regret


def handle_payoff(payoff, final, window_size):
    indices = np.where(np.array(final) == 0)[0]
    selected_payoffs = np.array(payoff)[indices]
    cumulative_payoff_list = np.zeros(len(indices))
    cumulative_sum = np.cumsum(selected_payoffs)
    for i in range(len(selected_payoffs)):
        start = max(0, i - window_size + 1)
        window_sum = cumulative_sum[i] - cumulative_sum[start] + selected_payoffs[start]
        cumulative_payoff_list[i] = window_sum / min(window_size, i + 1)
    cumulative_payoff = cumulative_sum[-1]
    return cumulative_payoff_list, cumulative_payoff



def performance_save(Algorithms, window_size, rounds, T, n, index, performance_index, matrix_name):
    names = [f'{Algorithm}_svd_{index}_{performance_index}_{n}_{T}' for Algorithm in Algorithms]
    data = dict()
    regret = dict()
    payoff = dict()
    final = dict()
    cumulative_regret_lists = dict()
    cumulative_regrets = dict()
    cumulative_payoff_lists = dict()
    cumulative_payoffs = dict()
    matrix_data = np.zeros((len(Algorithms), len(rounds)))
    for name_index, name in enumerate(names):
        dic_index = f'{name}'
        data[dic_index] = np.load(f'npz_data/{name}.npz')
        regret[dic_index] = data[dic_index]['regret']
        payoff[dic_index] = data[dic_index]['payoff']
        final[dic_index] = np.zeros(T)
        cumulative_regret_lists[dic_index], cumulative_regrets[dic_index] = handle_regret(regret[dic_index],
                                                                                          final[dic_index],
                                                                                          window_size)
        cumulative_payoff_lists[dic_index], cumulative_payoffs[dic_index] = handle_payoff(payoff[dic_index],
                                                                                          final[dic_index],
                                                                                          window_size)
        for t_index, round in enumerate(rounds):
            matrix_data[name_index][t_index] = cumulative_payoff_lists[dic_index][round]

    np.save(f'raw_data/{matrix_name}.npy', matrix_data)


