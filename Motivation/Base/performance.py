import os
import numpy as np
import cv2
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

from Base.video import get_txt_path

def get_file_num(folder_path, prefix, suffix):
    """
    Count the number of files in a folder with a given prefix and suffix.
    """
    return sum(1 for filename in os.listdir(folder_path) if filename.startswith(prefix) and filename.endswith(suffix))


# for detection to get result
def compute_iou(rec1, rec2):
    """
    Calculate the intersection over union (IOU) of two rectangles.
    :param rec1: (xc, yc, w, h) representing the coordinates of the first rectangle.
    :param rec2: (xc, yc, w, h) representing the coordinates of the second rectangle.
    :return: The IOU (intersection over union) of the two rectangles.
    """
    ans1 = [(rec1[0] - rec1[2] / 2), (rec1[1] - rec1[3] / 2), (rec1[0] + rec1[2] / 2), (rec1[1] + rec1[3] / 2)]
    ans2 = [(rec2[0] - rec2[2] / 2), (rec2[1] - rec2[3] / 2), (rec2[0] + rec2[2] / 2), (rec2[1] + rec2[3] / 2)]

    left_column_max = max(ans1[0], ans2[0])
    right_column_min = min(ans1[2], ans2[2])
    up_row_max = max(ans1[1], ans2[1])
    down_row_min = min(ans1[3], ans2[3])
    # if the two rectangles have no overlapping region.
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    else:
        S1 = (ans1[2] - ans1[0]) * (ans1[3] - ans1[1])
        S2 = (ans2[2] - ans2[0]) * (ans2[3] - ans2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    return S_cross / (S1 + S2 - S_cross)


def check_file_status(filename):
    """
    Check if the file exists and is not empty.

    Args:
    - filename (str): The path to the file to check.

    Returns:
    - int: 0 if the file doesn't exist or is empty, 1 if the file exists and is not empty.
    """
    # Check if the file exists
    if not os.path.exists(filename):
        return 0
    # Check if the file is empty (file size is 0)
    elif os.path.getsize(filename) == 0:
        return 0
    else:
        return 1


def performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold, confidence=0.5):
    path1 = os.path.join(stdpath, f'{stdtxt}.txt')
    path2 = os.path.join(testpath, f'{testtxt}.txt')

    # Check if the standard file exists and is not empty
    if not check_file_status(path1):
        print(f'{path1} doesn\'t exist or is empty')

        # Check if the test file exists and is not empty
        if not check_file_status(path2):
            return 0, 0, 0, 0, 0
        else:
            testfile = np.loadtxt(path2).reshape(-1, 6)
            testfile = testfile[testfile[:, 5] >= confidence]
            testfile = testfile[testfile[:, 0] == label]
            return 0, 0, len(testfile), 0, 0

    # Check if the test file exists and is not empty
    elif not check_file_status(path2):
        print(f'{path2} doesn\'t exist or is empty')
        stdfile = np.loadtxt(path1).reshape(-1, 6)
        stdfile = stdfile[stdfile[:, 5] >= confidence]
        stdfile = stdfile[stdfile[:, 0] == label]
        FN = len(stdfile)
        return 0, FN, 0, 0, 0

    # Both files exist and are not empty
    else:
        stdfile = np.loadtxt(path1).reshape(-1, 6)
        stdfile = stdfile[stdfile[:, 5] >= confidence]
        stdfile = stdfile[stdfile[:, 0] == label]
        testfile = np.loadtxt(path2).reshape(-1, 6)
        testfile = testfile[testfile[:, 5] >= confidence]
        testfile = testfile[testfile[:, 0] == label]

        TP, FN, iou_accum, iou_recall = 0, 0, 0, 0
        matched = np.ones(len(testfile))  # Track which test boxes are available

        for line in stdfile:
            iou_list = [compute_iou(line[1:5], tline[1:5]) for tline in testfile]
            if iou_list:
                iou_list = np.array(iou_list)
                avi_iou_list = iou_list * matched
                result = np.max(avi_iou_list)
                max_index = np.argmax(avi_iou_list)
                iou_accum += result
                if result >= threshold:
                    iou_recall += result
                    TP += 1
                    matched[max_index] = 0
                else:
                    FN += 1
            else:
                FN += 1

        if TP != 0:
            return TP, FN, len(testfile) - TP, iou_accum / (TP + FN), iou_recall / TP
        else:
            return TP, FN, len(testfile) - TP, 0, 0


# for segment to get result
def get_segment_mask(image_path):
    # we use paddle segment model, and the mask use green color.
    image = cv2.imread(image_path)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[image[:, :, 1] == 128] = 1
    return mask


def Segment_F1_element(stdimg, testimg):
    ground_truth = get_segment_mask(stdimg)
    predicted = get_segment_mask(testimg)
    TP = ((ground_truth == 1) & (predicted == 1)).sum()
    FN = ((ground_truth == 1) & (predicted == 0)).sum()
    FP = ((ground_truth == 0) & (predicted == 1)).sum()
    return TP, FN, FP

def segment_F1(stdpath: str, testpath: str, src_name: str, output_name: str):
    png_files = [f for f in os.listdir(stdpath) if f.startswith(src_name) and f.endswith('.png')]
    total_TP, total_FN, total_FP = 0,0,0
    for j in range(len(png_files) - 1):
        stdimg = os.path.join(stdpath, src_name + f'_{j}.png')
        testimg = os.path.join(testpath, output_name + f'_{j}.png')
        TP, FN, FP = Segment_F1_element(stdimg, testimg)
        total_TP += TP
        total_FN += FN
        total_FP += FP
    return element2performance(total_TP, total_FN, total_FP)


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
    TP, FN, FP, mAP, mAR = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
    precision, recall, F1 = element2performance(TP, FN, FP)
    return precision, recall, F1, mAP, mAR


def performance_accumulate(stdpath: str, testpath: str, src_name: str, output_name: str, label: int, threshold: float,
                           frame_begin=1, frame_end=0):
    # we can decide whether we should stop at fixed frame
    if frame_end == 0:
        txt_files = [f for f in os.listdir(stdpath) if f.startswith(src_name) and f.endswith('.txt')]
        TP, FN, FP, mAP_a, mAP_recall = 0, 0, 0, 0, 0
        for count in range(frame_begin, len(txt_files) + 1):
            stdtxt = f'{src_name}_{count}'
            testtxt = f'{output_name}_{count}'
            TP_, FN_, FP_, mAP_, mAR_ = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
            TP += TP_
            FN += FN_
            FP += FP_
            mAP_a += mAP_ * (TP_ + FN_)
            mAP_recall += mAR_ * TP_
        precision, recall, F1 = element2performance(TP, FN, FP)
        if TP != 0:
            return precision, recall, F1, mAP_a / (TP + FN), mAP_recall / TP
        else:
            return precision, recall, F1, 0, 0
    else:
        TP, FN, FP, mAP_a, mAP_recall = 0, 0, 0, 0, 0
        for count in range(frame_begin, frame_end + 1):
            stdtxt = f'{src_name}_{count}'
            testtxt = f'{output_name}_{count}'
            TP_, FN_, FP_, mAP_, mAR_ = performance_element(stdpath, testpath, stdtxt, testtxt, label, threshold)
            TP += TP_
            FN += FN_
            FP += FP_
            mAP_a += mAP_ * (TP_ + FN_)
            mAP_recall += mAR_ * TP_
        precision, recall, F1 = element2performance(TP, FN, FP)
        if TP != 0:
            return precision, recall, F1, mAP_a / (TP + FN), mAP_recall / TP
        else:
            return precision, recall, F1, 0, 0


def videos_performance_accumulate(video_names, dnn, version, label, confidence_threshold, is_free_viewpoint=0, gt='yolov5'):
    tmp_data = np.zeros((len(video_names), 5))
    src_additional_tag = ''
    out_additional_tag = ''
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