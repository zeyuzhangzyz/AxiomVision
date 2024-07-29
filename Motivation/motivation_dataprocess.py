import cv2
import subprocess
import json
import numpy as np

from Base.video import *
from Base.performance import *

def run_different_dnn_models(source_dir,video_names,versions):
    faster_rcnn(source_dir, video_names)
    mmdetection(source_dir, video_names)
    # ssd(source_dir, video_names)
    for version in versions:
        yolov5(source_dir, video_names, version)

def run_data_figure1(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["source_dir"]
    video_names = config["figure1_video_names"]
    versions = config["yolov5_versions"]
    run_different_dnn_models(source_dir,video_names,versions)


def save_data_figure1(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    versions = config['performance_versions']
    gts = config['gts']
    dnns = config['dnns']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    video_names = config['figure1_video_names']
    figure1_data_name = config['figure1_data_name']
    figure1_data = np.zeros((len(gts),len(dnns),len(video_names),5))
    for gt_index, gt in enumerate(gts):
        for dnn_index,dnn in enumerate(dnns):
            figure1_data[gt_index,dnn_index] = videos_performance_accumulate(video_names, dnn, versions[dnn_index],label , confidence_threshold,is_free_viewpoint = 0 , gt = gt)
    np.save(os.path.join(data_des_path, figure1_data_name + '.npy'),figure1_data)

def run_data_figure2(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["source_dir"]
    video_names = config["figure2_video_names"]
    versions = config["yolov5_versions"]
    run_different_dnn_models(source_dir,video_names,versions)

def save_data_figure2(config_path='motivation_config.json'):
    config = load_config(config_path)
    gts = config['gts']
    dnns = config['dnns']
    video_names = config["figure2_video_names"]
    versions = config['performance_versions']
    data_des_path = config['data_des_path']
    figure2_data_name = config['figure2_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure2_data = np.zeros((len(gts),len(dnns),len(video_names), 20, 5))
    for gt_index, gt in enumerate(gts):
        for dnn_index,dnn in enumerate(dnns):
            figure2_data[gt_index, dnn_index] = segment_performance(video_names, dnn, versions[dnn_index], label, confidence_threshold, is_free_viewpoint = 0, gt=gt)

    np.save(os.path.join(data_des_path, figure2_data_name + '.npy'),figure2_data)

def run_data_figure3(config_path='motivation_config.json'):

    config = load_config(config_path)
    source_dir = config["source_dir"]
    video_names = config["figure3_video_names"]
    versions = config["figure3_versions"]
    for version in versions:
        yolo_retrain(source_dir, video_names, version)
    run_different_dnn_models(source_dir,video_names,["s","x"])
def save_data_figure3(config_path='motivation_config.json'):
    config = load_config(config_path)
    video_names = config["figure3_video_names"]
    versions = config["figure3_versions"]
    gts = config['gts']
    data_des_path = config['data_des_path']
    figure3_data_name = config['figure3_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])

    figure3_data = np.zeros((len(gts),len(versions), len(video_names), 20, 5))
    for gt_index, gt in enumerate(gts):
        for version_index,version in enumerate(versions):
            figure3_data[gt_index, version_index] = segment_performance(video_names, 'yolov5', version, label, confidence_threshold, is_free_viewpoint = 0, gt = gt)
    np.save(os.path.join(data_des_path, figure3_data_name + '.npy'),figure3_data)



def run_data_figure4(config_path='motivation_config.json'):
    config = load_config(config_path)
    video_names = config["figure4_video_names"]
    versions = config["figure4_versions"]
    source_dir = config["source_dir"]
    run_different_dnn_models(source_dir, video_names, versions)

def save_data_figure4(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    figure4_data_name = config['figure4_data_name']
    video_names = config["figure4_video_names"]
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure4_data = videos_performance_accumulate(video_names, 'yolov5', 's' , label, confidence_threshold,is_free_viewpoint = 1, gt = 'yolov5')
    np.save(os.path.join(data_des_path, figure4_data_name + '.npy'),figure4_data)

def run_data_figure5(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config['source_dir']
    video_names = config["figure5_video_names"]
    versions = config["figure5_versions"]
    run_different_dnn_models(source_dir, video_names, versions)

def save_data_figure5(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config['data_des_path']
    video_names = config["figure5_video_names"]
    gts = config['gts']
    versions = config["figure5_versions"]
    figure5_data_name = config['figure5_data_name']
    label = int(config['car_label'])
    confidence_threshold = float(config['confidence_threshold'])
    figure5_data = np.zeros((len(gts),len(versions),len(video_names),5))
    for gt_index, gt in enumerate(gts):
        for version_index,version in enumerate(versions):
            figure5_data[gt_index,version_index] = videos_performance_accumulate(video_names, 'yolov5', version, label, confidence_threshold ,is_free_viewpoint = 0 , gt = gt)
    np.save(os.path.join(data_des_path, figure5_data_name + '.npy'),figure5_data)


def run_segmentation(image_path, save_dir, config_path, model_path):
    command = f'conda activate Paddle & python D:/code/PycharmProjects/Relighting-Base-Environment/tools/predict.py --config {config_path} --model_path {model_path} --image_path {image_path} --save_dir {save_dir}'
    os.system(command)

def figure6_resize_videos(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        for i in range(12):
            name = str(i) + '.mp4'
            input_name = os.path.join(input_path, name)
            output_name = os.path.join(output_path, name)
            only_change_resolution(input_name, output_name, 720, 480)
    extract_frames(input_path, os.path.join(input_path, 'extract_frames'))
    extract_frames(output_path, os.path.join(output_path, 'extract_frames'))

def segmentation_imgs(path, config_path='motivation_config.json'):
    config = load_config(config_path)
    segmentation_dnns = config["figure6_segmentation_dnns"]
    segmentation_versions = config["figure6_segmentation_versions"]
    segmentation_params_dir = config["figure6_segmentation_params_dir"]
    segmentation_config_dir = config["figure6_segmentation_config_dir"]
    segmentation_model_dir = config["figure6_segmentation_model_dir"]
    for dnn_index, dnn in enumerate(segmentation_dnns):
        dir = os.path.join(path, dnn)
        run_segmentation(os.path.join(path, 'extract_frames'), dir, os.path.join(segmentation_config_dir,
              segmentation_versions[dnn_index]), os.path.join(segmentation_model_dir, segmentation_params_dir[dnn_index],"model.pdparams"))

def get_num(stdpath, prefix, suffix):
    count = 0
    for filename in os.listdir(stdpath):
        if filename.startswith(prefix) and filename.endswith(suffix):
            count += 1
    return count

def main_segmentation(config_path='motivation_config.json'):
    config = load_config(config_path)
    folder_path = config["figure6_source_dir"]
    subfolder_names = get_subfolder_names(folder_path)
    releases =  config["figure6_releases"]
    for file_name in subfolder_names:
        for release in releases:
            path = os.path.join(folder_path, file_name, release)
            segmentation_imgs(path, config_path='motivation_config.json')

def save_segmentation_data(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config["data_des_path"]
    video_names = config["figure6_video_names"]
    folder_path = config['figure6_source_dir']
    versions = config["figure6_versions"]
    releases = config["figure6_releases"]
    data_name =  config["figure6_data_name"]
    data_matrix = np.zeros((len(video_names),12, len(versions),3))

    for video_index, file_name in enumerate(video_names):
        for release_index, release in enumerate(releases):
            path = os.path.join(folder_path, file_name, release)
            for i in range(12):
                stdpath = os.path.join(path, "server1", "pseudo_color_prediction")
                testpath = os.path.join(path, "seq_lite", "pseudo_color_prediction")
                src_name = f"{i}_frame"
                output_name  = f"{i}_frame"
                data_matrix[video_index, i, release_index] = segment_F1(stdpath, testpath, src_name, output_name)
    np.save(os.path.join(data_des_path, data_name + '_segmentation.npy'),data_matrix)


def main_detection(config_path='motivation_config.json'):
    config = load_config(config_path)
    source_dir = config["figure6_source_dir"]
    versions = config["figure6_versions"]
    video_names = config["figure6_video_names"]
    releases =  config["figure6_releases"]
    for file_name in video_names:
        for version_index, version in enumerate(versions):
            input_path = os.path.join(source_dir, file_name, releases[version_index])
            extract_frames(input_path, os.path.join(input_path, 'extract_frames'))
            run_yolo_imgs(os.path.join(input_path, 'extract_frames'), f"{file_name}_{version}", 's')
            run_yolo_imgs(os.path.join(input_path, 'extract_frames'), f"{file_name}_{version}", 'x')


def save_detection_data(config_path='motivation_config.json'):
    config = load_config(config_path)
    data_des_path = config["data_des_path"]
    video_names = config["figure6_video_names"]
    yolov5_free_viewpoint_project_dir = config['yolov5_free_viewpoint_project_dir']
    versions = config["figure6_versions"]
    label = int(config['people_label'])
    confidence_threshold = float(config['confidence_threshold'])
    data_name =  config["figure6_data_name"]
    data_matrix = np.zeros((len(video_names),12, len(versions),5))
    for video_index, file_name in enumerate(video_names):
        for i in range(12):
            for version_index, version in enumerate(versions):
                stdpath = os.path.join(yolov5_free_viewpoint_project_dir, f"{file_name}_{version}_x", "labels")
                testpath = os.path.join(yolov5_free_viewpoint_project_dir, f"{file_name}_{version}_s",  "labels")
                src_name = f"{i}_frame"
                output_name  = f"{i}_frame"
                print(file_name,i,version)
                data_matrix[video_index, i, version_index, :] = performance_accumulate(stdpath, testpath, src_name,
                    output_name, label=label, threshold=confidence_threshold)
    np.save(os.path.join(data_des_path, data_name + '_detect.npy'),data_matrix)

def run_data_figure6(config_path='motivation_config.json'):
    config = load_config(config_path)
    folder_path = config["figure6_source_dir"]
    subfolder_names = get_subfolder_names(folder_path)
    for file_name in subfolder_names:
        input_path = os.path.join(folder_path, file_name, 'RGB')
        output_path = os.path.join(folder_path, file_name, 'create1')
        figure6_resize_videos(input_path, output_path)
    main_detection()
    main_segmentation()

def save_data_figure6():
    save_detection_data()
    save_segmentation_data()


if __name__ == "__main__":

    # run_data_figure1()
    # run_data_figure2()
    # run_data_figure3()
    # run_data_figure4()
    # run_data_figure5()
    # run_data_figure6()

    # save_data_figure1()
    # save_data_figure2()
    # save_data_figure3()
    # save_data_figure4()
    # save_data_figure5()
    # save_data_figure6()
    pass