import os
import cv2
import subprocess
import json

def run_yolo_imgs(source_dir, des_name, version, config_path='motivation_config.json'):
    # Load configuration from JSON file
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    weights_dir = config['weights_dir']
    yolov5_free_viewpoint_project_dir = config['yolov5_free_viewpoint_project_dir']
    subprocess.run(
        [
            "python",
            yolov5_script_path,
            "--weights",
            os.path.join(weights_dir, f"yolov5{version}.pt"),
            "--project",
            yolov5_free_viewpoint_project_dir,
            "--name",
            f"{des_name}_{version}",
            "--source",
            f"{source_dir}",
            "--nosave",
            "--save-txt",
            "--save-conf",
        ]
    )


def extract_frames(video_folder, output_folder):
    """
    Extract frames from videos in the given folder and save them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        for video_file in os.listdir(video_folder):
            if video_file.endswith(".mp4") or video_file.endswith(".avi"):
                video_path = os.path.join(video_folder, video_file)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = 0
                true_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    interval = int(fps)  # Extract one frame per second
                    if count % interval == 0:
                        frame_name = f"{os.path.splitext(video_file)[0]}_frame_{true_count}.jpg"
                        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                        true_count += 1
                    count += 1
                cap.release()
        return true_count

def set_video_frame(src_path, src_name, des_path, des_name, frame_number, frame_begin_number=0):
    """
    Split a video from a specified frame and save it to another folder.
    """
    video = os.path.join(src_path, f"{src_name}.mp4")
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    videowriter = cv2.VideoWriter(os.path.join(des_path, f"{des_name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),
                                  frame_count, (frame_width, frame_height))

    success, _ = cap.read()
    count = -frame_begin_number

    while success:
        success, img = cap.read()
        if count >= 0:
            videowriter.write(img)
        if count == frame_number - 1:
            break
        count += 1

def set_video_segment(src_path, src_name, des_path, des_name, frame_number, segment_number, frame_begin_number=0):
    """
    Split a video from a specified frame and save it to multiple segments.
    """
    if not os.path.exists(os.path.join(des_path, f"{des_name}_Segment_{segment_number - 1}.mp4")):
        video = os.path.join(src_path, f"{src_name}.mp4")
        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        success, _ = cap.read()
        count = -frame_begin_number

        if frame_begin_number != 0:
            while success:
                success, _ = cap.read()
                if count >= 0:
                    break
        for segment in range(segment_number):
            videowriter = cv2.VideoWriter(os.path.join(des_path, f"{des_name}_Segment_{segment}.mp4"),
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            while success:
                success, img = cap.read()
                videowriter.write(img)
                count += 1
                if count == (segment + 1) * frame_number:
                    break

def set_video_frame_rate(src_path, src_name, des_path, des_name, fps_list):
    """
    Adjust the frame rate of a video and save to a new file.
    """
    video = os.path.join(src_path, f"{src_name}.mp4")
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print(f"Error opening video file {video}")
        return

    src_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for target_fps in fps_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if target_fps <= 0:
            print("Target FPS must be greater than 0.")
            continue

        frame_interval = src_fps / target_fps
        frame_list = [round(i * frame_interval) for i in range(target_fps)]

        out_video = os.path.join(des_path, f"{des_name}_fps_{target_fps}.mp4")
        videowriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), target_fps,
                                      (frame_width, frame_height))

        if not videowriter.isOpened():
            print(f"Error opening video writer for {out_video}")
            continue

        success, img = cap.read()
        frame_id = 0

        while success and frame_interval * frame_id <= src_fps:
            if frame_id in frame_list:
                videowriter.write(img)

            success, img = cap.read()
            frame_id += 1

        videowriter.release()

    cap.release()

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


def load_config(config_path='motivation_config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def get_subfolder_names(folder_path):
    subfolders = [os.path.basename(f.path) for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def only_change_resolution(input_path, output_path, new_width, new_height):
    command = ['ffmpeg', '-i', input_path, '-vf', 'scale={}:{}'.format(new_width, new_height), '-an',
               output_path]
    subprocess.call(command)



def ssd(source_dir, video_names, config_path='motivation_config.json'):
    config = load_config(config_path)
    ssd_script_path = config['ssd_script_path']
    ssd_project_dir = config['ssd_project_dir']

    for video_name in video_names:
        subprocess.run(
            [
                "python",
                ssd_script_path,
                "--project",
                ssd_project_dir,
                "--name",
                f"{str(video_name)}",
                "--videofile",
                os.path.join(source_dir, f"{video_name}.mp4"),
            ]
        )

def faster_rcnn(source_dir, video_names, config_path='motivation_config.json'):
    config = load_config(config_path)
    faster_rcnn_script_path = config['faster_rcnn_script_path']
    faster_rcnn_project_dir = config['faster_rcnn_project_dir']
    confidence_threshold = config['confidence_threshold']

    for video_name in video_names:
        subprocess.run(
            [
                "python",
                faster_rcnn_script_path,
                "--project",
                faster_rcnn_project_dir,
                "--name",
                f"{video_name}",
                "--videofile",
                os.path.join(source_dir, f"{video_name}.mp4"),
                "--threshold",
                confidence_threshold,
            ]
        )

def mmdetection(source_dir, video_names, config_path='motivation_config.json'):
    config = load_config(config_path)
    mmdetection_script_path = config['mmdetection_script_path']
    mmdetection_project_dir = config['mmdetection_project_dir']
    confidence_threshold = config['confidence_threshold']

    for video_name in video_names:
        subprocess.run(
            [
                "python",
                mmdetection_script_path,
                "--images_dir",
                os.path.join(source_dir, f"{video_name}"),
                "--name",
                f"{video_name}",
                "--output",
                mmdetection_project_dir,
                "--threshold",
                confidence_threshold,
                # "--only_change_confidence"
            ]
        )

def yolo_retrain(source_dir, video_names, version, config_path='motivation_config.json'):
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    train_weights_dir = config['train_weights_dir']
    yolov5_project_dir = config['yolov5_project_dir']

    for video_name in video_names:
        subprocess.run(
            [
                "python",
                yolov5_script_path,
                "--weights",
                os.path.join(train_weights_dir, version, "weights", "best.pt"),
                "--project",
                yolov5_project_dir,
                "--name",
                f"{str(video_name)}_{version}",
                "--source",
                os.path.join(source_dir, f"{video_name}.mp4"),
                "--nosave",
                "--save-txt",
                "--save-conf",
            ]
        )

def yolov5(source_dir, video_names, version, config_path='motivation_config.json'):
    # Load configuration from JSON file
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    weights_dir = config['weights_dir']
    yolov5_project_dir = config['yolov5_project_dir']

    for video_name in video_names:
        subprocess.run(
            [
                "python",
                yolov5_script_path,
                "--weights",
                os.path.join(weights_dir, f"yolov5{version}.pt"),
                "--project",
                yolov5_project_dir,
                "--name",
                f"{str(video_name)}_{version}",
                "--source",
                os.path.join(source_dir, f"{video_name}.mp4"),
                "--nosave",
                "--save-txt",
                "--save-conf",
            ]
        )

def get_txt_path(video_name, dnn, version, config_path='motivation_config.json', is_free_viewpoint = 0):
    config = load_config(config_path)
    if is_free_viewpoint == 0 :
        project_dirs = {
            'yolov5': config['yolov5_project_dir'],
            'ssd': config['ssd_project_dir'],
            'faster_rcnn': config['faster_rcnn_project_dir'],
            'mmdet': config['mmdetection_project_dir']
        }
    else:
        project_dirs = {
            'yolov5': config['yolov5_free_viewpoint_project_dir'],
            'ssd': config['ssd_project_dir'],
            'faster_rcnn': config['faster_rcnn_project_dir'],
            'mmdet': config['mmdetection_project_dir']
        }

    if dnn == 'yolov5' and version is not None:
        txt_path = os.path.join(project_dirs[dnn], f"{video_name}_{version}", "labels")
    else:
        txt_path = os.path.join(project_dirs[dnn], video_name, "labels")
    return txt_path




