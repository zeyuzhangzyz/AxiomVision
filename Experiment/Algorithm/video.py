import os
import cv2
import subprocess
import json
import yaml
from concurrent.futures import ProcessPoolExecutor
import shutil
import numpy as np
def run_yolo_imgs(source_dir, des_name, version, config_path='experiment_prepare_config.json'):
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

def extract_frames_single(video_folder, video_file, output_folder, img_path):
    """
    Extract frames from videos in the given folder and save them to the output folder.
    """
    ensure_dir(output_folder)
    ensure_dir(os.path.join(output_folder,img_path))

    if video_file.endswith(".mp4") or video_file.endswith(".avi"):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        true_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            interval = int(fps)  # Extract one frame per second
            if count % interval == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame_{true_count}.jpg"
                cv2.imwrite(os.path.join(output_folder,img_path,frame_name), frame)
                true_count += 1
            count += 1
        cap.release()
    return true_count

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

def load_config(config_path='experiment_prepare_config.json'):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def get_subfolder_names(folder_path):
    subfolders = [os.path.basename(f.path) for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def only_change_resolution(input_path, output_path, new_width, new_height):
    command = ['ffmpeg', '-i', input_path, '-vf', 'scale={}:{}'.format(new_width, new_height), '-an',
               output_path]
    subprocess.call(command)

def ssd(source_dir, video_names, config_path='experiment_prepare_config.json'):
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

def faster_rcnn(source_dir, video_names, config_path='experiment_prepare_config.json'):
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

def mmdetection(source_dir, video_names, config_path='experiment_prepare_config.json'):
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

def ssd_images(source_dir, video_names, config_path='experiment_prepare_config.json'):
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
                "--images_dir",
                os.path.join(source_dir, video_name)
            ]
        )

def faster_rcnn_images(source_dir, video_names, config_path='experiment_prepare_config.json'):
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
                "--images_dir",
                os.path.join(source_dir, video_name),
                "--threshold",
                confidence_threshold,
            ]
        )

def mmdetection_images(source_dir, video_names, config_path='experiment_prepare_config.json'):
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

def yolo_train(epoch,version, config_path='experiment_prepare_config.json'):
    config = load_config(config_path)
    yolov5_train_script_dir = config["yolov5_train_script_dir"]
    yolov5_train_config_dir = config["yolov5_train_config_dir"]
    yolov5_base_weight_dir = config["yolov5_base_weight_dir"]
    command = [
        'python', 
        yolov5_train_script_dir, 
        '--img', '640', 
        '--epochs', f"{epoch}", 
        '--batch', '16',
        '--weights', yolov5_base_weight_dir, 
        '--cache', 
        '--data', os.path.join(yolov5_train_config_dir, f'{version}.yaml'), 
        '--name', f'{version}_{epoch}'
    ]
    subprocess.run(command)

def yolov5_retrained(source_dir, video_names, version, config_path='experiment_prepare_config.json'):
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    train_weights_dir = config['train_weights_dir']
    yolov5_project_dir = config['yolov5_project_dir']

    for video_name in video_names:
        video_path = os.path.join(source_dir, f"{video_name}.mp4")
        if os.path.exists(video_path):
            source_path = video_path
        else:
            source_path = os.path.join(source_dir, video_name)

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
                source_path,
                "--nosave",
                "--save-txt",
                "--save-conf",
            ]
        )

def yolov5(source_dir, video_names, version, config_path='experiment_prepare_config.json'):
    # Load configuration from JSON file
    config = load_config(config_path)
    yolov5_script_path = config['yolov5_script_path']
    weights_dir = config['weights_dir']
    yolov5_project_dir = config['yolov5_project_dir']

    for video_name in video_names:
        video_path = os.path.join(source_dir, f"{video_name}.mp4")
        if os.path.exists(video_path):
            source_path = video_path
        else:
            source_path = os.path.join(source_dir, video_name)
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
                source_path,
                "--nosave",
                "--save-txt",
                "--save-conf",
            ]
        )

def get_txt_path(video_name, dnn, version, config_path='experiment_prepare_config.json', is_free_viewpoint = 0):
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

def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def add_gaussian_noise(image, mean=0, std=10):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_image(img_path, target_dir, gamma):
    img = cv2.imread(img_path)
    adjusted = adjust_gamma(img, gamma)
    noisy = add_gaussian_noise(adjusted)
    cv2.imwrite(os.path.join(target_dir, os.path.basename(img_path)), noisy)

def copy_labels(source_labels_dir, target_labels_dir):
    ensure_dir(target_labels_dir)
    for label_name in os.listdir(source_labels_dir):
        src_path = os.path.join(source_labels_dir, label_name)
        dst_path = os.path.join(target_labels_dir, label_name)
        shutil.copy(src_path, dst_path)

def process_images_with_gamma_and_noise(source_dir, target_dir, gamma):
    ensure_dir(target_dir)
    with ProcessPoolExecutor() as executor:
        for img_name in os.listdir(source_dir):
            img_path = os.path.join(source_dir, img_name)
            executor.submit(process_image, img_path, target_dir, gamma)
    print(f"Images processed with gamma {gamma} and noise added.")

def update_and_save_yaml(yolov5_train_config_dir, base_name, new_name, new_path):
    with open(os.path.join(yolov5_train_config_dir, base_name+".yaml"), 'r') as file:
        data = yaml.safe_load(file)
    data['path'] = f'{new_path}'
    yaml_new_path = os.path.join(yolov5_train_config_dir, new_name+".yaml")
    with open(yaml_new_path, 'w') as new_file:
        yaml.dump(data, new_file, sort_keys=False)
    print("YAML files updated and saved.")

def split_filename(filename):
    frame_index = filename.find('_frame_')

    if frame_index == -1:
        return None, None

    before_frame = filename[:frame_index]
    after_frame = filename[frame_index:]

    return before_frame, after_frame
# VR
def radians(degrees):
    return degrees * np.pi / 180

class Equi2Rect:
    def __init__(self, pan, tilt):
        self.w = 1280
        self.h = 720
        self.yaw = radians(pan)
        self.pitch = radians(tilt)
        self.roll = radians(0.0)
        self.Rot = self.eul2rotm(self.pitch, self.yaw, self.roll)
        self.f = 800
        self.K = np.array([[self.f, 0, self.w / 2],
                           [0, self.f, self.h / 2],
                           [0, 0, 1]])
        self.img_interp = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def eul2rotm(self, rotx, roty, rotz):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rotx), -np.sin(rotx)],
                        [0, np.sin(rotx), np.cos(rotx)]])

        R_y = np.array([[np.cos(roty), 0, np.sin(roty)],
                        [0, 1, 0],
                        [-np.sin(roty), 0, np.cos(roty)]])

        R_z = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                        [np.sin(rotz), np.cos(rotz), 0],
                        [0, 0, 1]])
        R = R_z.dot(R_y).dot(R_x)
        return R

    def set_image(self, img):
        self.img_src = img
        self.img_interp = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def vectorized_reprojection(self):
        x_img, y_img = np.meshgrid(np.arange(self.w), np.arange(self.h))
        xyz = np.stack([x_img.flatten(), y_img.flatten(), np.ones_like(x_img).flatten()], axis=-1)
        xyz_norm = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

        RK = self.Rot @ np.linalg.inv(self.K)
        ray3d = RK @ xyz_norm.T

        xp, yp, zp = ray3d
        theta = np.arctan2(yp, np.sqrt(xp ** 2 + zp ** 2))
        phi = np.arctan2(xp, zp)

        x_sphere = ((phi + np.pi) * self.img_src.shape[1] / (2 * np.pi)).reshape(self.h, self.w)
        y_sphere = ((theta + np.pi / 2) * self.img_src.shape[0] / np.pi).reshape(self.h, self.w)

        return x_sphere, y_sphere


    def perform_interpolation(self):
        x_sphere, y_sphere = self.vectorized_reprojection()
        map_x = np.float32(x_sphere)
        map_y = np.float32(y_sphere)
        self.img_interp = cv2.remap(self.img_src, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def adjust_gamma(self, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        self.img_interp = cv2.LUT(self.img_interp, table)

    def add_gaussian_noise(self, mean=0, std=10):
        row, col, ch = self.img_interp.shape
        gauss = np.random.normal(mean, std, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = self.img_interp + gauss
        noisy = np.clip(noisy, 0, 255).astype('uint8')
        self.img_interp = noisy



def process_vr_images(input_folder, output_folder, pan, tilt):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    equi2rect = Equi2Rect(pan, tilt)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            before_frame, after_frame = split_filename(filename)
            output_path = os.path.join(output_folder, f"{before_frame}_{pan}_{tilt}{after_frame}" )

            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Could not load image {input_path}!")
                continue

            equi2rect.set_image(img)
            equi2rect.perform_interpolation()
            cv2.imwrite(output_path, equi2rect.img_interp)
    print("All images processed.")

def process_vr_brightness_images(input_folder, output_folder, gamma, pan, tilt):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    equi2rect = Equi2Rect(pan, tilt)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            before_frame, after_frame = split_filename(filename)
            output_path = os.path.join(output_folder, f"{before_frame}_{pan}_{tilt}{after_frame}" )

            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Could not load image {input_path}!")
                continue

            equi2rect.set_image(img)
            equi2rect.perform_interpolation()
            equi2rect.adjust_gamma(gamma)
            equi2rect.add_gaussian_noise()

            cv2.imwrite(output_path, equi2rect.img_interp)
    print("All images processed.")
