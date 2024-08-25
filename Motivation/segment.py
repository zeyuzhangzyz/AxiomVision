import cv2
import os
import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import subprocess
from Base.video import *
from Base.performance import *



def calculate_average_jpg_size(folder_path):
    total_size = 0
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_count += 1

    if file_count > 0:
        average_size = total_size / file_count
    else:
        average_size = 0

    return average_size


def main(config_path='motivation_config.json'):
    config = load_config(config_path)
    pans = config["pans"]
    tilts = config["tilts"]
    brightness_descriptions = config["brightness_descriptions"]
    gamma_values = config["gamma_values"]
    video_names = config["free_viewpoint_video_names"]
    video_src_path = config["free_viewpoint_video_src_path"]
    video_des_path = config["free_viewpoint_video_des_path"]
    img_des_path = config["free_viewpoint_img_des_path"]

    video_names_short = [f"{video_name}_short" for video_name in video_names]

    # for video_name in video_names:
    #     set_video_frame(video_src_path, video_name , video_des_path, f"{video_name}_short", 3000)
    #     extract_frames_single(video_des_path, f"{video_name}_short.mp4",img_des_path, f"{video_name}_short")
    tasks = []
    for brightness, gamma in zip(brightness_descriptions, gamma_values):
        for video_name in video_names_short:
            for pan in pans:
                for tilt in tilts:
                    video_folder = os.path.join(img_des_path, video_name)
                    output_folder = os.path.join(img_des_path, f'{video_name}_{brightness}_{pan}_{tilt}')
                    tasks.append((video_folder, output_folder, gamma, pan, tilt))

    # Use ProcessPoolExecutor to process tasks in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = {executor.submit(process_vr_brightness_images, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(f"Task {task} completed successfully.")
            except Exception as e:
                print(f"Task {task} generated an exception: {e}")

    # img_size = np.zeros((len(video_names_short),len(gamma_values), len(pans),len(tilts)))
    # for i in range(len(video_names_short)):
    #     for env_light_index, env_light in enumerate(brightness_descriptions):
    #         for pan_index,pan in enumerate(pans):
    #             for tilt_index, tilt in enumerate(tilts):
    #                 target_images_dir = os.path.join(img_des_path, env_light, f'{video_name}_{pan}_{tilt}')
    #                 img_size[i,env_light_index,pan_index,tilt_index] = calculate_average_jpg_size(target_images_dir)/ 1024
    # np.save('E:/dataset/VR_youtube/img_size.npy', img_size)
if __name__ == '__main__':
    main()