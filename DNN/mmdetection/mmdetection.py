import subprocess
# for name in names:
import json
import argparse
from pathlib import Path
import sys
import os
import time
import random
import numpy as np
from PIL import Image
import shutil
import cv2


def mmdet(images_dir, output_dir, save_img, threshold):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if save_img == '':
        subprocess.run(
            [
                "python",
                "DNN/mmdetection/demo/image_demo.py",
                f"{images_dir}",
                "DNN/mmdetection/rtmdet_tiny_8xb32-300e_coco.py",
                "--weights",
                "DNN/mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
                "--device",
                "cuda",
                "--out-dir",
                f"{output_dir}",
                "--conf-score-thr",
                f"{threshold}",
            ]
        )
    else:
        subprocess.run(
            [
                "python",
                "DNN/mmdetection/demo/image_demo.py",
                f"{images_dir}",
                "DNN/mmdetection/rtmdet_tiny_8xb32-300e_coco.py",
                "--weights",
                "DNN/mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
                "--device",
                "cuda",
                "--out-dir",
                f"{output_dir}",
                "--no-save-vis",
                "--conf-score-thr",
                f"{threshold}",
            ]
        )

def normalize_boxes(boxes, width, height):
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # Center X coordinate
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # Center Y coordinate
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # Width
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # Height

    # Normalize coordinates
    normalized_xywh = np.zeros_like(xywh, dtype=float)
    normalized_xywh[:, 0] = xywh[:, 0] / width
    normalized_xywh[:, 1] = xywh[:, 1] / height
    normalized_xywh[:, 2] = xywh[:, 2] / width
    normalized_xywh[:, 3] = xywh[:, 3] / height

    return normalized_xywh

def save_txt_like_yolo(labels, xywh, scores, file_name,threshold):
    with open(file_name, 'w') as f:
        for i, label in enumerate(labels):
            if (label == 0 or label == 2) and scores[i]>threshold:  
                line = (label, *xywh[i], scores[i])
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

def json2txt(json_dir, output, name, width, height,threshold):
    threshold = float(threshold)
    labels_dir = os.path.join(output, name, 'labels')
    if os.path.exists(labels_dir):
        # If the directory exists, delete it and its contents
        shutil.rmtree(labels_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    num_files = len(json_files)

    for json_file in json_files:
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
        file_name = os.path.splitext(json_file)[0]
        labels = data['labels']
        scores = data['scores']
        bboxes = np.array(data['bboxes'])
        normalized_bboxes = normalize_boxes(bboxes, width, height)

        file = f'{output}/{name}/labels/{file_name}.txt'
        save_txt_like_yolo(data['labels'], normalized_bboxes, data['scores'], file,threshold)

def save_selected_frames(video_folder, videofile, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if videofile.endswith(".mp4") or videofile.endswith(".avi"):  
        video_path = os.path.join(video_folder, videofile)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        true_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            interval = int(fps)  
            if count % interval == 0:
                frame_name = f"{os.path.splitext(videofile)[0]}_frame_{true_count}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                true_count += 1
            count += 1
        cap.release()
    return true_count

def save_all_frames(video_folder, videofile, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    if videofile.endswith(".mp4") or videofile.endswith(".avi"):  
        video_path = os.path.join(video_folder, videofile)
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_name = f"{os.path.splitext(videofile)[0]}_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            count += 1
        cap.release()
    return count

def main():
    """
    zeyuzhang 2024/2/25
    add video processing
    """
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    parser = argparse.ArgumentParser(description="mmdetection")
    parser.add_argument("--images_dir", default='DNN/mmdetection/demo', type=str,
                        help='Specify an image dir to do prediction.')
    parser.add_argument("--name", default='', type=str, help='save results to project/name.')
    parser.add_argument('--project', default='DNN/mmdetection/runs',
                        help='Save results to project/name')
    parser.add_argument("--output", default='DNN/mmdetection/runs', type=str,
                        help='Specify a dir to save txts.')
    parser.add_argument("--threshold", default='0.25', type=str,
                        help='confidence threshold')
    parser.add_argument("--only_change_confidence",action='store_true', help='confidence threshold')
    parser.add_argument('--save_img', default='no', type=str, help='show results')
    parser.add_argument("--videofile", default='none', type=str, help='video file path')
    
    args = parser.parse_args()
    if not os.path.exists(args.project):
        os.makedirs(args.project, exist_ok=True)
    output_dir = f'{args.project}/{args.name}'

    if args.videofile != 'none':
        video_folder, videofile = os.path.split(args.videofile)
        temp_frames_dir = os.path.join(output_dir, 'temp_frames')
        save_all_frames(video_folder, videofile, temp_frames_dir)
        args.images_dir = temp_frames_dir

    if not args.only_change_confidence:
        mmdet(args.images_dir, output_dir, args.save_img, args.threshold)
    
    json_dir = f'{args.project}/{args.name}/preds'

    image_files = [f for f in os.listdir(args.images_dir) if f.endswith('.jpg')]
    if not image_files:
        print("No jpg files found in the specified directory.")
        return
        
    random_image = random.choice(image_files)
    image_path = os.path.join(args.images_dir, random_image)
    image = Image.open(image_path)
    width, height = image.size
    
    json2txt(json_dir, args.output, args.name, width, height, args.threshold)

    if args.videofile != 'none':
        shutil.rmtree(temp_frames_dir)

if __name__ == '__main__':
    main()
    
