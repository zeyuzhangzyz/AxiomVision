import glob
import os
import time

import cv2
import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
from pathlib import Path
import sys

# zeyuzhang 2024/2/25

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# zeyuzhang 2024/2/25

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
    # cv2.destroyAllWindows()
    return true_count

# zeyuzhang 2024/2/25

def save_all_frames(video_folder, videofile, output_folder):
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
            
            frame_name = f"{os.path.splitext(videofile)[0]}_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            count += 1
        cap.release()
    # cv2.destroyAllWindows()
    return count

# zeyuzhang 2024/2/25

def save_txt_like_yolo(cls, xywh, conf, file):
    
    with open(file, 'w+') as f:
        for i in range(len(cls)):
            if cls[i] == 1 or cls[i] == 3:
                line = (cls[i]-1, *xywh[i], conf[i])
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
# zeyuzhang 2024/2/25

def normalize_boxes(boxes, width, height):
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  

    
    normalized_xywh = np.zeros_like(xywh, dtype=float)
    normalized_xywh[:, 0] = xywh[:, 0] / width
    normalized_xywh[:, 1] = xywh[:, 1] / height
    normalized_xywh[:, 2] = xywh[:, 2] / width
    normalized_xywh[:, 3] = xywh[:, 3] / height

    return normalized_xywh



@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type, save_img, video_file):

    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)
    
    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)
    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    # zeyuzhang 2024/2/25
    

    video_folder, videofile = os.path.split(video_file)
    # zeyuzhang 2024/2/25
    
    for i, image_path in enumerate(image_paths):
        image = np.array(Image.open(image_path).convert("RGB"))
        images = transforms(image)[0].unsqueeze(0)
        result = model(images.to(device))[0]
        break
    total_time = 0
    if videofile == 'none':
        for i, image_path in enumerate(image_paths):
            start = time.time()

            image_name = os.path.basename(image_path)

            image_bgr = cv2.imread(image_path)
     
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0)
            load_time = time.time() - start

            start = time.time()

            result = model(images.to(device))[0]
            inference_time = time.time() - start
            total_time += inference_time
            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            meters = ' | '.join(
                [
                    'objects {:02d}'.format(len(boxes)),
                    'load {:03d}ms'.format(round(load_time * 1000)),
                    'inference {:03d}ms'.format(round(inference_time * 1000)),
                    'FPS {}'.format(round(1.0 / inference_time))
                ]
            )
            print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))
            
            xywh = normalize_boxes(boxes, width, height)

            if not os.path.exists(f'{output_dir}/labels/'):
                os.mkdir(f'{output_dir}/labels/')
            file = f'{output_dir}/labels/{image_name[:-4]}.txt'
            save_txt_like_yolo(labels, xywh, scores, file)
            if save_img:
                drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
                Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
        
        file_path = "ssd_time.txt"
        
        with open(file_path, "a+") as file:
            file.write(f"{total_time}, {image_name}, ssd\n")
    else:
        
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
                start = time.time()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                images = transforms(image)[0].unsqueeze(0)
                load_time = time.time() - start
                # print(load_time)
                start = time.time()

                result = model(images.to(device))[0]
                inference_time = time.time() - start
                # print(inference_time)
                total_time += inference_time
                result = result.resize((width, height)).to(cpu_device).numpy()
                boxes, labels, scores = result['boxes'], result['labels'], result['scores']

                indices = scores > score_threshold
                boxes = boxes[indices]
                labels = labels[indices]
                scores = scores[indices]
                meters = ' | '.join(
                    [
                        'objects {:02d}'.format(len(boxes)),
                        'load {:03d}ms'.format(round(load_time * 1000)),
                        'inference {:03d}ms'.format(round(inference_time * 1000)),
                        'FPS {}'.format(round(1.0 / inference_time))
                    ]
                )
                # videofile[:-4] means the video name without the extension(.mp4)
                print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), f'{videofile[:-4]}_{count}', meters))

                xywh = normalize_boxes(boxes, width, height)
                if not os.path.exists(f'{output_dir}/labels/'):
                    os.mkdir(f'{output_dir}/labels/')
                file = f'{output_dir}/labels/{videofile[:-4]}_{count}.txt'
                save_txt_like_yolo(labels, xywh, scores, file)
                if save_img:
                    drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
                    Image.fromarray(drawn_image).save(os.path.join(output_dir, f'{videofile[:-4]}_{count}'))

                count += 1
            cap.release()

            # zeyuzhang 2024/2/25
            
            
            file_path = "ssd_time.txt"
            
            with open(file_path, "a+") as file:
                file.write(f"{total_time}, {os.path.basename(video_file)}, ssd\n")


def main():
    """
    zeyuzhang 2024/2/25
    Modified config-file default settings and added 5 new variables:
    dataset_type, project, name, videofile, save_img
    """
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default=str(ROOT)+"/configs/vgg_ssd512_coco_trainval35k.yaml",
        metavar="FILE",
       help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=str(ROOT)+'/model/vgg_ssd512_coco_trainval35k.pth', help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=.25)
    parser.add_argument("--images_dir", default=str(ROOT)+ '/demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--dataset_type", default="coco", type=str, help='Specify dataset type. Currently support voc and coco.')
    parser.add_argument('--project', default=str(ROOT)+  '/runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument("--output_dir", default='', type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument("--videofile", default='none', type=str, help='video')
    parser.add_argument('--save_img', action='store_true', help='show results')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.output_dir = f'{args.project}/{args.name}'
    print(args.output_dir)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type,
             save_img = args.save_img,
             video_file = args.videofile)


if __name__ == '__main__':
    main()
    # zeyuzhang 2024/2/25
    
   ### python DNN/SSD/demo.py  --project DNN/SSD/runs/  --name 002 --images_dir D:/code/PycharmProjects/MM/source/0_10
   ##  python DNN/SSD/demo.py  --project DNN/SSD/runs/  --name 002 --videofile D:/code/PycharmProjects/MM/source/0_10.mp4

