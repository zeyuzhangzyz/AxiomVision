import glob
import os
import time
import json
import argparse
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



# zeyuzhang 2024/2/25

def calculate_coco91_to_coco80_mapping():
    """
    Calculate mapping from COCO91 to COCO80
    """
    ROOT = "DNN/faster_rcnn"
    label_json_path = str(ROOT) + '/coco91_indices.json'
    with open(label_json_path, 'r') as f:
        coco91_dict = json.load(f)
    
    na_classes = set()
    for cls_id, cls_name in coco91_dict.items():
        if cls_name == "N/A":
            na_classes.add(int(cls_id))
    
    mapping = {}
    offset = 0
    for cls_id in range(1, 91):
        if cls_id in na_classes:
            offset += 1
            continue
        else:
            mapped_id = cls_id - offset - 1
            mapping[cls_id] = mapped_id
    return mapping

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


# zeyuzhang 2024/2/25

from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(ROOT)
# zeyuzhang 2024/2/25


def save_all_frames(video_folder, video_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
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
            
            frame_name = f"{os.path.splitext(video_file)[0]}_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            count += 1
        cap.release()
    # cv2.destroyAllWindows()
    return count
# zeyuzhang 2024/2/25

def save_txt_like_yolo(cls, xywh, conf, file, threshold, saveconf, target_cls=None):
    threshold = float(threshold)
    with open(file, 'w+') as f:
        for i in range(len(cls)):
            
            if conf[i] > threshold:
                if target_cls is not None and cls[i] not in target_cls:
                    continue
                    
                if saveconf:
                    line = (cls[i], *xywh[i], conf[i])
                else:
                    line = (cls[i], *xywh[i])
                f.write(('%g ' * len(line)).rstrip() % line + '\n')


def main():
    """
      zeyuzhang 2024/2/25
      add 6 new variables
      images_dir,dataset_type,project,name,videofile,save_img
    """
    parser = argparse.ArgumentParser(description="FRCNN Demo.")

    parser.add_argument("--images_dir", default=str(ROOT)+ '/pictures', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument('--project', default=str(ROOT)+  '/runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument("--videofile", default='none', type=str, help='video')
    parser.add_argument('--save_img', action='store_true', help='show results')
    parser.add_argument("--output_dir", default='', type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument("--threshold", default='0.25', type=str,
                        help='confidence threshold')
    parser.add_argument("--saveconf", action='store_true',
                        help='save confidence')
    parser.add_argument("--target_cls", type=str, default=None,
                        help='target classes to save, comma separated (e.g., "0,1,2" for person,bicycle,car)')
    args = parser.parse_args()

    args.output_dir = f'{args.project}/{args.name}'
    
    # 处理目标类别参数
    target_cls = None
    if args.target_cls is not None:
        target_cls = [int(x) for x in args.target_cls.split(',')]
        print(f"Only saving classes: {target_cls}")
    
    if args.videofile != 'none':
        video_folder, video_file = os.path.split(args.videofile)
        # frames_folder = f'{video_folder}/{video_file[:-4]}'
        # save_all_frames(video_folder, video_file, frames_folder)
        # args.images_dir = frames_folder

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=91)

    # zeyuzhang 2024/2/25
    # load train weights
    weights_path = str(ROOT)+ "/save_weights/fasterrcnn_resnet50_fpn_coco.pth"

    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = str(ROOT) + '/coco91_indices.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)
    # zeyuzhang 2024/2/25
    category_index = {str(v): str(k) for v, k in class_dict.items()}
    # zeyuzhang 2024/2/25
    # load image
    image_paths = glob.glob(os.path.join(args.images_dir, '*.jpg')) + glob.glob(os.path.join(args.images_dir, '*.png'))
    if not os.path.exists(args.project):
        os.mkdir(args.project)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # danger! change the resolution please if you want to use other images with different resolution.
    # zeyuzhang 2024/10/21
    original_img = Image.open(str(ROOT) +'/pictures/1_short_0.jpg')

    # from pil image to tensor, do not normalize image
    coco91to80 = calculate_coco91_to_coco80_mapping()
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()  
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

    total_time = 0
    with torch.no_grad():
        # zeyuzhang 2024/2/25
        
        if args.videofile == 'none':
            for i, image_path in enumerate(image_paths):
                t_start = time_synchronized()
                image_name = os.path.basename(image_path)
                original_img = Image.open(image_path)
                img_width, img_height = original_img.size
                # from pil image to tensor, do not normalize image
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)



                predictions = model(img.to(device))[0]
                t_end = time_synchronized()

                inference_time = t_end - t_start
                total_time += inference_time
                print("inference+NMS time: {}".format(t_end - t_start))

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                boxes = predict_boxes
                xywh = normalize_boxes(boxes, img_width, img_height)
                if not os.path.exists(f'{args.output_dir}/labels/'):
                    os.mkdir(f'{args.output_dir}/labels/')
                file = f'{args.output_dir}/labels/{image_name[:-4]}.txt'
                predict_classes = [coco91to80[cls] for cls in predict_classes]
                save_txt_like_yolo(predict_classes, xywh, predict_scores, file, args.threshold, args.saveconf, target_cls)



                if args.save_img:
                    plot_img = draw_objs(original_img,
                                         predict_boxes,
                                         predict_classes,
                                         predict_scores,
                                         category_index=category_index,
                                         box_thresh=0.5,
                                         line_thickness=3,
                                         font='arial.ttf',
                                         font_size=20)
                    # plt.imshow(plot_img)
                    # plt.show()
                    # Image.fromarray(plot_img).save(os.path.join(args.output_dir, image_name))
                    plot_img.save(os.path.join(args.output_dir, image_name))
            file_path = "faster_rcnn_time.txt"
            
            with open(file_path, "a+") as file_write:
                file_write.write(f"{total_time}, {args.images_dir}, faster_rcnn\n")
        else:
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
                    t_start = time_synchronized()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image
                    original_img = Image.fromarray(np.uint8(frame))
                    img_width, img_height = original_img.size

                    # from pil image to tensor, do not normalize image
                    data_transform = transforms.Compose([transforms.ToTensor()])
                    img = data_transform(original_img)
                    # expand batch dimension
                    img = torch.unsqueeze(img, dim=0)

                    predictions = model(img.to(device))[0]
                    t_end = time_synchronized()

                    # zeyuzhang 2024/2/25
                    
                    inference_time = t_end - t_start
                    total_time+=inference_time
                    print("inference+NMS time: {}".format(inference_time))
                    predict_boxes = predictions["boxes"].to("cpu").numpy()
                    predict_classes = predictions["labels"].to("cpu").numpy()
                    predict_scores = predictions["scores"].to("cpu").numpy()
                    boxes = predict_boxes
                    # if len(predict_boxes) == 0:
                    xywh = normalize_boxes(boxes, img_width, img_height)
                    if not os.path.exists(f'{args.output_dir}/labels/'):
                        os.mkdir(f'{args.output_dir}/labels/')
                    file = f'{args.output_dir}/labels/{video_file[:-4]}_{count}.txt'
                    predict_classes = [coco91to80[cls] for cls in predict_classes]
                    save_txt_like_yolo(predict_classes, xywh, predict_scores, file, args.threshold, args.saveconf, target_cls)
                    if args.save_img:

                        plot_img = draw_objs(original_img,
                                             predict_boxes,
                                             predict_classes,
                                             predict_scores,
                                             category_index=category_index,
                                             box_thresh=0.5,
                                             line_thickness=3,
                                             font='arial.ttf',
                                             font_size=20)
                        plt.imshow(plot_img)
                        plt.show()
                        plot_img.save(os.path.join(args.output_dir, f'{video_file[:-4]}_{count}'))
                    count+= 1
                cap.release()
            # zeyuzhang 2024/2/25
            file_path = "faster_rcnn_time.txt"
            with open(file_path, "a+") as file_write:
                file_write.write(f"{total_time}, {video_file[:-4]}, faster_rcnn\n")
if __name__ == '__main__':
    main()
    # zeyuzhang 2024/2/25
    ## python DNN/faster_rcnn/predict.py  --project DNN/faster_rcnn/runs/  --name 003 --videofile D/0_10.mp4
