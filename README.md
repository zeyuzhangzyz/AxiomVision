# AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics


This repository contains the source code for reproducing the results of our paper titled **AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics**, which is accepted by ACM MM 2024.

## 🔧Prerequisites

To get started, ensure that you can run multiple target detection models (such as YOLOV5, SSD, etc.) or other computer vision tasks, and importantly, modify their source code to ensure that the output TXT detection results have the same format.

For example, in the output TXT file of YOLOv5, if confidence is included, the format is as follows: First column: class label, Second column: normalized center x-coordinate, Third column: normalized center y-coordinate, Fourth column: normalized width, Fifth column: normalized height, Sixth column: confidence score. In contrast, MMdet's format outputs a JSON file for each image frame. This JSON file contains a dictionary with three keys: labels, scores, and bboxes. The values of bboxes are the coordinates before normalization and sequentially represent the x and y coordinates of the top-left corner, followed by the x and y coordinates of the bottom-right corner.

We modified the source code of these programs to standardize the output file format by altering the way the results are saved. 


We also used the Paddle segmentation model and created an environment named "Paddle" You only need to prepare this setup if you want to test segmentation tasks.

## 📚 Dataset: 
We searched for online traffic cameras on YouTube and found four different views monitoring the same intersection. We saved scenes from various environments, including daytime, dusk, night, and snowy conditions.



|  Source  | Type    | URL                                                         |
|---------|---------|-------------------------------------------------------------|
| Youtube | Videos | https://www.youtube.com/watch?v=1EiC9bvVGnk |
| Youtube | Videos | https://www.youtube.com/watch?v=FmoclK_hKz8 |
| Youtube | Videos | https://www.youtube.com/watch?v=BN7gzH-i-zo |
| Youtube | Videos | https://www.youtube.com/watch?v=Zj0pXlq2-jI |
| Github |   Videos  | https://github.com/sjtu-medialab/Free-Viewpoint-RGB-D-Video-Dataset |
| Coco128 Dataset | Images  | https://ultralytics.com/assets/coco128.zip |


## 👉 Code Structure

```
.
├── README.md
├── Motivation
│   ├── Base
│   │   ├── video.py
│   │   └── performance.py
│   ├── data
│   │   ├── figure1_data.npy
│   │   ├── ...
│   │   └── figure6_data_segmentation.npy
│   ├── figure4_src_img
│   ├── figure5_src_img
│   ├── motivation_config.json
│   ├── motivation_dataprocess.py
│   └── motivation_plot.py
└── Experiment (cleaning up, coming soon)

```

## 🚀How to Run

Execute the following command to reproduce the figures in the paper:

Motivation:
1. set up your enviroument
2. prepare your dataset (if need)
3. python motivation_plot.py (activate function you want to run)
4. python motivation_dataprocess.py (activate function you want to run)

