# AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics

This repository contains the source code for reproducing the results of our paper titled [**AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics**](https://arxiv.org/abs/2407.20124), which is accepted by ACM MM 2024.

![](design.png)

AxiomVision is a general online streaming framework that adaptively selects the most effective visual model based on the end-edge-cloud architecture to ensure the accuracy of video stream analysis in diverse scenarios.

## 🔧Prerequisites

This project integrates multiple computer vision models and standardizes their output formats for consistent evaluation and comparison.

### Output Format Standardization

All models have been modified to output detection results in a unified YOLO-style format:
- Class label (0-based index)
- Normalized center x-coordinate
- Normalized center y-coordinate
- Normalized width
- Normalized height
- Confidence score (optional)

### Integrated Models

1. **Faster R-CNN**
   - Source: [deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
   - Modifications:
     - Standardized output format in `predict.py`
     - Added image/video support
     - Implemented model warm-up

2. **YOLOv5**
   - Source: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
   - Modifications:
     - Enhanced `detect.py` for result saving
     - Added inference time tracking

3. **SSD**
   - Source: [lufficc/SSD](https://github.com/lufficc/SSD)
   - Modifications:
     - Standardized output format in `demo.py`
     - Added image/video support

4. **PaddleSeg**
   - Source: [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
   - Setup:
     - Pre-trained models available in `DNN/PaddleSeg/pretrained_models`
     - Requires "Paddle" environment for segmentation tasks to avoid the difference with other models.

5. **MMDetection**
   - Source: [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
   - Modifications:
     - Created `mmdetection.py` for standardized output
     - Enhanced `demo/image_demo.py` and `apis/det_inferencer.py`
     - Added score threshold parameter

### Note

The original model outputs have been modified to ensure consistent evaluation. For example:
- YOLOv5: Modified TXT output format
- MMDetection: Converted JSON output to standardized TXT format
- All models: Normalized coordinates and unified class indexing

Three large models are not included in this repository due to their size. You can download them from the following link based on the path in the [link](https://drive.google.com/drive/folders/1lRYIBUrhWHA8jWTqQOIZd1hcsHoIwXMR?usp=drive_link). Please move the downloaded models to the corresponding folder in the repository. Such as 'yolov5x.pt' is in 'DNN/yolov5/' in the link, so please make sure the path of the model is 'DNN/yolov5/yolov5x.pt'.


## 📚 Dataset: 
We searched for online traffic cameras on YouTube and found four different views monitoring the same intersection. We saved scenes from various environments, including daytime, dusk, night, and snowy conditions.



|  Source  | Type    | URL                                                         |
|---------|---------|-------------------------------------------------------------|
| Youtube | Videos | https://www.youtube.com/watch?v=1EiC9bvVGnk |
| Youtube | Videos | https://www.youtube.com/watch?v=FmoclK_hKz8 |
| Youtube | Videos | https://www.youtube.com/watch?v=BN7gzH-i-zo |
| Youtube | Videos | https://www.youtube.com/watch?v=Zj0pXlq2-jI |
| Github | Videos | https://github.com/sjtu-medialab/Free-Viewpoint-RGB-D-Video-Dataset |
| CADC  Dataset |   Videos  | http://cadcd.uwaterloo.ca/ |
| COCO Dataset | Images  | http://images.cocodataset.org/zips/train2017.zip |
| RESIDE-β | Images  | https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2 (RTTS part)|

You can download those datasets from the links above. But you need to process them by yourself to adapt those lables or images. You can directly use our processed datasets from google drive: [link](https://drive.google.com/drive/folders/1WPPuVA9wclDjcmMxeMlkkPy7pKU01Olm?usp=drive_link)





## 👉 Code Structure

```
.

|-- README.md
├── Motivation
│   ├── Base
│   │   ├── video.py
│   │   └── performance.py
│   ├── data
│   │   ├── figure1_1_data.npy
│   │   ├── figure1_2_data.npy
│   │   ├── figure1_3_data.npy
│   │   ├── figure1_4_data.npy
│   │   ├── figure2_1_data.npy
│   │   └── figure2_2_data.npy
│   ├── figure4_src_img
│   │   ├── light1.jpg
│   │   ├── ...
│   │   └── light4.jpg
│   ├── figure5_src_img
│   │   ├── perspective_1.jpg
│   │   ├── ...
│   │   └── perspective_4.jpg
│   ├── motivation_config.json
│   ├── motivation_dataprocess.py
│   └── motivation_plot.py
|-- Experiment
|   |-- Algorithm
|   |   |-- Axiomvision.py
|   |   |-- ... (different algorithms)
|   |   |-- Greedy.py
|   |   |-- paras.py
|   |   |-- experiment_config.json
|   |   |-- performance.py
|   |   |-- video.py
|   |   |-- Base.py
|   |   |-- Enviroment.py
|   |-- retrained_models
|   |   |-- coco5k_Significantly_Darker_100
|   |   |-- ...
|   |   |-- coco5k_Extremely_Bright_100
|   |-- raw_data (some basic results)
|   |   |-- real_payoff_0_1_17.npy
|   |   |-- ...
|   |   |-- svd_results_0_1_17.npz
|   |-- main.py
|   |-- experiment_prepare_config.json
|   |-- experiment_dataprocess.py
|   |-- experiment_plot.py
|   |-- demo.py % Example of modifying model source code for time and accuracy analysis
|   |-- svd_decomposition.py
|-- DNN
|   |-- faster_rcnn
|   |-- mmdetection
|   |-- SSD
|   |-- yolov5
|   |-- PaddleSeg
|-- Dataset
|   |-- coco5k
|   |-- RTTS
|   |-- snowy
|   |-- Free-Viewpoint-RGB-D-Video-Dataset
|   |-- time_vary
|   |-- VR_youtube
|   |-- youtube_differemt_environment
|   |-- youtube_different_perspective
├── Motivation
│   ├── Base
│   │   ├── video.py
│   │   └── performance.py
│   ├── data
│   │   ├── figure1_data.npy
│   │   ├── ...
│   │   └── figure6_data_segmentation.npy
│   ├── figure1_1_src_img
│   │   ├── light1.jpg
│   │   ├── ...
│   │   └── light4.jpg
│   ├── figure5_src_img
│   │   ├── perspective_1.jpg
│   │   ├── ...
│   │   └── perspective_4.jpg
│   ├── motivation_config.json
│   ├── motivation_dataprocess.py
│   └── motivation_plot.py
└── Experiment
    ├── Algorithm
    │   ├── Axiomvision.py
    │   ├── wo_Group.py
    │   ├── ... (different algorithms)
    │   ├── Greedy.py
    │   ├── paras.py
    │   ├── experiment_config.json
    │   ├── performance.py
    │   ├── video.py
    │   ├── Base.py
    │   └── Enviroment.py
    ├── retrained_models
    │   ├── coco5k_Significantly_Darker_100
    │   ├── ...
    │   └── coco5k_Extremely_Bright_100
    ├── main.py
    ├── experiment_prepare_config.json
    ├── experiment_dataprocess.py
    ├── demo.py (This is an example of modifying different model's source code to analyze both time and accuracy together.)
    └── svd_decomposition.py

```

## 🚀How to Run

Execute the following command to reproduce the figures in the paper:


## Motivation:

### Getting Started
1. Clone the original repositories and change the code as we mentioned above or use our modified code.
2. Set up your environment
3. Prepare your dataset (if needed)
4. Run `motivation_dataprocess.py` (activate the function you want to run)
5. Run `motivation_plot.py` (activate the function you want to run)

###  Experiment:

1. Run `experiment_dataprocess.py` (activate the function you want to run)
2. Run `svd_decomposition.py`
3. Run `main.py` (activate the function you want to run)

To compare various algorithms, please consider three dimensions: algorithm_time, average_payoff, and dnn_time.


## 🌟 Citation

If you find this work helpful to your research, please kindly consider citing our paper.

```
@inproceedings{dai2024axiomvision,
  title={AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics},
  author={Dai, Xiangxiang and Zhang, Zeyu and Yang, Peng and Xu, Yuedong and Liu, Xutong and Lui, John CS},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={7229--7238},
  year={2024}
}
```
