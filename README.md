# AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics

This repository contains the source code for reproducing the results of our paper titled [**AxiomVision: Accuracy-Guaranteed Adaptive Visual Model Selection for Perspective-Aware Video Analytics**](https://arxiv.org/abs/2407.20124), which is accepted by ACM MM 2024.

![](design.png)

AxiomVision is a general online streaming framework that adaptively selects the most effective visual model based on the end-edge-cloud architecture to ensure the accuracy of video stream analysis in diverse scenarios.



## 🚀 Quick Start Guide

To quickly reproduce the main results, follow these steps:

1. **Environment Setup**  
   Set up both the main (AxiomVision) and segmentation (Paddle) environments as described in the [Environment Setup](#️-environment-setup) section below.

2. **Dataset Preparation**  
   See the [Dataset](#-dataset) section for details. You can directly download our processed datasets from [Google Drive](https://drive.google.com/drive/folders/1WPPuVA9wclDjcmMxeMlkkPy7pKU01Olm?usp=drive_link).

3. **Motivation Experiments**  
   Activate the main environment (AxiomVision) and run the following commands (activate the required functions as indicated by comments in the scripts):
   ```bash
   python Motivation/motivation_dataprocess.py
   python Motivation/motivation_plot.py
   ```

4. **Main Experiments**  
   Run the following commands (activate the required functions as indicated by comments in the scripts):
   ```bash
   python Experiment/experiment_dataprocess.py
   python Experiment/svd_decomposition.py
   python Experiment/main.py
   ```

> **Note:**
> - Please refer to the comments in each script for details on which functions to activate.
> - When comparing algorithms, consider three key metrics: `algorithm_time`, `average_payoff`, and `dnn_time`.


## ⚙️ Environment Setup

This project is tested on a Windows 11 system equipped with an NVIDIA 4060 Laptop GPU and CUDA 11.8. The following instructions describe the setup of two separate environments to ensure compatibility and reproducibility.

### 1. Main Environment (AxiomVision)

This environment is used for the majority of model inference and training tasks.

```bash
# Create the main environment
git clone https://github.com/zeyuzhangzyz/AxiomVision.git

conda create -n AxiomVision python=3.10.10 -y
conda activate AxiomVision

# Install dependencies
cd AxiomVision
pip install -r requirements.txt

# Install PyTorch (CUDA 11.8 version)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install mmcv (ensure strict version compatibility with CUDA/PyTorch; use the official wheel)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
# Note: This step may require compilation and can take 10-15 minutes.
```

> **Note:**
> - It is essential to install mmcv using the provided wheel link. Avoid installing via `pip install mmcv` directly, as this may result in missing C++/CUDA extensions and subsequent inference errors.
> - When running the project, the default working directory should be the root of the AxiomVision project, and all relative paths are based on this location.

### 2. Segmentation Environment (Paddle)

Due to dependency conflicts between PaddleSeg and the main environment, a separate environment is required. 

```bash
# Create the Paddle environment
conda create -n Paddle python=3.11.5 -y
conda activate Paddle

# Install dependencies
cd DNN/PaddleSeg
pip install -r requirements.txt

# Install PaddlePaddle (CUDA 11.8 version)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

> **Note:**
> - Activate the Paddle environment only when performing segmentation tasks (e.g., `run_data_figure2_2`).
> - For all other tasks, always use the main (AxiomVision) environment.
> - If you encounter the error `No module named 'mmcv._ext'`, please ensure that mmcv is installed using the official wheel and that the CUDA and PyTorch versions are strictly matched as specified above.


## 🔧Prerequisites

### Model Weights Download & Placement

Due to the large size of model weights, they are not included in this repository. Please follow the steps below to obtain and place the weights:

1. **Download Weights**  
   Visit the [Google Drive link](https://drive.google.com/drive/folders/1lRYIBUrhWHA8jWTqQOIZd1hcsHoIwXMR?usp=drive_link) to download all required model weights.

2. **Place Weights**  
   After downloading, place each model weight file into the corresponding directory in your local repository, matching the folder structure shown in Google Drive. For example:
   - If you download `yolov5x.pt` from `DNN/yolov5/` in the Drive, place it at `DNN/yolov5/yolov5x.pt` in your local repository.
   - For PaddleSeg, Faster R-CNN, SSD, MMDetection, etc., follow the same rule: strictly match the directory structure.

3. **Notes**  
   - **The relative paths must match exactly**, otherwise the code may not find the models.
   - If you encounter loading errors, please first check whether the file paths and names are consistent with the repository structure.



This project integrates multiple computer vision models and standardizes their output formats for consistent evaluation and comparison. 

The following sections provide detailed instructions for model weights download and output format standardization.

<details>
<summary><b>🔗 Output Format Standardization (Click to expand)</b></summary>

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
</details>


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
|-- motivation_config.json
|-- requirements.txt
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
│   │   ├── figure1_1_data.npy
│   │   ├── figure1_2_data.npy
│   │   ├── figure1_3_data.npy
│   │   ├── figure1_4_data_label_0.npy
│   │   ├── figure1_4_data_label_1.npy
│   │   ├── figure1_4_data_label_2.npy
│   │   ├── figure2_1_data.npy
│   │   ├── figure2_2_data.npy
│   │   ├── overall_data_label_0.npy
│   │   └── overall_data_label_2.npy
│   ├── figure1_1_src
│   ├── figure2_1_src
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
    ├── data_label_0.npy
    ├── data_label_2.npy
    ├── heatmap_data_0.npy
    ├── heatmap_data_2.npy
    ├── main.py
    ├── experiment_prepare_config.json
    ├── experiment_dataprocess.py
    └── svd_decomposition.py

```


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
