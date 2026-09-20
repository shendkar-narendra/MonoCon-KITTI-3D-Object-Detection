# MonoCon-KITTI-3D-Object-Detection

<img width="1241" height="376" alt="000047" src="https://github.com/user-attachments/assets/5a37d43e-7a20-4bcd-afe3-45a651348321" />
BEV : <img width="900" height="1100" alt="bev_000047" src="https://github.com/user-attachments/assets/d43de90c-3743-4900-95bc-0c27c20390b5" />
BEV_GT :<img width="900" height="1100" alt="gt_000047" src="https://github.com/user-attachments/assets/60558946-e87f-4a9d-acae-320fc648ddab" />
BEV_PRED : <img width="900" height="1100" alt="pred_000047" src="https://github.com/user-attachments/assets/0be99e3d-e70f-4788-9ad4-d608619134af" />



# MonoCon 3D Object Detection on KITTI

This repository contains my implementation, training setup, configuration, and experimental files for **camera-based monocular 3D object detection using MonoCon on the KITTI dataset**.

The experiment is part of a comparative study of camera-based, LiDAR-based, and camera-LiDAR fusion methods for 3D object detection.

The MonoCon model was trained using **MMDetection3D 0.14.0** with a **DLA-34 backbone** for the following KITTI object classes:

- Pedestrian
- Cyclist
- Car

The final training was completed for **200 epochs**.

---

# 1. Project Overview

MonoCon is a monocular 3D object detector that estimates 3D object information using only an RGB camera image.

In this project, MonoCon is used as the **camera-only 3D object detection baseline**.

## Input

RGB camera image from KITTI.

## Output

The model predicts information including:

- Object class
- 2D bounding box
- 3D bounding box
- Object dimensions
- Object depth
- Object orientation
- Keypoints
- Object center

The three evaluated KITTI classes are:

```text
Pedestrian
Cyclist
Car
```

---

# 2. Repository Structure

```text
MonoCon-KITTI-3D-Object-Detection/
│
├── checkpoint/
│   └── README.md
│
├── configs/
│   ├── monocon_dla34_windows_train.py
│   └── monocon_dla34_windows_train_200e.py
│
├── create_data_tools_monocon/
│   ├── create_data.py
│   └── data_converter/
│
├── mmdet3d/
│   ├── apis/
│   ├── core/
│   ├── datasets/
│   ├── models/
│   ├── ops/
│   ├── utils/
│   ├── __init__.py
│   └── version.py
│
├── requirements/
│
├── results/
│   └── last_training.log
│
├── tools/
│   ├── train.py
│   └── test.py
│
├── LICENSE
├── MANIFEST.in
├── README.md
├── requirements.txt
├── setup.cfg
└── setup.py
```

---

# 3. Experimental Configuration

The final experiment used the following configuration.

| Parameter | Value |
|---|---|
| Model | MonoCon |
| Detector type | CenterNetMono3D |
| Backbone | DLA-34 |
| Neck | DLAUp |
| Detection head | MonoConHead |
| Dataset | KITTI |
| Input modality | Camera only |
| Classes | Pedestrian, Cyclist, Car |
| Number of classes | 3 |
| Epochs | 200 |
| Samples per GPU | 2 |
| Workers per GPU | 0 |
| Optimizer | AdamW |
| Initial learning rate | 0.000225 |
| Weight decay | 1e-5 |
| LR policy | Cyclic |
| Gradient clipping | Max norm 35 |
| Validation interval | Every 5 epochs |
| Checkpoint interval | Every 5 epochs |
| Random seed | 0 |

The main final configuration file is:

```text
configs/monocon_dla34_windows_train_200e.py
```

---

# 4. Model Architecture

The model uses:

```text
RGB image
   │
   ▼
DLA-34 Backbone
   │
   ▼
DLAUp Neck
   │
   ▼
MonoCon Head
   │
   ├── Center heatmap
   ├── 2D width / height
   ├── Center offset
   ├── Keypoint prediction
   ├── Object dimensions
   ├── Depth estimation
   └── Orientation estimation
   │
   ▼
3D Object Predictions
```

The configured detector is:

```python
type='CenterNetMono3D'
```

with:

```python
backbone=dict(
    type='DLA',
    depth=34
)
```

and:

```python
bbox_head=dict(
    type='MonoConHead',
    num_classes=3
)
```

---

# 5. Training Environment

The experiment was performed using the following software environment:

```text
Operating System : Windows
Python           : 3.8.20
PyTorch          : 1.8.1+cu111
TorchVision      : 0.9.1+cu111
CUDA Runtime     : 11.1
CuDNN            : 8.0.5
OpenCV           : 4.5.5
MMCV             : 1.4.0
MMDetection      : 2.11.0
MMDetection3D    : 0.14.0
```

Hardware used during training:

```text
GPU : NVIDIA GeForce GTX 1650
```

The experiment was run on a single GPU.

---

# 6. Installation

## 6.1 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MonoCon-KITTI-3D-Object-Detection.git
cd MonoCon-KITTI-3D-Object-Detection
```

Replace `YOUR_USERNAME` with the GitHub account hosting the repository.

---

## 6.2 Create Python Environment

The original experiment used Python 3.8.20.

Using Conda:

```bash
conda create -n monocon python=3.8.20
conda activate monocon
```

---

## 6.3 Install PyTorch

The original experiment used:

```text
PyTorch 1.8.1 + CUDA 11.1
TorchVision 0.9.1 + CUDA 11.1
```

Install versions compatible with your CUDA and operating system.

The exact experiment environment used:

```text
torch==1.8.1+cu111
torchvision==0.9.1+cu111
```

---

## 6.4 Install Required OpenMMLab Packages

The experiment used:

```text
mmcv==1.4.0
mmdet==2.11.0
mmsegmentation==0.13.0
mmdet3d==0.14.0
```

The exact versions are important because newer versions of MMDetection3D have significantly different APIs and configuration formats.

---

## 6.5 Install Repository Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

The repository also contains the individual requirement files inside:

```text
requirements/
```

---

## 6.6 Install MMDetection3D Source

From the repository root:

```bash
pip install -v -e .
```

This uses:

```text
setup.py
```

and builds the required MMDetection3D CUDA/C++ operations.

> Building MMDetection3D CUDA extensions on Windows can depend strongly on the installed CUDA Toolkit, PyTorch version, Microsoft Visual C++ Build Tools and Windows SDK.

The environment used for this experiment was based on CUDA 11.1 and PyTorch 1.8.1+cu111.

---

# 7. KITTI Dataset

The KITTI dataset itself is **not included in this repository**.

Download the KITTI object detection dataset from the official KITTI website.

The dataset should be organized under:

```text
data/kitti/
```

A typical structure is:

```text
data/kitti/
│
├── training/
│   ├── image_2/
│   ├── label_2/
│   ├── calib/
│   └── velodyne/
│
├── testing/
│   ├── image_2/
│   ├── calib/
│   └── velodyne/
│
└── ImageSets/
    ├── train.txt
    └── val.txt
```

For this MonoCon experiment, camera images are used as the model input.

LiDAR is not used by MonoCon during inference.

---

# 8. KITTI Train / Validation Split

The experiment uses the KITTI training data divided into:

```text
Training split
Validation split
```

The KITTI official test labels are not publicly available, therefore training and validation experiments use a split of the labeled KITTI training set.

The configuration references:

```text
data/kitti/kitti_infos_train.pkl
data/kitti/kitti_infos_val.pkl
```

and:

```text
data/kitti/kitti_infos_train_mono3d.coco.json
data/kitti/kitti_infos_val_mono3d.coco.json
```

---

# 9. MonoCon Dataset Preparation

MonoCon requires additional monocular 3D annotation information.

The data preparation code used in this project is contained in:

```text
create_data_tools_monocon/
```

Important files include:

```text
create_data_tools_monocon/create_data.py
create_data_tools_monocon/data_converter/
```

Before running the data conversion, inspect the supported arguments:

```bash
python create_data_tools_monocon/create_data.py --help
```

The preprocessing stage should generate the dataset information and MonoCon COCO-format annotation files required by the configuration:

```text
kitti_infos_train.pkl
kitti_infos_val.pkl

kitti_infos_train_mono3d.coco.json
kitti_infos_val_mono3d.coco.json
```

The exact generated files used by the final configuration are located logically under:

```text
data/kitti/
```

The KITTI data itself is intentionally not stored in this GitHub repository.

---

# 10. Data Augmentation

The training pipeline applies several augmentation operations.

## Photometric Distortion

```text
Brightness delta : 32
Contrast range   : 0.5 - 1.5
Saturation range : 0.5 - 1.5
Hue delta        : 18
```

## Random Image Shift

```text
Probability : 0.5
Maximum shift : 32 pixels
```

## Horizontal Flip

```text
Probability : 0.5
```

## Image Normalization

Mean:

```text
123.675
116.28
103.53
```

Standard deviation:

```text
58.395
57.12
57.375
```

---

# 11. Dataset Filtering

The training configuration applies the following object filtering:

```text
Minimum object height : 25 pixels
Minimum depth         : 2 m
Maximum depth         : 65 m
Maximum truncation    : 0.5
Maximum occlusion     : 2
```

---

# 12. Training

The main training entry point is:

```text
tools/train.py
```

The final configuration is:

```text
configs/monocon_dla34_windows_train_200e.py
```

## Important Before Training From Scratch

The uploaded configuration is retained as an experimental record and contains the checkpoint that was used when one of the original training sessions was resumed.

Therefore, for a **new training run from epoch 1**, open:

```text
configs/monocon_dla34_windows_train_200e.py
```

and change:

```python
resume_from = 'work_dirs\\monocon_dla34_windows_train\\epoch_115.pth'
```

to:

```python
resume_from = None
```

Also set the desired output directory, for example:

```python
work_dir = './work_dirs/monocon_dla34_windows_train'
```

Do not overwrite the historical config if you want to preserve the exact experimental record. A copy can be created for a new experiment.

---

# 13. Train From Scratch

After setting:

```python
resume_from = None
```

run:

```bash
python tools/train.py configs/monocon_dla34_windows_train_200e.py
```

The configuration contains:

```python
runner = dict(
    type='EpochBasedRunner',
    max_epochs=200
)
```

so the complete training schedule runs for 200 epochs.

---

# 14. Checkpoints

Checkpoints are generated every 5 epochs:

```python
checkpoint_config = dict(interval=5)
```

Examples include:

```text
epoch_5.pth
epoch_10.pth
epoch_15.pth
...
epoch_195.pth
epoch_200.pth
```

The final model from this project is:

```text
epoch_200.pth
```

The checkpoint is not stored directly in the Git repository because of its large file size.

It can instead be provided through GitHub Releases.

---

# 15. Resuming Training

Training did not have to run for 200 epochs in a single uninterrupted session.

MMDetection3D checkpoints store information necessary to continue training.

One command used during this project was:

```bash
python tools/train.py work_dirs/monocon_dla34_windows_train/monocon_dla34_windows_train_200e.py --resume-from work_dirs/monocon_dla34_windows_train/epoch_100.pth
```

A later training session was also resumed from:

```text
epoch_115.pth
```

Resuming restores training state including the model checkpoint and allows training to continue toward epoch 200.

A general resume command is:

```bash
python tools/train.py CONFIG_FILE --resume-from CHECKPOINT_FILE
```

For example:

```bash
python tools/train.py configs/monocon_dla34_windows_train_200e.py --resume-from work_dirs/monocon_dla34_windows_train/epoch_100.pth
```

---

# 16. Validation During Training

Validation was configured to run every 5 epochs:

```python
evaluation = dict(interval=5)
```

This means the model weights are trained on the training split, while the validation split is periodically used to measure performance.

Validation does **not** perform optimizer updates.

Training weight updates occur during the training stage using:

```text
forward pass
→ loss calculation
→ backpropagation
→ optimizer update
```

The validation results are therefore used to measure model performance, not to directly update model weights.

---

# 17. Optimizer

The experiment uses AdamW:

```python
optimizer = dict(
    type='AdamW',
    lr=0.000225,
    betas=(0.95, 0.99),
    weight_decay=1e-05
)
```

Gradient clipping is enabled:

```python
optimizer_config = dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2
    )
)
```

---

# 18. Learning Rate Schedule

A cyclic learning-rate schedule is used:

```python
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4
)
```

A corresponding cyclic momentum schedule is also applied.

---

# 19. MonoCon Losses

MonoCon optimizes multiple objectives simultaneously.

The configured losses include:

```text
Center heatmap loss
2D width/height loss
Center offset loss
Center-to-keypoint offset loss
Keypoint heatmap loss
Keypoint heatmap offset loss
Dimension loss
Depth loss
Orientation classification loss
Orientation regression loss
```

Examples of the implemented loss functions include:

```text
CenterNetGaussianFocalLoss
L1Loss
DimAwareL1Loss
LaplacianAleatoricUncertaintyLoss
CrossEntropyLoss
```

These different objectives allow the network to estimate both 2D image information and 3D object geometry.

---

# 20. Inference Configuration

The MonoCon head uses:

```text
Top-K predictions       : 30
Local maximum kernel    : 3
Maximum objects/image   : 30
Detection threshold     : 0.4
```

---


# 22. KITTI Evaluation Metrics

The project evaluates 3D object detection using KITTI metrics.

Relevant metrics include:

- 2D Bounding Box AP
- Bird's-Eye-View AP
- 3D Bounding Box AP
- Average Orientation Similarity
- Easy
- Moderate
- Hard difficulty levels

For the strict KITTI evaluation thresholds:

```text
Car        : IoU 0.70
Pedestrian : IoU 0.50
Cyclist    : IoU 0.50
```

The final epoch-200 evaluation results will be added after testing.

---

# 23. Training Logs

Training logs are stored under:

```text
results/
```

The current training record is:

```text
results/last_training.log
```

The log contains information including:

```text
Epoch
Iteration
Learning rate
Individual loss values
Total loss
Gradient norm
Validation results
Environment information
```

---

# 24. Current Project Status

| Stage | Status |
|---|---|
| Environment setup | Completed |
| KITTI dataset preparation | Completed |
| MonoCon data conversion | Completed |
| MonoCon configuration | Completed |
| 200-epoch training | Completed |
| Final checkpoint generation | Completed |
| Intermediate validation | Completed |
| Final epoch-200 evaluation | Pending |
| Final prediction generation | Pending |
| Bounding-box visualization | Pending |
| Research-paper comparison | Pending |

---

# 25. Reproducing the Experiment

A student wishing to reproduce the project should follow these steps:

```text
1. Clone this repository or download original repo from MMDetection framework for MonoCon

2. Create a Python 3.8 environment

3. Install the compatible PyTorch/CUDA stack

4. Install:
   MMCV 1.4.0
   MMDetection 2.11.0
   MMSegmentation 0.13.0

5. Install this MMDetection3D source using:
   pip install -v -e .

6. Download the KITTI dataset

7. Create the KITTI train/validation split

8. Run the MonoCon KITTI data-preparation scripts

9. Verify that the generated .pkl and MonoCon COCO JSON files exist

10. Set resume_from = None in a copy of the 200-epoch configuration

11. Run:
    python tools/train.py configs/monocon_dla34_windows_train_200e.py

12. Allow training to complete to epoch 200

13. Evaluate epoch_200.pth using tools/test.py

14. Generate visual predictions

15. Record KITTI 2D, BEV and 3D AP results
```

---

# 26. Important Reproducibility Notes

This repository contains the code and configuration used for the experiment, but several factors can affect exact numerical reproducibility:

- GPU model
- CUDA version
- CuDNN version
- PyTorch version
- Random initialization
- Data-loader behavior
- KITTI train/validation split
- Dependency versions

The original experiment used random seed:

```text
0
```

with:

```text
deterministic = False
```

Therefore extremely small numerical differences between independent training runs may occur.

---

# 27. Dataset License

The KITTI dataset is not redistributed with this repository.

Users must obtain KITTI separately and follow the KITTI dataset's terms and licensing requirements.

---

# 28. Acknowledgements

This project builds upon:

- MonoCon
- MMDetection3D
- MMDetection
- MMCV
- KITTI Vision Benchmark Suite

The underlying framework and third-party source code remain subject to their respective licenses.

The MMDetection3D-derived code contained in this repository is distributed according to the included license.

---



---

# 30. Project Purpose

This repository forms the camera-based component of a larger scientific project comparing:

```text
Camera-only 3D detection
vs.
LiDAR-only 3D detection
vs.
Camera-LiDAR fusion
```

using the KITTI dataset.

The camera-based method used here is:

```text
MonoCon
```

The LiDAR-based and multi-modal fusion experiments are maintained separately.

---

# Citation

If this repository or its underlying methods are used in academic work, please cite the original MonoCon, MMDetection3D and KITTI publications as appropriate.
