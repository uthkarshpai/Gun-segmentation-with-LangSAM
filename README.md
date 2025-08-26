# GunDex-Plus Gun Segmentation
 A two-stage pipeline that first detects firearms (GunDex) and then refines those detections with pixel-level segmentation for precise mask generation.

---

## Table of Contents

* [Project overview](#project-overview)
* [Features](#features)
* [Repository structure](#repository-structure)
* [Requirements](#requirements)
* [Dataset and preparation](#dataset-and-preparation)
* [Model architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Inference / Demo](#inference--demo)
* [Performance tips & troubleshooting](#performance-tips--troubleshooting)
* [Licensing & ethics](#licensing--ethics)
* [Citation](#citation)

---

## Project overview

We built **GunDex-Plus**, a two-stage pipeline for firearm detection and segmentation. First, a lightweight detection model (GunDex) identifies potential gun regions in an image or video. Then, a segmentation network refines these regions into accurate pixel-level masks. This approach combines the **speed and recall of object detection** with the **precision of segmentation**, making it useful for research in security, forensic analysis, and safety-critical vision tasks.

---

## Features

* Two-stage pipeline: fast detector + accurate segmentation
* End-to-end training scripts and evaluation utilities
* Inference scripts with options for batch images and video
* Checkpointing and logging (TensorBoard / WandB)
* Config-driven experiments (YAML)

---

## Repository structure

```
├── README.md                    # This file
├── configs/                      # YAML experiment configs (dataset, model, training)
├── datasets/                     # dataset download / preparation scripts
├── src/
│   ├── detector/                 # GunDex detection model & utils
│   ├── segmenter/                # segmentation model and losses
│   ├── trainer/                  # training loops, schedulers, checkpointing
│   ├── inference/                # inference / visualization tools
│   └── utils/                    # common helpers (IO, metrics, transforms)
├── experiments/                  # logs, checkpoints produced by runs
├── notebooks/                    # EDA and prototyping notebooks
├── requirements.txt              # pinned Python dependencies
└── setup.py                      # package install helper
```

---

## Requirements

Recommended: Python 3.9+ and a CUDA-enabled GPU for training.

Basic install (pip):

```bash
python -m venv venv
source venv/bin/activate        # or `venv\\Scripts\\activate` on Windows
pip install -r requirements.txt
```

Example `requirements.txt` highlights (keep exact versions in repo):

```
torch>=2.0
torchvision
opencv-python
numpy
pyyaml
tqdm
albumentations
tensorboard
wandb        # optional
scipy
```

If you rely on detectron2 / mmseg / mmdetection, include the appropriate installation instructions in `configs/README.md`.

---

## Dataset and preparation

This project expects a dataset structured as follows (adapt to your dataset):

```
/datasets/guns/
  ├── images/
  │    ├── train/
  │    ├── val/
  │    └── test/
  └── annotations/
       ├── train.json    # COCO-style detection + segmentation or separate masks
       ├── val.json
       └── masks/        # optional directory of per-image mask PNGs if not using COCO polygons
```

Supported annotation formats:

* COCO (preferred) with `annotations.segmentation` polygons for masks
* Pascal VOC style masks
* Custom CSV + mask folder (use `datasets/convert_*` helpers)

Run the prepare script to convert raw annotations to the repository's expected COCO format:

```bash
python datasets/prepare_dataset.py --src /path/to/raw --dst datasets/guns --format coco
```

---

## Model architecture

**GunDex (Detector)**

* Lightweight backbone (e.g., ResNet-50 / MobileNetV3) with FPN
* Anchor/anchor-free head configurable in `configs/detector.yaml`
* Trained for high recall (loss weighting, increased NMS thresholds)

**Segmentation head**

* U-Net / DeepLabV3 style encoder-decoder operating on cropped detection boxes
* Optional mask refinement module (e.g., CRF post-processing)
* Losses: combination of DiceLoss + BCE / focal loss

**Pipeline flow**

1. Run GunDex to predict bounding boxes and scores
2. For each box above a score threshold, crop and resize region
3. Run segmentation head to predict mask for the crop
4. Uncrop/rescale mask back to original image coordinates
5. Merge overlapping masks (by score or mask IoU threshold)

Model hyperparameters and architecture choices are stored in `configs/*.yaml`.

---

## Training

### Quick-start (single-GPU)

1. Edit `configs/train_gundex_plus.yaml` to point to dataset and set hyperparameters.
2. Start the training run:

```bash
python src/trainer/train.py --config configs/train_gundex_plus.yaml --exp_name gundex_plus_experiment
```

Key flags you may pass:

* `--resume /path/to/checkpoint.pth`
* `--batch-size N`
* `--lr 0.001`
* `--epochs 50`

Training saves checkpoints to `experiments/<exp_name>/checkpoints/` and logs to `experiments/<exp_name>/logs/` (TensorBoard compatible).

### Multi-GPU / distributed

We support `torch.distributed.launch` for multi-GPU training. Example:

```bash
python -m torch.distributed.run --nproc_per_node=4 src/trainer/train.py --config configs/train_gundex_plus.yaml --exp_name gundex_plus_dist
```

---

## Evaluation

We provide detection and segmentation metrics.

* Detection: mAP (COCO-style AP @ IoU=0.5:0.95), recall, precision
* Segmentation: mean IoU, Dice score, per-class IoU

Run evaluation on a checkpoint:

```bash
python src/trainer/eval.py --config configs/train_gundex_plus.yaml --ckpt experiments/gundex_plus_experiment/checkpoints/last.pth --split val
```

Outputs: JSON metric summary and a visual grid of predicted masks vs ground truth saved to `experiments/<exp_name>/eval_visuals/`.

---

## Inference / Demo

### Single image

```bash
python src/inference/run_inference.py --image ./samples/test1.jpg --ckpt experiments/gundex_plus_experiment/checkpoints/best.pth --out ./out/test1_pred.png --vis
```

### Directory / Video

Process a folder of images or a video (MP4):

```bash
python src/inference/run_inference.py --input ./data/images/ --ckpt ... --out ./out/ --batch 8
# or
python src/inference/video_inference.py --video ./videos/clip.mp4 --ckpt ... --out ./out/clip_masked.mp4
```

### Inference options

* `--det-score-thr` : detector threshold (default 0.25)
* `--mask-merge-iou` : IoU threshold when merging overlapping masks
* `--crop-padding` : padding applied when cropping detection boxes for segmentation

---

## Example usage (Python API)

```python
from src.inference import Predictor

pred = Predictor(model_ckpt='experiments/gundex_plus_experiment/checkpoints/best.pth', device='cuda')
img = '/path/to/image.jpg'
results = pred.predict(img)
# results = [{ 'box': [x1,y1,x2,y2], 'score':0.98, 'mask': np.array(H,W,1) }, ...]

# save visualization
pred.visualize(results, save_path='out/vis.png')
```

---

## Performance tips & troubleshooting

* If segmentation masks are poor on small guns, increase detector resolution or add multi-scale training.
* For high recall: lower detector score threshold, increase anchor sizes, or augment with synthetic crops.
* If training is unstable: reduce learning rate, use gradient clipping, or switch to AdamW.
* Use mixed precision (`torch.cuda.amp`) to reduce memory and speed up training.

Common checks:

* Ensure dataset format matches config paths
