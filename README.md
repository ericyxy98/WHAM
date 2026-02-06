# WHAM Inference (Python 3.11)

This repository contains a streamlined, inference-only version of WHAM with a modern `pyproject.toml` setup. It provides:

1. **Inference API** that consumes a video + pre-extracted 2D keypoints/bounding boxes and outputs SMPL parameters.
2. **Multi-view SMPL renderer** that exports a 4-view video (front, back, left, right).

## Requirements

* Python **3.11**
* PyTorch **2.5.1**

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For visualization (4-view rendering), install the optional dependencies:

```bash
pip install -e ".[viz]"
```

> **Note:** PyTorch3D wheels are platform-specific. If the optional install fails, follow the official PyTorch3D install guide for your CUDA/PyTorch version, then re-run the renderer.

## Model Assets

You need SMPL assets and the WHAM/HMR2 checkpoints. The helper script below downloads only the inference-time assets:

```bash
bash fetch_demo_data.sh
```

This script uses `gdown`, so you may need to install it:

```bash
pip install gdown
```

## Input Pose Data Format

The inference API expects a `.npz` (recommended) or `.pkl` containing:

* `keypoints`: `(T, J, 3)` array of 2D keypoints in pixel space `(x, y, conf)`
* `bboxes`: either `(T, 3)` **cxcys** (`cx`, `cy`, `scale`) or `(T, 4)` **xyxy**
* `frame_ids` (optional): `(T,)` frame indices

Example NPZ creation:

```python
import numpy as np

np.savez(
    "pose_data.npz",
    keypoints=keypoints,  # (T, J, 3)
    bboxes=bboxes,        # (T, 3) cxcys or (T, 4) xyxy
    frame_ids=np.arange(len(keypoints)),
)
```

## Inference (API + CLI)

### CLI

```bash
wham-infer \
  --video /path/to/video.mp4 \
  --pose-data /path/to/pose_data.npz \
  --output-dir output/inference
```

The command writes:

* `output/inference/wham_output.npz` (single-subject SMPL params)
* `output/inference/wham_results.pkl` (full dictionary output)

### Python API

```python
from wham.inference import WHAMInference

runner = WHAMInference(device="cuda")
outputs = runner(
    video="video.mp4",
    pose_data="pose_data.npz",
    output_dir="output/inference",
)
```

## SMPL Multi-View Visualization (4 Views)

Render a 2x2 grid video with **front**, **back**, **left**, and **right** views:

```bash
wham-render \
  --smpl-output output/inference/wham_output.npz \
  --output output/multiview.mp4
```

You can control output size and FPS:

```bash
wham-render --smpl-output output/inference/wham_output.npz --size 720 --fps 30
```

## Output Format

The `wham_output.npz` file contains:

* `pose`: `(T, 72)` axis-angle (root + body)
* `betas`: `(T, 10)` shape parameters (broadcast if constant)
* `trans`: `(T, 3)` translation
* `frame_ids`: `(T,)` frame indices

Use these values directly with SMPL for downstream applications.
