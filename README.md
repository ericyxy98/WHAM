# WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion

This is the refurbished WHAM repo for personal use, supporting newer versions of dependencies. The original repo is at [here](https://github.com/yohanshin/WHAM)

## Environment Setup

Run `fetch_demo_data.sh`, or manually prepare data, including:

```
.
├── checkpoints
│   ├── dpvo.pth
│   ├── hmr2a.ckpt
│   ├── vitpose-h-multi-coco.pth
│   ├── wham_vit_bedlam_w_3dpw.pth.tar
│   ├── wham_vit_w_3dpw.pth.tar
│   └── yolov8x.pt
├── dataset
│   └── body_models
│       ├── coco_aug_dict.pth
│       ├── J_regressor_coco.npy
│       ├── J_regressor_feet.npy
│       ├── J_regressor_h36m.npy
│       ├── J_regressor_wham.npy
│       ├── smpl
│       │   ├── SMPL_FEMALE.pkl
│       │   ├── SMPL_MALE.pkl
│       │   └── SMPL_NEUTRAL.pkl
│       ├── smpl_mean_params.npz
│       └── smplx2smpl.pkl
└── examples
    ├── drone_calib.txt
    ├── drone_video.mp4
    ├── IMG_9730.mov
    ├── IMG_9731.mov
    └── IMG_9732.mov
```  

Then, set up the environment as follows:

```bash
conda create -n wham python=3.11
conda activate wham
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
pip install -e --no-build-isolation .
```

## Install as a package

```bash
pip install -e .
```

## Python API

```python
import numpy as np
from wham import WHAMRunner

runner = WHAMRunner()

# 1) Same flow as demo_pose_npz.py (load 2D poses from npz)
results, tracking_results, slam_results = runner.run(
    video="examples/drone_video.mp4",
    pose_npz="path/to/poses.npz",
)

# 2) Same flow as demo_pose_npz.py, but pass keypoints directly as ndarray
keypoints = np.random.rand(1, 100, 17, 3).astype(np.float32)
results, tracking_results, slam_results = runner.run(
    video="examples/drone_video.mp4",
    pose_keypoints=keypoints,
)

# 3) Fallback to original WHAM behavior (detector/tracker preprocessing)
results, tracking_results, slam_results = runner.run(
    video="examples/drone_video.mp4",
)
```
