# WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion

This is the refurbished WHAM repo for personal use, supporting newer versions of dependencies. The original repo is at [here](https://github.com/yohanshin/WHAM)

## Environment Setup

Run `fetch_demo_data.sh`, or manually prepare data, including:

```
.
├── checkpoints
│   ├── dpvo.pth
│   ├── hmr2a.ckpt
│   ├── wham_vit_bedlam_w_3dpw.pth.tar
│   ├── wham_vit_w_3dpw.pth.tar
│   └── (optional) other custom assets
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

Create conda env (optional):
```bash
conda create -n wham python=3.11
conda activate wham
```

Install as a package:

```bash
pip install -e .
```

For visualization and SLAM:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
pip install --no-build-isolation -e .
```

## CLI Usage

Run with pose NPZ input:

```bash
wham \
  --video examples/drone_video.mp4 \
  --pose_npz path/to/poses.npz
```

Run in local-coordinate mode (disable global SLAM):

```bash
wham \
  --video examples/drone_video.mp4 \
  --pose_npz path/to/poses.npz \
  --estimate_local_only
```

Enable visualization and save output:

```bash
wham \
  --video examples/drone_video.mp4 \
  --pose_npz path/to/poses.npz \
  --visualize \
  --save_pkl
```

## Python API

```python
import numpy as np
from wham import WHAMRunner

runner = WHAMRunner()

# 1) Pose-NPZ flow (load 2D poses from npz)
results, tracking_results, slam_results = runner.run(
    video="examples/drone_video.mp4",
    pose_npz="path/to/poses.npz",
    visualize=True,
)

# 2) Pose-array flow (pass keypoints directly as ndarray)
keypoints = np.random.rand(1, 100, 17, 3).astype(np.float32)
results, tracking_results, slam_results = runner.run(
    video="examples/drone_video.mp4",
    pose_keypoints=keypoints,
    visualize=True,
)

# Note: pose input is required (pose_npz or pose_keypoints).
```
