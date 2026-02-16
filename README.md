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
