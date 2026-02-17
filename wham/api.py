from __future__ import annotations

import os
import os.path as osp
from typing import Any

import numpy as np

from configs.config import get_cfg_defaults
from lib.models import build_body_model, build_network


def prepare_cfg(config_path: str = "configs/yamls/demo.yaml"):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    return cfg


class WHAMRunner:
    """High-level API for running WHAM inference.

    If `pose_npz` is given, use keypoints from NPZ.
    If `pose_keypoints` is given, use the provided numpy array directly.
    One of `pose_npz` or `pose_keypoints` must be provided.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or prepare_cfg()
        smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        self.network = build_network(self.cfg, smpl)
        self.network.eval()

    def run(
        self,
        video: str,
        output_dir: str = "output/demo",
        pose_npz: str | os.PathLike | None = None,
        pose_keypoints: np.ndarray | None = None,
        calib: str | None = None,
        run_global: bool = True,
        save_pkl: bool = False,
        visualize: bool = False,
        run_smplify: bool = False,
    ) -> tuple[dict[int, dict[str, Any]], dict, Any]:
        if pose_npz is not None and pose_keypoints is not None:
            raise ValueError("Provide only one of `pose_npz` or `pose_keypoints`.")
        if pose_npz is None and pose_keypoints is None:
            raise ValueError(
                "Pose input is required. Provide `pose_npz` or `pose_keypoints`."
            )

        sequence = ".".join(video.split("/")[-1].split(".")[:-1])
        output_pth = osp.join(output_dir, sequence)
        os.makedirs(output_pth, exist_ok=True)

        from wham.pipeline import run_pose_demo

        pose_data = pose_npz if pose_npz is not None else np.asarray(pose_keypoints)
        return run_pose_demo(
            self.cfg,
            video,
            output_pth,
            self.network,
            pose_data,
            calib,
            run_global=run_global,
            save_pkl=save_pkl,
            visualize=visualize,
            run_smplify=run_smplify,
        )


def run_wham(*args, **kwargs):
    """Convenience function around :class:`WHAMRunner`."""
    return WHAMRunner().run(*args, **kwargs)
