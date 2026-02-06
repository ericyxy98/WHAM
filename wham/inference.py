from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.models import build_body_model, build_network
from lib.models.preproc.extractor import FeatureExtractor
from lib.utils.transforms import matrix_to_axis_angle

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "yamls" / "demo.yaml"


@dataclass
class InferenceOutput:
    results: dict[int, dict[str, Any]]
    tracking_results: dict[int, dict[str, Any]]
    slam_results: np.ndarray


def _resolve_repo_path(path: str | Path) -> Path:
    return (Path(__file__).resolve().parents[1] / path).resolve()


def prepare_cfg(config_path: str | Path | None, device: str, flip_eval: bool) -> Any:
    cfg = get_cfg_defaults()
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    config_path = config_path.resolve()
    cfg.merge_from_file(str(config_path))

    cfg.MODEL_CONFIG = str(_resolve_repo_path(cfg.MODEL_CONFIG))
    cfg.TRAIN.CHECKPOINT = str(_resolve_repo_path(cfg.TRAIN.CHECKPOINT))
    cfg.DEVICE = device
    cfg.FLIP_EVAL = flip_eval
    return cfg


def _load_pose_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        keypoints = np.asarray(data["keypoints"])
        bboxes = np.asarray(data["bboxes"] if "bboxes" in data else data["bbox"])
        frame_ids = np.asarray(
            data["frame_ids"] if "frame_ids" in data else np.arange(len(keypoints))
        )
        return keypoints, bboxes, frame_ids
    if path.suffix.lower() in {".pkl", ".pickle"}:
        data = joblib.load(path)
        keypoints = np.asarray(data["keypoints"])
        bboxes = np.asarray(data.get("bboxes", data.get("bbox")))
        frame_ids = np.asarray(data.get("frame_ids", np.arange(len(keypoints))))
        return keypoints, bboxes, frame_ids
    raise ValueError(f"Unsupported pose file: {path}")


def _ensure_keypoint_confidence(keypoints: np.ndarray) -> np.ndarray:
    if keypoints.shape[-1] == 2:
        conf = np.ones((*keypoints.shape[:-1], 1), dtype=keypoints.dtype)
        return np.concatenate([keypoints, conf], axis=-1)
    return keypoints


def _bboxes_to_cxcys(bboxes: np.ndarray, scale_factor: float = 1.2) -> np.ndarray:
    if bboxes.shape[-1] == 3:
        return bboxes
    if bboxes.shape[-1] != 4:
        raise ValueError("Bboxes must be (T, 3) cxcys or (T, 4) xyxy.")
    x1, y1, x2, y2 = bboxes.T
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    size = np.maximum(x2 - x1, y2 - y1)
    scale = size * scale_factor / 200.0
    return np.stack([cx, cy, scale], axis=-1)


def _build_tracking_results(
    cfg: Any,
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    frame_ids: np.ndarray,
) -> dict[int, dict[str, Any]]:
    tracking_results: dict[int, dict[str, Any]] = {
        0: {
            "frame_id": frame_ids,
            "bbox": bboxes,
            "keypoints": keypoints,
            "features": [],
        }
    }
    if cfg.FLIP_EVAL:
        tracking_results[0].update(
            {"flipped_bbox": [], "flipped_keypoints": [], "flipped_features": []}
        )
    return tracking_results


def run_inference(
    video: str | Path,
    pose_data: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    device: str = "cuda",
    flip_eval: bool = False,
    save_npz: bool = True,
    save_pkl: bool = True,
) -> InferenceOutput:
    cfg = prepare_cfg(config_path, device=device, flip_eval=flip_eval)

    keypoints, bboxes, frame_ids = _load_pose_data(pose_data)
    keypoints = _ensure_keypoint_confidence(keypoints)
    bboxes = _bboxes_to_cxcys(bboxes)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if fps <= 0:
        fps = 30.0

    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
    tracking_results = _build_tracking_results(cfg, keypoints, bboxes, frame_ids)
    tracking_results = extractor.run(str(video), tracking_results)

    if length <= 0:
        length = int(frame_ids.max()) + 1

    slam_results = np.zeros((length, 7), dtype=np.float32)
    slam_results[:, 3] = 1.0

    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    results: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for subj in range(len(dataset)):
            batch = dataset.load_data(subj)
            if batch is None:
                continue
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            pred = network(
                x,
                inits,
                features,
                mask=mask,
                init_root=init_root,
                cam_angvel=cam_angvel,
                return_y_up=True,
                **kwargs,
            )

            pred_body_pose = (
                matrix_to_axis_angle(pred["poses_body"]).cpu().numpy().reshape(-1, 69)
            )
            pred_root = (
                matrix_to_axis_angle(pred["poses_root_cam"]).cpu().numpy().reshape(-1, 3)
            )
            pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
            pred_trans = (pred["trans_cam"] - network.output.offset).cpu().numpy()

            results[_id] = {
                "pose": pred_pose,
                "betas": pred["betas"].cpu().squeeze(0).numpy(),
                "trans": pred_trans,
                "frame_ids": frame_id,
            }
            if "poses_root_world" in pred:
                pred_root_world = (
                    matrix_to_axis_angle(pred["poses_root_world"]).cpu().numpy().reshape(-1, 3)
                )
                results[_id]["pose_world"] = np.concatenate(
                    (pred_root_world, pred_body_pose), axis=-1
                )
            if "trans_world" in pred:
                results[_id]["trans_world"] = pred["trans_world"].cpu().squeeze(0).numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_pkl:
        joblib.dump(results, output_dir / "wham_results.pkl")
    if save_npz and len(results) == 1:
        only = next(iter(results.values()))
        np.savez(
            output_dir / "wham_output.npz",
            pose=only["pose"],
            betas=only["betas"],
            trans=only["trans"],
            frame_ids=only["frame_ids"],
        )

    logger.info("Inference complete. Outputs written to {}", output_dir)
    return InferenceOutput(results=results, tracking_results=tracking_results, slam_results=slam_results)


class WHAMInference:
    def __init__(
        self,
        config_path: str | Path | None = None,
        device: str = "cuda",
        flip_eval: bool = False,
    ) -> None:
        self.config_path = config_path
        self.device = device
        self.flip_eval = flip_eval

    def __call__(
        self,
        video: str | Path,
        pose_data: str | Path,
        output_dir: str | Path,
        save_npz: bool = True,
        save_pkl: bool = True,
    ) -> InferenceOutput:
        return run_inference(
            video=video,
            pose_data=pose_data,
            output_dir=output_dir,
            config_path=self.config_path,
            device=self.device,
            flip_eval=self.flip_eval,
            save_npz=save_npz,
            save_pkl=save_pkl,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WHAM inference with pre-extracted poses.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--pose-data",
        required=True,
        help="NPZ/PKL file with keypoints and bboxes.",
    )
    parser.add_argument("--output-dir", default="output/inference", help="Output directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Override config path (defaults to configs/yamls/demo.yaml).",
    )
    parser.add_argument("--device", default="cuda", help="torch device (cuda or cpu).")
    parser.add_argument(
        "--flip-eval",
        action="store_true",
        help="Run flip evaluation for inference.",
    )
    parser.add_argument("--no-npz", action="store_true", help="Disable NPZ output.")
    parser.add_argument("--no-pkl", action="store_true", help="Disable PKL output.")

    args = parser.parse_args()
    run_inference(
        video=args.video,
        pose_data=args.pose_data,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
        flip_eval=args.flip_eval,
        save_npz=not args.no_npz,
        save_pkl=not args.no_pkl,
    )


if __name__ == "__main__":
    main()
