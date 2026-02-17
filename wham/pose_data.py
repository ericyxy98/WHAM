from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from loguru import logger

VIS_THRESH = 0.3


def _compute_bbox_from_keypoints(keypoints, s_factor=1.2):
    # keypoints: (T, J, 3)
    mask = keypoints[..., -1] > VIS_THRESH
    bbox = np.zeros((len(keypoints), 3), dtype=np.float32)
    for i, (kp, m) in enumerate(zip(keypoints, mask)):
        if not np.any(m):
            bbox[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            continue
        bb = [kp[m, 0].min(), kp[m, 1].min(), kp[m, 0].max(), kp[m, 1].max()]
        cx, cy = [(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2]
        bb_w = bb[2] - bb[0]
        bb_h = bb[3] - bb[1]
        s = np.stack((bb_w, bb_h)).max()
        bbox[i] = np.array([cx, cy, s], dtype=np.float32)
    bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
    return bbox


def _sanitize_bbox_xyxy(bb):
    # Ensure x1<=x2, y1<=y2 and positive sizes
    x1 = np.minimum(bb[:, 0], bb[:, 2])
    y1 = np.minimum(bb[:, 1], bb[:, 3])
    x2 = np.maximum(bb[:, 0], bb[:, 2])
    y2 = np.maximum(bb[:, 1], bb[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1)


def _xywh_to_xyxy(bb):
    # (x, y, w, h) -> (x1, y1, x2, y2)
    w = np.abs(bb[:, 2])
    h = np.abs(bb[:, 3])
    w = np.where(w > 0, w, 1.0)
    h = np.where(h > 0, h, 1.0)
    x1 = bb[:, 0]
    y1 = bb[:, 1]
    x2 = bb[:, 0] + w
    y2 = bb[:, 1] + h
    return np.stack([x1, y1, x2, y2], axis=1)


def sanitize_tracking_bboxes(tracking_results, width, height, min_scale=0.05):
    # Ensure bbox center/scale are finite and within reasonable bounds
    max_dim = max(width, height)
    max_scale = max_dim / 50.0  # generous upper bound (200*scale ~ 4*max_dim)
    for _id, res in tracking_results.items():
        bb = res.get("bbox", None)
        if bb is None:
            continue
        bb = np.asarray(bb, dtype=np.float32)
        bad = ~np.isfinite(bb).all(axis=1)
        if bad.any():
            bb[bad, 0] = width / 2.0
            bb[bad, 1] = height / 2.0
            bb[bad, 2] = min_scale
        bb[:, 0] = np.clip(bb[:, 0], 0.0, width - 1.0)
        bb[:, 1] = np.clip(bb[:, 1], 0.0, height - 1.0)
        bb[:, 2] = np.clip(bb[:, 2], min_scale, max_scale)
        res["bbox"] = bb


def _normalize_keypoints(kp):
    kp = np.asarray(kp)
    if kp.dtype == object:
        elems = list(kp)
        if len(elems) > 0 and hasattr(elems[0], "shape"):
            try:
                kp = np.stack(elems, axis=0)
            except Exception:
                pass
    if kp.ndim == 2:
        if kp.shape[1] % 3 == 0:
            j = kp.shape[1] // 3
            kp = kp.reshape(kp.shape[0], j, 3)
        elif kp.shape[1] % 2 == 0:
            j = kp.shape[1] // 2
            kp = kp.reshape(kp.shape[0], j, 2)
        else:
            raise ValueError("keypoints 2D array must have width divisible by 2 or 3")
        kp = kp[None, ...]

    if kp.ndim == 3:
        c_last = kp.shape[2]
        c_mid = kp.shape[1]
        if 2 <= c_last <= 8:
            pass
        elif 2 <= c_mid <= 8:
            kp = np.transpose(kp, (0, 2, 1))
        else:
            raise ValueError(
                f"keypoints must have a channel axis (size 2~8) in dim 1 or 2, got shape {kp.shape}"
            )

        if kp.shape[2] == 2:
            conf = np.ones((kp.shape[0], kp.shape[1], 1), dtype=kp.dtype)
            kp = np.concatenate([kp, conf], axis=2)
        kp = kp[None, ...]

    elif kp.ndim == 4:
        c_last = kp.shape[3]
        c_mid = kp.shape[2]
        if 2 <= c_last <= 8:
            pass
        elif 2 <= c_mid <= 8:
            kp = np.transpose(kp, (0, 1, 3, 2))
        else:
            raise ValueError(
                f"keypoints must have a channel axis (size 2~8) in dim 2 or 3, got shape {kp.shape}"
            )

        if kp.shape[3] == 2:
            conf = np.ones((kp.shape[0], kp.shape[1], kp.shape[2], 1), dtype=kp.dtype)
            kp = np.concatenate([kp, conf], axis=3)
    else:
        raise ValueError(f"keypoints must have ndim 2, 3, or 4 (got {kp.ndim}, shape {kp.shape})")
    return kp


def load_pose_data(pose_data, max_frames=None):
    if isinstance(pose_data, (str, os.PathLike)):
        data = np.load(pose_data, allow_pickle=True)
        if "keypoints" in data:
            keypoints = data["keypoints"]
        elif "poses2d" in data:
            keypoints = data["poses2d"]
        elif "pose2d" in data:
            keypoints = data["pose2d"]
        else:
            raise ValueError("pose npz must contain keypoints/poses2d/pose2d")

        if "frame_ids" in data:
            frame_id = np.array(data["frame_ids"])
        elif "frame_id" in data:
            frame_id = np.array(data["frame_id"])
        else:
            frame_id = None

        if "bbox" in data:
            bbox = np.array(data["bbox"])
        elif "bboxes" in data:
            bbox = np.array(data["bboxes"])
        else:
            bbox = None
    else:
        keypoints = np.asarray(pose_data)
        frame_id = None
        bbox = None

    keypoints = _normalize_keypoints(keypoints)

    if keypoints.shape[-1] == 4:
        # Input format: (x, y, confidence, visibility)
        conf = keypoints[..., 2]
        vis = keypoints[..., 3]
        conf = conf * (vis > 0).astype(keypoints.dtype)
        keypoints = np.stack([keypoints[..., 0], keypoints[..., 1], conf], axis=-1)
    elif keypoints.shape[-1] > 4:
        logger.warning("Keypoints have more than 4 channels, only using first 3 (x,y,conf)")
        keypoints = keypoints[..., [0, 1, 2]]

    n_people, n_frames = keypoints.shape[0], keypoints.shape[1]

    if max_frames is not None:
        if n_frames > max_frames:
            logger.warning(f"Pose file has {n_frames} frames, trimming to {max_frames}.")
            keypoints = keypoints[:, :max_frames]
            n_frames = max_frames
        elif n_frames < max_frames:
            logger.warning(f"Pose file has {n_frames} frames, video has {max_frames}. Using {n_frames} frames.")

    if frame_id is None:
        frame_id = np.arange(n_frames)
    if frame_id.ndim == 2:
        frame_id = frame_id[0]
    frame_id = frame_id[:n_frames]

    tracking_results = defaultdict(dict)
    for pid in range(n_people):
        tracking_results[pid]["keypoints"] = keypoints[pid]
        tracking_results[pid]["frame_id"] = frame_id
        tracking_results[pid]["features"] = []
        tracking_results[pid]["flipped_features"] = []
        tracking_results[pid]["flipped_bbox"] = []
        tracking_results[pid]["flipped_keypoints"] = []

        if bbox is not None:
            if bbox.ndim == 2:
                bb = bbox[:n_frames]
            elif bbox.ndim == 3:
                bb = bbox[pid][:n_frames]
            else:
                raise ValueError("bbox must have shape (T,3/4) or (N,T,3/4)")

            if bb.shape[1] == 4:
                bb = _xywh_to_xyxy(bb)
                bb = _sanitize_bbox_xyxy(bb)
                cx = (bb[:, 0] + bb[:, 2]) / 2.0
                cy = (bb[:, 1] + bb[:, 3]) / 2.0
                s = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
                bb = np.stack([cx, cy, s], axis=1).astype(np.float32)
                bb[:, 2] = bb[:, 2] * 1.2 / 200.0
            elif bb.shape[1] != 3:
                raise ValueError("bbox must have 3 (cx,cy,scale) or 4 (x,y,w,h) columns")

            bb[:, 2] = np.clip(bb[:, 2], 1e-4, None)
            tracking_results[pid]["bbox"] = bb
        else:
            tracking_results[pid]["bbox"] = _compute_bbox_from_keypoints(keypoints[pid])

    return tracking_results


def load_pose_npz(pose_npz, max_frames=None):
    return load_pose_data(pose_npz, max_frames=max_frames)
