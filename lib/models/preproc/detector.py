from __future__ import annotations

import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

# Prefer bundled ViTPose's mmpose implementation; its config/checkpoint pair
# uses legacy TopDown APIs that are incompatible with newer pip mmpose.
if osp.isdir(VIT_DIR) and VIT_DIR not in sys.path:
    sys.path.insert(0, VIT_DIR)

try:
    # mmpose <=0.x style
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, get_track_id
    _MMPOSE_V1 = False
except ImportError:
    # mmpose >=1.x style
    from mmpose.apis import inference_topdown, init_model
    try:
        from mmpose.apis import get_track_id
    except ImportError:
        get_track_id = None
    _MMPOSE_V1 = True

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30
MINIMUM_JOINTS = 6

class DetectionModel(object):
    def __init__(self, device):
        
        # ViTPose
        pose_model_cfg = osp.join(VIT_DIR, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py')
        pose_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'vitpose-h-multi-coco.pth')
        if _MMPOSE_V1:
            self.pose_model = init_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        else:
            self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device.lower())
        
        # YOLO
        bbox_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'yolov8x.pt')
        self.bbox_model = YOLO(bbox_model_ckpt)
        
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])
        
    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox
    
    def track(self, img, fps, length):
        
        # bbox detection
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()
        bboxes = [{'bbox': bbox} for bbox in bboxes]
        
        # keypoints detection
        if _MMPOSE_V1:
            xyxy = np.array([b["bbox"] for b in bboxes], dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            pose_results = inference_topdown(self.pose_model, img, bboxes=xyxy, bbox_format="xyxy")
            pose_results = self._to_legacy_pose_dicts(pose_results)
        else:
            pose_results, _ = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results=bboxes,
                format='xyxy',
                return_heatmap=False,
                outputs=None)

        # person identification
        if get_track_id is not None:
            pose_results, self.next_id = get_track_id(
                pose_results,
                self.pose_results_last,
                self.next_id,
                use_oks=False,
                tracking_thr=TRACKING_THR,
                use_one_euro=True,
                fps=fps)
        else:
            pose_results = self._assign_track_ids_by_iou(pose_results)
        
        for pose_result in pose_results:
            n_valid = (pose_result['keypoints'][:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS: continue
            
            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'])
        
        self.frame_id += 1
        self.pose_results_last = pose_results

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(ix2 - ix1, 0.0)
        ih = max(iy2 - iy1, 0.0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = max((ax2 - ax1), 0.0) * max((ay2 - ay1), 0.0)
        ub = max((bx2 - bx1), 0.0) * max((by2 - by1), 0.0)
        denom = ua + ub - inter
        return inter / denom if denom > 0 else 0.0

    def _assign_track_ids_by_iou(self, pose_results):
        used_prev = set()
        for cur in pose_results:
            best_iou = 0.0
            best_idx = -1
            for i, prev in enumerate(self.pose_results_last):
                if i in used_prev:
                    continue
                iou = self._bbox_iou(cur["bbox"], prev["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= TRACKING_THR:
                cur["track_id"] = self.pose_results_last[best_idx]["track_id"]
                used_prev.add(best_idx)
            else:
                cur["track_id"] = self.next_id
                self.next_id += 1
        return pose_results

    @staticmethod
    def _to_legacy_pose_dicts(results):
        out = []
        for sample in results:
            pred = sample.pred_instances
            kpts = np.asarray(pred.keypoints)
            if kpts.ndim == 3:
                kpts = kpts[0]
            if hasattr(pred, "keypoint_scores"):
                scores = np.asarray(pred.keypoint_scores)
                if scores.ndim == 2:
                    scores = scores[0]
            else:
                scores = np.ones((kpts.shape[0],), dtype=kpts.dtype)
            if kpts.shape[-1] == 2:
                kpts = np.concatenate([kpts, scores[:, None]], axis=1)
            bbox = np.asarray(pred.bboxes)
            if bbox.ndim == 2:
                bbox = bbox[0]
            out.append({"bbox": bbox, "keypoints": kpts})
        return out
    
    def process(self, fps):
        for key in ['id', 'frame_id', 'keypoints']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        self.compute_bboxes_from_keypoints()
            
        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output
