import os
import argparse
import os.path as osp
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from pprint import pprint
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

try:
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except Exception:
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

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


def _sanitize_tracking_bboxes(tracking_results, width, height, min_scale=0.05):
    # Ensure bbox center/scale are finite and within reasonable bounds
    max_dim = max(width, height)
    max_scale = max_dim / 50.0  # generous upper bound (200*scale ~ 4*max_dim)
    for _id, res in tracking_results.items():
        bb = res.get('bbox', None)
        if bb is None:
            continue
        bb = np.asarray(bb, dtype=np.float32)
        # Replace non-finite with center of image and small scale
        bad = ~np.isfinite(bb).all(axis=1)
        if bad.any():
            bb[bad, 0] = width / 2.0
            bb[bad, 1] = height / 2.0
            bb[bad, 2] = min_scale
        # Clamp center to image bounds
        bb[:, 0] = np.clip(bb[:, 0], 0.0, width - 1.0)
        bb[:, 1] = np.clip(bb[:, 1], 0.0, height - 1.0)
        # Clamp scale
        bb[:, 2] = np.clip(bb[:, 2], min_scale, max_scale)
        res['bbox'] = bb


def _normalize_keypoints(kp):
    kp = np.asarray(kp)
    if kp.dtype == object:
        # Try to stack list/array of arrays
        elems = list(kp)
        if len(elems) > 0 and hasattr(elems[0], "shape"):
            try:
                kp = np.stack(elems, axis=0)
            except Exception:
                # leave as-is; fall through to error for clarity
                pass
    if kp.ndim == 2:
        # (T, J*2) or (T, J*3)
        if kp.shape[1] % 3 == 0:
            j = kp.shape[1] // 3
            kp = kp.reshape(kp.shape[0], j, 3)
        elif kp.shape[1] % 2 == 0:
            j = kp.shape[1] // 2
            kp = kp.reshape(kp.shape[0], j, 2)
        else:
            raise ValueError('keypoints 2D array must have width divisible by 2 or 3')
        kp = kp[None, ...]

    if kp.ndim == 3:
        # (T,J,2/3) or (T,2/3,J)
        if kp.shape[2] in (2, 3):
            pass
        elif kp.shape[1] in (2, 3):
            kp = np.transpose(kp, (0, 2, 1))
        else:
            raise ValueError('keypoints must have shape (T,J,2/3) or (T,2/3,J)')

        if kp.shape[2] == 2:
            conf = np.ones((kp.shape[0], kp.shape[1], 1), dtype=kp.dtype)
            kp = np.concatenate([kp, conf], axis=2)
        kp = kp[None, ...]

    elif kp.ndim == 4:
        # (N,T,J,2/3) or (N,T,2/3,J)
        if kp.shape[3] in (2, 3):
            pass
        elif kp.shape[2] in (2, 3):
            kp = np.transpose(kp, (0, 1, 3, 2))
        else:
            raise ValueError('keypoints must have shape (N,T,J,2/3) or (N,T,2/3,J)')

        if kp.shape[3] == 2:
            conf = np.ones((kp.shape[0], kp.shape[1], kp.shape[2], 1), dtype=kp.dtype)
            kp = np.concatenate([kp, conf], axis=3)
    else:
        raise ValueError(f'keypoints must have ndim 2, 3, or 4 (got {kp.ndim}, shape {kp.shape})')
    return kp


def load_pose_npz(pose_npz, max_frames=None):
    data = np.load(pose_npz, allow_pickle=True)
    if 'keypoints' in data:
        keypoints = data['keypoints']
    elif 'poses2d' in data:
        keypoints = data['poses2d']
    elif 'pose2d' in data:
        keypoints = data['pose2d']
    else:
        raise ValueError('pose npz must contain keypoints/poses2d/pose2d')
    
    if keypoints.shape[2] > 3:
        logger.warning('Keypoints have more than 3 channels, only using first 3 (x,y,conf)')
        keypoints = keypoints[..., [0, 1, 2]]

    keypoints = _normalize_keypoints(keypoints)
    n_people, n_frames = keypoints.shape[0], keypoints.shape[1]

    if max_frames is not None:
        if n_frames > max_frames:
            logger.warning(f'Pose file has {n_frames} frames, trimming to {max_frames}.')
            keypoints = keypoints[:, :max_frames]
            n_frames = max_frames
        elif n_frames < max_frames:
            logger.warning(f'Pose file has {n_frames} frames, video has {max_frames}. Using {n_frames} frames.')

    if 'frame_ids' in data:
        frame_id = np.array(data['frame_ids'])
    elif 'frame_id' in data:
        frame_id = np.array(data['frame_id'])
    else:
        frame_id = np.arange(n_frames)

    if frame_id.ndim == 2:
        frame_id = frame_id[0]
    frame_id = frame_id[:n_frames]

    if 'bbox' in data:
        bbox = np.array(data['bbox'])
    elif 'bboxes' in data:
        bbox = np.array(data['bboxes'])
    else:
        bbox = None

    tracking_results = defaultdict(dict)
    for pid in range(n_people):
        tracking_results[pid]['keypoints'] = keypoints[pid]
        tracking_results[pid]['frame_id'] = frame_id
        tracking_results[pid]['features'] = []
        tracking_results[pid]['flipped_features'] = []
        tracking_results[pid]['flipped_bbox'] = []
        tracking_results[pid]['flipped_keypoints'] = []
        if bbox is not None:
            if bbox.ndim == 2:
                bb = bbox[:n_frames]
            elif bbox.ndim == 3:
                bb = bbox[pid][:n_frames]
            else:
                raise ValueError('bbox must have shape (T,3/4) or (N,T,3/4)')

            # If xyxy, convert to (cx, cy, scale)
            if bb.shape[1] == 4:
                # Treat input as xywh
                bb = _xywh_to_xyxy(bb)
                bb = _sanitize_bbox_xyxy(bb)
                cx = (bb[:, 0] + bb[:, 2]) / 2.0
                cy = (bb[:, 1] + bb[:, 3]) / 2.0
                s = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
                bb = np.stack([cx, cy, s], axis=1).astype(np.float32)
                bb[:, 2] = bb[:, 2] * 1.2 / 200.0
            elif bb.shape[1] != 3:
                raise ValueError('bbox must have 3 (cx,cy,scale) or 4 (x1,y1,x2,y2) columns')

            # Guard against invalid/zero/negative scale
            bb[:, 2] = np.clip(bb[:, 2], 1e-4, None)

            tracking_results[pid]['bbox'] = bb
        else:
            tracking_results[pid]['bbox'] = _compute_bbox_from_keypoints(keypoints[pid])

    return tracking_results


def run(cfg,
        video,
        output_pth,
        network,
        pose_npz,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False):

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    with torch.no_grad():
        # Build tracking results from pose npz
        tracking_results = load_pose_npz(pose_npz, max_frames=length)
        _sanitize_tracking_bboxes(tracking_results, width, height)

        # SLAM for global trajectory (optional)
        if run_global:
            logger.info('Running SLAM for global trajectory estimation...')
            slam = SLAMModel(video, output_pth, width, height, calib)
            while cap.isOpened():
                flag, _img = cap.read()
                if not flag:
                    break
                slam.track()
            slam_results = slam.process()
        else:
            slam_results = np.zeros((length, 7))
            slam_results[:, 3] = 1.0  # Unit quaternion

        # Extract image features and init pose from image
        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
        tracking_results = extractor.run(video, tracking_results)
        logger.info('Complete Data preprocessing from pose npz!')

        joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
        joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Save processed data at {output_pth}')

    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

    # run WHAM
    results = defaultdict(dict)

    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)

                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)

                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2

                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch

                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)

        if args.run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)

            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()

        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id

    if save_pkl:
        joblib.dump(results, osp.join(output_pth, 'wham_output.pkl'))

    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str,
                        default='examples/demo_video.mp4',
                        help='input video path or youtube link')

    parser.add_argument('--pose_npz', type=str, required=True,
                        help='npz file with 2D keypoints to skip pose detection')

    parser.add_argument('--output_pth', type=str, default='output/demo',
                        help='output folder to write results')

    parser.add_argument('--calib', type=str, default=None,
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')

    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')

    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')

    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)

    run(cfg,
        args.video,
        output_pth,
        network,
        args.pose_npz,
        args.calib,
        run_global=not args.estimate_local_only,
        save_pkl=args.save_pkl,
        visualize=args.visualize)

    print()
    logger.info('Done !')
