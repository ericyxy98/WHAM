from __future__ import annotations

import os.path as osp
from collections import defaultdict

import cv2
import joblib
import numpy as np
import torch
from loguru import logger

from lib.data.datasets import CustomDataset
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from wham.pose_data import load_pose_data, sanitize_tracking_bboxes

try:
    from lib.models.preproc.slam import SLAMModel

    _run_global = True
except Exception:
    logger.warning("DPVO is not properly installed. Only estimate in local coordinates !")
    _run_global = False


def _run_inference(
    cfg,
    video,
    output_pth,
    network,
    tracking_results,
    slam_results,
    width,
    height,
    fps,
    run_global,
    save_pkl=False,
    visualize=False,
    run_smplify=False,
):
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    results = defaultdict(dict)

    n_subjs = len(dataset)
    for subj in range(n_subjs):
        with torch.no_grad():
            if cfg.FLIP_EVAL:
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                batch = dataset.load_data(subj)
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

                flipped_pose, flipped_shape = flipped_pred["pose"].squeeze(0), flipped_pred["betas"].squeeze(0)
                pose, shape = pred["pose"].squeeze(0), pred["betas"].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred["contact"][..., [2, 3, 0, 1]] + pred["contact"]) / 2

                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            else:
                batch = dataset.load_data(subj)
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

        if run_smplify:
            smpl = network.smpl
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]["keypoints"]
            pred = smplify.fit(pred, input_keypoints, **kwargs)

            with torch.no_grad():
                network.pred_pose = pred["pose"]
                network.pred_shape = pred["betas"]
                network.pred_cam = pred["cam"]
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        pred_body_pose = matrix_to_axis_angle(pred["poses_body"]).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred["poses_root_cam"]).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred["poses_root_world"]).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred["trans_cam"] - network.output.offset).cpu().numpy()

        results[_id]["pose"] = pred_pose
        results[_id]["trans"] = pred_trans
        results[_id]["pose_world"] = pred_pose_world
        results[_id]["trans_world"] = pred["trans_world"].cpu().squeeze(0).numpy()
        results[_id]["betas"] = pred["betas"].cpu().squeeze(0).numpy()
        results[_id]["verts"] = (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).cpu().numpy()
        results[_id]["frame_ids"] = frame_id

    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))

    if visualize:
        from lib.vis.run_vis import run_vis_on_demo

        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)

    return results, tracking_results, slam_results


def run_pose_demo(
    cfg,
    video,
    output_pth,
    network,
    pose_data,
    calib=None,
    run_global=True,
    save_pkl=False,
    visualize=False,
    run_smplify=False,
):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"Faild to load video file {video}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    run_global = run_global and _run_global

    with torch.no_grad():
        tracking_results = load_pose_data(pose_data, max_frames=length)
        sanitize_tracking_bboxes(tracking_results, width, height)

        if run_global:
            logger.info("Running SLAM for global trajectory estimation...")
            slam = SLAMModel(video, output_pth, width, height, calib)
            while cap.isOpened():
                flag, _img = cap.read()
                if not flag:
                    break
                slam.track()
            slam_results = slam.process()
        else:
            slam_results = np.zeros((length, 7))
            slam_results[:, 3] = 1.0

        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
        tracking_results = extractor.run(video, tracking_results)
        logger.info("Complete Data preprocessing from pose npz!")

        joblib.dump(tracking_results, osp.join(output_pth, "tracking_results.pth"))
        joblib.dump(slam_results, osp.join(output_pth, "slam_results.pth"))
        logger.info(f"Save processed data at {output_pth}")

    return _run_inference(
        cfg,
        video,
        output_pth,
        network,
        tracking_results,
        slam_results,
        width,
        height,
        fps,
        run_global,
        save_pkl=save_pkl,
        visualize=visualize,
        run_smplify=run_smplify,
    )
