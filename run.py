import argparse

from wham import WHAMRunner, prepare_cfg


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="input video path or youtube link",
    )
    parser.add_argument(
        "--pose_npz",
        type=str,
        required=True,
        help="npz file with 2D keypoints",
    )
    parser.add_argument("--output_pth", type=str, default="output/demo", help="output folder to write results")
    parser.add_argument("--calib", type=str, default=None, help="Camera calibration file path")
    parser.add_argument(
        "--estimate_local_only",
        action="store_true",
        help="Only estimate motion in camera coordinate if True",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the output mesh if True")
    parser.add_argument("--save_pkl", action="store_true", help="Save output as pkl file")
    parser.add_argument("--run_smplify", action="store_true", help="Run Temporal SMPLify for post processing")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/yamls/demo.yaml",
        help="path to WHAM config yaml",
    )
    return parser


def _run_from_args(args):
    cfg = prepare_cfg(args.config)
    runner = WHAMRunner(cfg=cfg)
    runner.run(
        video=args.video,
        output_dir=args.output_pth,
        pose_npz=args.pose_npz,
        calib=args.calib,
        run_global=not args.estimate_local_only,
        save_pkl=args.save_pkl,
        visualize=args.visualize,
        run_smplify=args.run_smplify,
    )


def main():
    args = _build_parser().parse_args()
    _run_from_args(args)


def main_pose_npz():
    # Backward-compatible entrypoint alias.
    main()


if __name__ == "__main__":
    main()
