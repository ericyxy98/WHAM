"""Minimal Python API usage example for WHAM."""

from wham import WHAMRunner


def main():
    runner = WHAMRunner()
    runner.run(
        video="tmp/r47s.mp4",
        pose_npz="tmp/r47s_pose.npz",
        visualize=True,
    )


if __name__ == "__main__":
    main()
