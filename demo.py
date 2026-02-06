from wham.inference import WHAMInference
from wham.visualize import render_multiview_video

runner = WHAMInference(device="cuda:0")
outputs = runner(
    video="tmp/cljwno0m600043n6lq54umusu.mp4",
    pose_data="tmp/cljwno0m600043n6lq54umusu_pose.npz",
    output_dir="output/inference",
)

# Render a simple multiview visualization from the inference output.
render_multiview_video(
    smpl_output_path="output/inference/wham_output.npz",
    output_path="output/inference/wham_multiview.mp4",
    device="cuda:0",
)
