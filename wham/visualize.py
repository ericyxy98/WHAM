from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio
import joblib
import numpy as np
import torch
from loguru import logger
from progress.bar import Bar
from lib.models import build_body_model


def _load_smpl_output(path: str | Path, subject_id: int | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return {
            "pose": data["pose"],
            "betas": data["betas"],
            "trans": data["trans"],
            "frame_ids": data.get("frame_ids", np.arange(len(data["pose"]))),
        }
    if path.suffix.lower() in {".pkl", ".pickle"}:
        data = joblib.load(path)
        if subject_id is None:
            if len(data) != 1:
                raise ValueError("Multiple subjects found; pass --subject-id.")
            subject_id = next(iter(data.keys()))
        return data[subject_id]
    raise ValueError(f"Unsupported output file: {path}")


def _broadcast_to_frames(values: np.ndarray, frames: int) -> np.ndarray:
    if values.ndim == 1:
        return np.tile(values[None, :], (frames, 1))
    if values.shape[0] == 1:
        return np.tile(values, (frames, 1))
    return values


def _compute_camera_transforms(
    centers: torch.Tensor, distance: float, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from pytorch3d.renderer.cameras import look_at_rotation

    device = centers.device
    centers = centers.to(device)
    offset = offset.to(device)
    positions = centers + offset
    rotation = look_at_rotation(positions, centers).mT.to(device)
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    return rotation, translation


def render_multiview_video(
    smpl_output_path: str | Path,
    output_path: str | Path,
    size: int = 512,
    fps: int = 30,
    device: str = "cuda",
    subject_id: int | None = None,
) -> Path:
    smpl_output = _load_smpl_output(smpl_output_path, subject_id=subject_id)
    pose = smpl_output["pose"]
    trans = smpl_output["trans"]
    betas = smpl_output["betas"]

    frames = pose.shape[0]
    betas = _broadcast_to_frames(np.asarray(betas), frames)
    trans = _broadcast_to_frames(np.asarray(trans), frames)

    pose = torch.from_numpy(pose).float().to(device)
    betas = torch.from_numpy(betas).float().to(device)
    trans = torch.from_numpy(trans).float().to(device)

    body_model = build_body_model(device, batch_size=frames)
    smpl_out = body_model.get_output(
        global_orient=pose[:, :3],
        body_pose=pose[:, 3:],
        betas=betas,
        transl=trans,
        pose2rot=True,
    )
    verts = smpl_out.vertices

    extent = (verts.max(dim=1).values - verts.min(dim=1).values).max().item()
    distance = max(extent * 2.5, 2.5)
    centers = verts.mean(dim=1)

    offsets = {
        "front": torch.tensor([0.0, 0.0, distance], device=device),
        "back": torch.tensor([0.0, 0.0, -distance], device=device),
        "left": torch.tensor([-distance, 0.0, 0.0], device=device),
        "right": torch.tensor([distance, 0.0, 0.0], device=device),
    }

    cameras = {
        name: _compute_camera_transforms(centers, distance, offset)
        for name, offset in offsets.items()
    }

    focal_length = (size**2 + size**2) ** 0.5
    from lib.vis.renderer import Renderer

    renderer = Renderer(size, size, focal_length, device, body_model.faces)
    background = np.ones((size, size, 3), dtype=np.uint8) * 255

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path), fps=fps, mode="I", format="FFMPEG", macro_block_size=1
    )
    bar = Bar("Rendering multiview", fill="#", max=frames)

    with torch.no_grad():
        for idx in range(frames):
            views = {}
            for name, (rotation, translation) in cameras.items():
                renderer.create_camera(rotation[idx], translation[idx])
                views[name] = renderer.render_mesh(verts[idx], background)

            top = np.concatenate((views["front"], views["back"]), axis=1)
            bottom = np.concatenate((views["left"], views["right"]), axis=1)
            grid = np.concatenate((top, bottom), axis=0)
            writer.append_data(grid)
            bar.next()

    writer.close()
    logger.info("Multiview video written to {}", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 4-view SMPL video.")
    parser.add_argument(
        "--smpl-output",
        required=True,
        help="wham_output.npz or wham_results.pkl from wham-infer.",
    )
    parser.add_argument("--output", default="output/multiview.mp4", help="Output mp4 path.")
    parser.add_argument("--size", type=int, default=512, help="Render size per view.")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS.")
    parser.add_argument("--device", default="cuda", help="torch device (cuda or cpu).")
    parser.add_argument("--subject-id", type=int, default=None, help="Subject id for PKL.")
    args = parser.parse_args()

    render_multiview_video(
        smpl_output_path=args.smpl_output,
        output_path=args.output,
        size=args.size,
        fps=args.fps,
        device=args.device,
        subject_id=args.subject_id,
    )


if __name__ == "__main__":
    main()
