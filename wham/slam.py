from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


def _resolve_dpvo_script(dpvo_script: str | Path | None) -> Path:
    if dpvo_script:
        return Path(dpvo_script).expanduser().resolve()

    env_script = os.environ.get("DPVO_SCRIPT")
    if env_script:
        return Path(env_script).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "third-party" / "DPVO" / "demo.py",
        repo_root / "third-party" / "DPVO" / "run.py",
        repo_root / "third-party" / "DPVO" / "run_dpvo.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "DPVO script not found. Set --slam-script or DPVO_SCRIPT to point to a DPVO demo script."
    )


def _build_dpvo_command(
    script: Path,
    video: str | Path,
    output: str | Path,
    device: str,
    config: str | Path | None,
    checkpoint: str | Path | None,
) -> list[str]:
    cmd: list[str] = ["python", str(script), "--video", str(video), "--output", str(output)]
    if config:
        cmd.extend(["--config", str(config)])
    if checkpoint:
        cmd.extend(["--checkpoint", str(checkpoint)])
    if device:
        cmd.extend(["--device", str(device)])
    return cmd


def _load_dpvo_output(output_path: str | Path) -> np.ndarray:
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".npy":
        traj = np.load(output_path)
    elif output_path.suffix.lower() == ".npz":
        data = np.load(output_path, allow_pickle=True)
        for key in ("traj", "trajectory", "poses"):
            if key in data:
                traj = data[key]
                break
        else:
            if len(data.files) != 1:
                raise ValueError(f"Unsupported DPVO npz keys: {data.files}")
            traj = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported DPVO output file: {output_path}")

    traj = np.asarray(traj)
    if traj.ndim != 2 or traj.shape[1] != 7:
        raise ValueError(f"DPVO output must be (T, 7), got {traj.shape}")
    return traj.astype(np.float32, copy=False)


def run_dpvo(
    video: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    dpvo_script: str | Path | None = None,
    output_name: str = "dpvo_traj.npy",
    extra_args: Iterable[str] | None = None,
) -> np.ndarray:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    script = _resolve_dpvo_script(dpvo_script)
    cmd = _build_dpvo_command(script, video, output_path, device, config, checkpoint)
    if extra_args:
        cmd.extend([str(arg) for arg in extra_args])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "DPVO failed with exit code "
            f"{result.returncode}. Stdout: {result.stdout}\nStderr: {result.stderr}"
        )

    if not output_path.exists():
        raise FileNotFoundError(f"DPVO did not produce output at {output_path}")

    return _load_dpvo_output(output_path)

