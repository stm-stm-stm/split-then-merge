"""Thin video / mask I/O helpers (numpy in, numpy out)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


def _to_numpy(batch) -> np.ndarray:
    # decord returns an NDArray (native bridge) or a torch tensor (torch bridge)
    if hasattr(batch, "asnumpy"):
        return batch.asnumpy()
    if hasattr(batch, "numpy"):
        return batch.numpy()
    return np.asarray(batch)


def video_info(path: Path | str) -> Tuple[int, float, int, int]:
    """Return (num_frames, fps, height, width)."""
    import decord

    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    first = _to_numpy(vr[0])
    return len(vr), float(vr.get_avg_fps()), int(first.shape[0]), int(first.shape[1])


def read_video(
    path: Path | str,
    num_frames: Optional[int] = None,
    start: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """Read frames as uint8 [T, H, W, 3] (RGB)."""
    import decord

    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    n = len(vr)
    if num_frames is None:
        idx = list(range(start, n, stride))
    else:
        idx = list(range(start, start + num_frames * stride, stride))
    idx = [i for i in idx if i < n]
    if not idx:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return _to_numpy(vr.get_batch(idx)).astype(np.uint8)


def write_video(path: Path | str, frames: Iterable[np.ndarray] | np.ndarray, fps: int = 16) -> None:
    """Write uint8 RGB frames with mediapy (H.264, high quality)."""
    import mediapy as media

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(list(frames)) if not isinstance(frames, np.ndarray) else frames
    media.write_video(str(path), frames, fps=fps)


def read_mask_video(path: Path | str, num_frames: Optional[int] = None) -> np.ndarray:
    """Read a (3-channel, 0/255) mask video as bool [T, H, W]."""
    frames = read_video(path, num_frames=num_frames)
    return frames[..., 0] > 127


def mask_to_frames(mask: np.ndarray) -> np.ndarray:
    """bool [T, H, W] -> uint8 [T, H, W, 3] (0/255)."""
    m = (mask.astype(np.uint8) * 255)[..., None]
    return np.repeat(m, 3, axis=-1)


def resize_video(frames: np.ndarray, height: int, width: int, is_mask: bool = False) -> np.ndarray:
    """Plain resize to (height, width) without cropping — the paper's convention ("videos are resized to
    49x480x720"; the released test videos are direct resizes of 854x480 DAVIS frames)."""
    if frames.shape[1] == height and frames.shape[2] == width:
        return np.ascontiguousarray(frames)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return np.stack([cv2.resize(f, (width, height), interpolation=interp) for f in frames])


def center_crop_resize(frames: np.ndarray, height: int, width: int, is_mask: bool = False) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize.

    Accepts [T, H, W, C] or [T, H, W]. Masks use nearest-neighbour resizing.
    """
    t, h, w = frames.shape[:3]
    target_ar = width / height
    if w / h > target_ar:  # too wide -> crop width
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        frames = frames[:, :, x0 : x0 + new_w]
    elif w / h < target_ar:  # too tall -> crop height
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        frames = frames[:, y0 : y0 + new_h]
    if frames.shape[1] == height and frames.shape[2] == width:
        return np.ascontiguousarray(frames)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    out = np.stack([cv2.resize(f, (width, height), interpolation=interp) for f in frames])
    return out


def motion_score(frames: np.ndarray, step: int = 4, size: int = 128) -> float:
    """Mean absolute temporal difference on a downsampled gray video (0-255)."""
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA) for f in frames]).astype(
        np.float32
    )
    if len(small) <= step:
        return 0.0
    return float(np.abs(small[step:] - small[:-step]).mean())


def save_first_frame(frames: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
