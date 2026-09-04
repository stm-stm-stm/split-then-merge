"""Shared helpers: result-folder discovery and frame I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from stm.data_generation.video_io import read_mask_video, read_video


@dataclass
class Sample:
    """One evaluation case: generated composition + the (fg, bg, prompt) inputs it was made from."""

    name: str
    generated: Path
    fg_gt: Path
    bg_gt: Path
    prompt: str
    mask_gt: Optional[Path] = None  # input foreground mask when available


def iter_samples(results_dir: Path, layout: str = "compose", method: str = "gen-recomposer") -> Iterator[Sample]:
    """Yield samples from a results folder.

    ``layout="compose"`` — folders written by ``inference/compose.py``:
        ``sample-<i>_gs-<gs>_<name>/{video_generated,video_fg_gt,video_bg_gt}.mp4 + prompt.txt``
    ``layout="released"`` — the released ``Organized_Results`` folders:
        ``sample-…/{fg,bg,<method>}.mp4 + prompt.txt``
    """
    for d in sorted(p for p in Path(results_dir).iterdir() if p.is_dir() and p.name.startswith("sample-")):
        prompt_file = d / "prompt.txt"
        prompt = prompt_file.read_text().strip() if prompt_file.exists() else ""
        if layout == "compose":
            gen, fg, bg = d / "video_generated.mp4", d / "video_fg_gt.mp4", d / "video_bg_gt.mp4"
        else:
            gen, fg, bg = d / f"{method}.mp4", d / "fg.mp4", d / "bg.mp4"
        if gen.exists() and fg.exists() and bg.exists():
            yield Sample(name=d.name, generated=gen, fg_gt=fg, bg_gt=bg, prompt=prompt)


def read_frames(path: Path, num_frames: Optional[int] = None) -> np.ndarray:
    """uint8 RGB frames [T, H, W, 3]."""
    return read_video(path, num_frames=num_frames)


def read_mask(path: Path, num_frames: Optional[int] = None) -> np.ndarray:
    """bool [T, H, W] from a 0/255 mask video."""
    return read_mask_video(path, num_frames=num_frames)


def sample_frames(frames: np.ndarray, num: int) -> np.ndarray:
    """Evenly sample ``num`` frames (repeat the last one if the clip is shorter)."""
    frames = list(frames)
    if len(frames) < num:
        frames = frames + [frames[-1]] * (num - len(frames))
    step = len(frames) // num
    return np.stack(frames[::step][:num])


def apply_mask(frames: np.ndarray, mask: np.ndarray, keep_foreground: bool) -> np.ndarray:
    """Black out the foreground (keep_foreground=False) or the background of ``frames``."""
    t = min(len(frames), len(mask))
    m = mask[:t].astype(np.uint8)[..., None]
    if not keep_foreground:
        m = 1 - m
    return frames[:t] * m


def summarize(rows: List[dict], keys: List[str]) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        out[k] = {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}
    return out
