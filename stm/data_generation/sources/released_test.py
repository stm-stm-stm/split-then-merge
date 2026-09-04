"""Build the paper's evaluation set from the released ``Organized_Results`` folders.

Each ``sample-<i>_gs-<gs>_<prompt>-<hash>/`` directory of the released results
contains the *inputs* of one test triplet (``fg.mp4`` foreground-on-black,
``bg.mp4`` background, ``prompt.txt``) next to the outputs of StM and every
baseline.  This converts them into a validation-dir manifest usable by
``inference/compose.py`` / the trainer's validation loop.

The foreground mask is recovered from the black-background foreground video
(non-black pixels, small holes closed); it only feeds the latent-space mask
(max-pooled 8x), so this approximation is harmless.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ..config import DecomposerConfig
from ..video_io import mask_to_frames, read_video, save_first_frame, write_video

_SAMPLE_RE = re.compile(r"^sample-(\d+)_gs-([\d.]+)_(.*)$")


def mask_from_black_fg(fg: np.ndarray, threshold: int = 12) -> np.ndarray:
    m = (fg.max(axis=-1) > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    return np.stack([cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel) for f in m]).astype(bool)


def import_released_test_set(cfg: DecomposerConfig, results_root: Path, dataset_name: str = "stm_test") -> int:
    out = cfg.data_root / "datasets" / dataset_name
    src = "released"
    dirs = sorted(p for p in results_root.iterdir() if p.is_dir() and _SAMPLE_RE.match(p.name))
    # every sample folder is a distinct (fg, bg, prompt) test case; the same prompt / foreground
    # is reused with several backgrounds, and each folder carries its selected guidance scale
    keep = []
    for d in dirs:
        idx, gs, tail = _SAMPLE_RE.match(d.name).groups()
        if not (d / "fg.mp4").exists() or not (d / "bg.mp4").exists():
            continue
        keep.append((int(idx), d, tail))
    keep.sort()
    lines = {k: [] for k in ("prompts.txt", "videos.txt", "fg.txt", "fg-black.txt", "bg.txt", "masks.txt", "images.txt")}
    for idx, d, tail in keep:
        name = f"sample{idx:04d}-{tail[:40]}"
        fg = read_video(d / "fg.mp4")
        bg = read_video(d / "bg.mp4")
        mask = mask_from_black_fg(fg)
        write_video(out / src / "fg" / f"{name}-00.mp4", fg, fps=cfg.fps)
        write_video(out / src / "bg" / f"{name}-bg.mp4", bg, fps=cfg.fps)
        write_video(out / src / "masks" / f"{name}-00.mp4", mask_to_frames(mask), fps=cfg.fps)
        write_video(out / src / "videos" / f"{name}.mp4", bg, fps=cfg.fps)  # no ground truth composite exists
        save_first_frame(bg, out / src / "first_frames" / f"{name}.png")
        prompt = (d / "prompt.txt").read_text().strip() if (d / "prompt.txt").exists() else ""
        lines["prompts.txt"].append(" ".join(prompt.split()) or "None")
        lines["videos.txt"].append(f"{src}/videos/{name}.mp4")
        lines["fg.txt"].append(f"{src}/fg/{name}-00.mp4")
        lines["fg-black.txt"].append(f"{src}/fg/{name}-00.mp4")
        lines["bg.txt"].append(f"{src}/bg/{name}-bg.mp4")
        lines["masks.txt"].append(f"{src}/masks/{name}-00.mp4")
        lines["images.txt"].append(f"{src}/first_frames/{name}.png")
    out.mkdir(parents=True, exist_ok=True)
    for k, v in lines.items():
        (out / k).write_text("\n".join(v) + "\n")
    print(f"[released-test] {len(keep)} triplets -> {out}")
    return len(keep)
