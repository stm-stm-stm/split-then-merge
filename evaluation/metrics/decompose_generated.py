"""Split generated videos into foreground / background layers with Segment-Any-Motion.

All metrics compare *layers*, so every generated video is decomposed with the same
Decomposer that produced the training data (``stm.data_generation.segmentation``).
Outputs per sample: ``<out>/<sample>/{mask,fg,bg}.mp4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from stm.data_generation.config import DecomposerConfig
from stm.data_generation.layers import compose_layers
from stm.data_generation.video_io import mask_to_frames, write_video

from .video_utils import Sample, read_frames


def layer_paths(out_dir: Path, sample: Sample) -> Dict[str, Path]:
    d = Path(out_dir) / sample.name
    return {"mask": d / "mask.mp4", "fg": d / "fg.mp4", "bg": d / "bg.mp4"}


def decompose_samples(samples: Iterable[Sample], out_dir: Path, cfg: Optional[DecomposerConfig] = None, device: str = "cuda") -> int:
    """Run the segmenter on every generated video that has no layers yet. Returns #decomposed."""
    from stm.data_generation.segmentation import SegAnyMoSegmenter

    cfg = cfg or DecomposerConfig()
    cfg.export_env()
    segmenter = None
    n = 0
    for s in samples:
        paths = layer_paths(out_dir, s)
        if all(p.exists() for p in paths.values()):
            continue
        if segmenter is None:
            segmenter = SegAnyMoSegmenter(
                cfg.seganymo_root,
                sam2_ckpt=cfg.sam2_ckpt,
                tapir_ckpt=cfg.tapir_ckpt,
                moseg_repo=cfg.moseg_repo,
                depth_model=cfg.depth_model,
                dino_model=cfg.dino_model,
                step=cfg.seganymo_step,
                device=device,
                seed=cfg.seed,
                tracks_per_query_frame=cfg.tracks_per_query_frame,
            )
        frames = read_frames(s.generated)
        mask, _ = segmenter.segment(frames, cfg.work_dir / "eval_decompose")
        if mask is None:  # no moving object found: empty foreground
            mask = np.zeros(frames.shape[:3], dtype=bool)
        fg, bg = compose_layers(frames, mask)
        write_video(paths["mask"], mask_to_frames(mask), fps=cfg.fps)
        write_video(paths["fg"], fg, fps=cfg.fps)
        write_video(paths["bg"], bg, fps=cfg.fps)
        n += 1
    return n
