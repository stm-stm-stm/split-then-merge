"""DAVIS 2017 -> StM clips with ground-truth masks (validation set).

The paper reserves DAVIS exclusively for validation, and for VOS datasets the
provided masks replace the motion segmenter.  Each sequence becomes one clip:
first 49 frames resized to 480x720 (no crop); the union of
all annotated objects is the foreground mask.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..chunking import ClipInfo
from ..config import DecomposerConfig
from ..video_io import mask_to_frames, resize_video, write_video


def import_davis(cfg: DecomposerConfig, davis_root: Path, source: str = "davis", split: Optional[str] = None) -> int:
    img_root = davis_root / "JPEGImages" / "480p"
    ann_root = davis_root / "Annotations" / "480p"
    seqs: List[str] = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if split:
        seqs = [s.strip() for s in (davis_root / "ImageSets" / "2017" / f"{split}.txt").read_text().split()]
    out_dir = cfg.clips_dir / source
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "source.json").write_text(json.dumps({"is_training": False, "origin": "DAVIS-2017"}))
    written = 0
    for seq in seqs:
        files = sorted((img_root / seq).glob("*.jpg"))[: cfg.num_frames]
        if len(files) < cfg.num_frames:
            print(f"[davis] {seq}: only {len(files)} frames, skipped")
            continue
        clip_path = out_dir / f"{seq}.mp4"
        if clip_path.exists():
            continue
        frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in files])
        masks = np.stack([np.array(Image.open(ann_root / seq / (f.stem + ".png")).convert("P")) > 0 for f in files])
        frames = resize_video(frames, cfg.height, cfg.width)
        masks = resize_video(masks.astype(np.uint8), cfg.height, cfg.width, is_mask=True).astype(bool)
        write_video(clip_path, frames, fps=cfg.fps)
        write_video(out_dir / "masks" / f"{seq}.mp4", mask_to_frames(masks), fps=cfg.fps)
        info = ClipInfo(clip_id=seq, source=source, source_video=str(img_root / seq), start_frame=0, stride=1, source_fps=24.0)
        (out_dir / f"{seq}.json").write_text(json.dumps(asdict(info), indent=1))
        written += 1
    print(f"[davis] wrote {written} clips -> {out_dir}")
    return written
