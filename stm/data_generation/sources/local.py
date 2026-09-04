"""Import an arbitrary folder of videos as a Decomposer source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..chunking import chunk_directory
from ..config import DecomposerConfig


def import_local(
    cfg: DecomposerConfig, raw_dir: Path, source: str, is_training: bool = True, max_videos: Optional[int] = None, seed: int = 0
) -> int:
    out_dir = cfg.clips_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source.json").write_text(json.dumps({"is_training": is_training, "origin": str(raw_dir)}))
    return chunk_directory(raw_dir, cfg, source, max_videos=max_videos, seed=seed)
