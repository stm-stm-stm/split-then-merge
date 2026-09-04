"""Split raw source videos into fixed-length StM clips.

Paper (A1.1): "Videos are either directly split into multiple chunks of 49
frames, or first downsampled in frame rate (by a factor of 2, 3, or 4) and
subsequently split into 49-frame chunks."  The stride is sampled once per
source video; chunks are non-overlapping; every chunk is centre-cropped to
3:2 and resized to 480x720; static chunks are discarded.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

from .config import DecomposerConfig
from .video_io import motion_score, read_video, resize_video, video_info, write_video


@dataclass
class ClipInfo:
    clip_id: str
    source: str  # dataset/source name (e.g. panda70m)
    source_video: str  # path of the raw video
    start_frame: int
    stride: int
    source_fps: float
    source_caption: Optional[str] = None  # caption shipped with the source (fallback only)


def iter_clips(
    video_path: Path,
    cfg: DecomposerConfig,
    source: str,
    rng: random.Random,
    max_clips: Optional[int] = None,
    source_caption: Optional[str] = None,
) -> Iterator[Tuple[ClipInfo, np.ndarray]]:
    """Yield (ClipInfo, frames[T,H,W,3]) for one raw video."""
    try:
        n, fps, h, w = video_info(video_path)
    except Exception as exc:  # corrupted download etc.
        print(f"[chunk] cannot open {video_path}: {exc}")
        return
    if min(h, w) < 240:
        return
    candidates = [s for s in cfg.strides if n >= cfg.num_frames * s]
    if not candidates:
        return
    stride = rng.choice(candidates)
    span = cfg.num_frames * stride
    starts = list(range(0, n - span + 1, span))
    if max_clips is not None and len(starts) > max_clips:
        # spread the selected chunks over the whole video
        idx = np.linspace(0, len(starts) - 1, max_clips).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    stem = video_path.stem
    for k, start in enumerate(starts):
        frames = read_video(video_path, num_frames=cfg.num_frames, start=start, stride=stride)
        if len(frames) < cfg.num_frames:
            continue
        frames = resize_video(frames, cfg.height, cfg.width)
        if motion_score(frames) < cfg.static_threshold:
            continue
        info = ClipInfo(
            clip_id=f"{stem}-{k:03d}",
            source=source,
            source_video=str(video_path),
            start_frame=start,
            stride=stride,
            source_fps=fps,
            source_caption=source_caption,
        )
        yield info, frames


def _chunk_one(vp: Path, cfg: DecomposerConfig, source: str, out_dir: Path, seed: int) -> int:
    """Chunk a single raw video (runs in a worker process)."""
    done_marker = out_dir / f".{vp.stem}.chunked"
    if done_marker.exists():
        return 0
    caption = None
    sidecar = vp.with_suffix(".json")
    if sidecar.exists():
        try:
            caption = json.loads(sidecar.read_text()).get("caption")
        except Exception:
            caption = None
    rng = random.Random(f"{seed}:{vp.stem}")
    written = 0
    try:
        for info, frames in iter_clips(vp, cfg, source, rng, cfg.max_clips_per_video, caption):
            clip_path = out_dir / f"{info.clip_id}.mp4"
            write_video(clip_path, frames, fps=cfg.fps)
            (out_dir / f"{info.clip_id}.json").write_text(json.dumps(asdict(info), indent=1))
            written += 1
    except Exception as exc:
        print(f"[chunk] failed on {vp.name}: {exc}")
    done_marker.touch()
    return written


def chunk_directory(
    raw_dir: Path,
    cfg: DecomposerConfig,
    source: str,
    max_videos: Optional[int] = None,
    seed: int = 0,
    exts: Sequence[str] = (".mp4", ".mkv", ".webm", ".mov"),
    workers: int = 16,
) -> int:
    """Chunk every video below ``raw_dir`` into ``cfg.clips_dir / source`` (multi-process).

    Sidecar ``<video>.json`` files (written by the downloaders) may carry a
    ``caption`` that is kept as ``source_caption``.  Returns #clips written.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = cfg.clips_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in exts)
    if max_videos:
        videos = videos[:max_videos]
    written = 0
    if workers <= 1:
        for vp in videos:
            written += _chunk_one(vp, cfg, source, out_dir, seed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_chunk_one, vp, cfg, source, out_dir, seed) for vp in videos]
            for i, f in enumerate(as_completed(futs)):
                written += f.result()
                if (i + 1) % 200 == 0:
                    print(f"[chunk] {source}: {i + 1}/{len(videos)} videos, {written} clips so far", flush=True)
    print(f"[chunk] {source}: wrote {written} clips from {len(videos)} videos -> {out_dir}")
    return written
