"""OpenVid-1M (Nan et al., 2024) as a Decomposer source.

OpenVid-1M is a curated, directly downloadable (Hugging Face, CC-BY-4.0)
subset of Panda-70M / ChronoMagic / Open-Sora-plan / CelebV-HQ with captions,
aesthetic / motion / temporal-consistency scores and camera-motion tags.  It is
the practical stand-in for the YouTube-hosted Panda-70M used in the paper
(YouTube blocks unauthenticated bulk downloads from cloud IPs).

Videos live in ``OpenVid_part<N>.zip`` archives (30-47 GB each, ~5-7k videos).
Only members that pass the caption / score filters are extracted.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..chunking import chunk_directory
from ..config import DecomposerConfig
from .panda70m import EXCLUDE_RE, SUBJECT_RE

# OpenVid captions are long VLM descriptions that mention "a man/woman" for almost every
# clip, so we additionally require either a clearly moving subject class (animal / vehicle)
# or an action verb, and drop static close-up / talking-head content.
ACTION_RE = re.compile(
    r"\b(walk|walks|walking|run|runs|running|jump|jumps|jumping|rid(e|es|ing)|swim|swims|swimming|fly|flies|flying|"
    r"danc(e|es|ing)|play(s|ing)? (football|soccer|basketball|tennis|volleyball|golf|hockey|baseball|frisbee|with)|"
    r"skat(e|es|ing)|surf(s|ing)?|climb(s|ing)?|kick(s|ing)?|throw(s|ing)?|driv(e|es|ing)|gallop(s|ing)?|"
    r"chas(e|es|ing)|div(e|es|ing)|slid(e|es|ing)|ski(s|ing)?|snowboard(s|ing)?|row(s|ing)?|paddl(e|es|ing)|"
    r"crawl(s|ing)?|hop(s|ping)?|leap(s|ing)?|sprint(s|ing)?|trot(s|ting)?|jog(s|ging)?|cycl(e|es|ing)|bik(e|es|ing)|"
    r"skateboard(s|ing)?|wrestl(e|es|ing)|box(es|ing)|fight(s|ing)?|march(es|ing)?|graz(e|es|ing)|feed(s|ing)?|"
    r"hunt(s|ing)?|dig(s|ging)?|roll(s|ing)?|spin(s|ning)?|flip(s|ping)?|carr(y|ies|ying)|push(es|ing)?|pull(s|ing)?|"
    r"lift(s|ing)?|exercis(e|es|ing)|stretch(es|ing)?|perform(s|ing)? (a |an )?(trick|stunt|flip|jump))\b",
    re.IGNORECASE,
)
MOVING_SUBJECT_RE = re.compile(
    r"\b(dog|dogs|puppy|cat|cats|kitten|horse|horses|bird|birds|cow|cows|sheep|goat|goats|pig|pigs|elephant|lion|tiger|bear|"
    r"monkey|deer|zebra|giraffe|duck|ducks|swan|chicken|rabbit|fish|dolphin|whale|shark|turtle|frog|snake|squirrel|fox|"
    r"wolf|camel|kangaroo|penguin|eagle|owl|parrot|butterfly|animal|animals|"
    r"car|cars|truck|bus|motorcycle|bicycle|bike|boat|ship|train|airplane|plane|helicopter|tractor|jeep|kayak|jet ski|drone|"
    r"player|players|skier|surfer|swimmer|runner|dancer|cyclist|rider|athlete|skateboarder|snowboarder|climber|gymnast)\b",
    re.IGNORECASE,
)
STATIC_RE = re.compile(
    r"\b(close-up|closeup|close up|talking|speaking|smiling at the camera|looking at the camera|sitting at a (desk|table)|"
    r"interview|kitchen|plate of|food|cooking|dish|recipe|makeup|hair|selfie|screen|monitor|text|logo|studio|"
    r"podcast|microphone|vlog|blurred background|portrait|headshot)\b",
    re.IGNORECASE,
)

# OpenVid-1M members by filename: Panda-70M-derived clips (`<youtube id>_<clip>_<start>to<end>.mp4`, 63%),
# CelebV-HQ face videos (`celebv_*`), Mixkit/Pexels/Pixabay stock landscapes and MagicTime time-lapses.
# Only the Panda-70M-derived part has moving subjects suitable for composition (as in the paper).
PANDA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{11}_\d+_\d+to\d+\.mp4$")

CSV_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
PART_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}.zip"


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def select_videos(
    df: pd.DataFrame,
    min_motion: float = 3.0,
    min_aesthetic: float = 4.5,
    min_seconds: float = 2.5,
    panda_only: bool = True,
) -> pd.DataFrame:
    """Keep videos with a plausible *moving* subject and decent quality."""
    keep = (df["motion_score"] >= min_motion) & (df["aesthetic_score"] >= min_aesthetic) & (df["seconds"] >= min_seconds)
    if panda_only:
        keep &= df["video"].astype(str).str.match(PANDA_NAME_RE)
    caps = df["caption"].fillna("")
    keep &= caps.str.contains(SUBJECT_RE) & ~caps.str.contains(EXCLUDE_RE) & ~caps.str.contains(STATIC_RE)
    keep &= caps.str.contains(MOVING_SUBJECT_RE) | caps.str.contains(ACTION_RE)
    return df[keep]


def load_selection(csv_path: Path, min_motion: float = 3.0, min_aesthetic: float = 4.5, refresh: bool = False) -> Dict[str, str]:
    """{video basename: caption} of the filtered videos, cached next to the CSV (the filter takes minutes)."""
    cache = csv_path.parent / f"selection_m{min_motion}_a{min_aesthetic}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    df = select_videos(load_metadata(csv_path), min_motion=min_motion, min_aesthetic=min_aesthetic)
    print(f"[openvid] {len(df)} / metadata rows pass the filters")
    captions = dict(zip(df["video"], df["caption"]))
    cache.write_text(json.dumps(captions))
    return captions


def extract_selected(zip_path: Path, wanted: Iterable[str], out_dir: Path, captions: Optional[Dict[str, str]] = None) -> int:
    """Extract only ``wanted`` members (basenames) of an OpenVid zip; write caption sidecars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(wanted)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name not in wanted:
                continue
            dst = out_dir / name
            if not dst.exists():
                with zf.open(info) as src, open(dst, "wb") as f:
                    while True:
                        chunk = src.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
            if captions and name in captions:
                dst.with_suffix(".json").write_text(json.dumps({"caption": captions[name], "origin": "OpenVid-1M"}))
            n += 1
    print(f"[openvid] extracted {n} selected videos from {zip_path.name} -> {out_dir}")
    return n


def import_openvid_remote(
    cfg: DecomposerConfig,
    parts: List[int],
    csv_path: Path,
    source: str = "openvid",
    min_motion: float = 3.0,
    min_aesthetic: float = 4.5,
    workers: int = 8,
    seed: int = 0,
) -> int:
    """Fetch only the filtered videos of the given zip parts via HTTP ranges (no zip download), then chunk."""
    from .openvid_index import fetch_selected_members

    captions = load_selection(csv_path, min_motion=min_motion, min_aesthetic=min_aesthetic)
    raw_dir = cfg.raw_dir / source / "videos"
    for p in parts:
        fetch_selected_members(p, captions, raw_dir, workers=workers)
    out_dir = cfg.clips_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source.json").write_text(json.dumps({"is_training": True, "origin": "OpenVid-1M"}))
    return chunk_directory(raw_dir, cfg, source, seed=seed)


def import_openvid(
    cfg: DecomposerConfig,
    zip_paths: List[Path],
    csv_path: Path,
    source: str = "openvid",
    min_motion: float = 3.0,
    min_aesthetic: float = 4.5,
    max_videos: Optional[int] = None,
    seed: int = 0,
) -> int:
    captions = load_selection(csv_path, min_motion=min_motion, min_aesthetic=min_aesthetic)
    raw_dir = cfg.raw_dir / source / "videos"
    for zp in zip_paths:
        extract_selected(zp, captions.keys(), raw_dir, captions)
    out_dir = cfg.clips_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source.json").write_text(json.dumps({"is_training": True, "origin": "OpenVid-1M"}))
    return chunk_directory(raw_dir, cfg, source, max_videos=max_videos, seed=seed)
