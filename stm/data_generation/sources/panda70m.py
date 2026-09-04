"""Panda-70M clip selection + download (yt-dlp).

The Panda-70M CSV (``videoID,url,timestamp,caption,matching_score``) lists,
per YouTube video, a list of semantically-consistent clips.  We keep clips
whose caption mentions a plausible foreground subject (animal / person /
vehicle) and does not look like a talking-head / screen recording, then
download each clip section at <=720p.
"""

from __future__ import annotations

import ast
import json
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SUBJECT_RE = re.compile(
    r"\b(dog|dogs|cat|cats|horse|horses|bird|birds|cow|cows|sheep|goat|goats|pig|pigs|elephant|lion|tiger|bear|"
    r"monkey|deer|zebra|giraffe|duck|ducks|swan|chicken|rabbit|fish|dolphin|whale|shark|turtle|frog|snake|"
    r"squirrel|fox|wolf|camel|kangaroo|penguin|eagle|owl|parrot|butterfly|animal|animals|puppy|kitten|"
    r"man|woman|person|people|boy|girl|child|children|kid|kids|player|players|skier|surfer|swimmer|runner|"
    r"dancer|cyclist|rider|athlete|skateboarder|snowboarder|climber|"
    r"car|cars|truck|bus|motorcycle|bicycle|bike|boat|ship|train|airplane|plane|helicopter|tractor|jeep|"
    r"kayak|jet ski|drone)\b",
    re.IGNORECASE,
)
EXCLUDE_RE = re.compile(
    r"\b(news|anchor|talking|interview|interviewed|speech|speaking|podcast|screen|screenshot|monitor|"
    r"cartoon|animated|animation|video game|gameplay|slideshow|text|logo|title|map|diagram|chart|"
    r"microphone|podium|desk|studio|webcam|zoom call|presentation|lecture|tutorial|graphic)\b",
    re.IGNORECASE,
)


def _parse_list(s: str) -> List:
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def _to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def select_clips(
    csv_path: Path,
    max_videos: int,
    min_score: float = 0.44,
    min_duration: float = 3.0,
    max_duration: float = 40.0,
    max_clips_per_video: int = 2,
    seed: int = 0,
    chunksize: int = 100_000,
) -> List[Dict]:
    """Stream the CSV and return a random subset of subject-bearing clips."""
    rng = random.Random(seed)
    candidates: List[Dict] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            stamps, caps, scores = _parse_list(row.timestamp), _parse_list(row.caption), _parse_list(row.matching_score)
            picks = []
            for k, (st, cap, sc) in enumerate(zip(stamps, caps, scores)):
                if sc < min_score or not SUBJECT_RE.search(cap) or EXCLUDE_RE.search(cap):
                    continue
                try:
                    t0, t1 = _to_seconds(st[0]), _to_seconds(st[1])
                except Exception:
                    continue
                if not (min_duration <= t1 - t0 <= max_duration):
                    continue
                picks.append({"videoID": row.videoID, "url": row.url, "k": k, "start": t0, "end": t1, "caption": cap, "score": sc})
            if picks:
                rng.shuffle(picks)
                candidates.append(picks[:max_clips_per_video])
    rng.shuffle(candidates)
    selected = [c for group in candidates[:max_videos] for c in group]
    print(f"[panda70m] {len(candidates)} candidate videos, selected {len(selected)} clips from {min(max_videos, len(candidates))} videos")
    return selected


def _resolve_ytdlp(ytdlp: str) -> str:
    """Prefer the yt-dlp installed next to the running interpreter."""
    import shutil
    import sys

    cand = Path(sys.executable).parent / ytdlp
    if cand.exists():
        return str(cand)
    found = shutil.which(ytdlp)
    if found is None:
        raise FileNotFoundError(f"{ytdlp} not found; `pip install yt-dlp`")
    return found


def _download_one(item: Dict, out_dir: Path, height: int, ytdlp: str, cookies: Optional[str] = None) -> Optional[Path]:
    out = out_dir / f"{item['videoID']}_{item['k']:02d}.mp4"
    sidecar = out.with_suffix(".json")
    if out.exists() and out.stat().st_size > 0:
        return out
    fail_marker = out_dir / f".{out.stem}.failed"
    if fail_marker.exists():
        return None
    section = f"*{item['start']:.3f}-{item['end']:.3f}"
    cmd = [
        ytdlp,
        "--no-warnings",
        "--quiet",
        "--no-playlist",
        "--no-part",
        "-f",
        f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]/b[height<={height}]/bv*/b",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "-o",
        str(out),
        item["url"],
    ]
    if cookies:  # YouTube frequently requires an authenticated session from cloud IPs
        cmd += ["--cookies", cookies]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            fail_marker.write_text((r.stderr or "")[-2000:])
            out.unlink(missing_ok=True)
            return None
    except Exception as exc:
        fail_marker.write_text(repr(exc))
        out.unlink(missing_ok=True)
        return None
    sidecar.write_text(json.dumps(item, indent=1))
    return out


def download(
    selected: List[Dict], out_dir: Path, workers: int = 4, height: int = 720, ytdlp: str = "yt-dlp", cookies: Optional[str] = None
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ytdlp = _resolve_ytdlp(ytdlp)
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_one, it, out_dir, height, ytdlp, cookies) for it in selected]
        for i, f in enumerate(as_completed(futs)):
            if f.result() is not None:
                ok += 1
            if (i + 1) % 25 == 0:
                print(f"[panda70m] {i + 1}/{len(selected)} attempted, {ok} ok")
    print(f"[panda70m] downloaded {ok}/{len(selected)} clips -> {out_dir}")
    return ok
