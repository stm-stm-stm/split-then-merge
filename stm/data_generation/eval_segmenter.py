"""Benchmark the motion segmenter against ground-truth masks (e.g. DAVIS clips).

For every clip that has a GT mask video next to it (``clips/<source>/masks/<clip>.mp4``)
the SegAnyMo port is run *without* the GT and the per-sequence IoU / foreground ratio
is recorded to ``<dataset_root>/segmenter_eval/<clip>.json``.  Runs one process per GPU.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DecomposerConfig
from .video_io import mask_to_frames, read_mask_video, read_video, write_video


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def _worker(worker_id: int, num_workers: int, jobs: List[Dict], cfg_dict: Dict, gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cfg = DecomposerConfig(**cfg_dict)
    cfg.export_env()
    from .segmentation import SegAnyMoSegmenter

    out_dir = cfg.dataset_root / ("segmenter_eval_fast" if cfg.fast_segmenter else "segmenter_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = SegAnyMoSegmenter(
        cfg.seganymo_root,
        sam2_ckpt=cfg.sam2_ckpt,
        tapir_ckpt=cfg.tapir_ckpt,
        moseg_repo=cfg.moseg_repo,
        depth_model=cfg.depth_model,
        dino_model=cfg.dino_model,
        step=cfg.seganymo_step,
        seed=cfg.seed,
        tracks_per_query_frame=cfg.tracks_per_query_frame,
        fast=cfg.fast_segmenter,
    )
    work = cfg.work_dir / f"segeval{worker_id}"
    for job in jobs[worker_id::num_workers]:
        clip, gt = Path(job["clip"]), Path(job["gt_mask"])
        res_path = out_dir / f"{clip.stem}.json"
        if res_path.exists():
            continue
        t0 = time.time()
        frames = read_video(clip)
        try:
            mask, n_obj = seg.segment(frames, work)
        except Exception as exc:  # noqa: BLE001
            res_path.write_text(json.dumps({"clip_id": clip.stem, "error": repr(exc)[:300]}))
            continue
        gt_mask = read_mask_video(gt, num_frames=len(frames))
        rec = {"clip_id": clip.stem, "source": job["source"], "seconds": round(time.time() - t0, 1), "num_objects": n_obj}
        if mask is None:
            rec.update(iou=0.0, pred_ratio=0.0, gt_ratio=float(gt_mask.mean()), no_dynamic_object=True)
        else:
            rec.update(iou=_iou(mask, gt_mask), pred_ratio=float(mask.mean()), gt_ratio=float(gt_mask.mean()), no_dynamic_object=False)
            write_video(out_dir / f"{clip.stem}-pred.mp4", mask_to_frames(mask), fps=cfg.fps)
        res_path.write_text(json.dumps(rec))
        print(
            f"[segeval {worker_id}] {clip.stem}: iou={rec['iou']:.3f} ratio={rec['pred_ratio']:.3f}/{rec['gt_ratio']:.3f} {rec['seconds']}s",
            flush=True,
        )


def run(cfg: DecomposerConfig, sources: Sequence[str], gpus: Sequence[int], limit: Optional[int] = None) -> None:
    from dataclasses import asdict

    import torch.multiprocessing as mp

    jobs = []
    for src in sources:
        cdir = cfg.clips_dir / src
        for clip in sorted(cdir.glob("*.mp4")):
            gt = cdir / "masks" / clip.name
            if gt.exists():
                jobs.append({"clip": str(clip), "gt_mask": str(gt), "source": src})
    if limit:
        jobs = jobs[:limit]
    print(f"[segeval] {len(jobs)} clips with GT masks, {len(gpus)} workers", flush=True)
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    mp.set_start_method("spawn", force=True)
    procs = []
    for w, gpu in enumerate(gpus):
        p = mp.Process(target=_worker, args=(w, len(gpus), jobs, cfg_dict, gpu), daemon=True)
        p.start()
        procs.append(p)
        time.sleep(10)
    for p in procs:
        p.join()
    summarize(cfg)


def summarize(cfg: DecomposerConfig) -> Dict:
    out_dir = cfg.dataset_root / ("segmenter_eval_fast" if cfg.fast_segmenter else "segmenter_eval")
    recs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    ok = [r for r in recs if "iou" in r]
    if not ok:
        print("[segeval] no results")
        return {}
    ious = np.array([r["iou"] for r in ok])
    summary = {
        "n": len(ok),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou>=0.5": float((ious >= 0.5).mean()),
        "no_dynamic_object": int(sum(r["no_dynamic_object"] for r in ok)),
        "errors": len(recs) - len(ok),
        "mean_seconds": float(np.mean([r["seconds"] for r in ok])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[segeval]", json.dumps(summary))
    for r in sorted(ok, key=lambda r: r["iou"])[:8]:
        print(f"  worst: {r['clip_id']:25s} iou={r['iou']:.3f} pred={r['pred_ratio']:.3f} gt={r['gt_ratio']:.3f}")
    return summary
