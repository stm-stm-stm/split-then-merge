"""Per-clip Decomposer state machine + multi-worker driver (single GPU)."""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import DecomposerConfig
from .layers import compose_layers, foreground_ratio
from .manifest import record_path
from .video_io import mask_to_frames, read_mask_video, read_video, save_first_frame, write_video

STAGES = ("caption", "mask", "layers", "inpaint")


def _load_json(p: Path) -> Optional[Dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _dump_json(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


class ClipProcessor:
    """Runs mask -> (fg-ratio gate) -> caption -> layers -> inpaint for one clip with resumable records."""

    def __init__(self, cfg: DecomposerConfig, device: str = "cuda", worker_id: int = 0):
        cfg.export_env()
        self.cfg = cfg
        self.device = device
        self.worker_id = worker_id
        self._captioner = None
        self._segmenter = None
        self._inpainter = None
        self.work_dir = cfg.work_dir / f"worker{worker_id}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # lazy model loading -----------------------------------------------------
    @property
    def captioner(self):
        if self._captioner is None:
            from .captioning import InternVLCaptioner

            self._captioner = InternVLCaptioner(
                self.cfg.caption_model,
                device=self.device,
                num_segments=self.cfg.caption_num_segments,
                max_new_tokens=self.cfg.caption_max_new_tokens,
                prompt=self.cfg.caption_prompt,
            )
        return self._captioner

    @property
    def segmenter(self):
        if self._segmenter is None:
            from .segmentation import SegAnyMoSegmenter

            self._segmenter = SegAnyMoSegmenter(
                self.cfg.seganymo_root,
                sam2_ckpt=self.cfg.sam2_ckpt,
                tapir_ckpt=self.cfg.tapir_ckpt,
                moseg_repo=self.cfg.moseg_repo,
                depth_model=self.cfg.depth_model,
                dino_model=self.cfg.dino_model,
                step=self.cfg.seganymo_step,
                device=self.device,
                seed=self.cfg.seed,
                tracks_per_query_frame=self.cfg.tracks_per_query_frame,
                fast=self.cfg.fast_segmenter,
            )
        return self._segmenter

    @property
    def inpainter(self):
        if self._inpainter is None:
            from .inpainting import MiniMaxRemover

            self._inpainter = MiniMaxRemover(
                self.cfg.minimax_root,
                repo_id=self.cfg.minimax_repo,
                device=self.device,
                num_inference_steps=self.cfg.inpaint_steps,
                dilation_iterations=self.cfg.inpaint_dilation_iters,
                seed=self.cfg.seed,
            )
        return self._inpainter

    # paths ------------------------------------------------------------------
    def _paths(self, source: str, clip_id: str) -> Dict[str, Path]:
        root = self.cfg.dataset_root / source
        return {
            "video": root / "videos" / f"{clip_id}.mp4",
            "mask": root / "masks" / f"{clip_id}-00.mp4",
            "fg": root / "fg" / f"{clip_id}-00.mp4",
            "bg": root / "bg" / f"{clip_id}-bg.mp4",
            "bg_inpainted": root / "bg-inpainted" / f"{clip_id}-bg.mp4",
            "first_frame": root / "first_frames" / f"{clip_id}.png",
        }

    # main -------------------------------------------------------------------
    def process(self, clip_path: Path, source: str, gt_mask_path: Optional[Path] = None, is_training: bool = True) -> Dict:
        cfg = self.cfg
        clip_id = clip_path.stem
        rec_p = record_path(cfg, clip_id)
        rec = _load_json(rec_p) or {}
        if rec.get("status") in ("done", "skipped"):
            return rec
        info = _load_json(clip_path.with_suffix(".json")) or {}
        rec.update(
            {
                "clip_id": clip_id,
                "source": source,
                "clip_path": str(clip_path),
                "is_training": is_training,
                "stages_done": rec.get("stages_done", []),
                "status": "running",
                "worker": self.worker_id,
                "clip_info": info,
            }
        )
        paths = self._paths(source, clip_id)
        t0 = time.time()
        frames = read_video(clip_path)
        T, H, W = frames.shape[:3]
        if T != cfg.num_frames or H != cfg.height or W != cfg.width:
            rec.update(status="skipped", reason=f"bad_shape:{T}x{H}x{W}")
            _dump_json(rec_p, rec)
            return rec
        rec.update(num_frames=T, height=H, width=W)
        done = set(rec["stages_done"])
        timings = rec.get("timings", {})

        # 1. mask (the expensive stage; done first so that clips failing the ratio gate cost nothing else)
        if "mask" not in done:
            t = time.time()
            if gt_mask_path is not None:
                mask = read_mask_video(gt_mask_path, num_frames=T)
                n_obj = 1
                rec["mask_source"] = "gt"
            else:
                mask, n_obj = self.segmenter.segment(frames, self.work_dir)
                rec["mask_source"] = "seganymo"
            timings["mask"] = round(time.time() - t, 2)
            if mask is None or mask.shape != (T, H, W):
                rec.update(status="skipped", reason="no_dynamic_object", timings=timings)
                _dump_json(rec_p, rec)
                return rec
            rec["num_objects"] = int(n_obj)
            rec["foreground_ratio"] = foreground_ratio(mask)
            write_video(paths["mask"], mask_to_frames(mask), fps=cfg.fps)
            done.add("mask")
            rec["stages_done"] = sorted(done)
            _dump_json(rec_p, {**rec, "timings": timings})
        else:
            mask = read_mask_video(paths["mask"], num_frames=T)

        lo, hi = cfg.fg_ratio_range
        ratio = rec["foreground_ratio"]
        if not (lo <= ratio <= hi):
            # keep the mask around for inspection, but do not spend GPU time on layers/inpainting
            rec.update(status="skipped", reason=f"fg_ratio:{ratio:.4f}", timings=timings)
            _dump_json(rec_p, rec)
            return rec

        # 2. caption ----------------------------------------------------------
        if "caption" not in done:
            t = time.time()
            rec["caption"] = self.captioner.caption(frames)
            timings["caption"] = round(time.time() - t, 2)
            done.add("caption")
            rec["stages_done"] = sorted(done)
            _dump_json(rec_p, {**rec, "timings": timings})

        # 3. layers -----------------------------------------------------------
        if "layers" not in done:
            t = time.time()
            fg, bg = compose_layers(frames, mask)
            if not paths["video"].exists():
                paths["video"].parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip_path, paths["video"])
            write_video(paths["fg"], fg, fps=cfg.fps)
            write_video(paths["bg"], bg, fps=cfg.fps)
            save_first_frame(frames, paths["first_frame"])
            timings["layers"] = round(time.time() - t, 2)
            done.add("layers")
            rec["stages_done"] = sorted(done)
            _dump_json(rec_p, {**rec, "timings": timings})

        # 4. inpaint ----------------------------------------------------------
        if "inpaint" not in done:
            t = time.time()
            inpainted = self.inpainter.inpaint(frames, mask)
            write_video(paths["bg_inpainted"], inpainted, fps=cfg.fps)
            timings["inpaint"] = round(time.time() - t, 2)
            done.add("inpaint")
            rec["stages_done"] = sorted(done)

        timings["total"] = round(time.time() - t0, 2)
        rec.update(status="done", timings=timings)
        rec.pop("reason", None)
        _dump_json(rec_p, rec)
        return rec


# ---------------------------------------------------------------------- driver
def discover_clips(cfg: DecomposerConfig, sources: Sequence[str]) -> List[Dict]:
    """List clips (with optional GT masks) that still need processing."""
    jobs = []
    for src in sources:
        cdir = cfg.clips_dir / src
        if not cdir.exists():
            print(f"[discover] no clips dir for source {src}: {cdir}")
            continue
        meta = _load_json(cdir / "source.json") or {}
        is_training = bool(meta.get("is_training", True))
        for clip in sorted(cdir.glob("*.mp4")):
            rec = _load_json(record_path(cfg, clip.stem))
            if rec and rec.get("status") in ("done", "skipped"):
                continue
            gt = cdir / "masks" / f"{clip.stem}.mp4"
            jobs.append({"clip": str(clip), "source": src, "gt_mask": str(gt) if gt.exists() else None, "is_training": is_training})
    return jobs


def _worker_main(worker_id: int, num_workers: int, jobs: List[Dict], cfg_dict: Dict, device: str, gpu: Optional[int] = None) -> None:
    if gpu is not None:  # must happen before torch initialises CUDA in this (spawned) process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cfg = DecomposerConfig(**cfg_dict)
    cfg.export_env()
    log_dir = cfg.dataset_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"worker{worker_id}.log", "a", buffering=1)
    my_jobs = jobs[worker_id::num_workers]
    print(f"[worker {worker_id}] {len(my_jobs)} jobs on {device} (gpu {gpu})", file=log)
    proc = ClipProcessor(cfg, device=device, worker_id=worker_id)
    n_done = n_skip = n_err = 0
    for i, job in enumerate(my_jobs):
        clip = Path(job["clip"])
        try:
            rec = proc.process(clip, job["source"], Path(job["gt_mask"]) if job["gt_mask"] else None, job["is_training"])
            if rec.get("status") == "done":
                n_done += 1
            else:
                n_skip += 1
            print(
                f"[worker {worker_id}] {i + 1}/{len(my_jobs)} {clip.stem}: {rec.get('status')} "
                f"{rec.get('reason', '')} ratio={rec.get('foreground_ratio', float('nan')):.3f} "
                f"timings={rec.get('timings')}",
                file=log,
            )
        except Exception as exc:  # keep going; record the failure
            n_err += 1
            rec_p = record_path(cfg, clip.stem)
            rec = _load_json(rec_p) or {"clip_id": clip.stem, "source": job["source"]}
            rec.update(status="error", reason=repr(exc)[:300])
            _dump_json(rec_p, rec)
            print(f"[worker {worker_id}] ERROR {clip.stem}: {exc}\n{traceback.format_exc()}", file=log)
            import torch

            torch.cuda.empty_cache()
    print(f"[worker {worker_id}] finished: done={n_done} skipped={n_skip} errors={n_err}", file=log)


def run(
    cfg: DecomposerConfig,
    sources: Sequence[str],
    gpus: Sequence[int] = (0,),
    workers_per_gpu: int = 1,
    limit: Optional[int] = None,
    stagger: float = 15.0,
) -> None:
    """Process all pending clips of ``sources`` with ``workers_per_gpu`` processes on each of ``gpus``.

    Each worker keeps its own copy of every Decomposer model (~15 GB); an 80 GB H100 comfortably
    hosts 4 workers.  Workers are killed together with the parent (daemon processes).
    """
    import torch.multiprocessing as mp

    cfg.export_env()
    jobs = discover_clips(cfg, sources)
    if limit:
        jobs = jobs[:limit]
    assignment = [g for g in gpus for _ in range(workers_per_gpu)]
    num_workers = len(assignment)
    print(
        f"[run] {len(jobs)} pending clips from {list(sources)} -> {cfg.dataset_root} (gpus {list(gpus)}, {num_workers} workers)", flush=True
    )
    if not jobs:
        return
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    if num_workers == 1:
        _worker_main(0, 1, jobs, cfg_dict, "cuda", assignment[0])
        return
    mp.set_start_method("spawn", force=True)
    procs = []
    for w, gpu in enumerate(assignment):
        p = mp.Process(target=_worker_main, args=(w, num_workers, jobs, cfg_dict, "cuda", gpu), daemon=True)
        p.start()
        procs.append(p)
        time.sleep(stagger)  # stagger model loading
    for p in procs:
        p.join()
