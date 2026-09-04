#!/usr/bin/env python
"""StM Decomposer CLI - build a multi-layer video dataset from unlabeled videos.

Typical flow (all paths default to <repo>/data):

    # 1. sources
    python data_processing/generate_data.py download-panda --max-videos 3000 --workers 6
    python data_processing/generate_data.py chunk --source panda70m --raw-dir data/raw/panda70m/videos
    python data_processing/generate_data.py import-davis --davis-root data/raw/davis/DAVIS

    # 2. decompose (caption + motion segmentation + layers + inpainting) on the GPU(s)
    python data_processing/generate_data.py process --sources openvid davis --gpus 0 --workers-per-gpu 4

    # 3. training manifest (txt lists + metadata.csv)
    python data_processing/generate_data.py manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stm.data_generation import DecomposerConfig  # noqa: E402


def _cfg(args) -> DecomposerConfig:
    kwargs = {}
    if args.data_root:
        kwargs["data_root"] = Path(args.data_root)
    if args.dataset_name:
        kwargs["dataset_name"] = args.dataset_name
    if getattr(args, "caption_model", None):
        kwargs["caption_model"] = args.caption_model
    if getattr(args, "fast", False):
        kwargs["fast_segmenter"] = True
    if getattr(args, "depth_model", None):
        kwargs["depth_model"] = args.depth_model
    if getattr(args, "tracks_per_query_frame", None):
        kwargs["tracks_per_query_frame"] = args.tracks_per_query_frame
    if getattr(args, "clips_per_video", None):
        kwargs["max_clips_per_video"] = args.clips_per_video
    cfg = DecomposerConfig(**kwargs)
    cfg.export_env()
    return cfg


def cmd_download_panda(args):
    from stm.data_generation.sources.panda70m import download, select_clips

    cfg = _cfg(args)
    csv_path = Path(args.csv) if args.csv else cfg.raw_dir / "panda70m" / "panda70m_train_2m.csv"
    out_dir = Path(args.out) if args.out else cfg.raw_dir / "panda70m" / "videos"
    selected = select_clips(csv_path, args.max_videos, min_score=args.min_score, max_clips_per_video=args.clips_per_video, seed=args.seed)
    (out_dir.parent / f"selection_seed{args.seed}_n{args.max_videos}.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / f"selection_seed{args.seed}_n{args.max_videos}.json").write_text(json.dumps(selected))
    download(selected, out_dir, workers=args.workers, height=args.height, ytdlp=args.ytdlp, cookies=args.cookies)


def cmd_chunk(args):
    from stm.data_generation.sources.local import import_local

    cfg = _cfg(args)
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.raw_dir / args.source / "videos"
    import_local(cfg, raw_dir, args.source, is_training=not args.validation, max_videos=args.max_videos, seed=args.seed)


def cmd_import_davis(args):
    from stm.data_generation.sources.davis import import_davis

    cfg = _cfg(args)
    davis_root = Path(args.davis_root) if args.davis_root else cfg.raw_dir / "davis" / "DAVIS"
    import_davis(cfg, davis_root, source=args.source, split=args.split)


def cmd_import_openvid(args):
    from stm.data_generation.sources.openvid import import_openvid

    cfg = _cfg(args)
    csv_path = Path(args.csv) if args.csv else cfg.raw_dir / "openvid" / "OpenVid-1M.csv"
    zips = [Path(z) for z in args.zips] if args.zips else sorted((cfg.raw_dir / "openvid").glob("OpenVid_part*.zip"))
    import_openvid(
        cfg,
        zips,
        csv_path,
        source=args.source,
        min_motion=args.min_motion,
        min_aesthetic=args.min_aesthetic,
        max_videos=args.max_videos,
        seed=args.seed,
    )


def cmd_import_openvid_remote(args):
    from stm.data_generation.sources.openvid import import_openvid_remote

    cfg = _cfg(args)
    csv_path = Path(args.csv) if args.csv else cfg.raw_dir / "openvid" / "OpenVid-1M.csv"
    import_openvid_remote(
        cfg,
        args.parts,
        csv_path,
        source=args.source,
        workers=args.workers,
        seed=args.seed,
        min_motion=args.min_motion,
        min_aesthetic=args.min_aesthetic,
    )


def cmd_import_test_set(args):
    from stm.data_generation.sources.released_test import import_released_test_set

    cfg = _cfg(args)
    root = Path(args.results_root) if args.results_root else cfg.data_root / "results" / "Organized_Results"
    import_released_test_set(cfg, root, dataset_name=args.name)


def cmd_process(args):
    from stm.data_generation.pipeline import run

    cfg = _cfg(args)
    run(cfg, args.sources, gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, limit=args.limit)


def cmd_eval_segmenter(args):
    from stm.data_generation.eval_segmenter import run, summarize

    cfg = _cfg(args)
    if args.summary_only:
        summarize(cfg)
    else:
        run(cfg, args.sources, gpus=args.gpus, limit=args.limit)


def cmd_manifest(args):
    from stm.data_generation.manifest import build_manifest, summarize

    cfg = _cfg(args)
    n, df = build_manifest(cfg)
    print(f"[manifest] {n} clips written to {cfg.dataset_root}")
    if n:
        print(df.groupby("dataset_type")[["main_train", "main_train_foreground_volume"]].sum())
    print("[records]", summarize(cfg))


def cmd_verify(args):
    from stm.data_generation.manifest import verify_dataset

    verify_dataset(_cfg(args), max_clips=args.max_clips)


def cmd_stats(args):
    from stm.data_generation.manifest import write_dataset_readme

    cfg = _cfg(args)
    print(write_dataset_readme(cfg))


def cmd_status(args):
    from stm.data_generation.manifest import load_records, summarize

    cfg = _cfg(args)
    print(summarize(cfg))
    recs = [r for r in load_records(cfg) if r.get("status") == "done" and r.get("timings")]
    if recs:
        import numpy as np

        for k in ("caption", "mask", "layers", "inpaint", "total"):
            v = [r["timings"].get(k) for r in recs if r["timings"].get(k) is not None]
            if v:
                print(f"  {k:8s} median {np.median(v):6.1f}s  mean {np.mean(v):6.1f}s  (n={len(v)})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="bulk storage root (default: <repo>/data)")
    p.add_argument("--dataset-name", default=None, help="name of the generated dataset (default: stm_gen)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-panda", help="select + download Panda-70M clips with yt-dlp")
    s.add_argument("--csv", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--max-videos", type=int, default=1000)
    s.add_argument("--clips-per-video", type=int, default=2)
    s.add_argument("--min-score", type=float, default=0.44)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ytdlp", default="yt-dlp")
    s.add_argument("--cookies", default=None, help="Netscape cookies file for YouTube (needed from cloud IPs)")
    s.set_defaults(func=cmd_download_panda)

    s = sub.add_parser("chunk", help="split raw videos of a source into 49x480x720 clips")
    s.add_argument("--source", required=True)
    s.add_argument("--raw-dir", default=None)
    s.add_argument("--max-videos", type=int, default=None)
    s.add_argument("--validation", action="store_true", help="mark clips as validation (is_training=False)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_chunk)

    s = sub.add_parser("import-davis", help="DAVIS-2017 sequences with GT masks (validation)")
    s.add_argument("--davis-root", default=None)
    s.add_argument("--source", default="davis")
    s.add_argument("--split", default=None, choices=[None, "train", "val"])
    s.set_defaults(func=cmd_import_davis)

    s = sub.add_parser("import-openvid", help="extract filtered OpenVid-1M videos from zip parts and chunk them")
    s.add_argument("--zips", nargs="*", default=None, help="OpenVid_part*.zip files (default: all under data/raw/openvid)")
    s.add_argument("--csv", default=None)
    s.add_argument("--source", default="openvid")
    s.add_argument("--min-motion", type=float, default=3.0)
    s.add_argument("--min-aesthetic", type=float, default=4.5)
    s.add_argument("--max-videos", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_import_openvid)

    s = sub.add_parser("import-openvid-remote", help="fetch only the filtered videos of OpenVid zip parts via HTTP ranges, then chunk")
    s.add_argument("--parts", type=int, nargs="+", required=True)
    s.add_argument("--csv", default=None)
    s.add_argument("--source", default="openvid")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-motion", type=float, default=3.0)
    s.add_argument("--min-aesthetic", type=float, default=4.5)
    s.add_argument("--clips-per-video", type=int, default=None)
    s.set_defaults(func=cmd_import_openvid_remote)

    s = sub.add_parser("import-test-set", help="paper test triplets (fg/bg/prompt) from the released Organized_Results")
    s.add_argument("--results-root", default=None)
    s.add_argument("--name", default="stm_test")
    s.set_defaults(func=cmd_import_test_set)

    s = sub.add_parser("process", help="run the Decomposer on all pending clips")
    s.add_argument("--sources", nargs="+", required=True)
    s.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU ids to use")
    s.add_argument("--workers-per-gpu", type=int, default=1, help="processes per GPU (~15 GB each; 4 fits an 80 GB H100)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--caption-model", default=None)
    s.add_argument("--fast", action="store_true", help="bf16 TAPIR + fp16 depth in the segmenter (NOT recommended: IoU drop)")
    s.add_argument(
        "--depth-model",
        default=None,
        help="e.g. depth-anything/Depth-Anything-V2-Small-hf (equivalent after the reference's 8-bit saturation)",
    )
    s.add_argument(
        "--tracks-per-query-frame", type=int, default=None, help="pre-sample query points (e.g. 1100); default = full grid as SegAnyMo"
    )
    s.set_defaults(func=cmd_process)

    s = sub.add_parser("eval-segmenter", help="mask IoU of the motion segmenter vs GT masks (e.g. DAVIS)")
    s.add_argument("--sources", nargs="+", default=["davis"])
    s.add_argument("--gpus", type=int, nargs="+", default=[0])
    s.add_argument("--summary-only", action="store_true")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--fast", action="store_true", help="bf16 TAPIR + fp16 depth")
    s.set_defaults(func=cmd_eval_segmenter)

    s = sub.add_parser("manifest", help="write txt lists + metadata.csv")
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("verify", help="check frames/resolution/fps/file sizes of every done clip")
    s.add_argument("--max-clips", type=int, default=None)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("stats", help="write <dataset>/README.md with structure + statistics")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("status", help="summarise per-clip records")
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
