#!/usr/bin/env python
"""Compute M1-M5 for a folder of generated compositions (Table 1 / 2 of the paper).

    python evaluation/run_eval.py --results outputs/validation/validation_checkpoint-20000 --out outputs/eval/ckpt20000
    python evaluation/run_eval.py --results data/results/Organized_Results --layout released --method gen-recomposer --out outputs/eval/paper

Steps: (1) decompose every generated video with Segment-Any-Motion (GPU), (2) ViCLIP M1/M2/M5,
VideoSwin M3, RAFT M4.  Writes ``<out>/scores.csv`` (per sample) and ``<out>/summary.json``.
Use ``--skip-decompose`` when layers already exist under ``<out>/decomposed``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.action_metric import ActionScorer, kl_divergence  # noqa: E402
from evaluation.metrics.decompose_generated import decompose_samples, layer_paths  # noqa: E402
from evaluation.metrics.flow_metric import FlowScorer  # noqa: E402
from evaluation.metrics.viclip_metrics import ViCLIPScorer  # noqa: E402
from evaluation.metrics.video_utils import apply_mask, iter_samples, read_frames, read_mask, summarize  # noqa: E402

METRICS = ("m1_fg_identity", "m2_bg_identity", "m3_action_kl", "m4_bg_flow_mse", "m5_text_alignment")


def evaluate(
    results: Path, out: Path, layout: str, method: str, device: str, skip_decompose: bool, metrics: list, limit: int | None
) -> dict:
    samples = list(iter_samples(results, layout, method))
    if limit:
        samples = samples[:limit]
    if not samples:
        raise SystemExit(f"no samples found in {results}")
    dec_dir = out / "decomposed"
    if not skip_decompose:
        decompose_samples(samples, dec_dir, device=device)

    want = set(metrics)
    viclip = ViCLIPScorer(device=device) if want & {"m1_fg_identity", "m2_bg_identity", "m5_text_alignment"} else None
    action = ActionScorer(device=device) if "m3_action_kl" in want else None
    flow = FlowScorer(device=device) if "m4_bg_flow_mse" in want else None

    rows = []
    for s in samples:
        lp = layer_paths(dec_dir, s)
        row = {"sample": s.name}
        gen = read_frames(s.generated)
        gen_mask = read_mask(lp["mask"], num_frames=len(gen)) if lp["mask"].exists() else np.zeros(gen.shape[:3], bool)
        gen_fg, gen_bg = read_frames(lp["fg"]), read_frames(lp["bg"])
        fg_gt, bg_gt = read_frames(s.fg_gt), read_frames(s.bg_gt)
        if viclip:
            if "m1_fg_identity" in want:
                row["m1_fg_identity"] = viclip.cosine(viclip.video_features(gen_fg), viclip.video_features(fg_gt))
            if "m2_bg_identity" in want:
                bg_gt_holed = apply_mask(bg_gt, gen_mask, keep_foreground=False)
                row["m2_bg_identity"] = viclip.cosine(viclip.video_features(gen_bg), viclip.video_features(bg_gt_holed))
            if "m5_text_alignment" in want and s.prompt:
                row["m5_text_alignment"] = viclip.cosine(viclip.video_features(gen), viclip.text_features(s.prompt))
        if action:
            row["m3_action_kl"] = kl_divergence(action.logits(fg_gt), action.logits(gen_fg))
        if flow:
            row["m4_bg_flow_mse"] = flow.score(bg_gt, gen_bg, valid_mask=gen_mask)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    keys = [m for m in METRICS if m in want]
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample"] + keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, keys)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[eval] summary:", json.dumps(summary))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="folder of sample-* result directories")
    p.add_argument("--out", required=True)
    p.add_argument("--layout", choices=["compose", "released"], default="compose")
    p.add_argument("--method", default="gen-recomposer", help="video name inside released folders (layout=released)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-decompose", action="store_true")
    p.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    evaluate(Path(a.results), Path(a.out), a.layout, a.method, a.device, a.skip_decompose, a.metrics, a.limit)


if __name__ == "__main__":
    main()
