#!/usr/bin/env python
"""Pairwise VLM-as-a-judge study (Table 3 / Table 8 of the paper).

    GEMINI_API_KEY=... python evaluation/run_vlm_judge.py --ours outputs/validation/ckpt --baseline outputs/baselines/copy_paste \
        --out outputs/judge/copy_paste --backend gemini --criteria core

``--ours`` / ``--baseline`` are result folders in the ``compose`` layout (or ``--layout released --method-ours ... --method-baseline ...``);
both must contain the same ``sample-*`` cases.  Prints and saves the win rate of ``ours`` per criterion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.video_utils import iter_samples  # noqa: E402
from evaluation.metrics.vlm_judge import CRITERIA, GeminiJudge, QwenJudge, judge_pairwise  # noqa: E402

CORE = ["fg_identity", "bg_identity", "fg_motion", "bg_motion", "harmony", "overall"]
AFFORDANCE = ["physical_interaction", "scale", "placement", "action_scene", "lighting_shadow", "occlusion_depth"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["gemini", "qwen"], default="gemini")
    p.add_argument("--model", default=None, help="gemini model id or Qwen2.5-VL checkpoint")
    p.add_argument("--criteria", choices=["core", "affordance", "all"], default="core")
    p.add_argument("--layout", choices=["compose", "released"], default="compose")
    p.add_argument("--method-ours", default="gen-recomposer")
    p.add_argument("--method-baseline", default="cogvideox-i2v-copy_paste")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ours = {s.name: s for s in iter_samples(Path(a.ours), a.layout, a.method_ours)}
    base = {s.name: s for s in iter_samples(Path(a.baseline), a.layout, a.method_baseline)}
    names = sorted(set(ours) & set(base))[: a.limit]
    cases = [
        {"name": n, "fg_ref": ours[n].fg_gt, "bg_ref": ours[n].bg_gt, "ours": ours[n].generated, "baseline": base[n].generated}
        for n in names
    ]
    criteria = {"core": CORE, "affordance": AFFORDANCE, "all": list(CRITERIA)}[a.criteria]
    judge = GeminiJudge(a.model or "gemini-2.5-pro") if a.backend == "gemini" else QwenJudge(a.model or "Qwen/Qwen2.5-VL-7B-Instruct")
    out = Path(a.out)
    rates = judge_pairwise(judge, cases, criteria, out / "answers.csv", seed=a.seed)
    (out / "win_rates.json").write_text(json.dumps(rates, indent=1))
    print("[judge] win rate of ours:", json.dumps(rates))


if __name__ == "__main__":
    main()
