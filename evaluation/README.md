# Evaluation

Quantitative metrics of the paper (Table 1) and the VLM-as-a-judge protocol (Tables 3 and 8).

```bash
# compositions produced by inference/compose.py  (sample-*/{video_generated,video_fg_gt,video_bg_gt}.mp4 + prompt.txt)
python evaluation/run_eval.py --results outputs/validation/validation_checkpoint-20000 --out outputs/eval/ckpt20000

# the released result folders (sample-*/{fg,bg,<method>}.mp4 + prompt.txt)
python evaluation/run_eval.py --results data/results/Organized_Results --layout released --method gen-recomposer --out outputs/eval/paper

# pairwise judge, win rate of ours vs a baseline
GEMINI_API_KEY=... python evaluation/run_vlm_judge.py --ours <results A> --baseline <results B> --out outputs/judge/B --criteria core
```

| Metric | Definition | Implementation |
|--------|------------|----------------|
| M1 FG identity ↑ | cos( ViCLIP(generated fg layer), ViCLIP(input fg) ) | `metrics/viclip_metrics.py` |
| M2 BG identity ↑ | cos( ViCLIP(generated bg layer), ViCLIP(input bg with the generated foreground blacked out) ) | `metrics/viclip_metrics.py` |
| M3 semantic action ↓ | KL( VideoSwin(input fg) ‖ VideoSwin(generated fg) ), Kinetics-400 classes | `metrics/action_metric.py` |
| M4 BG motion ↓ | MSE of optical flow on background pixels, input bg vs generated bg | `metrics/flow_metric.py` (RAFT; the paper used Perceiver IO flow) |
| M5 text alignment ↑ | cos( ViCLIP(generated video), ViCLIP(prompt) ) | `metrics/viclip_metrics.py` |

Generated videos are first split into layers with Segment-Any-Motion (`metrics/decompose_generated.py`,
the same segmenter as the training data; Appendix A1.4 discusses other decomposers).  Outputs:
`<out>/scores.csv` per sample and `<out>/summary.json` (means, counts).

`run_vlm_judge.py` implements the exact prompts of Appendix A2.3 (identity / motion / harmony / overall)
and A2.4 (six affordance criteria); the judge answers A / B / N with the left–right order shuffled per
question.  Backends: Gemini 2.5 Pro (`google-genai`, as in the paper) or a local Qwen2.5-VL checkpoint.
