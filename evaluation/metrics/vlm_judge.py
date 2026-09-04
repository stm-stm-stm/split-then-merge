"""VLM-as-a-judge pairwise protocol (Appendix A2.3 / A2.4 of the paper).

For every test case the judge sees the reference foreground / background videos and two
generated videos (A, B — order shuffled per question) and answers ``A``, ``B`` or ``N``.
Backends: ``gemini`` (Gemini 2.5 Pro through ``google-genai``, as in the paper) or ``qwen``
(a local Qwen2.5-VL checkpoint through ``transformers``).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List

PREAMBLE = (
    "You are a video analysis tool. You will be given a Reference Foreground video, a Reference Background video "
    "and two generated videos (Video A, Video B). IMPORTANT: Judge *only* this specific metric. Do not let overall "
    "visual quality influence your choice. "
)
CRITERIA: Dict[str, str] = {
    # core criteria (Table 3)
    "fg_identity": "Metric: Foreground Identity Consistency. Question: Which generated video (Video A or Video B) better preserves the appearance of the subject (person, animal, object) from the Reference Foreground video?",
    "fg_motion": "Metric: Foreground Motion Consistency. Question: Which generated video (Video A or Video B) better preserves the subject's motion from the Reference Foreground video and looks more physically believable?",
    "bg_identity": "Metric: Background Identity Consistency. Question: Which generated video (Video A or Video B) better preserves the appearance of the background scene from the Reference Background video?",
    "bg_motion": "Metric: Background Motion Consistency. Question: Which generated video (Video A or Video B) better preserves the background's motion (or camera movement) from the Reference Background video, and which looks smoother?",
    "harmony": "Metric: Affordance-aware Generation. Question: Which generated video (Video A or Video B) shows a more believable interaction between the subject and the background? (e.g., sitting *on* a chair, not *through* it; not walking through walls).",
    "overall": "Metric: Overall Quality. Question: Which generated video (Video A or Video B) has the best overall visual quality, clarity, and fewest visual artifacts (like flickering, blurring, or blockiness)?",
    # affordance criteria (Table 8)
    "physical_interaction": "Metric: Physical Interaction. Question: Which generated video (Video A or Video B) shows more physically plausible contact between the subject and the scene (e.g., feet making firm ground contact and bearing weight, water rippling or splashing where the subject touches it), rather than the subject hovering, sinking, or sliding unnaturally?",
    "scale": "Metric: Scale. Question: Which generated video (Video A or Video B) renders the subject at a more believable size relative to the surrounding scene and its objects, avoiding a subject that is implausibly large or small for the environment?",
    "placement": "Metric: Placement. Question: Which generated video (Video A or Video B) positions the subject in a location that is consistent with where it could plausibly exist in the scene (e.g., a boat on water rather than on a road, an animal on the ground rather than floating in the air)?",
    "action_scene": "Metric: Action-Scene Affordance. Question: Which generated video (Video A or Video B) better matches the subject's action to what the scene actually affords (e.g., swimming where there is water, walking where there is a walkable surface), so that the action is consistent with the environment rather than physically impossible within it?",
    "lighting_shadow": "Metric: Lighting & Shadow Consistency. Question: Which generated video (Video A or Video B) better harmonizes the subject's shading and shadows with the scene, such that shadow direction, color, and hardness, as well as the lighting on the subject, are consistent with the background's light sources?",
    "occlusion_depth": "Metric: Occlusion / Depth Ordering. Question: Which generated video (Video A or Video B) better respects depth ordering, with the subject correctly passing behind or in front of scene objects according to their relative depth, rather than appearing pasted on top of everything?",
}
ANSWER_FORMAT = " Answer with a single character: A, B, or N (no preference)."


def parse_answer(text: str) -> str:
    t = text.strip().upper()
    for ch in ("A", "B", "N"):
        if t.startswith(ch):
            return ch
    return "N"


class GeminiJudge:
    def __init__(self, model: str = "gemini-2.5-pro"):
        from google import genai  # pip install google-genai; GEMINI_API_KEY in the environment

        self.client = genai.Client()
        self.model = model

    def ask(self, question: str, videos: Dict[str, Path]) -> str:
        parts = []
        for label, path in videos.items():
            f = self.client.files.upload(file=str(path))
            parts += [f"{label}:", f]
        parts.append(PREAMBLE + question + ANSWER_FORMAT)
        return self.client.models.generate_content(model=self.model, contents=parts).text


class QwenJudge:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def ask(self, question: str, videos: Dict[str, Path]) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        for label, path in videos.items():
            content += [{"type": "text", "text": f"{label}:"}, {"type": "video", "video": f"file://{path}"}]
        content.append({"type": "text", "text": PREAMBLE + question + ANSWER_FORMAT})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=4)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]


def judge_pairwise(judge, cases: List[dict], criteria: List[str], out_csv: Path, seed: int = 0) -> Dict[str, float]:
    """cases: [{name, fg_ref, bg_ref, ours, baseline}] -> win rate of ``ours`` per criterion.

    Left/right order is shuffled per question; ``N`` counts as half a win (tie), as in a pairwise study.
    """
    rng = random.Random(seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wins: Dict[str, List[float]] = {c: [] for c in criteria}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "criterion", "ours_is_A", "answer", "ours_wins"])
        for case in cases:
            for c in criteria:
                ours_is_a = rng.random() < 0.5
                a, b = (case["ours"], case["baseline"]) if ours_is_a else (case["baseline"], case["ours"])
                ans = parse_answer(
                    judge.ask(
                        CRITERIA[c],
                        {"Reference Foreground": case["fg_ref"], "Reference Background": case["bg_ref"], "Video A": a, "Video B": b},
                    )
                )
                win = 0.5 if ans == "N" else float((ans == "A") == ours_is_a)
                wins[c].append(win)
                w.writerow([case["name"], c, ours_is_a, ans, win])
    return {c: (sum(v) / len(v) if v else float("nan")) for c, v in wins.items()}
