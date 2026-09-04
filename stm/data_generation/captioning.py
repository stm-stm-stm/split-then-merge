"""Video captioning with InternVL2.5 (the StM Decomposer captioner).

Reproduces the authors' preprocessing: 12 uniformly sampled frames, one
448x448 tile per frame (``max_num=1``), prompt "Describe the video in detail",
sampling decode with ``max_new_tokens=1024``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _frame_indices(num_frames: int, num_segments: int) -> np.ndarray:
    """Same as the authors' ``get_index`` (bound=None): centre of each segment."""
    max_frame = num_frames - 1
    seg_size = float(max_frame) / num_segments
    idx = np.array([int(seg_size / 2 + np.round(seg_size * i)) for i in range(num_segments)])
    return np.clip(idx, 0, max_frame)


class InternVLCaptioner:
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-1B",
        device: str = "cuda",
        num_segments: int = 12,
        input_size: int = 448,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        prompt: str = "Describe the video in detail",
    ) -> None:
        import transformers

        self.device = device
        self.num_segments = num_segments
        self.input_size = input_size
        self.prompt = prompt
        self.generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        self.model = (
            transformers.AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
            .eval()
            .to(device)
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        if not hasattr(self.tokenizer, "convert_tokens_to_ids"):
            raise RuntimeError(
                f"Tokenizer of {model_id} could not be loaded (got {type(self.tokenizer)}). "
                "InternVL2_5-2B's InternLM2 sentencepiece model is incompatible with this environment; "
                "use the 1B/4B (Qwen tokenizer) variants."
            )
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _prepare(self, frames: np.ndarray) -> Tuple[torch.Tensor, List[int]]:
        idx = _frame_indices(len(frames), self.num_segments)
        tiles = [self.transform(Image.fromarray(frames[i])) for i in idx]  # max_num=1 -> one tile / frame
        pixel_values = torch.stack(tiles).to(torch.bfloat16).to(self.device)
        return pixel_values, [1] * len(tiles)

    @torch.no_grad()
    def caption(self, frames: np.ndarray) -> str:
        pixel_values, num_patches_list = self._prepare(frames)
        video_prefix = "".join(f"Frame-{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        question = video_prefix + self.prompt
        response, _ = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return " ".join(response.split())
