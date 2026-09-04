"""M1 / M2 / M5 with ViCLIP (InternVid, ``OpenGVLab/ViCLIP-L-14-hf``).

* M1  FG identity preservation  = cos( ViCLIP(gen fg layer), ViCLIP(input fg layer) )
* M2  BG identity preservation  = cos( ViCLIP(gen bg layer), ViCLIP(input bg with the generated
      foreground region blacked out) ) — both sides share the same hole
* M5  textual alignment         = cos( ViCLIP(generated video), ViCLIP(prompt) )
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .video_utils import sample_frames

_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def frames_to_tensor(frames: np.ndarray, num: int = 8, size: int = 224) -> torch.Tensor:
    """uint8 RGB [T, H, W, 3] -> float [1, num, 3, size, size] (ImageNet-normalised)."""
    picked = sample_frames(frames, num)
    imgs = [(cv2.resize(f, (size, size)) / 255.0 - _MEAN) / _STD for f in picked]
    tube = np.stack(imgs)[None].transpose(0, 1, 4, 2, 3)
    return torch.from_numpy(tube).float()


class ViCLIPScorer:
    def __init__(self, model_id: str = "OpenGVLab/ViCLIP-L-14-hf", device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel

        self.device = torch.device(device)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # the remote code resolves the BPE vocab relative to the CWD; point it at the cached snapshot instead
        config.tokenizer_path = hf_hub_download(model_id, "bpe_simple_vocab_16e6.txt.gz")
        self.model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True).to(self.device).eval()
        self.tokenizer = self.model.tokenizer

    @torch.no_grad()
    def video_features(self, frames: np.ndarray) -> torch.Tensor:
        return self.model.get_vid_features(frames_to_tensor(frames).to(self.device))

    @torch.no_grad()
    def text_features(self, text: str) -> torch.Tensor:
        return self.model.get_text_features(text, self.tokenizer, {})

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.flatten()[None], b.flatten()[None]).item())
