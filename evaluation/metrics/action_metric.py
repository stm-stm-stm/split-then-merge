"""M3 — semantic action alignment.

KL divergence between the Kinetics-400 action distributions predicted by a Video Swin
Transformer (torchvision ``swin3d_b``, Kinetics-400 weights) for the input foreground layer
and the generated foreground layer:  KL( p(input fg) || p(generated fg) ).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .video_utils import sample_frames


def kl_divergence(logits_ref: torch.Tensor, logits_gen: torch.Tensor) -> float:
    """KL(p_ref || p_gen) from two logit vectors."""
    p_ref = F.log_softmax(logits_ref.float(), dim=-1)
    p_gen = F.log_softmax(logits_gen.float(), dim=-1)
    return float(F.kl_div(p_gen, p_ref, log_target=True, reduction="sum").item())


class ActionScorer:
    def __init__(self, device: str = "cuda", num_frames: int = 32):
        from torchvision.models.video import Swin3D_B_Weights, swin3d_b

        self.device = torch.device(device)
        weights = Swin3D_B_Weights.KINETICS400_V1
        self.model = swin3d_b(weights=weights).to(self.device).eval()
        self.transform = weights.transforms()
        self.num_frames = num_frames

    @torch.no_grad()
    def logits(self, frames: np.ndarray) -> torch.Tensor:
        clip = torch.from_numpy(sample_frames(frames, self.num_frames)).permute(0, 3, 1, 2)  # T, C, H, W uint8
        x = self.transform(clip).unsqueeze(0).to(self.device)  # 1, C, T, H, W
        return self.model(x)[0].cpu()
