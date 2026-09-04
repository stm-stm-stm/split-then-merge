"""Background inpainting with MiniMax-Remover (Zi et al., 2025)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


class MiniMaxRemover:
    def __init__(
        self,
        minimax_root: Path,
        repo_id: str = "zibojia/minimax-remover",
        device: str = "cuda",
        num_inference_steps: int = 12,
        dilation_iterations: int = 6,
        seed: int = 42,
    ) -> None:
        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import UniPCMultistepScheduler
        from huggingface_hub import snapshot_download

        if str(minimax_root) not in sys.path:
            sys.path.insert(0, str(minimax_root))
        from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: E402
        from transformer_minimax_remover import Transformer3DModel  # noqa: E402

        snap = Path(snapshot_download(repo_id, allow_patterns=["vae/*", "transformer/*", "scheduler/*"]))
        vae = AutoencoderKLWan.from_pretrained(snap / "vae", torch_dtype=torch.float16)
        transformer = Transformer3DModel.from_pretrained(snap / "transformer", torch_dtype=torch.float16)
        scheduler = UniPCMultistepScheduler.from_pretrained(snap / "scheduler")
        self.pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler).to(device)
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.dilation_iterations = dilation_iterations
        self.seed = seed

    @torch.no_grad()
    def inpaint(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """frames uint8 [T,H,W,3], mask bool [T,H,W] -> inpainted uint8 [T,H,W,3]."""
        t, h, w = mask.shape
        assert (t - 1) % 4 == 0 and h % 16 == 0 and w % 16 == 0, "Wan VAE needs (T-1)%4==0 and H,W%16==0"
        images = torch.from_numpy(frames).float() / 127.5 - 1.0
        masks = torch.from_numpy(mask.astype(np.float32))[..., None]
        out = self.pipe(
            images=images,
            masks=masks,
            num_frames=t,
            height=h,
            width=w,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
            iterations=self.dilation_iterations,
        ).frames[0]
        out = np.asarray(out)
        if out.dtype != np.uint8:
            out = (np.clip(out, 0, 1) * 255.0).round().astype(np.uint8)
        return out
