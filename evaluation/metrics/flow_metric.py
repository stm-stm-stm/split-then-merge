"""M4 — background motion alignment.

Mean squared error between the optical-flow fields of the input background and of the
generated background, restricted to background pixels (the generated foreground region is
excluded on both sides).  The paper used the Perceiver IO flow model; this implementation
uses torchvision's RAFT-large (``Raft_Large_Weights.C_T_SKHT_V2``) — pass another
``(frame_a, frame_b) -> flow`` callable to ``FlowScorer`` to swap it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def flow_mse(flow_ref: np.ndarray, flow_gen: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """MSE between two flow fields [H, W, 2]; ``valid`` [H, W] restricts the average."""
    err = ((flow_ref - flow_gen) ** 2).sum(-1)
    if valid is not None:
        if valid.sum() == 0:
            return float("nan")
        err = err[valid]
    return float(err.mean())


class FlowScorer:
    def __init__(self, device: str = "cuda", flow_fn: Optional[Callable] = None, pair_stride: int = 4):
        self.device = torch.device(device)
        self.pair_stride = pair_stride
        if flow_fn is None:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            weights = Raft_Large_Weights.C_T_SKHT_V2
            self.model = raft_large(weights=weights).to(self.device).eval()
            self.transform = weights.transforms()
            flow_fn = self._raft
        self.flow_fn = flow_fn

    @torch.no_grad()
    def _raft(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ta = torch.from_numpy(a).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b).permute(2, 0, 1)[None]
        ta, tb = self.transform(ta, tb)
        flow = self.model(ta.to(self.device), tb.to(self.device))[-1]
        return flow[0].permute(1, 2, 0).cpu().numpy()

    def score(self, ref_frames: np.ndarray, gen_frames: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
        """Average flow MSE over frame pairs (t, t+1) sampled every ``pair_stride`` frames."""
        t = min(len(ref_frames), len(gen_frames))
        vals = []
        for i in range(0, t - 1, self.pair_stride):
            fr = self.flow_fn(ref_frames[i], ref_frames[i + 1])
            fg = self.flow_fn(gen_frames[i], gen_frames[i + 1])
            valid = None if valid_mask is None else ~valid_mask[i]
            vals.append(flow_mse(fr, fg, valid))
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
