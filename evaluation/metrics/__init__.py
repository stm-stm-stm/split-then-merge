"""Metric implementations (Table 1 of the paper).

* M1 / M2 — foreground / background identity preservation: ViCLIP cosine similarity of the
  generated layers vs the input layers (:mod:`viclip_metrics`)
* M3 — semantic action alignment: KL divergence between VideoSwin action distributions of the
  input and generated foreground (:mod:`action_metric`)
* M4 — background motion alignment: MSE between optical-flow fields of the input and generated
  background (:mod:`flow_metric`)
* M5 — textual alignment: ViCLIP text-video cosine similarity (:mod:`viclip_metrics`)

Generated videos are first split into layers with Segment-Any-Motion
(:mod:`decompose_generated`), exactly like the training data.
"""

from .video_utils import Sample, iter_samples, read_frames, read_mask, sample_frames

__all__ = ["Sample", "iter_samples", "read_frames", "read_mask", "sample_frames"]
