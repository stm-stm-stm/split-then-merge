"""StM Decomposer: automatic multi-layer video dataset generation.

Splits unlabeled videos into (caption, foreground layer, mask, background layer)
using off-the-shelf models, exactly as described in Sec. 3.3 / 4.1 of
"Layer-Aware Video Composition via Split-then-Merge":

* captioning        -> InternVL2.5 (``OpenGVLab/InternVL2_5-1B`` by default)
* motion segmentation -> Segment-Any-Motion (SegAnyMo: Depth-Anything-V2 +
                         BootsTAPIR + DINOv2 + moseg + SAM2)
* background inpainting -> MiniMax-Remover (``zibojia/minimax-remover``)

The output follows the layout consumed by ``stm.data.I2VDatasetWithAugmentation``.
"""

from .config import DecomposerConfig

__all__ = ["DecomposerConfig"]
