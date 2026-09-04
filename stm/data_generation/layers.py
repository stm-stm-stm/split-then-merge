"""Foreground / background layer composition from a binary mask."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def foreground_ratio(mask: np.ndarray) -> float:
    """Fraction of foreground pixels over the whole video volume."""
    return float(mask.mean()) if mask.size else 0.0


def compose_layers(frames: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fg, bg): the video multiplied by the mask / its complement.

    Both layers have a black (0) background, matching the ``fg-black`` layers
    used to train the StM Composer (``FG_COLUMN='fg-black.txt'``).
    """
    m = mask.astype(np.uint8)[..., None]
    fg = frames * m
    bg = frames * (1 - m)
    return fg.astype(np.uint8), bg.astype(np.uint8)
