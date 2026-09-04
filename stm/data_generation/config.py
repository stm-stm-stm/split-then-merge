"""Configuration for the StM Decomposer (data generation) pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass
class DecomposerConfig:
    """All knobs of the Decomposer. Paths default to ``<repo>/data`` (``STM_DATA_ROOT``),
    typically a symlink to bulk storage."""

    # ---------------------------------------------------------------- paths
    data_root: Path = field(default_factory=lambda: _env_path("STM_DATA_ROOT", REPO_ROOT / "data"))
    # local (non-NFS) scratch used for per-clip intermediates of SegAnyMo/SAM2
    work_dir: Path = field(default_factory=lambda: _env_path("STM_WORK_DIR", Path.home() / ".cache" / "stm_work"))
    hf_home: Path = field(default_factory=lambda: _env_path("HF_HOME", Path.home() / ".cache" / "huggingface"))
    seganymo_root: Path = THIRD_PARTY / "SegAnyMo"
    minimax_root: Path = THIRD_PARTY / "MiniMax-Remover"
    dataset_name: str = "stm_gen"  # data_root/datasets/<dataset_name>

    # -------------------------------------------------------------- clip spec
    num_frames: int = 49  # CogVideoX: 8N+1
    height: int = 480
    width: int = 720
    fps: int = 16  # output fps (all StM videos are written at 16 fps)
    strides: Tuple[int, ...] = (1, 2, 3, 4)  # random frame-rate downsampling (paper A1.1)
    max_clips_per_video: int = 4
    static_threshold: float = 1.5  # mean abs diff (0-255) below which a clip is "static"

    # ---------------------------------------------------------------- models
    caption_model: str = "OpenGVLab/InternVL2_5-1B"
    caption_prompt: str = "Describe the video in detail"
    caption_num_segments: int = 12
    caption_max_new_tokens: int = 1024
    # Exactly as in Segment-Any-Motion. (The reference pipeline saturates the depth map through a 16-bit ->
    # 8-bit PIL conversion, so "depth-anything/Depth-Anything-V2-Small-hf" gives identical inputs 3x faster;
    # opt in with --depth-model.)
    depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf"
    dino_model: str = "dinov2_vitb14"
    moseg_repo: str = "Changearthmore/moseg"
    minimax_repo: str = "zibojia/minimax-remover"
    seganymo_step: int = 10  # query-frame stride of SegAnyMo (default of the authors)
    # None = track the full ~9600-point grid from every query frame, exactly as the reference scripts.
    # (SegAnyMo's classifier keeps a uniform random 1/len(q_ts) of these tracks, capped at 5000, so
    # pre-sampling ~1100 points per query frame is distribution-equivalent and ~5x faster: opt in with
    # --tracks-per-query-frame 1100.)
    tracks_per_query_frame: Optional[int] = None
    fast_segmenter: bool = False  # bf16 BootsTAPIR + fp16 depth (see SegAnyMoSegmenter)
    inpaint_steps: int = 12
    inpaint_dilation_iters: int = 6
    seed: int = 42

    # --------------------------------------------------------------- filters
    fg_ratio_range: Tuple[float, float] = (0.01, 0.45)  # ``main_train_foreground_volume``

    def __post_init__(self) -> None:
        # dataclass may be re-hydrated from a JSON-able dict (multiprocessing)
        for name in ("data_root", "work_dir", "hf_home", "seganymo_root", "minimax_root"):
            setattr(self, name, Path(getattr(self, name)).expanduser())
        self.strides = tuple(int(s) for s in self.strides)
        self.fg_ratio_range = tuple(float(x) for x in self.fg_ratio_range)

    # ------------------------------------------------------------ derived
    @property
    def dataset_root(self) -> Path:
        return self.data_root / "datasets" / self.dataset_name

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def clips_dir(self) -> Path:
        return self.data_root / "clips"

    @property
    def ckpt_dir(self) -> Path:
        return self.data_root / "ckpts"

    @property
    def torch_home(self) -> Path:
        return self.ckpt_dir / "torch_hub"

    @property
    def tapir_ckpt(self) -> Path:
        return self.ckpt_dir / "bootstapir_checkpoint_v2.pt"

    @property
    def sam2_ckpt(self) -> Path:
        return self.ckpt_dir / "sam2_hiera_large.pt"

    def export_env(self) -> None:
        """Make HF / torch.hub caches point at bulk storage."""
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("TORCH_HOME", str(self.torch_home))
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
