"""CPU-only checks of the pure-python parts of the pipeline (no models, no GPU).

    pytest tests -q
Set STM_TEST_MODELS=1 to also run the metric models (ViCLIP, VideoSwin, RAFT) on CPU with tiny inputs.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stm.data_generation.chunking import iter_clips  # noqa: E402
from stm.data_generation.config import DecomposerConfig  # noqa: E402
from stm.data_generation.layers import compose_layers, foreground_ratio  # noqa: E402
from stm.data_generation.manifest import build_manifest, record_path  # noqa: E402
from stm.data_generation.sources.released_test import mask_from_black_fg  # noqa: E402
from stm.data_generation.video_io import mask_to_frames, read_mask_video, read_video, resize_video, video_info, write_video  # noqa: E402
from stm.models.stm_composer import downsample_mask_to_latent, expand_patch_embed  # noqa: E402


def synthetic_video(t=60, h=240, w=360, size=40, seed=0, pan=2):
    """Moving square on a textured background that pans ``pan`` px/frame (camera motion)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([(xx / w) * 255, (yy / h) * 255, ((xx + yy) / (w + h)) * 255], -1)
    for _ in range(12):  # a few soft blobs so the background has structure but compresses well
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(15, 40)
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2)))[..., None] * rng.integers(-80, 80, 3)
        bg = bg + blob
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    frames, masks = [], []
    for i in range(t):
        f = np.roll(bg, i * pan, axis=1)
        x = 20 + i * 4
        f[100 : 100 + size, x : x + size] = (255, 40, 40)
        m = np.zeros((h, w), bool)
        m[100 : 100 + size, x : x + size] = True
        frames.append(f)
        masks.append(m)
    return np.stack(frames), np.stack(masks)


def test_video_io_roundtrip(tmp_path):
    frames, masks = synthetic_video(t=49, h=480, w=720)
    p = tmp_path / "v.mp4"
    write_video(p, frames, fps=16)
    assert video_info(p) == (49, 16.0, 480, 720)
    back = read_video(p)
    assert back.shape == frames.shape
    assert np.abs(back.astype(int) - frames.astype(int)).mean() < 20  # lossy but close
    mp = tmp_path / "m.mp4"
    write_video(mp, mask_to_frames(masks), fps=16)
    assert (read_mask_video(mp) == masks).mean() > 0.99


def test_resize_video_shapes():
    frames, masks = synthetic_video(t=3, h=480, w=854)
    assert resize_video(frames, 480, 720).shape == (3, 480, 720, 3)
    assert resize_video(masks.astype(np.uint8), 480, 720, is_mask=True).dtype == np.uint8


def test_chunking_and_static_filter(tmp_path):
    frames, _ = synthetic_video(t=120)
    p = tmp_path / "raw.mp4"
    write_video(p, frames, fps=30)
    cfg = DecomposerConfig(data_root=tmp_path, work_dir=tmp_path / "w", strides=(1,), max_clips_per_video=4)
    clips = list(iter_clips(p, cfg, "test", random.Random(0), 4))
    assert 1 <= len(clips) <= 4
    info, clip = clips[0]
    assert clip.shape == (49, 480, 720, 3) and info.stride == 1
    static = np.repeat(frames[:1], 60, axis=0)
    write_video(tmp_path / "static.mp4", static, fps=30)
    assert list(iter_clips(tmp_path / "static.mp4", cfg, "test", random.Random(0), 4)) == []


def test_layers_and_ratio():
    frames, masks = synthetic_video(t=4)
    fg, bg = compose_layers(frames, masks)
    assert fg[~masks].max() == 0 and bg[masks].max() == 0
    assert abs(foreground_ratio(masks) - masks.mean()) < 1e-9


def test_mask_from_black_fg():
    frames, masks = synthetic_video(t=4)
    fg, _ = compose_layers(frames, masks)
    rec = mask_from_black_fg(fg)
    assert (rec == masks).mean() > 0.99


def test_manifest(tmp_path):
    cfg = DecomposerConfig(data_root=tmp_path, work_dir=tmp_path / "w", dataset_name="ds")
    for cid, ratio, training in [("a", 0.1, True), ("b", 0.6, True), ("c", 0.2, False)]:
        rec = {
            "clip_id": cid,
            "source": "src",
            "status": "done",
            "caption": f"caption {cid}",
            "num_frames": 49,
            "height": 480,
            "width": 720,
            "foreground_ratio": ratio,
            "is_training": training,
        }
        p = record_path(cfg, cid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec))
    n, df = build_manifest(cfg)
    assert n == 3 and len((cfg.dataset_root / "prompts.txt").read_text().splitlines()) == 3
    assert (cfg.dataset_root / "fg-black.txt").read_text().splitlines()[0] == "src/fg/a-00.mp4"
    assert df.set_index("vid_name")["main_train_foreground_volume"].to_dict() == {"a": True, "b": False, "c": False}


def test_latent_mask_downsampling():
    m = torch.zeros(1, 1, 49, 480, 720)
    m[:, :, 5, 96:104, 200:208] = 1  # one aligned 8x8 patch in frame 5 -> latent frame 2, cell (12, 25)
    lat = downsample_mask_to_latent(m)
    assert lat.shape == (1, 1, 13, 60, 90)
    assert lat[0, 0, 2, 12, 25] == 1 and lat.sum() == 1


def test_expand_patch_embed():
    class T(torch.nn.Module):
        pass

    class PE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(32, 8, kernel_size=2, stride=2)

    t = T()
    t.patch_embed = PE()
    old = t.patch_embed.proj.weight.data.clone()
    expand_patch_embed(t, 1)
    w = t.patch_embed.proj.weight.data
    assert w.shape == (8, 48, 2, 2)
    assert torch.equal(w[:, :32], old) and torch.equal(w[:, 32:48], old[:, 16:32])


def test_metric_helpers():
    from evaluation.metrics.action_metric import kl_divergence
    from evaluation.metrics.flow_metric import flow_mse
    from evaluation.metrics.viclip_metrics import frames_to_tensor

    assert kl_divergence(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])) < 1e-6
    assert kl_divergence(torch.tensor([5.0, 0.0]), torch.tensor([0.0, 5.0])) > 1
    ref = np.zeros((4, 4, 2))
    gen = np.ones((4, 4, 2))
    assert flow_mse(ref, gen) == 2.0
    valid = np.zeros((4, 4), bool)
    valid[0, 0] = True
    assert flow_mse(ref, gen, valid) == 2.0
    frames, _ = synthetic_video(t=5)
    assert frames_to_tensor(frames).shape == (1, 8, 3, 224, 224)


@pytest.mark.skipif(os.environ.get("STM_TEST_MODELS") != "1", reason="set STM_TEST_MODELS=1 to run the metric models on CPU")
def test_metric_models_cpu():
    from evaluation.metrics.action_metric import ActionScorer, kl_divergence
    from evaluation.metrics.flow_metric import FlowScorer
    from evaluation.metrics.viclip_metrics import ViCLIPScorer

    frames, masks = synthetic_video(t=9, h=240, w=360)
    fg, bg = compose_layers(frames, masks)
    v = ViCLIPScorer(device="cpu")
    assert 0.99 < v.cosine(v.video_features(fg), v.video_features(fg)) <= 1.0001
    assert -1 <= v.cosine(v.video_features(frames), v.text_features("a red square moving right")) <= 1
    a = ActionScorer(device="cpu", num_frames=8)
    assert kl_divergence(a.logits(fg), a.logits(fg)) < 1e-4
    f = FlowScorer(device="cpu", pair_stride=4)
    assert f.score(bg, bg, valid_mask=masks) < 1e-6
