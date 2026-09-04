"""In-process wrapper around Segment-Any-Motion (SegAnyMo, Huang et al. CVPR'25).

The authors' reference implementation (``third_party/SegAnyMo``) is a chain of
five stand-alone scripts that each reload their model.  This module keeps every
model resident and runs the identical algorithm per clip:

1. Depth-Anything-V2 (relative disparity)            ``core/utils/run_depth.py``
2. DINOv2 ViT-B/14 patch features (query frames)     ``core/utils/dino_feat.py``
3. BootsTAPIR dense point tracks from query frames   ``preproc/run_tapir.py``
4. moseg trajectory motion classifier                ``inference.py``
5. SAM2 mask densification + object merging          ``sam2/run_sam2.py``

Only the intermediate PNG frames needed by SAM2's video predictor touch disk.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

UINT16_MAX = 65535


def _to_uint16(disp: np.ndarray) -> np.ndarray:
    lo, hi = disp.min(), disp.max()
    if hi - lo > np.finfo("float").eps:
        return (UINT16_MAX * (disp - lo) / (hi - lo)).astype(np.uint16)
    return np.zeros(disp.shape, dtype=np.uint16)


class SegAnyMoSegmenter:
    def __init__(
        self,
        seganymo_root: Path,
        sam2_ckpt: Path,
        tapir_ckpt: Path,
        moseg_repo: str = "Changearthmore/moseg",
        depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf",
        dino_model: str = "dinov2_vitb14",
        step: int = 10,
        device: str = "cuda",
        seed: int = 0,
        tapir_chunk: int = 4096,
        tracks_per_query_frame: Optional[int] = None,
        fast: bool = False,
    ) -> None:
        root = Path(seganymo_root)
        self.tapir_chunk = tapir_chunk
        # fast=True runs BootsTAPIR under bf16 autocast and Depth-Anything in fp16 (~2x faster mask stage;
        # tracks differ by ~1 px on average, mask IoU on DAVIS is unchanged within noise)
        self.fast = fast
        # The reference tracks a full grid (~9600 points) from every query frame, after which
        # ``inference.py`` keeps a uniform random 1/len(q_ts) of them and caps the total at 5000.
        # Sampling the same number of query points *before* tracking is equivalent in
        # distribution and avoids ~80% of the (dominant) TAPIR cost.  None = track the full grid.
        self.tracks_per_query_frame = tracks_per_query_frame
        for p in (root, root / "preproc", root / "sam2"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        os.environ.setdefault("WANDB_MODE", "disabled")

        import run_sam2  # noqa: E402  (third_party/SegAnyMo/sam2/run_sam2.py)
        from core.dataset.data_utils import normalize_point_traj_torch  # noqa: E402
        from core.dataset.kubric import parse_tapir_track_info  # noqa: E402
        from core.utils.dino_feat import ViTExtractor  # noqa: E402
        from core.utils.utils import get_feat, load_config_file  # noqa: E402
        from huggingface_hub import snapshot_download
        from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
        from tapnet_torch import tapir_model  # noqa: E402
        from tapnet_torch import transforms as tapir_transforms
        from train_seq import setup_model  # noqa: E402
        from transformers import pipeline as hf_pipeline

        self._run_sam2 = run_sam2
        run_sam2.args = SimpleNamespace(vis=False)  # module-level global used inside its helpers
        self._normalize_point_traj_torch = normalize_point_traj_torch
        self._parse_tapir_track_info = parse_tapir_track_info
        self._get_feat = get_feat
        self._tapir_transforms = tapir_transforms

        self.device = torch.device(device)
        self.step = step
        self.seed = seed

        # 1. depth
        self.depth_pipe = hf_pipeline(
            task="depth-estimation", model=depth_model, device=self.device, torch_dtype=torch.float16 if fast else torch.float32
        )
        # 2. dino
        self.dino = ViTExtractor(model_type=dino_model, stride=7)
        # 3. tracks
        self.tapir = tapir_model.TAPIR(pyramid_level=1)
        self.tapir.load_state_dict(torch.load(tapir_ckpt, map_location="cpu"))
        self.tapir = self.tapir.to(self.device).eval()
        # 4. moseg
        cfg = load_config_file(str(root / "configs" / "example_train.yaml"))
        cfg.resume_path = str(Path(snapshot_download(moseg_repo)) / "moseg.pth")
        self.moseg_cfg = cfg
        self.moseg = setup_model(cfg)
        ckpt = torch.load(cfg.resume_path, map_location="cpu")
        self.moseg.load_state_dict(ckpt["model_state_dict"])
        self.moseg = self.moseg.to(self.device).eval()
        # 5. sam2
        self.sam2 = build_sam2_video_predictor("sam2_hiera_l.yaml", str(sam2_ckpt), device=str(self.device))

    # ------------------------------------------------------------------ stages
    @torch.no_grad()
    def _depths(self, frames: np.ndarray, batch_size: int = 8) -> torch.Tensor:
        """[1, 1, H, W, T] normalised depth, matching inference.py's PNG round-trip."""
        depth_list = []
        h, w = frames.shape[1:3]
        images = [Image.fromarray(f) for f in frames]
        outputs = self.depth_pipe(images, batch_size=batch_size)
        for out in outputs:
            disp = out["predicted_depth"]
            if disp.ndim == 2:
                disp = disp[None]
            disp = torch.nn.functional.interpolate(disp.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False)
            disp = disp.squeeze().cpu().numpy()
            disp16 = _to_uint16(disp)
            # the reference implementation writes uint16 PNGs and reloads them with PIL.convert('L')
            d8 = np.array(Image.fromarray(disp16).convert("L")).astype(np.float32)
            d8 = (d8 - d8.min()) / max(d8.max() - d8.min(), 1e-8)
            depth_list.append(torch.from_numpy(d8))
        return torch.stack(depth_list, dim=0).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def _dino_feats(self, frames: np.ndarray, q_ts) -> Dict[int, np.ndarray]:
        import torchvision.transforms as T

        h, w = frames.shape[1:3]
        h14, w14 = (h + 13) // 14 * 14, (w + 13) // 14 * 14
        prep = T.Compose([T.ToTensor(), T.Resize([h14, w14]), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        feats = {}
        for q in q_ts:
            batch = prep(Image.fromarray(frames[q]))[None].to(self.device)
            desc = self.dino.extract_descriptors(batch, [11], "key", include_cls=False)
            desc = desc.reshape(desc.shape[0], self.dino.num_patches[0], self.dino.num_patches[1], -1).squeeze()
            feats[q] = desc.cpu().numpy().astype(np.float16)
        return feats

    @torch.no_grad()
    def _tracks(self, frames: np.ndarray, q_ts) -> Dict[int, np.ndarray]:
        """BootsTAPIR tracks for every query frame: {q_t: (N, T, 4) = x, y, occ, dist}."""
        import mediapy as media

        num_frames, height, width = frames.shape[:3]
        rh = rw = 256
        video = media.resize_video(frames, (rh, rw))
        video = torch.from_numpy(np.asarray(video)).to(self.device).float() / 255 * 2 - 1
        video = video[None]
        grid_size = max(1, int(math.sqrt((height * width) / 9000)))
        y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
        y_resize = y / (height - 1) * (rh - 1)
        x_resize = x / (width - 1) * (rw - 1)
        query_xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        out: Dict[int, np.ndarray] = {}
        # The reference script calls ``model(video, points)`` per 128-point chunk, which recomputes
        # the video feature pyramid every time.  Compute the feature grids once per clip instead
        # and only run query-feature extraction + trajectory refinement per chunk (identical math).
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.fast):
            feature_grids = self.tapir.get_feature_grids(video, False, None)
        p = self.tapir.num_pips_iter
        for t in q_ts:
            pts = np.stack([t * np.ones_like(y), y_resize, x_resize], axis=-1).reshape(-1, 3).astype(np.float32)
            xy_t = query_xy
            if self.tracks_per_query_frame and self.tracks_per_query_frame < len(pts):
                sel = np.sort(rng.choice(len(pts), self.tracks_per_query_frame, replace=False))
                pts, xy_t = pts[sel], query_xy[sel]
            chunks = []
            for chunk in np.array_split(pts, max(1, math.ceil(len(pts) / self.tapir_chunk)), axis=0):
                points = torch.from_numpy(chunk)[None].to(self.device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.fast):
                    query_features = self.tapir.get_query_features(video, False, points, feature_grids, None)
                    # query_chunk_size is only a memory knob of TAPIR's refinement (default 64); 256 is ~1.6x faster, same results
                    traj = self.tapir.estimate_trajectories(video.shape[-3:-1], False, feature_grids, query_features, points, 256)
                    tracks = torch.mean(torch.stack(traj["tracks"][p::p]), dim=0)
                    occ = torch.mean(torch.stack(traj["occlusion"][p::p]), dim=0)
                    dist = torch.mean(torch.stack(traj["expected_dist"][p::p]), dim=0)
                tracks = tracks[0].float().cpu().numpy()
                occ = occ[0].float().cpu().numpy()
                dist = dist[0].float().cpu().numpy()
                tracks = self._tapir_transforms.convert_grid_coordinates(tracks, (rw - 1, rh - 1), (width - 1, height - 1))
                chunks.append(np.concatenate([tracks, occ[..., None], dist[..., None]], axis=-1))
            arr = np.concatenate(chunks, axis=0)  # (N, T, 4)
            arr[:, t, :2] = xy_t
            out[t] = arr.astype(np.float32)
        return out

    @torch.no_grad()
    def _motion_seg(self, depths: torch.Tensor, tracks: Dict[int, np.ndarray], dinos: Dict[int, np.ndarray], hw):
        """Port of ``inference.py`` (bootstapir branch, no GT / no visualisation)."""
        cfg = self.moseg_cfg
        H, W = hw
        num_frames = depths.shape[-1]
        q_ts = sorted(tracks)
        ratio = 1.0 if self.tracks_per_query_frame else 1 / len(q_ts)  # already sub-sampled before tracking
        g = torch.Generator().manual_seed(self.seed)
        sampled_tracks, sampled_vis, sampled_conf, sampled_vv, sampled_cv, dino_feats = [], [], [], [], [], []
        for q_t in q_ts:
            tracks_2d = torch.from_numpy(tracks[q_t])
            track_2d, occs, dists = tracks_2d[..., :2], tracks_2d[..., 2], tracks_2d[..., 3]
            visibles, _, confidences, visib_value, confi_value = self._parse_tapir_track_info(occs, dists, 0.5)
            pts_num = track_2d.shape[0]
            sel = torch.randperm(pts_num, generator=g)[: int(pts_num * ratio)]
            sampled_track = track_2d[sel]
            sampled_tracks.append(sampled_track)
            sampled_vis.append(visibles[sel])
            sampled_conf.append(confidences[sel])
            sampled_vv.append(visib_value[sel])
            sampled_cv.append(confi_value[sel])
            if cfg.dino:
                dino_list = torch.from_numpy(dinos[q_t]).unsqueeze(0).unsqueeze(0)
                factor = (dino_list[0].shape[1] / H, dino_list[0].shape[2] / W)
                gt_dino = self._get_feat(factor, sampled_track.permute(2, 0, 1)[..., q_t : q_t + 1], dino_list)
                if not cfg.dino_later:
                    gt_dino = gt_dino.repeat(1, 1, 1, num_frames)
                dino_feats.append(gt_dino)
        track_2d = torch.cat(sampled_tracks, 0)
        visibles = torch.cat(sampled_vis, 0)
        confidences = torch.cat(sampled_conf, 0)
        visib_value = torch.cat(sampled_vv, 0)
        confi_value = torch.cat(sampled_cv, 0)
        dino = torch.cat(dino_feats, dim=2) if cfg.dino else None

        sel = torch.randperm(track_2d.shape[0], generator=g)[:5000]
        track_2d, visibles, confidences = track_2d[sel], visibles[sel], confidences[sel]
        visib_value, confi_value = visib_value[sel], confi_value[sel]
        if cfg.dino:
            dino = dino[:, :, sel, :]
        keep = ~torch.all(~visibles, dim=1)
        track_2d, visibles, confidences = track_2d[keep], visibles[keep], confidences[keep]
        visib_value, confi_value = visib_value[keep], confi_value[keep]
        if cfg.dino:
            dino = dino[:, :, keep, :]
        if track_2d.shape[0] == 0:
            return None
        cols_all_false = torch.all(~visibles.permute(1, 0), dim=1)
        for t in range(cols_all_false.size(0)):
            if cols_all_false[t]:
                _, idx = torch.max(confidences[:, t], dim=0)
                visibles[idx, t] = True

        track = track_2d.permute(2, 0, 1).unsqueeze(0)  # [1, 2, N, T]
        mask = (~visibles).unsqueeze(0).unsqueeze(0)
        traj = self._normalize_point_traj_torch(track, [H, W])
        batch = {
            "traj": traj.float().to(self.device),
            "mask": mask.float().to(self.device),
            "depth": depths.float().to(self.device),
        }
        if cfg.extra_info:
            batch["visib_value"] = visib_value.unsqueeze(0).unsqueeze(0).float().to(self.device)
            batch["confi_value"] = confi_value.unsqueeze(0).unsqueeze(0).float().to(self.device)
        if cfg.dino:
            batch["dino"] = dino.float().to(self.device)
        pred = self.moseg(batch).detach().cpu()

        L = track.shape[-1]
        thresholds = [0.99, 0.98, 0.97, 0.96, 0.95] if L > 300 else [0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7]
        for thres in thresholds:
            m = pred > thres
            if m.sum() >= 10:
                pred = m
                break
        else:
            _, topk = torch.topk(pred.view(-1), 3)
            pred = torch.zeros_like(pred, dtype=torch.bool)
            pred.view(-1)[topk] = True
        d_mask = pred.squeeze(0).squeeze(0)
        visibles = visibles & (confidences > 0.9)
        dynamic_traj = track.squeeze(0)[:, d_mask, :].numpy()  # [2, N, T]
        d_visibility = visibles[d_mask, :].numpy()  # [N, T]
        d_confidences = confidences[d_mask, :].numpy()  # [N, T]
        if d_visibility.shape[0] == 0:
            return None
        return dynamic_traj, d_visibility, d_confidences

    @torch.no_grad()
    def _sam2_masks(self, images_dir: Path, traj, visible_mask, confidences, hw) -> Tuple[Optional[np.ndarray], int]:
        """Port of ``sam2/run_sam2.py::main`` -> (bool mask [T,H,W], num_objects)."""
        rs = self._run_sam2
        H, W = hw
        _, N, T = traj.shape
        predictor = self.sam2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(str(images_dir))
            q_ts = list(range(0, T, 16))
            max_iterations = min(max(len(q_ts), 5), 10)
            memory_dict = rs.process_invisible_traj(
                traj,
                visible_mask,
                confidences,
                state,
                predictor,
                dilation_size=6,
                max_iterations=max_iterations,
                timestep=T,
                downscale_factor=None,
            )
            if len(memory_dict) == 0:
                return None, 0
            video_segments: Dict[int, Dict[int, np.ndarray]] = {}
            for obj_id_valid, pkg in memory_dict.items():
                q_ts = list(range(0, T, 16))
                predictor.reset_state(state)
                time = pkg["time"]
                if time in q_ts:
                    q_ts.remove(time)
                q_ts.insert(0, time)
                pts_trajs, vis_trajs = pkg["pts_trajs"], pkg["vis_trajs"]
                require_reverse = True
                prompted = False
                for t in q_ts:
                    visible_points = pts_trajs[:, :, t][vis_trajs[:, t]]
                    if visible_points.shape[0] == 0:
                        continue
                    prompted = True
                    if t == 0:
                        require_reverse = False
                    prompt_points = rs.find_dense_pts(visible_points)
                    labels = np.ones(prompt_points.shape[0], dtype=np.int32)
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=t, obj_id=obj_id_valid, points=prompt_points, labels=labels
                    )
                    prompt_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    in_mask = rs.find_pts_in_mask(prompt_mask, visible_points)
                    if in_mask.sum() < visible_points.shape[0] * 0.7:
                        nearest_point, _, farthest_points, _, _ = rs.find_centroid_and_nearest_farthest(visible_points)
                        prompt_points = np.concatenate((nearest_point, farthest_points), axis=0)
                        labels = np.ones(prompt_points.shape[0], dtype=np.int32)
                        _, _, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=t, obj_id=obj_id_valid, points=prompt_points, labels=labels
                        )

                if not prompted:  # tracks invisible on every query frame -> nothing to prompt SAM2 with
                    continue

                def _collect(iterator):
                    for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
                        seg = video_segments.setdefault(out_frame_idx, {})
                        for i, oid in enumerate(out_obj_ids):
                            m = (out_mask_logits[i] > 0.0).cpu().numpy()
                            seg[oid] = m if oid not in seg else (seg[oid] | m)

                _collect(predictor.propagate_in_video(state))
                if require_reverse:
                    _collect(predictor.propagate_in_video(state, reverse=True))
            predictor.reset_state(state)
        if not video_segments:
            return None, 0
        video_segments = dict(sorted(video_segments.items()))
        merges = rs.analyze_frame_merges(video_segments, iou_threshold=0.9)
        merged = rs.merge_masks(video_segments, merges)
        masks = np.zeros((T, H, W), dtype=bool)
        for frame_idx, per_obj in merged.items():
            if per_obj:
                masks[frame_idx] = rs.put_per_obj_mask(per_obj, H, W) > 0
        return masks, len(merges)

    # ------------------------------------------------------------------ public
    def segment(self, frames: np.ndarray, work_dir: Path) -> Tuple[Optional[np.ndarray], int]:
        """frames uint8 [T,H,W,3] -> (bool mask [T,H,W] or None if no moving object, #objects)."""
        T, H, W = frames.shape[:3]
        assert max(H, W) <= 1000, "efficiency mode expects max side <= 1000"
        work_dir = Path(work_dir)
        images_dir = work_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(images_dir / f"{i:05d}.png")
        try:
            q_ts = list(range(0, T, self.step))
            depths = self._depths(frames)
            dinos = self._dino_feats(frames, q_ts)
            tracks = self._tracks(frames, q_ts)
            res = self._motion_seg(depths, tracks, dinos, (H, W))
            if res is None:
                return None, 0
            traj, vis, conf = res
            mask, n_obj = self._sam2_masks(images_dir, traj, vis, conf, (H, W))
            return mask, n_obj
        finally:
            shutil.rmtree(images_dir, ignore_errors=True)
