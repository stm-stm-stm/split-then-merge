"""StM Composer: CogVideoX-I2V fine-tuned to merge foreground/background layers.

Implements Sec. 3.4 of the paper on top of :class:`stm.trainer.Trainer`:

* **Multi-layer conditional fusion** -- the I2V image-conditioning channels are
  replaced by the (augmented) foreground latent and the background latent,
  concatenated channel-wise with the noisy target latent and fed through the
  (widened) patch-embedding projection (Eq. 1).
* **Transformation-aware augmentation** -- performed by
  :class:`stm.data.I2VDatasetWithAugmentation`; the conditioning foreground is
  therefore re-encoded here from the augmented frames.
* **Identity-preservation loss** -- foreground / background region losses
  normalised by their areas and mixed with weight ``alpha`` (Eq. 3-4;
  ``--apply_mask_on_orig_loss true --fg_weight_on_orig_loss 0.5``).

Additional, optional knobs (noisy conditioning layers, masking the conditioning
latents, extra fg/bg reconstruction losses, multi-foreground and depth
conditioning) are available behind the corresponding ``Args`` flags; see
``training/configs/recipes``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from stm.models.cogvideox.pipeline_i2v import CogVideoXImageToVideoPipeline
from stm.models.cogvideox.transformer_i2v import CogVideoXTransformer3DModel
from stm.schemas import Components
from stm.trainer import Trainer
from stm.utils import unwrap_model

# number of *extra* 16-channel latent layers prepended to the I2V conditioning
# (the pretrained I2V model already has one image-conditioning layer)
EXTRA_CONDITION_LAYERS = {
    "original": 0,
    "replace_img_with_fg": 0,
    "replace_img_with_noisy_fg": 0,
    "replace_img_with_fg_bg": 1,
    "replace_img_with_fg_bg_noisy": 1,
    "fg_plain_bg_noisy": 1,  # <- paper setting (fg + bg layers)
    "fg_plain_bg_noisy_fgimg_noaug": 2,
    "fg_noisy_bg_noisy_fgimg_noaug": 2,
    "fg_bg_depth_noisy": 2,
    "replace_img_with_3fg_bg_noisy": 3,
}


def expand_patch_embed(transformer: CogVideoXTransformer3DModel, extra_layers: int, latent_channels: int = 16) -> None:
    """Widen ``patch_embed.proj`` by ``extra_layers * latent_channels`` input channels.

    New channels are initialised with a copy of the pretrained image-conditioning
    weights so that training starts from a sensible operating point.  The model
    config is intentionally left untouched (``in_channels`` stays 32) so that the
    diffusers pipeline keeps computing ``latent_channels = in_channels // 2``.
    """
    if extra_layers <= 0:
        return
    old = transformer.patch_embed.proj
    in_channels = old.in_channels + extra_layers * latent_channels
    new = torch.nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    weight = torch.zeros(new.out_channels, in_channels, *new.kernel_size, dtype=old.weight.dtype)
    weight[:, : old.in_channels] = old.weight.data
    cond = old.weight.data[:, old.in_channels - latent_channels : old.in_channels]
    for i in range(extra_layers):
        s = old.in_channels + i * latent_channels
        weight[:, s : s + latent_channels] = cond.clone()
    new.weight = torch.nn.Parameter(weight)
    if old.bias is not None:
        new.bias = torch.nn.Parameter(old.bias.data.clone())
    transformer.patch_embed.proj = new


def downsample_mask_to_latent(mask: torch.Tensor, temporal_compression: int = 4, spatial_compression: int = 8) -> torch.Tensor:
    """Pixel-space mask [B, 1, F, H, W] -> latent-space mask [B, 1, F', H', W'].

    CogVideoX's causal VAE maps frame 0 to latent frame 0 and every following
    group of ``temporal_compression`` frames to one latent frame; a latent cell
    is marked foreground if *any* pixel it covers is foreground (max-pooling),
    so the identity loss / latent masking never cuts into the subject.
    """
    b, c, f, h, w = mask.shape
    first = mask[:, :, :1]
    rest = mask[:, :, 1:]
    n_rest = rest.shape[2]
    if n_rest % temporal_compression != 0:  # pad by repeating the last frame
        pad = temporal_compression - n_rest % temporal_compression
        rest = torch.cat([rest, rest[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
    rest = rest.reshape(b, c, -1, temporal_compression, h, w).amax(dim=3)
    m = torch.cat([first, rest], dim=2)
    m = F.max_pool3d(m, kernel_size=(1, spatial_compression, spatial_compression))
    return m


class StMComposerTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    # ------------------------------------------------------------ components
    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)
        components.pipeline_cls = CogVideoXImageToVideoPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
        expand_patch_embed(components.transformer, EXTRA_CONDITION_LAYERS[self.args.i2v_training_type])
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        return CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # [B, C, F, H, W] -> [B, C', F', H', W']
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        return latent_dist.sample() * vae.config.scaling_factor

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        return self.components.text_encoder(ids.to(self.components.text_encoder.device))[0]

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = [
            "prompt_embedding",
            "encoded_video",
            "encoded_video_fg",
            "encoded_video_bg",
            "frames_fg",
            "frames_bg",
            "mask",
            "mask_augmented",
            "num_fgs",
            "encoded_depth",
        ]
        batch: Dict[str, Any] = {}
        for k in keys:
            if all(k in s for s in samples):
                batch[k] = torch.stack([s[k] for s in samples])
        if "mask_augmented" not in batch and "mask" in batch:
            batch["mask_augmented"] = batch["mask"]
        return batch

    # ----------------------------------------------------------------- helpers
    def _encode_frames(self, frames: torch.Tensor, noise_mean: Optional[float]) -> torch.Tensor:
        """frames [B, 3, F, H, W] in [-1, 1] -> latent [B, F', C, H', W'] (optionally noised first)."""
        vae = self.components.vae
        frames = frames.to(vae.device, dtype=vae.dtype)
        if noise_mean is not None:
            sigma = torch.exp(torch.normal(mean=noise_mean, std=0.5, size=(1,), device=frames.device)).to(frames.dtype)
            frames = frames + torch.randn_like(frames) * sigma
        latent = vae.encode(frames).latent_dist.sample() * vae.config.scaling_factor
        return latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

    def _pad_temporal(self, latent: torch.Tensor) -> torch.Tensor:
        """CogVideoX-1.5: pad latent frames (dim 1) to a multiple of patch_size_t."""
        p_t = self.state.transformer_config.patch_size_t
        if p_t is None or latent.shape[1] % p_t == 0:
            return latent
        n_pad = p_t - latent.shape[1] % p_t
        return torch.cat([latent[:, :1].repeat(1, n_pad, 1, 1, 1), latent], dim=1)

    def _latent_mask(self, mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """dataset mask [B, F, 1, H, W] (0/1) -> [B, F', 1, H', W'] aligned with ``like`` [B, F', C, H', W']."""
        m = downsample_mask_to_latent(mask.permute(0, 2, 1, 3, 4).float())  # [B,1,F',H',W']
        m = m.permute(0, 2, 1, 3, 4)
        if m.shape[1] != like.shape[1]:  # temporal padding applied to latents
            n_pad = like.shape[1] - m.shape[1]
            m = torch.cat([m[:, :1].repeat(1, n_pad, 1, 1, 1), m], dim=1)
        return m.to(like.device, like.dtype)

    def prepare_rotary_positional_embeddings(
        self, height: int, width: int, num_frames: int, transformer_config, vae_scale_factor_spatial: int, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)
        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t
        return get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

    def _build_condition(self, batch: Dict[str, Any], latent: torch.Tensor) -> torch.Tensor:
        """Assemble the conditioning latents [B, F, C_cond, H, W] for the current ``i2v_training_type``."""
        args = self.args
        tt = args.i2v_training_type
        fg_noise = args.noise_mean_fg if args.apply_fg_noise else None
        bg_noise = args.noise_mean_bg if args.apply_bg_noise else None

        if tt == "original":  # plain I2V: first frame, noised, zero-padded in time
            first = batch["frames_fg"][:, :, :1] if "images" not in batch else batch["images"].unsqueeze(2)
            image_latents = self._encode_frames(first, noise_mean=-3.0)
            pad = image_latents.new_zeros((latent.shape[0], latent.shape[1] - 1, *latent.shape[2:]))
            return torch.cat([image_latents, pad], dim=1)

        # the conditioning foreground is *re-encoded* from the augmented frames
        # (``batch["encoded_video_fg"]`` holds the un-augmented latent used by the fg loss)
        if tt == "replace_img_with_3fg_bg_noisy":
            frames_fg = batch["frames_fg"]  # [B, 3fg, 3, F, H, W]
            latent_fg = torch.stack([self._encode_frames(frames_fg[:, i], fg_noise) for i in range(frames_fg.shape[1])], dim=1)
            latent_fg = einops.rearrange(latent_fg, "B FG F C H W -> B F (FG C) H W")
        else:
            latent_fg = self._encode_frames(batch["frames_fg"], fg_noise)
        if args.apply_mask_on_latents_fg:
            latent_fg = latent_fg * self._latent_mask(batch["mask_augmented"], latent_fg)
        if tt in ("replace_img_with_fg", "replace_img_with_noisy_fg"):
            return latent_fg

        if bg_noise is not None or "encoded_video_bg" not in batch:
            latent_bg = self._encode_frames(batch["frames_bg"], bg_noise)
        else:
            latent_bg = batch["encoded_video_bg"].permute(0, 2, 1, 3, 4).to(latent_fg.dtype)
        if args.apply_mask_on_latents_bg:
            latent_bg = latent_bg * (1.0 - self._latent_mask(batch["mask"], latent_bg))

        layers = [latent_fg, latent_bg]
        if tt in ("fg_plain_bg_noisy_fgimg_noaug", "fg_noisy_bg_noisy_fgimg_noaug"):
            first = batch["frames_fg"][:, :, :1].expand(-1, -1, batch["frames_fg"].shape[2], -1, -1)
            layers.insert(0, self._encode_frames(first.contiguous(), None))
        if tt == "fg_bg_depth_noisy":
            layers.append(batch["encoded_depth"].permute(0, 2, 1, 3, 4).to(latent_fg.dtype))
        return torch.cat(layers, dim=2)

    # ----------------------------------------------------------------- loss
    @override
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.args
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_video"].permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        latent = self._pad_temporal(latent)
        batch_size, num_frames, num_channels, height, width = latent.shape
        prompt_embedding = prompt_embedding.view(batch_size, prompt_embedding.shape[1], -1).to(dtype=latent.dtype)

        condition = self._pad_temporal(self._build_condition(batch, latent)).to(latent.dtype)
        assert condition.shape[1] == num_frames and condition.shape[3:] == latent.shape[3:], (condition.shape, latent.shape)

        scheduler = self.components.scheduler
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device).long()
        noise = torch.randn_like(latent)
        latent_noisy = scheduler.add_noise(latent, noise, timesteps)
        model_input = torch.cat([latent_noisy, condition], dim=2)

        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        ofs_emb = None if transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        predicted_noise = self.components.transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]
        latent_pred = scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)  # v-pred -> x0

        alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        weights = (1 / (1 - alphas_cumprod)).view(-1, *([1] * (latent_pred.dim() - 1)))
        se = weights * (latent_pred - latent) ** 2  # [B, F, C, H, W]
        eps = 1e-6

        def region_mean(err: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
            # mean over the region, per sample (Eq. 3), then over the batch
            num = (err * region).sum(dim=(1, 2, 3, 4))
            den = region.sum(dim=(1, 2, 3, 4)) * err.shape[2] + eps
            return (num / den).mean()

        latent_mask = self._latent_mask(batch["mask"], latent) if "mask" in batch else None

        # --- identity-preservation loss (Eq. 3-4) --------------------------------
        if args.apply_mask_on_orig_loss and latent_mask is not None:
            alpha = args.fg_weight_on_orig_loss
            if args.apply_adaptive_mask_on_orig_loss:
                # re-implemented ablation: weight the foreground inversely to its area
                fg_area = latent_mask.mean(dim=(1, 2, 3, 4)).clamp(eps, 1 - eps)
                alpha = (1.0 - fg_area).view(-1, 1, 1, 1, 1)
                l_fg = (se * latent_mask).sum(dim=(1, 2, 3, 4)) / (latent_mask.sum(dim=(1, 2, 3, 4)) * num_channels + eps)
                l_bg = (se * (1 - latent_mask)).sum(dim=(1, 2, 3, 4)) / ((1 - latent_mask).sum(dim=(1, 2, 3, 4)) * num_channels + eps)
                loss_orig = (alpha.view(-1) * l_fg + (1 - alpha.view(-1)) * l_bg).mean()
            else:
                loss_orig = alpha * region_mean(se, latent_mask) + (1 - alpha) * region_mean(se, 1 - latent_mask)
        else:
            loss_orig = se.reshape(batch_size, -1).mean(dim=1).mean()

        # --- optional layer reconstruction losses (3-loss ablations) --------------
        zero = latent.new_zeros(())
        loss_fg, loss_bg = zero, zero
        if args.weight_fg > 0 and "encoded_video_fg" in batch:
            target_fg = batch["encoded_video_fg"].permute(0, 2, 1, 3, 4)
            if target_fg.dim() == 6:  # multi-fg: use the first foreground
                target_fg = target_fg[:, 0]
            target_fg = self._pad_temporal(target_fg).to(latent.dtype)
            err = weights * (latent_pred - target_fg) ** 2
            loss_fg = region_mean(err, latent_mask) if (args.apply_mask_on_fg_bg_loss and latent_mask is not None) else err.mean()
        if args.weight_bg > 0 and "encoded_video_bg" in batch:
            target_bg = self._pad_temporal(batch["encoded_video_bg"].permute(0, 2, 1, 3, 4)).to(latent.dtype)
            err = weights * (latent_pred - target_bg) ** 2
            loss_bg = region_mean(err, 1 - latent_mask) if (args.apply_mask_on_fg_bg_loss and latent_mask is not None) else err.mean()

        loss = args.weight_orig * loss_orig + args.weight_fg * loss_fg + args.weight_bg * loss_bg
        return loss, loss_fg, loss_bg, loss_orig

    # ----------------------------------------------------------- validation
    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline, guidance_scale: float = 6.0
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        args = self.args
        common = dict(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=eval_data["prompt"],
            generator=self.state.generator,
            num_inference_steps=args.inference_steps,
            guidance_scale=guidance_scale,
        )
        if args.i2v_training_type == "original":
            video = pipe(image=eval_data["image"], **common).frames[0]
            return [("video", video)]

        encoded_fg = eval_data["encoded_video_fg"]  # [1, C, F, H, W] (or [1, FG, C, F, H, W])
        encoded_bg = eval_data["encoded_video_bg"]
        if eval_data.get("mask") is not None:
            mask = eval_data["mask"].unsqueeze(0)  # [1, F, 1, H, W]
            m = self._latent_mask(mask, encoded_fg.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)  # [1,1,F',H',W']
            if args.apply_mask_on_latents_fg:
                encoded_fg = encoded_fg * m
            if args.apply_mask_on_latents_bg:
                encoded_bg = encoded_bg * (1.0 - m)
        encoded_layers = {"encoded_video_fg": encoded_fg, "encoded_video_bg": encoded_bg}
        if "encoded_video_depth" in eval_data:
            encoded_layers["encoded_video_depth"] = eval_data["encoded_video_depth"]
        if "encoded_fg_initial_repeated" in eval_data:
            encoded_layers["encoded_fg_initial_repeated"] = eval_data["encoded_fg_initial_repeated"]
        video = pipe(image=None, encoded_layers=encoded_layers, i2v_training_type=args.i2v_training_type, **common).frames[0]
        return [("video", video)]
