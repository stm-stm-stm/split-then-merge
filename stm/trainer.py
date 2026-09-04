"""Generic accelerate-based trainer for the StM Composer (image/video-to-video).

Model-specific behaviour (component loading, loss, validation sampling) lives in
subclasses registered in :mod:`stm.models`; this class owns distributed setup,
data loading, optimisation, checkpointing / resuming and the validation loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import einops
import torch
import transformers
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from stm.constants import LOG_LEVEL, LOG_NAME
from stm.data import I2VDatasetWithAugmentation, I2VDatasetWithResize
from stm.data.utils import (
    load_prompts,
    load_videos,
    load_videos_fg,
    preprocess_depth_with_resize,
    preprocess_image_with_resize,
    preprocess_mask_with_resize,
    preprocess_video_with_resize,
)
from stm.schemas import Args, Components, State_I2V
from stm.utils import (
    cast_training_params,
    export_to_video_artifacts,
    find_files,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Trainer:
    # components to keep on CPU during training (see `Components`)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State_I2V(
            weight_dtype=self.__get_training_dtype(),
            train_frames=args.train_resolution[0],
            train_height=args.train_resolution[1],
            train_width=args.train_resolution[2],
        )
        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None
        self.optimizer = None
        self.lr_scheduler = None

        # every launch gets a numbered run directory <output_dir>/0001, 0002, ...;
        # --resume_from_checkpoint (any value) resumes the latest run instead.
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        run_ids = [int(f) for f in os.listdir(self.args.output_dir) if f.isdigit()]
        max_index = max(run_ids, default=0)
        if self.args.resume_from_checkpoint is not None and max_index > 0:
            self.args.output_dir = self.args.output_dir / f"{max_index:04d}"
            ckpts = find_files(self.args.output_dir)
            self.args.resume_from_checkpoint = ckpts[-1] if ckpts else None
        else:
            self.args.output_dir = self.args.output_dir / f"{max_index + 1:04d}"
            self.args.resume_from_checkpoint = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()
        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    # ------------------------------------------------------------------ setup
    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout))
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if str(self.args.report_to).lower() == "none" else self.args.report_to
        self.accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False
        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=LOG_LEVEL)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        self.args.output_dir = Path(self.args.output_dir)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            logger.info(f"Output directory: {self.args.output_dir}")
            save_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.args.model_dump().items()}
            (self.args.output_dir / "args.json").write_text(json.dumps(save_dict, indent=4))

    def check_setting(self) -> None:
        if self.UNLOAD_LIST is None:
            logger.warning("No UNLOAD_LIST specified for this Trainer. All components will be loaded to GPU during training.")
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")
        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()
        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        if self.args.model_type != "i2v":
            raise ValueError(f"Invalid model type: {self.args.model_type}")
        dataset_cls = (
            I2VDatasetWithAugmentation
            if (self.args.aug_config_path and os.path.exists(self.args.aug_config_path))
            else I2VDatasetWithResize
        )
        logger.info(f"Using dataset class {dataset_cls.__name__}")
        self.dataset = dataset_cls(
            **self.args.model_dump(),
            device=self.accelerator.device,
            max_num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            trainer=self,
        )

        # VAE / text encoder are used by the dataset for on-the-fly encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(self.accelerator.device, dtype=self.state.weight_dtype)

        # Pre-compute (and cache on disk) all prompt embeddings while the text encoder is on the GPU,
        # then move it back to the CPU for the rest of training.
        if hasattr(self.dataset, "precompute_prompt_embeddings"):
            logger.info("Precomputing prompt embeddings ...")
            self.dataset.precompute_prompt_embeddings(rank=self.accelerator.process_index, world_size=self.accelerator.num_processes)
            self.accelerator.wait_for_everyone()
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        weight_dtype = self.state.weight_dtype
        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            raise ValueError("bfloat16 is not supported on MPS.")
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                component.requires_grad_(self.args.training_type == "sft" and attr_name == "transformer")
        if self.args.training_type == "lora":
            lora_config = LoraConfig(
                r=self.args.rank, lora_alpha=self.args.lora_alpha, init_lora_weights=True, target_modules=self.args.target_modules
            )
            self.components.transformer.add_adapter(lora_config)
            self.__prepare_saving_loading_hooks(lora_config)
        ignore_list = ["transformer"] + (self.UNLOAD_LIST or [])
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)
        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")
        cast_training_params([self.components.transformer], dtype=torch.float32)
        trainable_parameters = [p for p in self.components.transformer.parameters() if p.requires_grad]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=[{"params": trainable_parameters, "lr": self.args.learning_rate}],
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True
        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes
        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler, optimizer=optimizer, total_num_steps=total_training_steps, num_warmup_steps=num_warmup_steps
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
        )
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        """Load the validation triplets (prompt, fg video, bg video [, mask, depth])."""
        val_dir = self.args.validation_dir
        prompts = load_prompts(val_dir / self.args.validation_prompts)
        limit = self.args.validation_num_samples
        if limit:
            prompts = prompts[:limit]
        n = len(prompts)
        self.state.validation_prompts = prompts
        if self.args.validation_images and (val_dir / self.args.validation_images).exists():
            from stm.data.utils import load_images

            self.state.validation_images = load_images(val_dir / self.args.validation_images)
        else:
            self.state.validation_images = [None] * n
        tt = self.args.i2v_training_type
        if tt == "original":
            self.state.validation_videos_fg = [None] * n
            self.state.validation_videos_bg = [None] * n
            return
        fg_loader = load_videos_fg if tt == "replace_img_with_3fg_bg_noisy" else load_videos
        fg_list = self.args.validation_fg_column or ("fg-black.txt" if (val_dir / "fg-black.txt").exists() else "fg.txt")
        self.state.validation_videos_fg = fg_loader(val_dir / fg_list)
        # like the training set, prefer the inpainted clean-plate background when it exists
        bg_paths = []
        for p in load_videos(val_dir / "bg.txt"):
            inpainted = p.parent.parent / "bg-inpainted" / p.name
            bg_paths.append(inpainted if inpainted.exists() else p)
        self.state.validation_videos_bg = bg_paths
        if self.args.load_depth:
            self.state.validation_videos_depth = load_videos(val_dir / "depth.txt")
        if (self.args.apply_mask_on_latents_fg or self.args.apply_mask_on_latents_bg) and (val_dir / "masks.txt").exists():
            self.state.validation_videos_mask = load_videos(val_dir / "masks.txt")
        else:
            self.state.validation_videos_mask = [None] * n
        for attr in (
            "validation_images",
            "validation_videos_fg",
            "validation_videos_bg",
            "validation_videos_depth",
            "validation_videos_mask",
        ):
            lst = getattr(self.state, attr)
            if lst and len(lst) > n:
                setattr(self.state, attr, lst[:n])

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "stm"
        allowed = (int, float, str, bool, torch.Tensor)
        config = {k: (v if isinstance(v, allowed) else str(v)) for k, v in self.args.model_dump().items()}
        self.accelerator.init_trackers(tracker_name, config=config)

    # --------------------------------------------------------------- training
    def train(self) -> None:
        logger.info("Starting training")
        logger.info(f"Memory before training start: {json.dumps(get_memory_statistics(), indent=4)}")
        self.state.total_batch_size_count = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        resume_path, initial_global_step, global_step, first_epoch = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint, num_update_steps_per_epoch=self.state.num_update_steps_per_epoch
        )
        if resume_path is not None:
            self.accelerator.load_state(resume_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )
        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")
            self.components.transformer.train()
            for step, batch in enumerate(self.data_loader):
                logs = {}
                with accelerator.accumulate([self.components.transformer]):
                    loss, loss_fg, loss_bg, loss_orig = self.compute_loss(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(self.components.transformer.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["loss_fg"] = float(loss_fg.detach().item())
                logs["loss_bg"] = float(loss_bg.detach().item())
                logs["loss_orig"] = loss_orig.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                if self.args.do_validation and accelerator.sync_gradients and global_step % self.args.validation_steps == 0:
                    del loss
                    free_memory()
                    self.validate(global_step)
                accelerator.log(logs, step=global_step)
                if global_step >= self.args.train_steps:
                    break
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(get_memory_statistics(), indent=4)}")
            if global_step >= self.args.train_steps:
                break

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)
        del self.components
        free_memory()
        logger.info(f"Memory after training end: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator.end_training()

    # ------------------------------------------------------------- validation
    def validate(self, step: int) -> None:
        logger.info("Starting validation...")
        if not self.state.validation_prompts:
            logger.warning("No validation samples found. Skipping validation.")
            return
        self.components.transformer.eval()
        torch.set_grad_enabled(False)
        logger.info(f"Memory before validation: {json.dumps(get_memory_statistics(), indent=2)}")

        pipe = self.initialize_pipeline()
        self._setup_pipeline_for_validation(pipe)
        self.accelerator.wait_for_everyone()
        n = len(self.state.validation_prompts)
        rank, world = self.accelerator.process_index, self.accelerator.num_processes
        for i in range(rank, n, world):  # shard samples across processes
            eval_data = self._prepare_data_for_step(i)
            for guidance_scale in self.args.validation_guidance_scales:
                logger.info(f"Validating sample {i + 1}/{n} (gs={guidance_scale}) on process {rank}.", main_process_only=False)
                artifacts = self.validation_step(eval_data, pipe, guidance_scale=guidance_scale)
                self._save_artifacts(step, i, eval_data, artifacts, guidance_scale=guidance_scale)
        self._cleanup_after_validation(pipe)
        logger.info(f"Memory after validation: {json.dumps(get_memory_statistics(), indent=2)}")
        torch.cuda.reset_peak_memory_stats(self.accelerator.device)

    def _setup_pipeline_for_validation(self, pipe):
        if self.state.using_deepspeed or self.accelerator.distributed_type == DistributedType.FSDP:
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            pipe.to(dtype=self.state.weight_dtype, device=self.accelerator.device)

    def _process_validation_media(self, media_path, media_type, transform):
        """Path(s) -> (normalised tensor, PIL frames, VAE latent)."""
        if media_path is None:
            return None, None, None
        is_video = media_type in ("video", "depth")
        paths = media_path if isinstance(media_path, list) else [media_path]
        tensors, pil_lists = [], []
        for path in paths:
            if media_type == "video":
                tensor = preprocess_video_with_resize(path, self.state.train_frames, self.state.train_height, self.state.train_width)
            elif media_type == "depth":
                tensor = preprocess_depth_with_resize(path, self.state.train_frames, self.state.train_height, self.state.train_width)
            else:
                tensor = preprocess_image_with_resize(path, self.state.train_height, self.state.train_width)
            u8 = tensor.round().clamp(0, 255).to(torch.uint8)
            pil = (
                [Image.fromarray(f.permute(1, 2, 0).cpu().numpy()) for f in u8]
                if is_video
                else Image.fromarray(u8.permute(1, 2, 0).cpu().numpy())
            )
            tensors.append(transform(tensor))
            pil_lists.append(pil)
        final = torch.stack(tensors) if len(tensors) > 1 else tensors[0]
        final_pil = pil_lists if len(pil_lists) > 1 else pil_lists[0]
        final = final.to(self.accelerator.device, dtype=self.state.weight_dtype)
        encoded = None
        if is_video:
            frames = (
                final.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous() if final.dim() == 4 else final.permute(0, 2, 1, 3, 4).contiguous()
            )
            encoded = self.encode_video(frames.to(self.accelerator.device, dtype=self.state.weight_dtype))
        return final, final_pil, encoded

    def _prepare_data_for_step(self, index: int) -> Dict[str, Any]:
        transform = transforms.Compose([lambda x: x / 255.0 * 2.0 - 1.0])
        eval_data: Dict[str, Any] = {"prompt": self.state.validation_prompts[index]}
        tt = self.args.i2v_training_type
        if tt == "original":
            _, eval_data["image"], _ = self._process_validation_media(self.state.validation_images[index], "image", transform)
            return eval_data
        eval_data["frames_fg"], eval_data["video_fg_list"], eval_data["encoded_video_fg"] = self._process_validation_media(
            self.state.validation_videos_fg[index], "video", transform
        )
        _, eval_data["video_bg_list"], eval_data["encoded_video_bg"] = self._process_validation_media(
            self.state.validation_videos_bg[index], "video", transform
        )
        if tt in ("fg_plain_bg_noisy_fgimg_noaug", "fg_noisy_bg_noisy_fgimg_noaug"):
            f = eval_data["frames_fg"].shape[0]
            first = einops.repeat(eval_data["frames_fg"][0], "C H W -> F C H W", F=f).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
            eval_data["encoded_fg_initial_repeated"] = self.encode_video(first.to(self.accelerator.device, dtype=self.state.weight_dtype))
        if tt == "fg_bg_depth_noisy":
            _, eval_data["video_depth_list"], eval_data["encoded_video_depth"] = self._process_validation_media(
                self.state.validation_videos_depth[index], "depth", transform
            )
        mask_path = self.state.validation_videos_mask[index] if self.state.validation_videos_mask else None
        if mask_path is not None:
            mask = preprocess_mask_with_resize(mask_path, self.state.train_frames, self.state.train_height, self.state.train_width)
            eval_data["mask"] = mask[:, 0:1]
        if tt == "replace_img_with_3fg_bg_noisy":
            num_fgs = eval_data["encoded_video_fg"].size(1)
            if 3 - num_fgs > 0:
                pad = torch.zeros(
                    (eval_data["encoded_video_fg"].shape[0], 3 - num_fgs, *eval_data["encoded_video_fg"].shape[2:]),
                    dtype=eval_data["encoded_video_fg"].dtype,
                    device=eval_data["encoded_video_fg"].device,
                )
                eval_data["encoded_video_fg"] = torch.cat([eval_data["encoded_video_fg"], pad], dim=1)
        return eval_data

    def _save_artifacts(self, step: int, sample_idx: int, eval_data: Dict, artifacts: List, guidance_scale: float) -> None:
        output_dir = self.args.output_dir / "validation_res" / f"{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt = eval_data["prompt"]
        prompt_filename = string_to_filename(prompt)[:25]
        hash_suffix = hashlib.md5(prompt[::-1].encode()).hexdigest()[:5]
        to_save = {"video_generated": artifacts[0][1]}
        if self.args.i2v_training_type == "replace_img_with_3fg_bg_noisy":
            to_save["video_bg_gt"] = eval_data.get("video_bg_list")
            for i, fg_list in enumerate(eval_data.get("video_fg_list", [])):
                to_save[f"video_fg_{i}_gt"] = fg_list
        elif self.args.i2v_training_type != "original":
            to_save["video_fg_gt"] = eval_data.get("video_fg_list")
            to_save["video_bg_gt"] = eval_data.get("video_bg_list")
        concat = []
        for key, value in to_save.items():
            if value is None:
                continue
            if isinstance(value, list):
                export_to_video(value, str(output_dir / f"{key}-{sample_idx}-{prompt_filename}-{hash_suffix}.mp4"), fps=self.args.gen_fps)
                concat.append((key, value))
            else:
                value.save(output_dir / f"{key}-{sample_idx}-{prompt_filename}-{hash_suffix}.png")
        if concat:
            export_to_video_artifacts(
                concat,
                str(output_dir / f"gs-{guidance_scale}-final-grid-{sample_idx}-{prompt_filename}-{hash_suffix}.mp4"),
                fps=self.args.gen_fps,
                prompt=prompt,
            )

    def _cleanup_after_validation(self, pipe):
        logger.info("Cleaning up after validation.")
        if self.state.using_deepspeed or self.accelerator.distributed_type == DistributedType.FSDP:
            del pipe
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
            cast_training_params([self.components.transformer], dtype=torch.float32)
        free_memory()
        self.accelerator.wait_for_everyone()
        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    # ------------------------------------------------------ subclass interface
    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:  # [B, C, F, H, W] -> [B, C', F', H', W']
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:  # -> [B, seq_len, dim]
        raise NotImplementedError

    def compute_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def validation_step(self, eval_data, pipe, guidance_scale: float) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    # --------------------------------------------------------------- private
    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        return _DTYPE_MAP[self.args.mixed_precision]

    def __move_components_to_device(self, dtype, ignore_list: List[str] = None):
        ignore = set(ignore_list or [])
        for name, component in self.components.model_dump().items():
            if not isinstance(component, type) and hasattr(component, "to") and name not in ignore:
                setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = None):
        unload = set(unload_list or [])
        for name, component in self.components.model_dump().items():
            if not isinstance(component, type) and hasattr(component, "to") and name in unload:
                setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                lora_layers = None
                for model in models:
                    if isinstance(unwrap_model(self.accelerator, model), type(unwrap_model(self.accelerator, self.components.transformer))):
                        lora_layers = get_peft_model_state_dict(unwrap_model(self.accelerator, model))
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")
                    if weights:
                        weights.pop()
                self.components.pipeline_cls.save_lora_weights(output_dir, transformer_lora_layers=lora_layers)

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                transformer_ = None
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(unwrap_model(self.accelerator, model), type(unwrap_model(self.accelerator, self.components.transformer))):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    self.args.model_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)
            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {k.replace("transformer.", ""): v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
            incompatible = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            unexpected = getattr(incompatible, "unexpected_keys", None) if incompatible is not None else None
            if unexpected:
                logger.warning(f"Loading adapter weights led to unexpected keys: {unexpected}.")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if must_save or global_step % self.args.checkpointing_steps == 0:
            save_path = get_intermediate_ckpt_path(
                checkpointing_limit=self.args.checkpointing_limit, step=global_step, output_dir=self.args.output_dir
            )
            self.accelerator.save_state(save_path, safe_serialization=False)
            self.accelerator.wait_for_everyone()


# backwards-compatible alias
Trainer_I2V = Trainer
