import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["i2v", "t2v"]
    training_type: Literal["lora", "sft"] = "lora"
    train_data_type: Literal["original", "with-fg-bg"] = "original"
    filtering_method: str | None = None
    pos_embed_is_4d: bool = False
    choose_dataset: str = "original"
    machine_type: str = "gcp"
    aug_config_path: str = None
    load_depth: bool = False
    prompt_drop_ratio: float = 0.0

    noise_mean_bg: float = -3.0
    noise_mean_fg: float = -1.0

    weight_fg: float = 0.0
    weight_bg: float = 0.0
    weight_orig: float = 1.0
    apply_fg_noise: bool = False
    apply_bg_noise: bool = False

    apply_mask_on_fg_bg_loss: bool = False
    apply_mask_on_latents_fg: bool = False
    apply_mask_on_latents_bg: bool = False
    apply_adaptive_mask_on_orig_loss: bool = False

    apply_mask_on_orig_loss: bool = False
    fg_weight_on_orig_loss: float = 0.0

    fg_column: str = "fg.txt"
    bg_column: str = "bg.txt"

    #
    i2v_training_type: str = "original"

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "finetrainer-cogvideo"

    ########## Data ###########
    data_root: Path
    caption_column: Path
    image_column: Path | None = None
    video_column: Path

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)

    #### deprecated args: video_resolution_buckets
    # if use bucket for training, should not be None
    # Note1: At least one frame rate in the bucket must be less than or equal to the frame rate of any video in the dataset
    # Note2:  For cogvideox, cogvideox1.5
    #   The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #   The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # video_resolution_buckets: List[Tuple[int, int, int]] | None = None

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None  # if set do_validation, should not be None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15
    inference_steps: int = 50  # denoising steps used for validation sampling
    validation_guidance_scales: List[float] = [6.0]
    validation_fg_column: str | None = None  # fg list inside validation_dir (default: fg-black.txt if present else fg.txt)
    validation_num_samples: int | None = None  # only validate on the first N samples

    #### deprecated args: gen_video_resolution
    # 1. If set do_validation, should not be None
    # 2. Suggest selecting the bucket from `video_resolution_buckets` that is closest to the resolution you have chosen for fine-tuning
    #        or the resolution recommended by the model
    # 3. Note:  For cogvideox, cogvideox1.5
    #        The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #        The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # gen_video_resolution: Tuple[int, int, int] | None  # shape: (frames, height, width)

    @field_validator("image_column")
    def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("model_type") == "i2v" and not v:
            logging.warning(
                "No `image_column` specified for i2v model. Will automatically extract first frames from videos as conditioning images."
            )
        return v

    @field_validator("validation_dir", "validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "i2v" and not v:
            raise ValueError("validation_images must be specified when do_validation is True and model_type is i2v")
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError("validation_videos must be specified when do_validation is True and model_type is v2v")
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> str:
        try:
            frames, height, width = v

            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError("For cogvideox-5b models, height must be 480 and width must be 720")

            return v

        except ValueError as e:
            if str(e) == "not enough values to unpack (expected 3, got 0)" or str(e) == "invalid literal for int() with base 10":
                raise ValueError("train_resolution must be in format 'frames x height x width'")
            raise e

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()

        # RECOMPOSER arguments
        parser.add_argument("--train_data_type", type=str, default="original")
        parser.add_argument("--filtering_method", type=str, default=None)
        parser.add_argument("--pos_embed_is_4d", type=lambda x: str(x).lower() == "true", default=True)
        parser.add_argument("--choose_dataset", type=str, default="original")
        parser.add_argument("--machine_type", type=str, default="gcp")
        parser.add_argument("--aug_config_path", type=str, default=None)
        parser.add_argument("--load_depth", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--prompt_drop_ratio", type=float, default=0.0)

        parser.add_argument("--fg_column", type=str, default="fg.txt")
        parser.add_argument("--bg_column", type=str, default="bg.txt")

        parser.add_argument("--noise_mean_bg", type=float, default=-3.0)
        parser.add_argument("--noise_mean_fg", type=float, default=-1.0)

        parser.add_argument("--apply_mask_on_fg_bg_loss", type=lambda x: str(x).lower() == "true", default=False)

        parser.add_argument("--apply_mask_on_latents_fg", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--apply_mask_on_latents_bg", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--apply_mask_on_orig_loss", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--apply_adaptive_mask_on_orig_loss", type=lambda x: str(x).lower() == "true", default=False)

        parser.add_argument("--fg_weight_on_orig_loss", type=float, default=0.0)

        parser.add_argument("--apply_fg_noise", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--apply_bg_noise", type=lambda x: str(x).lower() == "true", default=False)

        parser.add_argument("--weight_fg", type=float, default=0.0)
        parser.add_argument("--weight_bg", type=float, default=0.0)
        parser.add_argument("--weight_orig", type=float, default=1.0)

        # I2V arguments
        parser.add_argument("--i2v_training_type", type=str, default="original")

        # Required arguments
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--data_root", type=str, required=True)
        parser.add_argument("--caption_column", type=str, required=True)
        parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--report_to", type=str, required=True)

        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)

        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=100)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)

        # Data loading
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--image_column", type=str, default=None)

        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=bool, default=True)
        parser.add_argument("--enable_slicing", type=bool, default=True)
        parser.add_argument("--enable_tiling", type=bool, default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)

        # LoRA parameters
        parser.add_argument("--rank", type=int, default=128)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument("--target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == "true", default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)
        parser.add_argument("--inference_steps", type=int, default=50)
        parser.add_argument("--validation_guidance_scales", type=float, nargs="+", default=[6.0])
        parser.add_argument("--validation_fg_column", type=str, default=None)
        parser.add_argument("--validation_num_samples", type=int, default=None)

        args = parser.parse_args()

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))
