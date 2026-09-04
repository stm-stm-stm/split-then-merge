#!/usr/bin/env python
"""Generate compositions from a trained StM Composer checkpoint (FSDP/accelerate state).

    python -m accelerate.commands.launch --config_file training/configs/accelerate/single_gpu.yaml \
        inference/compose.py \
        --checkpoint_path outputs/stm_main/0001/checkpoint-20000 \
        --validation_dir data/datasets/stm_gen \
        --output_dir outputs/validation --guidance_scales 2 4 6 8

The experiment directory (parent of the checkpoint) must contain the
``args.json`` written at training time; ``--validation_dir`` overrides the
validation set (a folder with ``prompts.txt``, ``fg-black.txt``/``fg.txt``,
``bg.txt`` and optionally ``masks.txt``).  Outputs mirror the released result
folders: ``sample-<i>_gs-<gs>_<prompt>-<hash>/{video_generated,video_fg_gt,video_bg_gt}.mp4``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stm.models import StMComposerTrainer  # noqa: E402
from stm.schemas import Args  # noqa: E402
from stm.utils import string_to_filename, unwrap_model  # noqa: E402
from stm.utils.file_utils import export_to_video_artifacts  # noqa: E402


def main(cli: argparse.Namespace):
    checkpoint_path = Path(cli.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    args_path = checkpoint_path.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found next to the checkpoint: {args_path}")
    train_args = Args(**json.loads(args_path.read_text()))
    train_args.resume_from_checkpoint = None
    train_args.output_dir = Path(cli.output_dir) / "_runs"  # the trainer creates a numbered run dir here
    if cli.validation_dir:
        train_args.validation_dir = Path(cli.validation_dir)
    if cli.num_inference_steps:
        train_args.inference_steps = cli.num_inference_steps
    if cli.num_samples:
        train_args.validation_num_samples = cli.num_samples

    validator = StMComposerTrainer(train_args)
    accelerator = validator.accelerator
    device = accelerator.device
    validator.prepare_models()

    seed = cli.seed if cli.seed is not None else train_args.seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    components = validator.components
    if train_args.training_type == "lora":
        # adds the LoRA adapter and registers the save/load hooks that read pytorch_lora_weights.safetensors
        validator.prepare_trainable_parameters()
    dummy_optimizer = torch.optim.AdamW(
        [p for p in components.transformer.parameters() if p.requires_grad] or list(components.transformer.parameters()), lr=1e-4
    )
    components.transformer, dummy_optimizer = accelerator.prepare(components.transformer, dummy_optimizer)
    accelerator.load_state(str(checkpoint_path))
    print(f"Loaded model state from {checkpoint_path}")

    pipe = validator.initialize_pipeline()
    weight_dtype = torch.bfloat16 if train_args.mixed_precision == "bf16" else torch.float16
    pipe.to(device, dtype=weight_dtype)
    components.vae.to(device, dtype=weight_dtype)
    components.text_encoder.to(device, dtype=weight_dtype)
    unwrap_model(accelerator, components.transformer).eval()

    validator.prepare_for_validation()
    prompts = validator.state.validation_prompts
    if not prompts:
        print("No validation prompts found.")
        return
    output_dir = Path(cli.output_dir) / f"validation_{checkpoint_path.name}"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    validator.state.generator = generator

    with torch.no_grad():
        for i in tqdm(range(len(prompts)), desc="Generating", disable=not accelerator.is_main_process):
            eval_data = validator._prepare_data_for_step(i)
            prompt = eval_data["prompt"]
            name = f"{string_to_filename(prompt)[:30]}-{hashlib.md5(prompt[::-1].encode()).hexdigest()[:5]}"
            for gs in cli.guidance_scales:
                final_path = output_dir / f"sample-{i}_gs-{gs}_{name}.mp4"
                skip = torch.tensor([int(final_path.exists())], device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(skip, src=0)
                if skip.item():
                    continue
                artifacts = validator.validation_step(eval_data, pipe, guidance_scale=gs)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    to_save: Dict[str, List[Image.Image]] = {"video_generated": artifacts[0][1]}
                    if train_args.i2v_training_type != "original":
                        to_save["video_fg_gt"] = eval_data.get("video_fg_list")
                        to_save["video_bg_gt"] = eval_data.get("video_bg_list")
                    export_to_video_artifacts(
                        [(k, v) for k, v in to_save.items() if v is not None], str(final_path), fps=train_args.gen_fps, prompt=prompt
                    )
                accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--validation_dir", default=None)
    p.add_argument("--output_dir", default="outputs/validation")
    p.add_argument("--guidance_scales", nargs="+", type=float, default=[2.0, 4.0, 6.0, 8.0])
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None, help="only the first N validation samples")
    main(p.parse_args())
