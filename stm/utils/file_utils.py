import os
import shutil
from pathlib import Path
from typing import List, Union

import mediapy
import numpy as np
from accelerate.logging import get_logger
from PIL import Image

from stm.constants import LOG_LEVEL, LOG_NAME

logger = get_logger(LOG_NAME, LOG_LEVEL)


def export_to_video_artifacts(artifact_list, output_path, fps=16, prompt=None):
    """
    Creates a concatenated video, saves individual video artifacts to a sub-folder,
    and saves the prompt to a text file in the same sub-folder.

    Args:
        artifact_list (list): A list of tuples, (str_key, list_of_pil_images).
        output_path (str): The path for the output concatenated .mp4 video.
        fps (int): Frames per second for all videos.
        prompt (str, optional): If provided, this text is saved to a 'prompt.txt' file.
                                Defaults to None.
    """
    if not artifact_list or not any(images for _, images in artifact_list):
        return

    # --- 1. Set up paths ---
    output_path_obj = Path(output_path)
    base_dir = output_path_obj.parent
    video_stem_name = output_path_obj.stem
    individual_parts_dir = base_dir / video_stem_name
    individual_parts_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Save the prompt to a text file if it exists ---
    if prompt is not None:
        prompt_file_path = individual_parts_dir / "prompt.txt"
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    # --- 3. Save individual video parts ---
    for key, image_list in artifact_list:
        if image_list:
            safe_key = key.replace(" ", "_").replace("/", "-")
            individual_video_path = individual_parts_dir / f"{safe_key}.mp4"
            frames_for_individual_video = [np.array(img) for img in image_list]
            mediapy.write_video(str(individual_video_path), frames_for_individual_video, fps=fps)

    # --- 4. Proceed with creating the concatenated video ---
    num_frames = max(len(images) for _, images in artifact_list if images)
    if num_frames == 0:
        return

    # --- Frame Dimension Calculation ---
    video_h = max(img.height for _, images in artifact_list if images for img in images)
    max_w = 0
    for i in range(num_frames):
        current_frame_width = sum(img_list[i].width for _, img_list in artifact_list if i < len(img_list))
        if current_frame_width > max_w:
            max_w = current_frame_width
    video_w = max_w

    if video_w == 0 or video_h == 0:
        return

    # --- Frame Generation (Text drawing logic removed) ---
    video_frames = []
    for i in range(num_frames):
        concatenated_image = Image.new("RGB", (video_w, video_h), "black")
        current_x = 0

        for key, image_list in artifact_list:
            if i < len(image_list):
                img = image_list[i]
                concatenated_image.paste(img, (current_x, 0))
                current_x += img.width

        video_frames.append(np.array(concatenated_image))

    # --- Write the final concatenated video ---
    mediapy.write_video(output_path, video_frames, fps=fps)


def find_files(dir: Union[str, Path], prefix: str = "checkpoint") -> List[str]:
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        return []
    checkpoints = os.listdir(dir.as_posix())
    checkpoints = [c for c in checkpoints if c.startswith(prefix)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    checkpoints = [dir / c for c in checkpoints]
    return checkpoints


def delete_files(dirs: Union[str, List[str], Path, List[Path]]) -> None:
    if not isinstance(dirs, list):
        dirs = [dirs]
    dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
    logger.info(f"Deleting files: {dirs}")
    for dir in dirs:
        if not dir.exists():
            continue
        shutil.rmtree(dir, ignore_errors=True)


def string_to_filename(s: str) -> str:
    return (
        s.replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(",", "-")
        .replace(";", "-")
        .replace("!", "-")
        .replace("?", "-")
    )
