import os
from pathlib import Path
from typing import Dict

import torch
import yaml
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


LOADED_PIPELINES = {}  # we persist pipes for the duration of the application as loading them is expensive
THIS_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = THIS_PATH / ".."


def get_pipeline(model_path: str, low_memory: bool = False) -> StableDiffusionImg2ImgPipeline:
    if model_path in LOADED_PIPELINES:
        return LOADED_PIPELINES[model_path]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else None
    config = get_config()
    hf_token = config["hf_token"]
    assert hf_token != "<INSERT_TOKEN_HERE>", "Edit the config.yaml file `hf_token` with a valid Huggingface token"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        # torch_dtype=dtype,
        use_auth_token=hf_token,
        safety_checker=None,  # sometimes erroneously detects NSFW content when working in low resolution
        requires_safety_checker=False
    )

    if low_memory and device == "cuda":
        pipe.enable_sequential_cpu_offload()

    pipe = pipe.to(device)
    LOADED_PIPELINES[model_path] = pipe

    return pipe


def get_config() -> Dict:
    with open(THIS_PATH / ".." / "config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def load_image(image_path: Path, width: int = 512) -> Image:
    image = Image.open(image_path).convert("RGB")

    init_width, init_height = image.size
    scaling_factor = init_width // width
    height = init_height // scaling_factor

    return image.resize((width, height))
