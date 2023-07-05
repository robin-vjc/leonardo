import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import yaml
from PIL import Image


THIS_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = THIS_PATH / ".."


def assign_random_model_to_images(image_files: List[str]) -> defaultdict[str, list]:
    config = get_config()
    models = config["models"]

    assignments = defaultdict(list)
    for file in image_files:
        assignments[random.choice(models)].append(file)

    return assignments


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
