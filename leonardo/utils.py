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
    """
    Calculates a randomized association of an image with an img2img processing model.
    :param image_files: a list of image files to be associated with a random model
    :return: assignments
    """
    config = get_config()
    models = config["models"]

    assignments = defaultdict(list)
    for file in image_files:
        assignments[random.choice(models)].append(file)

    return assignments


def get_config() -> Dict:
    """
    Utility to retrieve the project's configuration stored in config.yaml
    :return: configuration
    """
    with open(THIS_PATH / ".." / "config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def load_image(image_path: Path, width: int = 512) -> Image:
    """
    Utility to load an image and resize it to a given dimension.
    :param image_path: path to the image file
    :param width: desired image width, in pixels
    :return: the loaded image
    """
    image = Image.open(image_path).convert("RGB")

    init_width, init_height = image.size
    scaling_factor = init_width // width
    height = init_height // scaling_factor

    return image.resize((width, height))
