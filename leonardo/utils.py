import random
from collections import defaultdict
from typing import List

from leonardo.processing import get_config


def assign_random_model_to_images(image_files: List[str]) -> defaultdict[str, list]:
    config = get_config()
    models = config["models"]

    assignments = defaultdict(list)
    for file in image_files:
        assignments[random.choice(models)].append(file)

    return assignments
