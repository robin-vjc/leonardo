import os
import random
from pathlib import Path

import typer

from leonardo.processing import PROJECT_PATH, get_config, get_pipeline, load_image

app = typer.Typer()


@app.command()
def process_folder(
    input_path: Path,
    output_path: Path,
    prompt: str,
    output_width: int = 128,
    strength: float = 0.2,
    guidance_scale: float = 1.5,
    low_memory: bool = False,
):
    images_files = os.listdir(PROJECT_PATH / input_path)
    config = get_config()

    for file_name in images_files[:2]:
        # random pipeline
        model = random.choice(config["models"])
        pipe = get_pipeline(model, low_memory=low_memory)

        # load image
        image = load_image(image_path=PROJECT_PATH / input_path / file_name, width=output_width)

        # perform processing
        output_image = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale
        ).images[0]

        # store result
        output_image.save(PROJECT_PATH / output_path / file_name)


if __name__ == "__main__":
    app()
