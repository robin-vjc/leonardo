import gc
import os
from pathlib import Path

import torch
import typer

from leonardo.processing import Img2ImgModel, CsvFileLogger
from leonardo.utils import assign_random_model_to_images, load_image, PROJECT_PATH

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
    """
    Run a pipeline that processes images in a given folder using a random img2img stable diffusion model.
    :param input_path: folder containing the input images to be processed
    :param output_path: folder where the processed images will be stored
    :param prompt: prompt to use to perform img2img processing
    :param output_width: output image width in pixels; the larger, the longer processing will take
    :param strength: strength parameter for the Huggingface img2img inference pipeline
    :param guidance_scale: guidance parameter for the Huggingface img2img inference pipeline
    :param low_memory: if set to true, will move models out of the GPU upon processing completion; saves GPU memory but results in a ~50% drop in inference speed
    :return: None
    """
    images_files = os.listdir(PROJECT_PATH / input_path)
    model_assignments = assign_random_model_to_images(images_files)

    for model_path, images_files in model_assignments.items():
        pipe = Img2ImgModel(model_path, low_memory=low_memory)
        file_logger = CsvFileLogger(output_file=PROJECT_PATH / "data" / "results.csv")
        pipe.register(file_logger)

        for file_name in images_files[:2]:
            # load image
            image = load_image(image_path=PROJECT_PATH / input_path / file_name, width=output_width)

            # perform processing
            with torch.no_grad():
                output_image = pipe(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale
                ).images[0]

            # store result
            output_image.save(PROJECT_PATH / output_path / file_name)

        # clear GPU cache after finishing with a model
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
