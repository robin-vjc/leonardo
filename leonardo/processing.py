import csv
import datetime
import time
from typing import Dict

import psutil
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# we persist pipes for the duration of the application as loading them is expensive
LOADED_PIPELINES = {}


def get_pipeline(model_path: str, low_memory: bool = False) -> StableDiffusionImg2ImgPipeline:
    """
    Retrieves a HuggingFace to perform img2img processing. Keeps models in RAM once loaded, as the loading operation is expensive.
    :param model_path: path to the model; Example: "stabilityai/stable-diffusion-2-1"
    :param low_memory: set to true to reduce memory consumption, at the cost of decreasing inference speed by 50%
    """
    if model_path in LOADED_PIPELINES:
        return LOADED_PIPELINES[model_path]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else None

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,  # sometimes erroneously detects NSFW content when working in low resolution
        requires_safety_checker=False,
    )

    if low_memory:
        pipe.enable_sequential_cpu_offload()

    pipe = pipe.to(device)
    LOADED_PIPELINES[model_path] = pipe

    return pipe


class Img2ImgModel:
    """
    Wrapper for HuggingFace pipelines that is observable. Observability facilitates things like performance logging.
    """
    def __init__(self, model_path: str, low_memory: bool = False):
        self.model_path = model_path
        self.pipe = get_pipeline(model_path=model_path, low_memory=low_memory)
        self._observers = []

    def __call__(self, *args, **kwargs):
        self._notify_observers({
            'name': 'pipeline called',
            'args': args,
            'kwargs': kwargs
        })

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        result = self.pipe(*args, **kwargs)
        end_time = time.time()
        gb_factor = 1_000_000_000

        self._notify_observers({
            'name': 'pipeline finished',
            'prompt': kwargs['prompt'],
            'strength': kwargs['strength'],
            'guidance_scale': kwargs['guidance_scale'],
            'model_path': self.model_path,
            'date': datetime.datetime.now(),
            'duration_in_sec': f"{end_time - start_time:.2f}",
            'gpu_memory_allocated_in_gb': f"{torch.cuda.memory_allocated()/gb_factor:.2f}" if torch.cuda.is_available() else None,
            'gpu_max_memory_allocated_in_gb': f"{torch.cuda.max_memory_allocated()/gb_factor:.2f}" if torch.cuda.is_available() else None,
            'ram_usage_in_gb': f"{psutil.virtual_memory()[3]/gb_factor:.2f}"
        })

        return result

    def register(self, observer):
        self._observers.append(observer)

    def _notify_observers(self, event: Dict):
        for observer in self._observers:
            observer.update(event)


class CsvFileLogger:
    """
    An observer that stores run information in a csv file upon pipeline completion
    """
    def __init__(self, output_file: str):
        self.output_file = output_file

    def update(self, event: Dict):
        if event['name'] == 'pipeline finished':
            with open(self.output_file, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(event.values())
