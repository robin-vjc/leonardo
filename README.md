# Leonardo

<p align="center">
  <strong>
    Recreate the image in "Futuristic style, trending on artstation"
    <br>
    <a href="data/images">Input Images</a> | <a href="data/output">Output Images</a> 
  </strong>
</p>

|                                                                                         Souce Image                                                                                          |                                                                                                                                                                 Output Image                                                                                                                                                                  |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/robin-vjc/leonardo/blob/master/data/images/Default_A_side_view_of_a_samurai_cat_with_a_flat_design_Artwork_of_t_0_cc6b0aa7-de9c-4d6e-8d3a-3aebd827b1cd_1.jpg" width="512"> | <img src="https://github.com/robin-vjc/leonardo/blob/master/data/output/Default_A_side_view_of_a_samurai_cat_with_a_flat_design_Artwork_of_t_0_cc6b0aa7-de9c-4d6e-8d3a-3aebd827b1cd_1.jpg" width="512"> |


# Introduction

This repo contains code that processes images using Huggingface img2img pipelines. For each image, a random diffusion
model is selected and used to produce the output image.

* The Google Colab notebook used to generate the images in [data/output](data%2Foutput) can be [found here](https://colab.research.google.com/drive/1AZh9PQb7DV9tkoYzL9d3Sl7hJ3CSTscG#scrollTo=Z-NOrKeJcS9m).
* The above Google Colab has sufficient memory to load all models at the same time. In environments with lower GPU memory, pipelines can be fetched with `get_pipeline(model, low_memory=True)`, and the model will be moved to GPU only during inference. Inference speed is ~halved in this mode.
* The CLI interface described below can be run locally, and it will use a GPU if available. Running with `--output-width=128` will allow the pipeline to run even on CPU in reasonable time (~10 secs per image), but results will not look great.

**Further notes:**
* The `setup.py`/`requirements.txt` files are the only infrastructure pieces (no dockerization) 
* In a production setting, we would not be loading all models in a single python process; rather, we'd spawn separate containers, one with only one model each, sitting behind a REST API (e.g. Flask or MLFlow serve). Processing images would send a request to a separate endpoint each time. Containers would be scaled depending on load.
* It is possible to perform batch processing by supplying a list of images and prompts to `pipe()` instead of individual ones. However, the input images would have to be of the same dimensions for batching to be possible. Since the aspect ratio of the input images provided varies, we wouldn't be able to batch much work in this case. It is something one would want to do in production to ensure GPU power is effectively utilized.

# Usage

Clone repo and install app as the editable (`-e`) `leonardo` package:
```bash
git clone git@github.com:robin-vjc/leonardo.git
cd leonardo
pip install -e .
```

To run the pipeline on the set of images in `data/images/`
```bash
# view options
>> python leonardo/cli.py --help
Usage: cli.py [OPTIONS] INPUT_PATH OUTPUT_PATH PROMPT

Arguments:
  INPUT_PATH   [required]
  OUTPUT_PATH  [required]
  PROMPT       [required]

Options:
  --output-width INTEGER          [default: 128]
  --strength FLOAT                [default: 0.2]
  --guidance-scale FLOAT          [default: 1.5]
  --low-memory / --no-low-memory  [default: no-low-memory]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

# run image processing
>> python leonardo\cli.py data\images\ data\output\ "Futuristic style, trending on artstation" --output-width=128 --strength=0.2 --guidance-scale=1.5
```


# ToDos
- [x] upload repo to git
- [x] process images folder pipeline (input_folder, prompt, output_folder)
- [x] store images in data/
- [x] make repo pip-installable
- [x] check on colab GPU processing works correctly
- [ ] only clear gpu if device is in fact gpu
- [ ] track memory/cpu usage
- [x] refactor so we do a randomized chunk of images with one model, then process the other chunks
- [ ] clean up docstrings everywhere
- [x] update README installation / usage
