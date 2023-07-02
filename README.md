# Leonardo

This repo contains code that processes images using Huggingface img2img pipelines. For each image, a random diffusion
model is selected and used to produce the output image.

* Sample input images are in [data/images](data%2Fimages); output images generated with the code in this repo are in [data/output](data%2Foutput). The Google Colab notebook used to generate them [is here](https://colab.research.google.com/drive/1AZh9PQb7DV9tkoYzL9d3Sl7hJ3CSTscG#scrollTo=Z-NOrKeJcS9m).
* The above Google Colab has sufficient memory to load all models at the same time. In environments with lower GPU memory, pipelines can be fetched with `get_pipeline(model, low_memory=True)`, and the model will be moved to GPU only during inference. Inference speed is ~halved in this mode.
* The CLI interface described below can be run locally, and it will use a GPU if available. Running with `--output-width=128` will allow the pipeline to run even on CPU in reasonable time (~10 secs per image), but results will not look great.

Further notes:
* No dockerization
* In a production setting, we would not be loading all models in a single python process; rather, we'd spawn separate containers, one with only one model each. Processing images would send a request to a separate container each time. Containers would be scaled depending on load.
* We do not attempt to parallelize the processing of images as the bottleneck (inference) is cpu/gpu bound.

# Usage

Clone repo and install app as the editable (`-e`) `leonardo` package:
```bash
git clone git@github.com:robin-vjc/leonardo.git
cd leonardo
pip install -e .
```

Update `config.yaml` with the Huggingface token
```bash
cp config.template.yaml config.yaml
nano config.yaml
```

Run the pipeline on the set of images in `data/images/`
```bash
# view options
python leonardo/cli.py --help

# run image processing
python leonardo\cli.py data\images\ data\output\ "Futuristic style, trending on artstation" --output-width=128 --strength=0.2 --guidance-scale=1.5
```


# ToDos
- [x] upload repo to git
- [x] process images folder pipeline (input_folder, prompt, output_folder)
- [x] store images in data/
- [x] make repo pip-installable
- [x] check on colab GPU processing works correctly
- [ ] track memory/cpu usage
- [ ] clean up docstrings everywhere
- [ ] update README installation / usage
