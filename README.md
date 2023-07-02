# Leonardo

This repo contains code that processes images using Huggingface img2img pipelines. For each image, a random diffusion
model is selected and used to produce the output image

# Usage

Clone repo:
```bash
git clone git@github.com:robin-vjc/leonardo.git
```

Update `config.yaml` with the Huggingface token
```bash
cd leonardo
cp config.template.yaml config.yaml
nano config.yaml
```

Run the pipeline on the set of images in `data/images`
```bash
# view options
python leonardo/cli.py --help

# run image processing
python leonardo/cli.py 
```

# Design decisions

- We do not attempt to parallelize the processing of images as the bottleneck (inference) is cpu/gpu bound.

# ToDos
- [x] upload repo to git
- [x] process images folder pipeline (input_folder, prompt, output_folder)
- [x] store images in data/
- [ ] make repo pip-installable
- [ ] clean up docstrings everywhere
- [ ] update README installation / usage
