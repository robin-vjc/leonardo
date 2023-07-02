# Leonardo

This repo contains code that processes images using Huggingface img2img pipelines. For each image, a random diffusion
model is selected and used to produce the output image

# Installation

Clone repo:
```bash
git clone ...
```

Update the `config.yaml` with the Huggingface token
```bash
cp config.template.yaml config.yaml
nano config.yaml
```


# Usage


# Design decision

- We do not attempt to parallelize the processing of images as the bottleneck (inference) is cpu/gpu bound.

# ToDos
- [x] upload repo to git
- [ ] process image (img_bytes, prompt, output_file)
- [ ] store images in data/
- [ ] make repo pip-installable
- [ ] create a cli that you supply input folder, output folder
- [ ] clean up docstrings everywhere
- [ ] update README installation / usage
