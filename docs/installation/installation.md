## Installation

### Manually installation

First, install some packages from their official site manually, mainly some packages related to cuda, and you have to choose the cuda version to use. 

#### Pytorch

Install [pytorch](https://pytorch.org/get-started/locally/) from their official site manually.

#### Detectron2

Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) from their official site manually. We highly recommand you to choose the `build from source` method. Because, they do not update the pre-build version for a long time, and the old version of detectron2 only support pytorch with a very old version, so, you may need to rechoose the version of pytorch, cudatoolkit and sometimes even python.

First, check whether you have `ninja` installed already, if you do not have `ninja` in your environment, and you are using `conda`, use follow command to install it:

```bash
conda install ninja
```

Use pip install detectron2 from github directly. Just use the follow command to build detectron2 from source on github:

```bash
TORCH_CUDA_ARCH_LIST="<cuda archs to support>" pip install 'git+https://github.com/facebookresearch/detectron2.git[@<branch/tag name or commit id>]'
```

If you only want to support gpus on your current machine, you can omit the `TORCH_CUDA_ARCH_LIST` env var, and detectron2 will infer it automatically.

If you want to install detectron2 with the latest version, you can omit the `@<part>` after the detectron2 github url.

### Automaticaly installation

Generally, you can just use the latest pacages in `requirements.txt` without specific their version, so you can install this project by two steps.

- Install required packages with `pip install -r requirements.txt`
- Install this project with `pip install -e .`
