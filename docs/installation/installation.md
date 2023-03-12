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
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

##### CUDA_ARCH

By default, detectron2 will infer `CUDA_ARCH` automatically according to gpus on your current machine. But, if you are building detectron2 on no-gpu node (like build on SLURM control node) or want to support gpus not on your current machine, you will need to set the `TORCH_CUDA_ARCH_LIST` env var manually, then, use command as follow:

```bash
TORCH_CUDA_ARCH_LIST="<cuda archs to support>" pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For the `CUDA_ARCH` of gpus, check [cuda arch doc](https://developer.nvidia.com/cuda-gpus) for details.

##### Specific version

Sometimes, you will need to select the version of detectron2, then just add the `branch name` or `tag name` or `ref id` after the github url. Use command as follow:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git[@<branch/tag name or commit id>]'
```

### Automaticaly installation

Generally, you can just use the latest pacages in `requirements.txt` without specific their version, so you can use command as follow to install this project and all required packages.

```bash
pip install -r requirements.txt
pip install -e .
```
