# ⚙️ Installation

> We recommend using conda to manage dependencies, which have been tested on NVIDIA RTX-3090/A5000/A6000 with CUDA 11.8, Python 3.10, under Ubuntu 22.04/24.04.

### Create environment

```bash
conda create --name neuralplane -y python=3.10
conda activate neuralplane
pip install --upgrade pip
```

#### Dependencies

a. Install PyTorch with CUDA and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Troubleshooting:
- **tiny-cuda-nn**: [Unable to install pytorch bindings - lcuda not found](https://github.com/NVlabs/tiny-cuda-nn/issues/183#issuecomment-1405541078).

b. Installing thirdparties

```bash
# install thirdparties from local clones

# detectron2
git clone https://github.com/facebookresearch/detectron2.git thirdparty
python -m pip install -e thirdparty/detectron2

# segment_anything
git clone https://github.com/facebookresearch/segment-anything.git thirdparty
pip install -e thirdparty/segment-anything

# hloc
git clone --recursive https://github.com/cvg/Hierarchical-Localization/ thirdparty
git -C thirdparty/Hierarchical-Localization checkout e3342201
python -m pip install -e thirdparty/Hierarchical-Localization
```

c. Installing **NeuralPlane**

```bash
# Under the repo root
pip install --upgrade pip setuptools

# Install this repo as well as dependencies such as pycolmap, Nerfstudio...
pip install -e .

# Test. It is all set if you see a tyro-based command-line interface
np-train --help
```
Troubleshooting:
- **open3d**: [no matching distribution found for open3d>=0.16.0.](https://github.com/nerfstudio-project/nerfstudio/issues/2046#issuecomment-1638333414) The version we use is `0.19.0`.


### Download Pretrain Models


```bash
np-download --dst ./checkpoints/
```