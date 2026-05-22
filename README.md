# FlashAvatar
**[Paper](https://arxiv.org/abs/2312.02214)|[Project Page](https://ustc3dv.github.io/FlashAvatar/)**

![teaser](exhibition/teaser.png)
Given a monocular video sequence, our proposed FlashAvatar can reconstruct a high-fidelity digital avatar in minutes which can be animated and rendered over 300FPS at the resolution of 512×512 with an Nvidia RTX 3090.

## Setup

This code has been tested on Nvidia RTX 3090. 

Create the environment:

```
conda env create --file environment.yml
conda activate FlashAvatar
```

Install PyTorch3D:

```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

### RunPod (pip, Python 3.11, Torch 2.4 / CUDA 12.4)

For RunPod GPU instances, use a pip-only workflow (no conda required):

```bash
git submodule update --init --recursive
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install --no-build-isolation -r requirements-runpod.txt
python -m pip install --no-build-isolation -e submodules/simple-knn
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
python scripts/verify_cuda_extensions.py
```

Equivalent one-shot setup:

```bash
bash scripts/setup_runpod_env.sh
```

Notes:
- `environment.yml` remains available for the legacy conda flow.
- If extension build fails, inspect the first `nvcc` error and verify torch CUDA build and host driver compatibility.
## Data Convention
The data is organized in the following form：
```
dataset
├── <id1_name>
    ├── alpha # raw alpha prediction
    ├── imgs # extracted video frames
    ├── parsing # semantic segmentation
├── <id2_name>
...
metrical-tracker
├── output
    ├── <id1_name>
        ├── checkpoint
    ├── <id2_name>
...
```
## Running
- **Evaluating pre-trained model**
```shell
python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
```
-  **Training on your own data** 
```shell
python train.py --idname <id_name>
```
Download the [example](https://drive.google.com/file/d/1_WLvlmHD73jOAO178N7eX5UQqlrL2ghD/view?usp=drive_link) with pre-processed data and pre-trained model for a try!

## Citation
```
@inproceedings{xiang2024flashavatar,
      author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
      title     = {FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2024},
  }
```
