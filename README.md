# Video-3DGS: Leveraging 3D Gaussian Splatting for Reconstructing and Editing Dynamic Monocular Videos
## [Project page](https://video-3dgs-project.github.io/) | [Paper]()
<img src='asset/teaser.png' width='600'>

This repository contains the official Pytorch implementation of the paper "Video-3DGS: Leveraging 3D Gaussian Splatting for Reconstructing and Editing Dynamic Monocular Videos".



## News

- 24/02/xx:  paper/website/code are released.

## TODO

- [X] Training and Inference code updated
- [X] MC-COLMAP pre-processed dataset uploaded
- [ ] Code for MC-COLMAP uploaded

## Dataset

According to our paper, we conducted two tasks with the following datasets.

- Video reconstruction: [DAVIS](https://davischallenge.org/davis2017/code.html) dataset (480x854)
- Video editing: [LOVEU-TGVE-2023](https://github.com/showlab/loveu-tgve-2023?tab=readme-ov-file) dataset (480x480)

There are two options for pre-processing the datasets.
1. You can download the datasets with the provided link and run MC-COLMAP (Code for MC-COLMAP will be updated later)
2. You directly download MC-COLMAP processed dataset from [here](https://drive.google.com/drive/folders/1uYmLWUn5veBlUES88-9NKgNibiHxhe_F) 

We organize the datasets as follows:

```shell
├── data
│   | recon
│     ├── DAVIS
│       ├── JPEGImages 
│         ├── 480p
│           ├── blackswan
│           ├── blackswan_pts_camera_from_deva
│           ├── ...
│   | edit


```

## Pipeline

<img src='asset/pipeline_v2.png' width='900'>


## Environments
Setting up environments for training contains three parts:

1.  Download [COLMAP](https://github.com/colmap/colmap) (dev version to minimize the randomness issue) and put it under "submodules". (change the folder name to "colmap_dev")
2.  3DGS related packages
3.  Download [Tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and put it under "submodules"

```shell
git clone https://github.com/dlsrbgg33/Video-3DGS.git --recursive
cd Video-3DGS

conda create -n video_3dgs python=3.8
conda activate video_3dgs

# install pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# install packages & dependencies
bash requirement.sh
```

Setting up environments for evaluation contains two parts:

1. download the pre-trained optical flow models (WarpSSIM)

```shells
cd models/optical_flow/RAFT
bash download_models.sh
unzip models.zip
```

2. download CLIP pre-trained models (CLIPScore, Qedit)

```shells
cd models/clipscore
git lfs install
git clone https://huggingface.co/openai/clip-vit-large-patch14
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

## Video reconstruction

```shell
bash sh_recon/davis.sh
```

Arguments:
  - iteration num
  - group size
  - number of random points

## Video editing
```shell
bash sh_edit/{initial_editor}/{dataset}.sh
```
We currently support three "initial editors": [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) / [TokenFlow](https://github.com/omerbt/TokenFlow) / [RAVE](https://rave-video.github.io/)

We recommend user to install related packages and modules of above initial editors in Video-3DGS framework to conduct initial video editing.

Arguments:
  - editing method
  - initial_editor
  - prompt
  - cate
  - progressive_num

## Citation
```shells

```

## Acknowledgement

[DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

[COLMAP](https://github.com/colmap/colmap)
 
[3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

[Deformable-3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians)

[Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero)

[TokenFlow](https://github.com/omerbt/TokenFlow)

[RAVE](https://rave-video.github.io/)
