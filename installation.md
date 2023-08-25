## Installation

We have tested the code with `python==3.7` and `pytorch=1.8`, other late versions may also work well. 
<br>
Welcome to provide feedback or suggestion for the version list!
<!-- Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch dependencies. 
Installing PyTorch and TorchVision with CUDA support is strongly recommended. -->

Install FoodSAM with the following steps:

a. Clone the repository locally:

```
git clone https://github.com/jamesjg/FoodSAM.git
```
b. Create a conda virtual environment and activate it
```
conda create -n FoodSAM python=3.7 -y
conda activate FoodSAM
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/). Here we use PyTorch 1.8.1 and CUDA 11.1. You may also switch to another version by specifying the version number.
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
d. Install MMCV following the [official instructions](https://mmcv.readthedocs.io/en/latest/#installation). 
```
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.8.0/index.html
```
e. Install SAM following official [SAM installation](https://github.com/facebookresearch/segment-anything).
```
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f
```
f. other requirements
```
pip install -r requirement.txt
```

e. Finally download three checkpoints, and move them to "ckpts/" folder as described.

[SAM-vit-h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

[FoodSeg103-SETR-MLA](https://smu-my.sharepoint.com/personal/xwwu_smu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxwwu%5Fsmu%5Fedu%5Fsg%2FDocuments%2Fcheckpoints%2Ezip&parent=%2Fpersonal%2Fxwwu%5Fsmu%5Fedu%5Fsg%2FDocuments&ga=1)

[UNIDET-Unified_learned_OCIM_RS200_6x+2x](https://drive.google.com/file/d/1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI/edit)

If the above links are not working, you also can download them in [Baidu Disk](https://pan.baidu.com/s/1o1w_Vejrtd7rvWVorSQZfg?pwd=pyyk) (code:`pyyk`).

## Dataset and configs
For UNIDET and FoodSeg103, the configs are already put into the [configs](configs/) folder. 
You can also download other ckpt and configs from their official links.

The default dataset we use is [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), other semantic segmentation food datasets like [UECFOODPIXCOMPLETE](https://mm.cs.uec.ac.jp/uecfoodpix/) can also be used. But you should change the  `args.category_txt and args.num_class`. The dataset should be put in the "dataset/"folder.

Your data, configs, and ckpt path should look like this:
````
FoodSAM
-- ckpts
   |-- SETR_MLA
   |   |-- iter_80000.pth
   |-- sam_vit_h_4b8939.pth
   |-- Unified_learned_OCIM_RS200_6x+2x.pth
-- configs
   |-- Base-CRCNN-COCO.yaml
   |-- Unified_learned_OCIM_RS200_6x+2x.yaml
   |-- SETR_MLA_768x768_80k_base.py
-- dataset
   |-- FoodSeg103
   |   |-- Images
   |   |   |-- ann_dir
   |   |   |-- img_dir  
-- FoodSAM
-- mmseg
-- UNIDET
   ...

````


