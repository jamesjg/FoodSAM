# FoodSAM


This repository contains code for FoodSAM. The model can segment and recognize multiple food or non-food instances in a given image.

[[`Paper`]()] 

![FoodSAM architecture](assets/model.jpg)

we propose a novel framework FoodSAM for food image segmentation.
This innovative approach integrates the coarse semantic mask with SAM-generated masks to enhance semantic segmentation quality.
Besides, we recognize that the ingredients in food can be supposed as independent individuals, which motivated us to perform instance segmentation on food images.Furthermore, FoodSAM extends its zero-shot capability to encompass panoptic segmentation by incorporating an object detector, which renders FoodSAM to effectively capture non-food object information.

  <img src="assets/foodsam.jpg" />

## Installation

The code requires `python==3.7`, as well as `pytorch>=1.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch dependencies. Installing PyTorch and TorchVision with CUDA support is strongly recommended.

Install FoodSAM:

a. Clone the repository locally:

```
git clone https://github.com/jamesjg/FoodSAM.git
```
b. Create a conda virtual environment and activate it
```
conda create -n FoodSAM python=3.7 -y
conda activate FoodSAM
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/). Here we use PyTorch 1.8.1 and CUDA 11.1. You may also switch to other version by specifying the version number.
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
d. Install MMCV following the [official instructions](https://mmcv.readthedocs.io/en/latest/#installation). 
```
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.8.0/index.html
```
e. Install SAM following official [SAM installation](https://github.com/facebookresearch/segment-anything).
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
f. other requirements
```
pip install -r requirement.txt
```
## <a name="GettingStarted"></a>Getting Started

First download three checkpoints:

[SAM-vit-h](https://github.com/facebookresearch/segment-anything)

[FoodSeg103-SETR-MLA](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)

[UNIDET-Unified_learned_OCIM_RS200_6x+2x](https://github.com/xingyizhou/UniDet/tree/master)

for UNIDET and FoodSeg103, you also need to download the related configs.

Then,  you can run the model for semantic segmentation in a few command lines, the panoramic and instance segmentation in FoodSAM/panoramic.py is similar
```
python FoodSAM/semantic.py --img_path <path/to/img> --output <path/to/output> --SAM_checkpoint <path/to/SAM_checkpoint> --semantic_checkpoint <path/to/FoodSeg103_checkpoint> --semantic_config <path/to/FoodSeg103_config>
```
Masks can also be generated for a folder of images by setting `args.data_root and args.img_dir`. Furthermore, by setting `args.eval` to true, the model can output the semantic masks and evaluate the metrics. 

Here are examples of semantic segmentation and instance segmentation on the FoodSeg103 dataset:
```
python FoodSAM/semantic.py --data_root dataset/FoodSeg103/Images --img_dir img_dir/test --ann_dir ann_dir/test --output FoodSAM/semantic_results --SAM_checkpoint  ckpts/sam_vit_l_0b3195.pth --semantic_checkpoint ckpts/SETR_MLA/iter_80000.pth --semantic_config ckpts/SETR_MLA/SETR_MLA_768x768_80k_base.py --eval 
```
```
python FoodSAM/semantic.py --data_root dataset/FoodSeg103/Images --img_dir img_dir/test --ann_dir ann_dir/test --output FoodSAM/semantic_results --SAM_checkpoint  ckpts/sam_vit_l_0b3195.pth --semantic_checkpoint ckpts/SETR_MLA/iter_80000.pth --semantic_config ckpts/SETR_MLA/SETR_MLA_768x768_80k_base.py --detection_config UNIDET/configs/Unified_learned_OCIM_RS200_6x+2x.yaml --opts MODEL.WEIGHTS ckpts/Unified_learned_OCIM_RS200_6x+2x.pth 
```
The default dataset we use is [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), other semantic segmentation food datasets like [UECFOODPIXCOMPLETE](https://mm.cs.uec.ac.jp/uecfoodpix/) can also be used. But you should change the  `args.category_txt and args.num_class`

## Main Results

### FoodSeg103
| Method | mIou | aAcc | mAcc 
| :-: | :- | -: | :-: |  
|[SETR_MLA(baseline)](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1) | 45.1 | 83.53 | 57.44
FoodSAM | 46.42 | 84.10 |  58.27

### UECFOODPIXCOMPLETE


| Method | mIou | aAcc | mAcc 
| :-: | :- | -: | :-: |  
|deeplabV3+ (baseline)| 65.61 |88.20| 77.56
FoodSAM | 66.14 |88.47 |78.01

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Citation
If you want to cite our work, please use this:

```
@article

```
