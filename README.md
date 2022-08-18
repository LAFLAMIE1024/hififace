# HifiFace &mdash; Unofficial Pytorch Implementation
>![](./assets/front_final.png)
>
> Image source: *[HifiFace: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping](https://arxiv.org/abs/2106.09965)* (figure 1, pg. 1)
![issueBadge](https://img.shields.io/github/issues/mindslab-ai/hififace)   ![starBadge](https://img.shields.io/github/stars/mindslab-ai/hififace)   ![repoSize](https://img.shields.io/github/repo-size/mindslab-ai/hififace)  ![lastCommit](https://img.shields.io/github/last-commit/mindslab-ai/hififace) 

This repository is an unofficial implementation of the face swapping model proposed by _Wang et. al_ in their paper [HifiFace: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping](https://arxiv.org/abs/2106.09965). This implementation makes use of the [Pytorch Lighting](https://www.pytorchlightning.ai/) library, a light-weight wrapper for [PyTorch](https://pytorch.org/).

## HifiFace Overview
The task of face swapping applies the face and the identity of the source person to the head of the target.

The HifiFace architecture can be broken up into three primary structures. The _3D shape-aware identity extractor_, the _semantic facial fusion module_, and an encoder-decoder structure. A high-level overview of the architecture can be seen in the image below.

> ![](./assets/hififace_arch_500.png)
>
> Image source: *[HifiFace: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping](https://arxiv.org/abs/2106.09965)* (figure 2, pg. 3)


## Changes from the original paper
### Dataset
In the paper, the author used VGGFace2 and Asian-Celeb as the training dataset. The Asian-Celeb dataset would be utilized in the  future.
And we use VGGFace2_224 downloaded from Simswap to train this hififace from scratch.

### Model
The paper proposes two versions of HifiFace model based on the output image size: 256x256 and 512x512 (referred to as Ours-256 and Ours-512 in the paper). The 512x512 model uses an extra data preprocessing before training. In this open source project, we implement the 256x256 model.
For the discriminator, the original paper uses the discriminator from [StarGAN v2](https://arxiv.org/abs/1912.01865). Our implementation uses the multi-scale discriminator from [SPADE](https://arxiv.org/abs/1903.07291).


## Installation

``` shell
git clone https://github.com/mindslab-ai/hififace 
cd hififace

git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch && git clone https://github.com/NVlabs/nvdiffrast && git clone https://github.com/deepinsight/insightface.git

cp -r insightface/recognition/arcface_torch/ Deep3DFaceRecon_pytorch/models/
cp -r insightface/recognition/arcface_torch/ ./model/
rm -rf insightface
cp -rf 3DMM/* Deep3DFaceRecon_pytorch
mv Deep3DFaceRecon_pytorch model/

rm -rf 3DMM
rm -rf nvdiffrast
```

### Pre-Trained Model for Deep3DFace PyTorch (3D face reconstruction network in 3D Shape-Aware Identity Extractor)

Follow the guideline in [Prepare prerequisite models](https://github.com/sicxu/Deep3DFaceRecon_pytorch#prepare-prerequisite-models).

Set up at `./model/Deep3DFaceRecon_pytorch/`

### Pre-Trained Models for ArcFace (face recognition network in 3D Shape-Aware Identity Extractor)
We used official Arcface per-trained pytorch [implementation](https://github.com/deepinsight/insightface/tree/415da817d127319a99aeb84927f2cd0fcbb3366c/recognition/arcface_torch).

Download pre-trained checkpoint from [onedrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d) (IResNet-100 trained on MS1MV3)

### Download HifiFace Pre-Trained Model
[google drive link](https://drive.google.com/file/d/1tZitaNRDaIDK1MPOaQJJn5CivnEIKMnB/view?usp=sharing)

This model was trained on VGGFace2, 300K iterations

## Training
### Dataset & Preprocessing
#### Align & Crop
We aligned the face images with the landmark extracted by [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2). The code will be added.

#### Face Segmentation Map
After finishing aligning the face images, you need to get the face segmentation map for each face images. We used face segmentation model provided by [PSFRGAN](https://github.com/chaofengc/PSFRGAN). You can use their codes and pre-trained mode

Here are the codes I ran on colab to download the whole repo, do not forget to [download their pre-trained model](https://drive.google.com/drive/folders/1Ubejhxd2xd4fxGc_M_LWl3Ux6CgQd9rP) too.

```python
!git clone https://github.com/chaofengc/PSFR-GAN.git
%cd PSFR-GAN

# Here you need to make a new directory
!mkdir pretrain_models 

# You just need to add a shortcut to your own google drive to put all those pre-trained models into this directory
!cp -r /content/drive/MyDrive/PSFR-GAN_pretrain_models/* /content/PSFR-GAN/pretrain_models/ 
```

#### Dataset Folder Structure
Each face image and the corresponding segmentation map should have the same name and the same relative path from the top-level directory.

```python
face_image_dataset_folder
└───identity1
│   │   image1.png
│   │   image2.png
│   │   ...
│   
└───identity2
│   │   image1.png
│   │   image2.png
│   │   ...
│ 
|   ...

face_segmentation_mask_folder
└───identity1
│   │   image1.png
│   │   image2.png
│   │   ...
│   
└───identity2
│   │   image1.png
│   │   image2.png
│   │   ...
│ 
|   ...
```
As I said, I planned to train on VGGFace2_224 provided by SimSwap. However, it's still too large for me and such time-consuming to parse each of this picture. So, I took about  760 identities out of this dataset and preprocess them with the models link offered by mindslab.ai .
And here is [my own cleaned dataset](https://drive.google.com/file/d/1tHjQALxc4q4klr9fKVwEjG31Ci6yg3Bl/view?usp=sharing)

### Wandb
[Wandb](https://www.wandb.com/) is a powerful tool to manage your model training. 
Please register a wandb account and a wandb project for training HifiFace with our training code.
Here I created a wandb account named 'marcocheung' and a project called 'hififace'.


### Changing the Configuration
* [config/model.yaml](config/model.yaml)
    * dataset.train.params.image_root: directory path to the training dataset images
    * dataset.train.params.parsing_root: directory path to the training dataset parsing images
    * dataset.validation.params.image_root: directory path to the validation dataset images
    * dataset.validation.params.parsing_root: directory path to the validation dataset parsing images

* [config/trainer.yaml](config/trainer.yaml)
    * checkpoint.save_dir: directory where the checkpoints will be saved
    * wandb: fill out your wandb entity and project name
    

### Run Training Code

```shell
python hififace_trainer.py --model_config config/model.yaml --train_config config/trainer.yaml -n hififace
```

## Inference

### Single Image

```shell
python hififace_inference --gpus 0 --model_config config/model.yaml --model_checkpoint_path hififace_opensouce_299999.ckpt --source_image_path assets/inference_samples/01_source.png --target_image_path assets/inference_samples/01_target.png --output_image_path ./01_result.png
```

### All Posible Pairs of Images in Directory

```shell
python hififace_inference --gpus 0 --model_config config/model.yaml --model_checkpoint_path hififace_opensouce_299999.ckpt  --input_directory_path assets/inference_samples --output_image_path ./result.png
```

### Interpolation
```shell
# interpolates both the identity and the 3D shape.
python hififace_inference --gpus 0 --model_config config/model.yaml --model_checkpoint_path hififace_opensouce_299999.ckpt --source_image_path assets/inference_samples/01_source.png --target_image_path assets/inference_samples/01_target.png --output_image_path ./01_result_all.gif  --interpolation_all 

# interpolates only the identity.
python hififace_inference --gpus 0 --model_config config/model.yaml --model_checkpoint_path hififace_opensouce_299999.ckpt --source_image_path assets/inference_samples/01_source.png --target_image_path assets/inference_samples/01_target.png --output_image_path ./01_result_identity.gif  --interpolation_identity

# interpolates only the 3D shape.
python hififace_inference --gpus 0 --model_config config/model.yaml --model_checkpoint_path hififace_opensouce_299999.ckpt --source_image_path assets/inference_samples/01_source.png --target_image_path assets/inference_samples/01_target.png --output_image_path ./01_result_3d.gif  --interpolation_3d

```

## To-Do List
- [ ] Pre-processing Code
- [ ] Colab Notebook 

## License

[BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
