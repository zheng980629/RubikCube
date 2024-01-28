# Rubik's Cube: High-Order Channel Interactions with a Hierarchical Receptive Field (NeurIPS2023)

Naishan Zheng, Man Zhou, Chong Zhou, Chen Change Loy

S-Lab, Nanyang Technological University   

---
>Image restoration techniques, spanning from the convolution to the transformer paradigm, have demonstrated robust spatial representation capabilities to deliver high-quality performance. Yet, many of these methods, such as convolution and the Feed Forward Network (FFN) structure of transformers, primarily leverage the basic first-order channel interactions and have not maximized the potential benefits of higher-order modeling. To address this limitation, our research dives into understanding relationships within the channel dimension and introduces a simple yet efficient, high-order channel-wise operator tailored for image restoration. Instead of merely mimicking high-order spatial interaction, our approach offers several added benefits: Efficiency: It adheres to the zero-FLOP and zero-parameter
principle, using a spatial-shifting mechanism across channel-wise groups. Simplicity: It turns the favorable channel interaction and aggregation capabilities into element-wise multiplications and convolution units with 1 × 1 kernel. Our new formulation expands the first-order channel-wise interactions seen in previous works to arbitrary high orders, generating a hierarchical receptive field akin to a Rubik’s cube through the combined action of shifting and interactions. Furthermore, our proposed Rubik’s cube convolution is a flexible operator that can be incorporated into existing image restoration networks, serving as a drop-in replacement for the standard convolution unit with fewer parameters overhead. We conducted experiments across various low-level vision tasks, including image denoising, low-light image enhancement, guided image super-resolution, and image de-blurring. The results consistently demonstrate that our Rubik’s cube operator enhances performance across all tasks.
---
<img src="./asserts/teaser.jpg" width="800px"/>
<img src="./asserts/receptiveField.jpg" width="800px"/>

## Applications
### 🚀: Low-Light Image Enhancement
#### 🐬: Prepare data
Download the training data and add the data path to the config file (/basicsr/option/train/LLIE/*.yml). Please refer to [LOL](https://daooshee.github.io/BMVC2018website/) and [Huawei](https://github.com/JianghaiSCU/R2RNet) for data download. 
#### 🐬: Training
```
python /RubikCube/train.py -opt /RubikCube/options/train/LLIE/SID_RubikConv.yml
python /RubikCube/train.py -opt /RubikCube/options/train/LLIE/DRBN_RubikConv.yml
```
#### 🐬: Inference
Download the pretrained low-light image enhancement model from [Google Drive](https://drive.google.com/drive/folders/1nPArtH3X291G0zAM1YyCZLS8bIqJAELs?usp=drive_link) and add the path to the config file (/RubikCube/options/test/LLIE/*.yml).
```
python /RubikCube/test.py -opt /RubikCube/options/test/LLIE/SID_RubikConv.yml
python /RubikCube/test.py -opt /RubikCube/options/test/LLIE/DRBN_RubikConv.yml
```
### 🚀: Image Denoising
#### 🐬: Prepare data
Please refer to Real Image Denoising in [Restormer](https://github.com/swz30/Restormer/tree/main/Denoising) for data download. 
#### 🐬: Training
```
python /RubikCube/train.py -opt /RubikCube/options/train/denoise/DNCNN_RubikConv.yml
python /RubikCube/train.py -opt /RubikCube/options/train/denoise/MPRNet_rubikCube_denoise.yml
python /RubikCube/train.py -opt /RubikCube/options/train/denoise/restormer_rubikCubeMul_denoise.yml
```
#### 🐬: Inference
Download the pretrained denoising model from [Google Drive](https://drive.google.com/drive/folders/1DLNu1p9epKUIUpn-8LtFQsqthcdNuSn-?usp=drive_link).
* To obtain denoised results
```
python test_DNCNN_rubikConv_sidd.py --save_images
```
* To reproduce PSNR/SSIM scores on SIDD
```
evaluate_sidd.m
```
### 🚀: Image Deblur
#### 🐬: Prepare data
Please refer to Motion Deblurring in [Restormer](https://github.com/swz30/Restormer/tree/main/Motion_Deblurring) for data download. 
#### 🐬: Training
```
python /RubikCube/train.py -opt /RubikCube/options/train/denoise/deepDeblur_RubikConv.yml
python /RubikCube/train.py -opt /RubikCube/options/train/denoise/MPRNet_rubikCube_deblur.yml
python /RubikCube/train.py -opt /RubikCube/options/train/denoise/restormer_rubikCubeMul_deblur.yml
```
#### 🐬: Inference
Download the pretrained de-blurring model from [Google Drive](https://drive.google.com/drive/folders/1tWilE5bBmIXLPlhxtsWVqcuYuGLYDfjl?usp=drive_link) and add the path to the config file(/RubikCube/options/test/deblur/*.yml).
* To obtain deblurred results
```
python /RubikCube/test.py -opt /RubikCube/options/test/deblur/deepDeblur_RubikCube.yml
python /RubikCube/test.py -opt /RubikCube/options/test/deblur/MPRNet_deblur_RubikCube.yml
```
* To reproduce PSNR/SSIM scores on GoPro/Hide
```
evaluate_GoPro.m
```
### 🚀: Classification
Training codes and pre-trained models of AlexNet, VGG-16, and ResNet-18 are released in [Google Drive](https://drive.google.com/drive/folders/1uGyYqR-VAeg9GD3_WpAFlZZJt4u6DZAh?usp=drive_link).
## Acknowledgement
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for sharing.
