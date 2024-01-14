# [FDNeRF](https://fdnerf.github.io/)
Official implementation for the paper "FDNeRF: Few-shot Dynamic Neural Radiance Fields for Face Reconstruction and Expression Editing".
Note: **This code is forked from [PixelNeRF](https://github.com/sxyu/pixel-nerf).**

<img src='https://github.com/FDNeRF/FDNeRF.github.io/blob/main/static/images/teaser.png'>
Abstract: We propose a Few-shot Dynamic Neural Radiance Field (FDNeRF), the first NeRF-based method capable of reconstruction and expression editing of 3D faces based on a small number of dynamic images. Unlike existing dynamic NeRFs that require dense images as input and can only be modeled for a single identity, our method enables face reconstruction across different persons with few-shot inputs. Compared to state-of-the-art few-shot NeRFs designed for modeling static scenes, the proposed FDNeRF accepts view-inconsistent dynamic inputs and supports arbitrary facial expression editing, i.e., producing faces with novel expressions beyond the input ones. To handle the inconsistencies between dynamic inputs, we introduce a well-designed conditional feature warping (CFW) module to perform expression conditioned warping in 2D feature space, which is also identity adaptive and 3D constrained. As a result, features of different expressions are transformed into the target ones. We then construct a radiance field based on these view-consistent features and use volumetric rendering to synthesize novel views of the modeled faces. Extensive experiments with quantitative and qualitative evaluation demonstrate that our method outperforms existing dynamic and few-shot NeRFs on both 3D face reconstruction and expression editing tasks.


---

## Pipeline
<img src='https://github.com/FDNeRF/FDNeRF.github.io/blob/main/static/images/pipeline_v4.png'>


## Installation
```
conda env create -f environment.yml
conda activate fdnerf
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

## Implementation

### train w/ semantic_window 
```
python train/train_fdnerf.py --resume --batch_size 8 --gpu_id 0 --datadir '[datasets path]' --dataset_prefix 'mixwild' --name '2Dimplicitdeform_reconstruct' --conf 'conf/exp/fp_mixexp_2D_implicit.conf' --chunk_size 4000
```

### train w/o semantic_window 
```
python train/train_fdnerf.py --semantic_window 27 --resume --batch_size 8 --gpu_id 0 --datadir '[datasets path]' --dataset_prefix 'mixwild' --name '2Dimplicitdeform_video' --conf 'conf/exp/fp_mixexp_2D_implicit_video.conf' --chunk_size 4000
```
