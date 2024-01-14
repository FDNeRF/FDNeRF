
### environment settings
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# train w/ semantic_window 
python train/train_fdnerf.py --resume --batch_size 8 --gpu_id 1 --datadir '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb' --dataset_prefix 'mixwild' --name '2Dimplicitdeform_reconstruct' --conf 'conf/exp/fp_mixexp_2D_implicit.conf' --chunk_size 4000

# train w/o semantic_window 
python train/train_fdnerf.py --semantic_window 27 --resume --batch_size 8 --gpu_id 1 --datadir '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb' --dataset_prefix 'mixwild' --name '2Dimplicitdeform_video' --conf 'conf/exp/fp_mixexp_2D_implicit_video.conf' --chunk_size 4000






























### aggregation type investigation
# multiexp joint modeling
python train/train2_PIRender.py --resume --batch_size 2 --name '1209_joint_concat' --gpu_id 2 --conf 'conf/exp/fs_multiexp_concat.conf' --Joint_Train --resume_init --resume_ckpt_init 'results/1208_fs_static_concat/checkpoints/pixel_nerf_latest'
python train/train2_PIRender.py --resume --batch_size 2 --name '1209_fixPI_multiexp_concat' --gpu_id 3 --conf 'conf/exp/fs_multiexp_concat.conf' --resume_init --resume_ckpt_init 'results/1208_fs_static_concat/checkpoints/pixel_nerf_latest'

python train/train2_PIRender.py --resume --batch_size 2 --name '1208_joint_weight' --gpu_id 2 --conf 'conf/exp/fs_multiexp_weight.conf' --Joint_Train --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'
python train/train2_PIRender.py --resume --batch_size 2 --name '1208_fixPI_multiexp_weight' --gpu_id 3 --conf 'conf/exp/fs_multiexp_weight.conf' --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'

# investigation of UVfeature selection
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_fixPI_multiexp_UVfeat_MAX_3' --gpu_id 0 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'max' --box_size_half 3 --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_joint_multiexp_UVfeat_MAX_3' --gpu_id 1 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'max' --box_size_half 3 --Joint_Train --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'

python train/train2_PIRender_featSml.py --resume --batch_size 2 --name '1208_fixPI_multiexp_UVfeat_MEAN_3' --gpu_id 2 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'mean' --box_size_half 3 --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'
python train/train2_PIRender_featSml.py --resume --batch_size 2 --name '1208_joint_multiexp_UVfeat_MEAN_3' --gpu_id 3 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'mean' --box_size_half 3 --Joint_Train --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'

python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_fixPI_multiexp_UVfeat_attention_3' --gpu_id '4 6' --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'attention' --box_size_half 3 --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_joint_multiexp_UVfeat_attention_3' --gpu_id '5 6' --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'attention' --box_size_half 3 --Joint_Train --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'







# TEST-Visual
python train/train1_static.py --resume --batch_size 1 --name '000_debug_static' --gpu_id 0 --conf 'conf/exp/fs_static.conf' --visual_path 'visuals_test' --chunk_size 1000 --only_test

python train/train1_static.py --resume --batch_size 1 --name '1220_real_static_test' --gpu_id 2 --conf 'conf/exp/fs_static_real.conf' --visual_path 'visuals_test' --chunk_size 1000 --only_test --resume_init --resume_ckpt_init 'checkpoints/00_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'

python train/train2_PIRender.py --resume --batch_size 1 --name '000_debug_PIRender' --gpu_id 0 --conf 'conf/exp/fs_multiexp.conf' --visual_path 'visuals_test' --chunk_size 1200 --only_test

python train/train2_PIRender.py --resume --batch_size 1 --name '1128_test_pirender_fixPI_trainNerf' --gpu_id 2 --conf 'conf/exp/fs_multiexp.conf' --visual_path 'visuals_test' --chunk_size 1000 --only_test

python train/train2_PIRender.py --resume --batch_size 1 --name '1216_PI_jointtrain' --gpu_id 3 --conf 'conf/exp/fs_multiexp.conf' --visual_path 'visuals_test' --chunk_size 1000 --only_test --Joint_Train --resume_init --resume_ckpt_init 'checkpoints/01_Joint_PI_PixelNeRF/pixel_nerf_latest'

python train/train1_static.py --resume --batch_size 1 --name '1201_fs_static_concat' --gpu_id 0 --conf 'conf/exp/fs_static_concat.conf' --visual_path 'visuals_test' --chunk_size 1000 --only_test

#
python train/train2_PIRender.py --resume --batch_size 1 --name '1209_joint_concat' --gpu_id 2 --conf 'conf/exp/fs_multiexp_concat.conf' --visual_path 'visuals_test2' --chunk_size 2000 --only_test --Joint_Train
#
python train/train2_PIRender.py --resume --batch_size 1 --name '1208_joint_weight' --gpu_id 0 --conf 'conf/exp/fs_multiexp_weight.conf' --visual_path 'visuals_test2' --chunk_size 2000 --only_test --Joint_Train
#
python train/train2_PIRender.py --resume --batch_size 1 --name '1209_fixPI_multiexp_concat' --gpu_id 2 --conf 'conf/exp/fs_multiexp_concat.conf' --visual_path 'visuals_test2' --chunk_size 2000 --only_test
#
python train/train2_PIRender.py --resume --batch_size 1 --name '1208_fixPI_multiexp_weight' --gpu_id 2 --conf 'conf/exp/fs_multiexp_weight.conf' --visual_path 'visuals_test2' --chunk_size 2000 --only_test

#
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_fixPI_multiexp_UVfeat_MAX_3' --gpu_id 3 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'max' --box_size_half 3 --visual_path 'visuals_test2' --chunk_size 256 --only_test
#
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_joint_multiexp_UVfeat_MAX_3' --gpu_id 3 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'max' --box_size_half 3 --visual_path 'visuals_test2' --chunk_size 256 --only_test --Joint_Train
#
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_fixPI_multiexp_UVfeat_MEAN_3' --gpu_id 2 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'mean' --box_size_half 3 --visual_path 'visuals_test2' --chunk_size 256 --only_test 
#
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_joint_multiexp_UVfeat_MEAN_3' --gpu_id 2 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'mean' --box_size_half 3 --visual_path 'visuals_test2' --chunk_size 256 --only_test --Joint_Train
#
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_fixPI_multiexp_UVfeat_attention_3' --gpu_id 3 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'attention' --box_size_half 3 --visual_path 'visuals_test2' --chunk_size 256 --only_test 
#
python train/train2_PIRender_featSml.py --resume --batch_size 1 --name '1208_joint_multiexp_UVfeat_attention_3' --gpu_id 2 --conf 'conf/exp/fs_multiexp.conf' --use_UVfea_box --UVfea_strategy 'attention' --box_size_half 3 --visual_path 'visuals_test2' --chunk_size 256 --only_test --Joint_Train


# TEST-Ours
python train/train2_PIRender_real.py --resume --batch_size 1 --name '1215_test_real' --gpu_id 2 --conf 'conf/exp/fs_multiexp_real.conf' --visual_path 'visuals_test_bg_test' --chunk_size 1000 --only_test --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'

python train/train2_PIRender_real.py --resume --batch_size 1 --name '1231_test_realwarp' --gpu_id 2 --conf 'conf/exp/fs_multiexp_real2.conf' --visual_path '02_visuals_test' --chunk_size 1000 --only_test --resume_init --resume_ckpt_init 'results/000_ckpt_pixelnerf_multiexp_fs/pixel_nerf_latest'

python train/train2_PIRender_real.py --resume --batch_size 1 --name '1217_test_real_joint' --gpu_id 2 --conf 'conf/exp/fs_multiexp_real2.conf' --visual_path '02_visuals_test' --chunk_size 1000 --only_test --Joint_Train --resume_init --resume_ckpt_init 'checkpoints/01_Joint_PI_PixelNeRF/pixel_nerf_latest'

