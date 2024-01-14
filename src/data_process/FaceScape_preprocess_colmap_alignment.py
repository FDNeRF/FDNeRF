"""
Author: Eckert ZHANG
Date: 2022-02-20 09:47:29
LastEditTime: 2022-03-01 20:03:37
LastEditors: Eckert ZHANG
Description: 
"""
import argparse
import os, sys
import torch
import torchvision.transforms as transforms
import imageio, glob, cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_colmap_pose(file):
    poses_bounds = np.load(file)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    hwfs = poses[:, :, -1:]
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses_bounds), 1, 1))
    M = np.concatenate([poses[:, :3, :4], last_row], 1)
    return M, bounds, hwfs


def main(args,
         folder_name_img_o='images_faces',
         folder_name_img_m='images_masked',
         folder_name_msk='masks_faces'):
    # total 359
    folder_standard = '1'
    folder_tochange = '2'
    view_stand = 25
    view_B = 25

    views_std_path = os.path.join(args.datapath, folder_standard, '1_neutral',
                                  folder_name_img_o, 'poses_bounds.npy')
    views_tcg_path = os.path.join(args.datapath, folder_tochange, '1_neutral',
                                  folder_name_img_o, 'poses_bounds.npy')
    poses_std, bounds_std, hwfs_std = read_colmap_pose(views_std_path)
    poses_tcg, bounds_tcg, hwfs_tcg = read_colmap_pose(views_tcg_path)
    P_std = poses_std[view_stand]
    P_tcg = poses_tcg[view_B]
    T = np.dot(np.linalg.inv(P_tcg), P_std)
    print('Translation Matrix:\n', T)

    num_tcg = poses_tcg.shape[0]
    poses_new = []
    for i in range(num_tcg):
        P = np.dot(poses_tcg[i], T)
        poses_new.append(P)
    poses_new = np.array(np.stack(poses_new))
    poses_bounds = np.concatenate([poses_new[:, :3, :], hwfs_tcg],
                                  2).reshape(-1, 15)
    poses_bounds = np.concatenate([poses_bounds, bounds_tcg], -1)

    # exps = sorted([
    #         x for x in os.listdir(root_dir)
    #         if os.path.isdir(os.path.join(root_dir, x))
    #     ])
    # img_neural_list = sorted([
    #         x for x in glob.glob(os.path.join(imgs_neural_path, "*"))
    #         if (x.endswith(".jpg") or x.endswith(".png"))
    #     ])


if __name__ == '__main__':
    """
    Extracts and aligns all faces from images, estimate the face pose for each image
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",
                        type=str,
                        default='/data/zhangjingbo/FaceScape_3dmm/images')
    parser.add_argument(
        "--savepath",
        type=str,
        default='/data/zhangjingbo/FaceScape_3dmm/images_align')
    args = parser.parse_args()

    main(args)