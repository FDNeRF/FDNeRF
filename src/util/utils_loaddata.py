"""
Author: Eckert ZHANG
Date: 2021-11-10 10:42:34
LastEditTime: 2022-02-20 21:33:07
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf-portrait/src/util/utils_loaddata.py
Description: 
"""
import os
import numpy as np
import torch


def save_list(list, save_path, filename):
    f = open(os.path.join(save_path, filename), 'w')
    for w in list:
        f.write(w + '\n')
    f.close
    print('--saving \'' + filename + '\' to \'' + save_path + '\'')


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:
                  3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]),
                       (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate
    poses_centered = np.linalg.inv(
        pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv


def colmap_pose_reading(filepath):
    '''
    Here, we need to center poses
    outputs:
    poses: c2w (N_images, 3, 4)
    hwfs: (N_images, 3)
    bounds: (N_images, 2)
    '''
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                               [0, 0, 0, 1]])
    blender2opengl = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    poses_bounds = np.load(os.path.join(filepath, 'poses_bounds.npy'))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    hwfs = poses[:, :, -1]
    # correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]],
                           -1)
    poses, _ = center_poses(poses, blender2opengl)

    return poses, hwfs, bounds


def list_sorting(lis):
    num = len(lis)
    list_tuple = []
    for ii in range(num):
        path = lis[ii]
        img_name = os.path.split(path)[-1]
        img_id = np.array(img_name.split('.')[0])
        list_tuple.append(('%2d' % img_id, path))
    list_sorted = sorted(list_tuple, key=lambda t: t[0])
    return list_sorted


# compute rotation matrix (c2w) based on 3 ruler angles
# input: angles with shape [1,3]
# output: rotation matrix with shape [1,3,3]
def Compute_rotation_matrix(angles):
    n_b = angles.shape[0]
    sinx = torch.sin(angles[:, 0])
    siny = torch.sin(angles[:, 1])
    sinz = torch.sin(angles[:, 2])
    cosx = torch.cos(angles[:, 0])
    cosy = torch.cos(angles[:, 1])
    cosz = torch.cos(angles[:, 2])
    rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1,
                                               1).view(3, n_b, 3, 3)
    if angles.is_cuda: rotXYZ = rotXYZ.cuda()
    rotXYZ[0, :, 1, 1] = cosx
    rotXYZ[0, :, 1, 2] = -sinx
    rotXYZ[0, :, 2, 1] = sinx
    rotXYZ[0, :, 2, 2] = cosx
    rotXYZ[1, :, 0, 0] = cosy
    rotXYZ[1, :, 0, 2] = siny
    rotXYZ[1, :, 2, 0] = -siny
    rotXYZ[1, :, 2, 2] = cosy
    rotXYZ[2, :, 0, 0] = cosz
    rotXYZ[2, :, 0, 1] = -sinz
    rotXYZ[2, :, 1, 0] = sinz
    rotXYZ[2, :, 1, 1] = cosz
    rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])
    return rotation.permute(0, 2, 1)


def pose_from_param_3dmm(param_3dmm):
    angle_part = torch.from_numpy(param_3dmm[:, 224:227])
    trans_part = torch.from_numpy(param_3dmm[:, 254:257])
    matrix_r = Compute_rotation_matrix(angle_part)
    num_B = angle_part.shape[0]
    pose = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(num_B, 1, 1)
    pose[:, :3, :3] = matrix_r
    pose[:, :3, 3] = trans_part
    # c2ws = torch.inverse(pose)
    c2ws = pose
    return np.array(c2ws[:, :3])
