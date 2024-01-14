"""
Author: Eckert ZHANG
Date: 2021-09-10 01:02:22
LastEditTime: 2022-03-30 14:06:33
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf-portrait/src/data/FSDataset_colmap_static.py
Description: 
"""
from logging import Filterer
import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2, random, json
from torchvision import transforms as T
from util import save_list, colmap_pose_reading, list_sorting


def pose_json_reading(filepath, ids):
    cont = json.load(open(filepath, "r"))
    camera_angle_x = float(cont['camera_angle_x'])
    frames = cont['frames']
    poses = []
    for ii in range(len(ids)):
        img_name = ids[ii]
        poses.append(np.array(frames[f'{img_name}_pose']))
    return np.array(poses), camera_angle_x


def face_pose_reading(jsonfile, invalid_ids=[]):
    cont = json.load(open(jsonfile, "r"))
    focal = np.array(cont['focal_len'])
    cx, cy = np.array(cont['cx']), np.array(cont['cy'])
    n, f = np.array(cont['near']), np.array(cont['far'])
    imgs_list, poses = [], []
    for frame in cont['frames']:
        if len(invalid_ids) > 0 and str(frame['img_id']) in invalid_ids:
            continue
        imgs_list.append(
            os.path.join(os.path.dirname(jsonfile),
                         str(frame['img_id']) + '.png'))
        poses.append(np.array(frame['transform_matrix']))
    poses = np.array(poses).astype(np.float32)
    return poses, focal, [float(cx), float(cy)], [float(n),
                                                  float(f)], imgs_list


random.seed(10)


class Facepose_WildData_static(torch.utils.data.Dataset):
    """
    Dataset from FaceScape
    """
    def __init__(
            self,
            path,
            stage="train",
            list_prefix="wild",
            load_img_folder="images_3dmm",
            load_para_folder="images_3dmm",
            image_size=(256, 256),  # w, h
    ):
        """
        Parameters:
            path: dataset root path
            stage: train | val | test
            list_prefix: prefix for split lists: <list_prefix>[train, val, test].lst
            image_size: result image size (resizes if different); None to keep original size
        """
        super().__init__()
        self.base_path = path
        self.load_img_folder = load_img_folder
        self.load_para_folder = load_para_folder
        assert os.path.exists(self.base_path)

        # list_prefix = 'few50'

        if os.path.exists(os.path.join(path, f"{list_prefix}_{stage}.lst")):
            print("Loading data on the basis of " +
                  f"{list_prefix}_{stage}.lst")
        else:
            cats = [
                x for x in os.listdir(path)
                if os.path.isdir(os.path.join(path, x))
            ]
            n_train = np.int(0.7 * len(cats))
            n_val = np.int(0.2 * len(cats))
            n_test = len(cats) - n_train - n_val
            cats_train = sorted(random.sample(cats, n_train))
            cats_val = sorted(
                random.sample(list(set(cats).difference(set(cats_train))),
                              n_val))
            cats_test = sorted(
                list(set(cats).difference(set(cats_train), set(cats_val))))
            save_list(cats_train, path, f"{list_prefix}_train.lst")
            save_list(cats_val, path, f"{list_prefix}_val.lst")
            save_list(cats_test, path, f"{list_prefix}_test.lst")
        file_list = os.path.join(path, f"{list_prefix}_{stage}.lst")

        self.invalid_ids = []

        self.build_metas(file_list)
        self.define_transforms()
        self.stage = stage
        self.image_size = image_size
        self.lindisp = False
        self.z_near, self.z_far = 8, 26

    def define_transforms(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        inplace=True),
        ])
        self.transform_mask = T.Compose([
            T.ToTensor(),
        ])

    def build_metas(self, scan_list_file):
        self.metas = []
        with open(scan_list_file, "r") as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        for scan in self.scans:
            expressions = [
                x for x in os.listdir(os.path.join(self.base_path, scan))
                if os.path.isdir(os.path.join(self.base_path, scan, x))
            ]
            for exp in expressions:
                view_path = os.path.join(self.base_path, scan, exp,
                                         self.load_img_folder)
                poses, focal, c, nf, img_files = face_pose_reading(
                    os.path.join(view_path, 'face_transforms_pose.json'),
                    self.invalid_ids)
                num_views = len(img_files)

                w2cs = []
                for view_id in range(num_views):
                    c2w = np.eye(4, dtype=np.float32)
                    c2w = np.array(poses[view_id])
                    w2c = np.linalg.inv(c2w)
                    w2cs.append(w2c)
                w2cs = np.stack(w2cs, 0)

                # Let each view could be as a ref-view OR skip
                skip_num = 1
                for ii in range(0, num_views, skip_num):
                    ref_view = int(ii)
                    w2c_ref = w2cs[ref_view]
                    angles_tuple = []
                    for jj in range(num_views):
                        if jj == ref_view:
                            continue
                        else:
                            w2c_cdd = w2cs[jj]
                            angle = np.arccos(
                                np.dot(w2c_ref[:, 2], w2c_cdd[:, 2]))
                            angles_tuple.append((angle, jj))
                    angle_sorted = sorted(
                        angles_tuple,
                        key=lambda t: t[0])  # angle small --> large
                    if angle_sorted[4][0] > np.pi / 6:
                        continue
                    src_views = [angle_sorted[x][1] for x in range(12)]
                    self.metas += [(scan, exp, num_views, ref_view, src_views)]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        scan, exp, num_views, target_view, src_views = self.metas[index]
        ids = torch.randperm(8)[:3]
        view_ids = [src_views[i] for i in ids] + [target_view]

        root_dir = os.path.join(self.base_path, scan, exp,
                                self.load_img_folder)

        poses, f, cxy, nf, rgb_paths = face_pose_reading(
            os.path.join(root_dir, 'face_transforms_pose.json'),
            self.invalid_ids)
        mask_paths = sorted(
            glob.glob(os.path.join(root_dir, "masks_faces", "*.jpg")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)
            mask_id = False
        else:
            mask_id = True

        rgb_paths0 = [rgb_paths[i] for i in view_ids]
        mask_paths0 = [mask_paths[i] for i in view_ids]
        rgb_paths = rgb_paths0
        if mask_id:
            mask_paths = mask_paths0

        all_imgs, all_poses, all_masks, all_nfs = [], [], [], []
        focal, c = [], []
        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths,
                                                        mask_paths)):
            img = imageio.imread(rgb_path)[..., :3]
            h, w, _ = img.shape
            scale = 1.0

            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

            if (w, h) != self.image_size:
                img = cv2.resize(img, self.image_size)
                scale = self.image_size[1] / h
                if mask_path is not None:
                    mask = cv2.resize(mask, self.image_size)

            pose = np.eye(4, dtype=np.float32)
            pose = poses[idx]
            pose = torch.tensor(pose, dtype=torch.float32)
            fx = torch.tensor(f) * scale
            fy = torch.tensor(f) * scale
            cx = torch.tensor(w / 2) * scale
            cy = torch.tensor(h / 2) * scale
            focal.append(torch.tensor((fx, fy), dtype=torch.float32))
            c.append(torch.tensor((cx, cy), dtype=torch.float32))

            img_tensor = self.transform(img)
            all_imgs.append(img_tensor)
            all_poses.append(pose)

            near, far = nf[0], nf[1]
            all_nfs.append(torch.tensor((near, far), dtype=torch.float32))
        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        focal = torch.stack(focal)
        c = torch.stack(c)
        all_nfs = torch.stack(all_nfs)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "c": c,
            "nfs": all_nfs,
        }
        if all_masks is not None:
            result["masks"] = all_masks

        return result
