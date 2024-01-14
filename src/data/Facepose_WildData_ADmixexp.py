"""
Author: Eckert ZHANG
Date: 2021-09-10 01:02:22
LastEditTime: 2022-04-01 10:55:49
LastEditors: Eckert ZHANG
Description: 
"""
import os, sys
import torch
import torch.nn.functional as F
import glob, pdb
import imageio
import numpy as np
import cv2, random, json, pickle, csv
from logging import Filterer
from PIL import Image
from torchvision import transforms as T
from util import save_list, colmap_pose_reading


def pose_json_reading(filepath, ids):
    cont = json.load(open(filepath, "r"))
    camera_angle_x = float(cont['camera_angle_x'])
    frames = cont['frames']
    poses = []
    for ii in range(len(ids)):
        img_name = ids[ii]
        poses.append(np.array(frames[f'{img_name}_pose']))
    return np.array(poses), camera_angle_x

def face_pose_reading_from_ids(jsonfile, imgs_path, ids):
    cont = json.load(open(jsonfile, "r"))
    focal = np.array(cont['focal_len'])
    cx, cy = np.array(cont['cx']), np.array(cont['cy'])
    n, f = np.array(cont['near']), np.array(cont['far'])
    imgs_list, poses = [], []
    for id in ids:
        for frame in cont['frames']:
            if str(frame['img_id']) == id:
                imgs_list.append(os.path.join(imgs_path,\
                                str(frame['img_id']) + '.png'))
                poses.append(np.array(frame['transform_matrix']))
                break
    poses = np.array(poses).astype(np.float32)
    return poses, focal, [float(cx), float(cy)], [float(n),
                                                  float(f)], imgs_list

def param_reading_3dmm_pkl(filepath, ids):
    f = open(os.path.join(filepath, 'params_3dmm.pkl'), 'rb')
    inf = pickle.load(f)
    num = len(ids)
    params = []
    for ii in range(num):
        params.append(np.array(inf['params'][ids[ii]]))
    return np.stack(params)

def obtain_seq_index(index, num_frames):
    seq = list(range(index-13, index+13+1))
    seq = [ min(max(item, 0), num_frames-1) for item in seq ]
    return seq

def load_ad_params(para_path, num_ids=None):
    params_dict = torch.load(os.path.join(para_path, 'track_params.pt'))
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    exps = params_dict['exp']
    params = torch.cat([exps, euler_angle, trans], dim=1)
    if num_ids is not None:
        params = params[num_ids]
    return params.numpy()

random.seed(10)
torch.manual_seed(10)


class FP_WildData_ADmixexp(torch.utils.data.Dataset):
    """
    Dataset from FaceScape
    """
    def __init__(
        self,
        path,
        stage="train",
        list_prefix="mixwild",
        image_size=(256, 256),  # w, h 
        load_img_folder="images_masked",
        load_para_folder="images_3dmm",
        n_view_in=3,
        with_mask=False,
        change_tar_id=False,
        sem_win=1,
        type_exp="tracking", # "tracking" or "3dmm"
    ):
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)
        self.stage = stage
        self.with_mask = with_mask
        self.change_tar_id = change_tar_id
        self.image_size = image_size
        self.load_img_folder = load_img_folder
        self.load_para_folder = load_para_folder
        self.parsing_folder = 'parsing'
        self.n_view_in = n_view_in
        self.lindisp = False
        self.z_near, self.z_far = 8, 26
        self.type_exp = type_exp

        self.use_num_id = False
        self.use_near_3dmm = True
        self.random_select = False
        if self.stage == 'test' or self.stage == 'val':
            self.random_select = True
        if sem_win > 1:
            self.use_near_3dmm = True

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

        self.build_metas(file_list, stage)
        self.define_transforms()

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

    def build_metas(self, scan_list_file, stage):
        self.metas = []
        self.valid_img_ids = {}
        with open(scan_list_file, "r") as f:
            self.scans = sorted([line.rstrip() for line in f.readlines()])
        for scan in self.scans:
            expressions = [
                x for x in os.listdir(os.path.join(self.base_path, scan))
                if os.path.isdir(os.path.join(self.base_path, scan, x))
            ]
            expressions = ['mixexp']

            exp = expressions[0]
            view_path = os.path.join(self.base_path, scan, exp,
                                     self.load_img_folder)
            para_path = os.path.join(self.base_path, scan, exp,
                                     self.load_para_folder)
            
            with open(os.path.join(para_path, 'valid_img_ids.txt'), "r") as f:
                valid_ids = sorted([line.rstrip() for line in f.readlines()])
            self.valid_img_ids[scan] = valid_ids
            poses, focal, c, nf, img_files = face_pose_reading_from_ids(
                os.path.join(para_path, 'face_transforms_pose.json'),
                view_path, valid_ids)
            num_views = len(img_files)

            if not self.random_select:
                w2cs = []
                for view_id in range(num_views):
                    c2w = np.eye(4, dtype=np.float32)
                    c2w = np.array(poses[view_id])
                    w2c = np.linalg.inv(c2w)
                    w2cs.append(w2c)
                w2cs = np.stack(w2cs, 0)

            # Let each view could be as a ref-view OR skip
            skip_num = 3
            for ii in range(0, num_views, skip_num):
                tar_id = valid_ids[ii]
                tar_num_id = int(ii)
                if not self.random_select:
                    w2c_ref = w2cs[ii]
                    angles_tuple = []
                    for jj in range(num_views):
                        if jj == ii:
                            continue
                        else:
                            w2c_cdd = w2cs[jj]
                            angle = np.arccos(np.dot(w2c_ref[:, 2], w2c_cdd[:, 2]))
                            angles_tuple.append((angle, jj))
                    # angle small --> large
                    angle_sorted = sorted(angles_tuple, key=lambda t: t[0])
                    n_candi = len(angle_sorted)
                    if n_candi < 20:
                        continue
                    middle_candi = int(n_candi/2)
                    # src_num_ids = [angle_sorted[x][1] for x in range(5, 20)]
                    # src_num_ids = [angle_sorted[x][1] for x in range(3, 7)]+[angle_sorted[x][1] for x in range(n_candi-4, n_candi)]+[angle_sorted[x][1] for x in range(middle_candi-2, middle_candi+2)]
                    id_list = [int(i/self.n_view_in * n_candi) for i in range(self.n_view_in)]
                    src_num_ids = [angle_sorted[x][1] for x in id_list]
                    src_ids = [valid_ids[x] for x in src_num_ids]
                else:
                    other_valid_ids = sorted(list(set(valid_ids) - {tar_id}))
                    src_ids = random.sample(other_valid_ids, self.n_view_in)
                    other_valid_num_ids = sorted(list(set(range(num_views)) - {tar_num_id}))
                    src_num_ids = random.sample(other_valid_num_ids, self.n_view_in)

                if self.use_num_id:
                    self.metas += [(scan, exp, num_views, tar_num_id, src_num_ids)]
                else:
                    self.metas += [(scan, exp, num_views, tar_id, src_ids)]
        random.shuffle(self.metas)
        # print("Meta0: ", self.metas[0])
        # print("Number of metas: ", len(self.metas))

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        if self.stage == 'test' or self.stage == 'val':
            ids = [i for i in range(self.n_view_in)]
        else:
            ids = torch.randperm(12)[:self.n_view_in]
        if self.use_num_id:
            scan, exp, num_views, tar_num_id, src_num_ids = self.metas[index]
            view_num_ids = [src_num_ids[i] for i in ids] + [tar_num_id]
        else:
            scan, exp, num_views, tar_id, src_ids = self.metas[index]
            view_ids = [src_ids[i] for i in ids] + [tar_id]

        view_path = os.path.join(self.base_path, scan, exp,
                                 self.load_img_folder)
        para_path = os.path.join(self.base_path, scan, exp,
                                 self.load_para_folder)
        valid_ids = self.valid_img_ids[scan]
        num_valid_ids = len(valid_ids)

        ### load img & parameters
        # in 2D editing, the final target exp is set as the ref_exp
        # load camera parameters (multiexp shares this para)
        if self.use_num_id:
            view_ids = [valid_ids[i] for i in view_num_ids]
        else:
            view_num_ids = [valid_ids.index(x) for x in view_ids]
            
        poses, f, cxy, nf, rgb_paths = face_pose_reading_from_ids(
            os.path.join(para_path, 'face_transforms_pose.json'), view_path,
            view_ids)
        if self.type_exp == "tracking":
            if self.use_near_3dmm:
                param = load_ad_params(para_path)
            else:
                param = load_ad_params(para_path, view_num_ids)
        else:
            if self.use_near_3dmm:
                param = param_reading_3dmm_pkl(para_path, valid_ids)
            else:
                param = param_reading_3dmm_pkl(para_path, view_ids)

        imgs_in, poses_in, nfs_in = [], [], []
        para_3dmm_in = []
        focal, c = [], []
        masks = []
        for i, vid in enumerate(view_ids):
            # load images
            img_file_in = rgb_paths[i]
            img_in = Image.open(img_file_in)
            parsing_file = img_file_in.replace(self.load_img_folder, self.parsing_folder)
            if os.path.exists(parsing_file) and self.with_mask:
                parse_img = Image.open(parsing_file)
            else:
                self.with_mask = False
            w, h = img_in.size
            scale = 1.0
            if img_in.size != self.image_size:
                scale = self.image_size[0] / img_in.size[0]
                img_in = img_in.resize(self.image_size, Image.BILINEAR)
                if self.with_mask:
                    parse_img = parse_img.resize(self.image_size, Image.BILINEAR)
            img_in_tensor = self.transform(img_in)
            imgs_in.append(img_in_tensor)
            if self.with_mask:
                parse_img =np.array(parse_img)
                bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
                mask = 1-bg
                masks.append(torch.tensor(mask))

            # load camera parameters
            pose = np.eye(4, dtype=np.float32)
            pose = poses[i]
            f0 = f
            nf0 = nf
            pose = torch.tensor(pose, dtype=torch.float32)
            fx = torch.tensor(f0) * scale
            fy = torch.tensor(f0) * scale
            cx = torch.tensor(w / 2) * scale
            cy = torch.tensor(h / 2) * scale
            near, far = nf0[0], nf0[1]
            poses_in.append(pose)
            focal.append(torch.tensor((fx, fy), dtype=torch.float32))
            c.append(torch.tensor((cx, cy), dtype=torch.float32))
            nfs_in.append(torch.tensor((near, far), dtype=torch.float32))
            
            if self.use_near_3dmm:
                img_num_id = view_num_ids[i]
                seq_near = obtain_seq_index(img_num_id, num_valid_ids)
                para_3dmm_in.append(param[seq_near].transpose(1,0))
            else:
                para_3dmm_in.append(param[i])  
        imgs_in = torch.stack(imgs_in).float()
        if self.with_mask:
            masks = torch.stack(masks)
        poses_in = torch.stack(poses_in)
        focal = torch.stack(focal)
        c = torch.stack(c)
        nfs_in = torch.stack(nfs_in)

        # process exp parameters, the exp of reference view is used as cdn_exp
        if self.type_exp == "tracking":
            semantic_in = np.stack(para_3dmm_in)
            semantic_src = semantic_in[:self.n_view_in]
            exp_part = semantic_in[:, :79]
            pose_part = semantic_in[:, 79:]
            exp_part_cdn = exp_part[-1:].repeat(semantic_in.shape[0], axis=0)
            semantic_cdn0 = np.concatenate(
                (exp_part_cdn, pose_part), axis=1)
        else:
            para_3dmm_in = np.stack(para_3dmm_in)
            exp_part = para_3dmm_in[:, 80:144]
            angle_part = para_3dmm_in[:, 224:227]
            trans_part = para_3dmm_in[:, 254:257]
            if para_3dmm_in.shape[1] > 257:
                crops_part = para_3dmm_in[:, -3:]
            else:
                crops_part = np.zeros_like(trans_part)
            semantic_in = np.concatenate(
                (exp_part, angle_part, trans_part, crops_part), axis=1)
            # 1st\2nd\3rd are as the source images
            semantic_src = semantic_in[:self.n_view_in]
            exp_part_cdn = exp_part[-1:].repeat(semantic_in.shape[0], axis=0)
            semantic_cdn0 = np.concatenate(
                (exp_part_cdn, angle_part, trans_part, crops_part), axis=1)
        semantic_cdn = semantic_cdn0[:self.n_view_in]
        semantic_tar = semantic_in[-1:].repeat(self.n_view_in, axis=0)
        if self.use_near_3dmm:
            semantic_src = torch.Tensor(semantic_src)
            semantic_cdn = torch.Tensor(semantic_cdn)
            semantic_tar = torch.Tensor(semantic_tar)
        else:
            semantic_src = torch.Tensor(semantic_src)[:, :, None].expand(-1, -1, 27)
            semantic_cdn = torch.Tensor(semantic_cdn)[:, :, None].expand(-1, -1, 27)
            semantic_tar = torch.Tensor(semantic_tar)[:, :, None].expand(-1, -1, 27)

        result = {
            "scan":
            scan,
            "img_id":
            view_ids,
            "images":
            imgs_in,
            "masks":
            masks,
            "poses":
            poses_in,
            "focal":
            focal,
            "c":
            c,
            "nfs":
            nfs_in,
            "semantic_src":
            semantic_src,
            "semantic_cdn":
            semantic_cdn,
            "semantic_tar":
            semantic_tar
        }

        return result
