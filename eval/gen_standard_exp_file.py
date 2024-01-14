import sys, os,pdb
import random

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F
import numpy as np
import imageio, json
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
from torchvision import transforms as T
from PIL import Image
import tqdm
from dotmap import DotMap

def load_ad_params(para_path, num_ids=None):
    params_dict = torch.load(os.path.join(para_path, 'track_params.pt'))
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    exps = params_dict['exp']
    params = torch.cat([exps, euler_angle, trans], dim=1)
    if num_ids is not None:
        params = params[num_ids]
    return params.numpy()

exps_dic = {}
# id10291#4aLg_keiGHw#001631#001901
# neutral
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10283#j8UugkSTzzk#001372#002396/mixexp/images_3dmm'
num_id_tar = 31  # 31, 1052
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['neutral'] = exp

# smile
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#001631#001901/mixexp/images_3dmm'
num_id_tar = 211  # 38 
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['smile'] = exp

# laugh
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#001631#001901/mixexp/images_3dmm'
num_id_tar = 54
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['laugh'] = exp

# mouth_stretch: 张嘴
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10283#j8UugkSTzzk#001372#002396/mixexp/images_3dmm'
num_id_tar = 430 # 944 #430 602 819
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['mouth_stretch'] = exp

# eye_closed id10291#4aLg_keiGHw#000101#000460 253
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#001217#001424/mixexp/images_3dmm'
num_id_tar = 8   # 8 9
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['eye_closed'] = exp

sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#001217#001424/mixexp/images_3dmm'
num_id_tar = 172
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['eye_closed_open_mouth'] = exp

# lip_funneler: 窝嘴
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#000101#000460/mixexp/images_3dmm'
num_id_tar = 12
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['lip_funneler'] = exp

# angle
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#uiBjIKX_0l8#001197#001806/mixexp/images_3dmm'
num_id_tar = 164
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['angle'] = exp

# eye_closed_mouse_pout
sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#000101#000460/mixexp/images_3dmm'
num_id_tar = 25
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['eye_closed_mouse_pout'] = exp

sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#kt4P4cyTpWQ#005725#005977/mixexp/images_3dmm'
num_id_tar = 61
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['minzui'] = exp

sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#000101#000460/mixexp/images_3dmm'
num_id_tar = 78
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['juezui'] = exp

sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#000101#000460/mixexp/images_3dmm'
num_id_tar = 293
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['biyanjuezui'] = exp

sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#4aLg_keiGHw#000101#000460/mixexp/images_3dmm'
num_id_tar = 74
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['waizui'] = exp

sample_path = '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test/id10291#kt4P4cyTpWQ#005725#005977/mixexp/images_3dmm'
num_id_tar = 3
param = load_ad_params(sample_path, num_id_tar)
exp = param[:79]
exps_dic['Ozui'] = exp

np.save('./data/std_exps.npy', exps_dic)