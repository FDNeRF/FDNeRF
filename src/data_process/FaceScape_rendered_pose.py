"""
Author: Eckert ZHANG
Date: 2022-02-20 09:47:29
LastEditTime: 2022-03-02 11:57:37
LastEditors: Eckert ZHANG
Description: 
"""
import json
import os, sys
import torch
import torchvision.transforms as transforms
import imageio, glob, cv2
import numpy as np

base_path = '/data/zhangjingbo/FaceScape_rendered'
stage = 'train'

data_path = os.path.join(base_path, stage, 'multi-view-images')
scans = sorted([
    x for x in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, x))
])
# scans = ['14']
for scan in scans:
    json_file = os.path.join(data_path, f'transforms_all_{scan}.json')
    expressions = sorted([
        x for x in os.listdir(os.path.join(data_path, scan))
        if os.path.isdir(os.path.join(data_path, scan, x))
    ])
    # expressions = [expressions[0]]

    with open(json_file, 'r') as fp:
        meta_data = json.load(fp)
    camera_angle_x = float(meta_data['camera_angle_x'])

    for exp in expressions:
        dis = {}
        dis['camera_angle_x'] = camera_angle_x
        frames_n = {}
        view_path = os.path.join(data_path, scan, exp)
        img_files = sorted([
            x for x in glob.glob(os.path.join(view_path, "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ])
        num = 0
        for img in img_files:
            img_name = img.split('/')[-1].split('.')[0]
            file_path_id = f'/{scan}/{exp}/{img_name}'
            for frame in meta_data['frames']:
                if frame['file_path'] != file_path_id:
                    continue
                else:
                    num += 1
                    frames_n[f'{img_name}_pose'] = frame['transform_matrix']
        assert len(img_files) == num
        print('Num:', num)
        dis['frames'] = frames_n
        para = json.dumps(dis, indent=2)
        f = open(os.path.join(view_path, 'transform_matrix.json'), 'w')
        f.write(para)
        f.close()
        print(f'Finish {scan}/{exp}!')
