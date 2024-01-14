"""
Author: Eckert ZHANG
Date: 2022-02-20 12:14:04
LastEditTime: 2022-03-18 11:14:35
LastEditors: Eckert ZHANG
Description: 
"""
# -*- encoding: utf-8 -*-
import cv2
import numpy as np


def vis_parsing_maps(h,
                     w,
                     im,
                     parsing_anno,
                     stride,
                     save_path='./',
                     save_im=False):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                   [150, 80, 0], [170, 120, 65], [220, 180, 210],
                   [255, 125, 125], [200, 100, 100], [215, 175, 125],
                   [125, 125, 125], [255, 150, 0], [255, 255,
                                                    0], [0, 255, 255],
                   [255, 225, 120], [125, 125, 255], [0, 255, 0], [0, 0, 255],
                   [0, 150, 80]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # vis_parsing_anno[vis_parsing_anno == 18] = 0
    # vis_parsing_anno[vis_parsing_anno == 17] = 0
    # vis_parsing_anno[vis_parsing_anno == 14] = 0
    # vis_parsing_anno[vis_parsing_anno == 15] = 0
    # vis_parsing_anno[vis_parsing_anno == 16] = 0
    # vis_parsing_anno[vis_parsing_anno == 4] = 0

    vis_parsing_anno = cv2.resize(vis_parsing_anno,
                                  None,
                                  fx=stride,
                                  fy=stride,
                                  interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # 新增的一段，用于纠正错误的左右眉毛和眼睛
    index_nose = np.where(vis_parsing_anno == 10)
    index_lefteb = np.where(vis_parsing_anno == 2)
    index_righteb = np.where(vis_parsing_anno == 3)
    index_lefteye = np.where(vis_parsing_anno == 4)
    index_righteye = np.where(vis_parsing_anno == 5)
    index_leftear = np.where(vis_parsing_anno == 7)
    index_rightear = np.where(vis_parsing_anno == 8)

    nose_x = np.mean(index_nose[1])
    if index_lefteb:
        ind_false = np.where(index_lefteb[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteb[0][ind_false],
                             index_lefteb[1][ind_false]] = 3

    if index_righteb:
        ind_false = np.where(index_righteb[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteb[0][ind_false],
                             index_righteb[1][ind_false]] = 2

    if index_lefteye:
        ind_false = np.where(index_lefteye[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteye[0][ind_false],
                             index_lefteye[1][ind_false]] = 5

    if index_righteye:
        ind_false = np.where(index_righteye[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteye[0][ind_false],
                             index_righteye[1][ind_false]] = 4

    if index_leftear:
        ind_false = np.where(index_leftear[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_leftear[0][ind_false],
                             index_leftear[1][ind_false]] = 8

    if index_rightear:
        ind_false = np.where(index_rightear[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_rightear[0][ind_false],
                             index_rightear[1][ind_false]] = 7

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    vis_im = vis_parsing_anno_color
    # Save result or not
    if save_im:
        vis_im = cv2.resize(vis_im, (w, h))
        cv2.imwrite(save_path, vis_im[:, :, ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return vis_im
