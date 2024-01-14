"""
Author: Eckert ZHANG
Date: 2022-02-20 09:47:29
LastEditTime: 2022-03-23 15:46:28
LastEditors: Eckert ZHANG
Description: 
"""
import argparse
import os, sys
import torch
import torchvision.transforms as transforms
import imageio, glob, cv2
import numpy as np

from AlignmentCode.wild_fit_base import fitting
from SegmentCode.model import BiSeNet
from SegmentCode.get_pair_parsing import vis_parsing_maps

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project(lm3d, pose, debug=False, save_path='./', img_name='img_ldm_test'):
    K = np.array([[1200, 0, 256], [0, 1200, 256], [0, 0, 1]])
    # get Rt
    Rt = np.eye(4)
    M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Rt[:3, :3] = pose[:3, :3].T  # pose[:3,:3] --> RT
    Rt[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])  # T = R.dot(C)

    # project 3d to 2d
    lm2d = K @ Rt[:3, :] @ (np.concatenate(
        [lm3d, np.ones([lm3d.shape[0], 1])], 1).T)
    lm2d_half = lm2d // lm2d[2, :]
    lm2d = np.round(lm2d_half).astype(
        np.long)[:2, :].T @ M[:2, :2]  # .T[:,:2]  #[68,2]

    lm2d[:, 1] = 512 + lm2d[:, 1]
    # print(lm2d.max(), lm2d.min())
    if debug == True:
        img = np.zeros([256, 256, 3])
        lm2d_view = lm2d.astype(np.long) // 2
        img[lm2d_view[:, 0], lm2d_view[:, 1], :] = np.ones([3])

        imageio.imsave(os.path.join(save_path, f'{img_name}.png'),
                       img.astype(np.int8))
    return lm2d


def main(args,
         folder_name_in='01_images_colmap',
         folder_name_out='images_align',
         folder_name_out_msk='images_masked'):

    # folders = sorted([
    #         x for x in os.listdir(args.datapath)
    #         if os.path.isdir(os.path.join(args.datapath, x))
    #     ])
    folders = ['eckert']

    size_tar = args.resolution_tar  # 512
    dshape = [size_tar, size_tar, 3]
    fitter = fitting(
        lm_file=
        "./src/data_process/AlignmentCode/shape_predictor_68_face_landmarks.dat"
    )

    # masked config
    n_classes = 19
    model_path = './src/data_process/SegmentCode/Seg_79999_iter.pth'
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    CLOTHES_CORLOR = np.array([0, 255, 0]).astype(np.uint8)
    BG_COLOR = np.array([0, 0, 0]).astype(np.uint8)
    color_list = [CLOTHES_CORLOR, BG_COLOR]

    for folder in folders:
        print(f'\n-------- Start to process {folder} ---------')
        root_dir = os.path.join(args.datapath, folder)
        save_dir = os.path.join(args.savepath, folder)
        if not os.path.exists(root_dir):
            print(f'The data path ({root_dir}) is NOT existing!')
            continue
        os.makedirs(save_dir, exist_ok=True)
        exps = sorted([
            x for x in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, x))
        ])
        '''
        ## ========== Pre-checking (To avoid no face cases) ===========
        # img_n_del_angle = []
        # print('***** checking:')
        # for exp in ['1_neutral']:
        #     img_paths = sorted([
        #         x for x in glob.glob(
        #             os.path.join(root_dir, exp, folder_name_in, '*'))
        #         if (x.endswith('.jpg') or x.endswith('.png'))
        #     ])
        #     for ii in range(len(img_paths)):
        #         img = imageio.imread(img_paths[ii])
        #         img_name_origi = img_paths[ii].split('/')[-1]
        #         if img_name_origi in img_n_del_angle:
        #             continue
        #         try:
        #             faces = fitter.detector(img, 1)
        #         except:
        #             img_n_del_angle.append(img_name_origi)
        #             continue
        #         if len(faces) < 1:
        #             img_n_del_angle.append(img_name_origi)
        #             continue
        #         pts = fitter.face_pred(img, faces[0])
        #         kp2d_raw = np.array(([[p.x, p.y] for p in pts.parts()]))
        #         M, scale = fitter.transformation_from_points(
        #             src_points=kp2d_raw, tmpt_points=fitter.tmpLM)
        #         out = fitter.warp_im(img, M, dshape)
        #         try:
        #             faces = fitter.detector(out, 1)
        #         except:
        #             img_n_del_angle.append(img_name_origi)
        #             continue
        #         if len(faces) < 1:
        #             img_n_del_angle.append(img_name_origi)
        #             continue
        # print('***** checking finish!')
        '''

        ## ========== alignment & pose-estimated (based on neural exp) ===========
        n_exp = 1
        for exp in exps:
            imgs_path = os.path.join(root_dir, exp, folder_name_in)
            img_save_path = os.path.join(save_dir, exp, folder_name_out)
            os.makedirs(img_save_path, exist_ok=True)
            mask_face_save_path = os.path.join(save_dir, exp, 'masks_face')
            os.makedirs(mask_face_save_path, exist_ok=True)
            parse_save_path = os.path.join(save_dir, exp, 'images_parsing')
            os.makedirs(parse_save_path, exist_ok=True)
            mask_save_path = os.path.join(save_dir, exp, 'masks')
            os.makedirs(mask_save_path, exist_ok=True)
            img_masked_save_path = os.path.join(save_dir, exp,
                                                folder_name_out_msk)
            os.makedirs(img_masked_save_path, exist_ok=True)

            img_list = sorted([
                x for x in glob.glob(os.path.join(imgs_path, "*"))
                if (x.endswith(".jpg") or x.endswith(".png"))
            ])
            face_poses = []
            num = 0
            for img in img_list:
                img_name = img.split('/')[-1]
                img = cv2.imread(img)

                # coarse alignment
                try:
                    kp2d, img_scaled, M1 = fitter.detect_kp2d(
                        img,
                        is_show_img=False,
                        dshape=dshape,
                    )
                except:
                    continue

                # pose-estimating (detect?可视化)
                pos, trans = fitter.get_pose_from_kp2d(kp2d)

                # keypoint-tuning
                lm3d_tmplate = fitter.fcFitter.tmpLM.copy()
                debug_path = os.path.join(save_dir, exp, 'ldm_visual')
                os.makedirs(debug_path, exist_ok=True)
                lm2d_tmplate = project(lm3d_tmplate,
                                       pos,
                                       debug=False,
                                       save_path=debug_path,
                                       img_name=img_name)
                try:
                    kp2d, img_scaled, M2 = fitter.detect_kp2d(
                        cv2.cvtColor(img_scaled, cv2.COLOR_RGB2BGR),
                        tar_kp=lm2d_tmplate,
                        is_show_img=False,
                        dshape=dshape,
                    )
                except:
                    continue
                face_poses.append(np.array(pos, dtype=np.float32))
                imageio.imsave(
                    os.path.join(img_save_path,
                                 '%02d_%05d.png' % (n_exp, num)), img_scaled)
                faces = fitter.detector(img_scaled, 1)
                img_noface = np.ones_like(img_scaled) * 255
                l, t, r, b = faces[0].left(), faces[0].top(), faces[0].right(
                ), faces[0].bottom()
                img_noface[t:b, l:r, :] = 0

                # debug
                if True:
                    img_with_lm = img_scaled.copy()
                    lm2d_view = lm2d_tmplate.astype(np.long)
                    img_with_lm[lm2d_view[:, 0],
                                lm2d_view[:, 1], :] = np.ones([3])
                    img_with_lm[lm2d_view[:, 0] - 1,
                                lm2d_view[:, 1], :] = np.ones([3])
                    img_with_lm[lm2d_view[:, 0],
                                lm2d_view[:, 1] - 1, :] = np.ones([3])
                    img_with_lm[lm2d_view[:, 0] + 1,
                                lm2d_view[:, 1], :] = np.ones([3])
                    img_with_lm[lm2d_view[:, 0],
                                lm2d_view[:, 1] + 1, :] = np.ones([3])
                    imageio.imsave(
                        os.path.join(debug_path,
                                     '%02d_%05d.png' % (n_exp, num)),
                        img_with_lm)

                ## ========= Parsing & masked =========
                with torch.no_grad():
                    img_tensor = to_tensor(img_scaled)
                    img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
                    out = net(img_tensor)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                img_parsing = vis_parsing_maps(dshape[0],
                                               dshape[1],
                                               img_scaled,
                                               parsing,
                                               stride=1,
                                               save_im=False)
                imageio.imsave(
                    os.path.join(parse_save_path,
                                 '%02d_%05d.png' % (n_exp, num)), img_parsing)

                mask = (np.ones_like(img_parsing) * 255).astype(np.uint8)
                mask[450:, ...] = 0
                for color in color_list:
                    index = np.where(np.all(img_parsing == color, axis=-1))
                    mask[index[0], index[1]] = 0
                img_masked = np.bitwise_and(mask, img_scaled)
                imageio.imsave(
                    os.path.join(mask_save_path,
                                 '%02d_%05d.png' % (n_exp, num)), mask)
                imageio.imsave(
                    os.path.join(img_masked_save_path,
                                 '%02d_%05d.png' % (n_exp, num)), img_masked)
                # img_noface = np.bitwise_and(mask, img_noface)
                imageio.imsave(
                    os.path.join(mask_face_save_path,
                                 '%02d_%05d.png' % (n_exp, num)), img_noface)
                num += 1
            n_exp += 1
            face_poses = np.array(np.stack(face_poses))
            pose_save_path = os.path.join(img_masked_save_path,
                                          'poses_face.npy')
            np.save(pose_save_path, face_poses)
            print(f'Finish exp [{exp}] !')
    print('Finish model', folder, '!')


if __name__ == '__main__':
    """
    Extracts and aligns all faces from images, estimate the face pose for each image
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",
                        type=str,
                        default='/home/zhangjingbo/Datasets/NeRF_data/ours')
    parser.add_argument("--savepath",
                        type=str,
                        default='/home/zhangjingbo/Datasets/NeRF_data/ours'
                        )  #/data/zhangjingbo/FaceScape_rendered/wild
    parser.add_argument("--resolution_tar", type=int, default=512)
    args = parser.parse_args()

    main(args)