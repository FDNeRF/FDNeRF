"""
Author: Eckert ZHANG
Date: 2022-02-20 09:47:29
LastEditTime: 2022-02-22 21:56:47
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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
         folder_name_in='images_standard',
         folder_name_out='images_align',
         folder_name_out_msk='images_masked'):
    # total 359
    folders = [str(x + 1) for x in range(0, 25)]
    # folders = ['1']
    dshape = [512, 512, 3]

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
        # exps = [
        #     '1_neutral',
        # ]

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

        ## ========== alignment & pose-estimated (based on neural exp) ===========
        exp_neural = '1_neutral'
        imgs_neural_path = os.path.join(root_dir, exp_neural, folder_name_in)
        img_neural_list = sorted([
            x for x in glob.glob(os.path.join(imgs_neural_path, "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ])
        poses = []
        num = 0
        for img_neural in img_neural_list:
            img_name = img_neural.split('/')[-1]
            # if img_name in img_n_del_angle:
            #     continue
            img = cv2.imread(img_neural)

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
            debug_path = os.path.join(save_dir, exp_neural, 'debug_visual')
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
            poses.append(np.array(pos, dtype=np.float32))

            debug = True
            if debug:
                img_with_lm = img_scaled.copy()
                lm2d_view = lm2d_tmplate.astype(np.long)
                img_with_lm[lm2d_view[:, 0], lm2d_view[:, 1], :] = np.ones([3])
                img_with_lm[lm2d_view[:, 0] - 1,
                            lm2d_view[:, 1], :] = np.ones([3])
                img_with_lm[lm2d_view[:, 0],
                            lm2d_view[:, 1] - 1, :] = np.ones([3])
                img_with_lm[lm2d_view[:, 0] + 1,
                            lm2d_view[:, 1], :] = np.ones([3])
                img_with_lm[lm2d_view[:, 0],
                            lm2d_view[:, 1] + 1, :] = np.ones([3])
                imageio.imsave(os.path.join(debug_path, '%05d.png' % num),
                               img_with_lm)

            for exp in exps:
                img_save_path = os.path.join(save_dir, exp, folder_name_out)
                os.makedirs(img_save_path, exist_ok=True)
                img_exp_path = os.path.join(root_dir, exp, folder_name_in,
                                            img_name)
                img_exp = cv2.imread(img_exp_path)
                img_exp_aligned = fitter.warp_im(img_exp, M1, dshape)
                img_exp_aligned = fitter.warp_im(img_exp_aligned, M2, dshape)
                img_exp_aligned = cv2.cvtColor(img_exp_aligned,
                                               cv2.COLOR_BGR2RGB)
                imageio.imsave(os.path.join(img_save_path, '%05d.png' % num),
                               img_exp_aligned)

                ## ========= Parsing & masked =========
                with torch.no_grad():
                    img_tensor = to_tensor(img_exp_aligned)
                    img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
                    out = net(img_tensor)[0]
                    parse_save_path = os.path.join(save_dir, exp,
                                                   'images_parsing')
                    os.makedirs(parse_save_path, exist_ok=True)
                    parsing = out.squeeze(0).cpu().numpy().argmax(0)
                    img_parsing = vis_parsing_maps(dshape[0],
                                                   dshape[1],
                                                   img_exp_aligned,
                                                   parsing,
                                                   stride=1,
                                                   save_path=os.path.join(
                                                       parse_save_path,
                                                       '%05d.png' % num),
                                                   save_im=False)
                    imageio.imsave(
                        os.path.join(parse_save_path, '%05d.png' % num),
                        img_parsing)

                    mask = (np.ones_like(img_parsing) * 255).astype(np.uint8)
                    mask[450:, ...] = 0
                    for color in color_list:
                        index = np.where(np.all(img_parsing == color, axis=-1))
                        mask[index[0], index[1]] = 0
                    img_masked = np.bitwise_and(mask, img_exp_aligned)
                    mask_save_path = os.path.join(save_dir, exp, 'masks')
                    os.makedirs(mask_save_path, exist_ok=True)
                    img_masked_save_path = os.path.join(
                        save_dir, exp, folder_name_out_msk)
                    os.makedirs(img_masked_save_path, exist_ok=True)
                    imageio.imsave(
                        os.path.join(mask_save_path, '%05d.png' % num), mask)
                    imageio.imsave(
                        os.path.join(img_masked_save_path, '%05d.png' % num),
                        img_masked)
            num += 1
        poses = np.array(np.stack(poses))
        for exp in exps:
            pose_save_path = os.path.join(save_dir, exp, folder_name_out_msk,
                                          'poses_face.npy')
            np.save(pose_save_path, poses)
    print('Finish model', folder, '!')


if __name__ == '__main__':
    """
    Extracts and aligns all faces from images, estimate the face pose for each image
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",
                        type=str,
                        default='/data/zhangjingbo/FaceScape_new/images')
    parser.add_argument("--savepath",
                        type=str,
                        default='/data/zhangjingbo/FaceScape_align3/images')
    args = parser.parse_args()

    main(args)