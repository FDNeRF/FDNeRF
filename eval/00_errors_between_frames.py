import numpy as np
from PIL import Image
import imageio
import cv2, os, glob


# ---------- calculate error
# data_path = 'results/0514_results_video_driven/visual_video_driven_other_indep/video_driven_id10283#j8UugkSTzzk#000812#001170_id10292#ENIHEvg_VLM#009003#009125'
# savepath = 'results/0519_driven_error/FNNeRF_win/video_driven_id10283#j8UugkSTzzk#000812#001170_id10292#ENIHEvg_VLM#009003#009125'
# os.makedirs(savepath, exist_ok=True)
# imgs_list = sorted([
#             x for x in glob.glob(os.path.join(data_path, "*"))
#             if x.endswith("_pre.png")
#         ])
# for i in range(0, len(imgs_list)-1):
#     img_path1 = imgs_list[i]
#     img_path2 = imgs_list[i+1]
#     name1 = img_path1.split('/')[-1].split('.')[0]
#     name2 = img_path2.split('/')[-1].split('.')[0]

#     img1 = cv2.imread(img_path1)
#     img2 = cv2.imread(img_path2)
#     err = cv2.absdiff(img1,img2)
#     ma = err.max()
#     mi = err.min()
#     err = (err - mi)/(ma - mi) * 255.

#     err_name = f'Err_{name1}_{name2}.png'

#     cv2.imwrite(os.path.join(savepath, err_name), err.astype(np.uint8))

data_path = '/home/zhangjingbo/Codes/pixelnerf-portrait-implicit/results/0516_results_video_driven(nowin)/visual_video_driven_other_indep(nowin)/video_driven_id10291#uiBjIKX_0l8#000067#000277_id10282#neQO6_CUY4w#002906#003048'
savepath = data_path
imgs_pre_list = sorted([
            x for x in glob.glob(os.path.join(data_path, "*"))
            if x.endswith("_pre.png")
        ])
imgs_gt_list = sorted([
            x for x in glob.glob(os.path.join(data_path, "*"))
            if x.endswith("_gt.png")
        ])
frames_pre, frames_gt = [], []
for i in range(0, len(imgs_pre_list)):
    img_gt = imageio.imread(imgs_gt_list[i])
    img_pre = imageio.imread(imgs_pre_list[i])
    frames_gt.append(img_gt)
    frames_pre.append(img_pre)
frames_gt = np.stack(frames_gt)
frames_pre = np.stack(frames_pre)

name_gt='video_gt.mp4'
name_pre='video_pre.mp4'
imageio.mimwrite(os.path.join(savepath, name_gt), frames_gt.astype(np.uint8),
                        fps=30,
                        quality=8)
imageio.mimwrite(os.path.join(savepath, name_pre), frames_pre.astype(np.uint8),
                        fps=30,
                        quality=8)



