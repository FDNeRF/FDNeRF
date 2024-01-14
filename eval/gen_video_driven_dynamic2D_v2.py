"""
Author: Eckert ZHANG
Date: 2021-11-04 21:49:57
LastEditTime: 2022-03-18 12:43:10
LastEditors: Eckert ZHANG
Description: 
"""
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


def set_seed(seed, base=0, is_set=True):
    seed += base
    assert seed >= 0, '{} >= {}'.format(seed, 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extra_args(parser):
    parser.add_argument(
        "--subset",
        "-S",
        type=int,
        default=0,
        help="Subset in data to use",)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Video scale relative to input size",)
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help=
        "Distance of camera from origin, default is average of z_far, z_near of dataset",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    parser.add_argument(
        "--ckpt_warp2D",
        type=str,
        default=
        'checkpoints/04_PIRender_rendered/epoch_00001_iteration_000400000_checkpoint.pt',
    )
    parser.add_argument(
        "--nviews",
        type=int,
        default=3,
        help=
        "Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--ids_src",
        default=None,
        nargs='+',
    )
    parser.add_argument(
        "--src_name",
        type=str,
        default='id10280#NXjT3732Ekg#001093#001192')
    parser.add_argument(
        "--tar_name",
        type=str,
        default=None)
    
    parser.add_argument(
        "--only_exp_driven",
        action="store_true",
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--method_gen_pose",
        type=str,
        default='standard')
    parser.add_argument(
        "--average_semantic",
        action="store_true",
    )
    parser.add_argument(
        "--fix_tar_pose",
        action="store_true",
    )
    parser.add_argument(
        "--semantic_window",
        type=int,
        default=1,
    )
    return parser


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

def gen_render_poses(near, far, num, method, poses=None):
    '''
    method: 'spiral' or 'spherical' or 'composite'
    '''
    if method == 'spiral':
        print("Computes target poses following a spiral path")
        render_poses = util.get_spiral(poses, [near, far],
                                    rads_scale=0.8,
                                    N_views=num)
    elif method == 'spherical':
        print("Computes target poses following a spherical path")
        render_poses = util.get_spherical_pose(poses, [near, far],
                                    rads_scale=0.8,
                                    N_views=num)
    elif method == 'composite':
        print("Computes target poses following a spherical and spiral path")
        render_poses = util.get_composite_pose(poses, [near, far],
                                    rads_scale=0.8,
                                    N_views=num)
    elif method == 'standard':
        print("Computes target poses following a standard path")
        render_poses = util.get_standard_pose(N_views=num)
    return render_poses

def load_ad_params(para_path, num_ids=None):
    params_dict = torch.load(os.path.join(para_path, 'track_params.pt'))
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    exps = params_dict['exp']
    params = torch.cat([exps, euler_angle, trans], dim=1)
    if num_ids is not None:
        params = params[num_ids]
    return params.numpy()

def obtain_seq_index(index, num_frames):
    seq = list(range(index-13, index+13+1))
    seq = [ min(max(item, 0), num_frames-1) for item in seq ]
    return seq

def load_driven_data(root_path, src_name, tar_name, nviews_in, ids_src=None, sem_win=1, image_size=(256, 256)):
    use_near_3dmm_src = False
    use_near_3dmm_tar = False
    if sem_win>1:
        use_near_3dmm_src = True
        use_near_3dmm_tar = True
    load_img_folder = "images_masked"
    load_para_folder="images_3dmm"
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        inplace=True),
        ])
    view_src_path = os.path.join(root_path, src_name, 'mixexp',
                                 load_img_folder)
    para_src_path = os.path.join(root_path, src_name, 'mixexp',
                                load_para_folder)
    view_tar_path = os.path.join(root_path, tar_name, 'mixexp',
                                 load_img_folder)
    para_tar_path = os.path.join(root_path, tar_name, 'mixexp',
                                load_para_folder)
    
    # src inputs
    with open(os.path.join(para_src_path, 'valid_img_ids.txt'), "r") as f:
        valid_ids = sorted([line.rstrip() for line in f.readlines()])
    num_views = len(valid_ids)
    src_num_ids,src_ids = [],[]
    for i in range(nviews_in):
        nid = int(num_views*i/nviews_in)
        src_num_ids.append(nid)
        src_ids.append(valid_ids[nid])
    if ids_src is not None and len(ids_src)==nviews_in:
        src_ids = ids_src
        src_num_ids = [valid_ids.index(x) for x in src_ids]
    poses, f, c, nf, img_files = face_pose_reading_from_ids(
                os.path.join(para_src_path, 'face_transforms_pose.json'),
                view_src_path, src_ids)
    if use_near_3dmm_src:
        param = load_ad_params(para_src_path)
    else:
        param = load_ad_params(para_src_path, src_num_ids)

    imgs_src, poses_src, nfs_src = [], [], []
    semantic_src = []
    focal, c = [], []
    for i, nid in enumerate(src_num_ids):
        img = Image.open(img_files[i])
        w, h = img.size
        scale = 1.0
        if img.size != image_size:
            scale = image_size[0] / img.size[0]
            img = img.resize(image_size, Image.BILINEAR)
        img_tensor = transform(img)
        imgs_src.append(img_tensor)
        pose = poses[i]
        f0 = f
        nf0 = nf
        pose = torch.tensor(pose, dtype=torch.float32)
        fx = torch.tensor(f0) * scale
        fy = torch.tensor(f0) * scale
        cx = torch.tensor(w / 2) * scale
        cy = torch.tensor(h / 2) * scale
        near, far = nf0[0], nf0[1]
        poses_src.append(pose)
        focal.append(torch.tensor((fx, fy), dtype=torch.float32))
        c.append(torch.tensor((cx, cy), dtype=torch.float32))
        nfs_src.append(torch.tensor((near, far), dtype=torch.float32))
        if use_near_3dmm_src:
            seq_near = obtain_seq_index(nid, num_views)
            semantic_src.append(param[seq_near].transpose(1,0))
        else:
            semantic_src.append(param[i])  
    imgs_src = torch.stack(imgs_src).float()
    poses_src = torch.stack(poses_src)
    focal = torch.stack(focal)
    c = torch.stack(c)
    nfs_src = torch.stack(nfs_src)

    semantic_src = np.stack(semantic_src)
    pose_part = semantic_src[:, 79:]

    # target inputs
    with open(os.path.join(para_tar_path, 'valid_img_ids.txt'), "r") as f:
        valid_ids = sorted([line.rstrip() for line in f.readlines()])
    num_views = len(valid_ids)
    poses_tar, f_tar, c_tar, nf_tar, img_files_tar = face_pose_reading_from_ids(
                os.path.join(para_tar_path, 'face_transforms_pose.json'),
                view_tar_path, valid_ids)
    semantic_tar = load_ad_params(para_tar_path)
    if use_near_3dmm_tar:
        semantic_tar_near = []
        for i in range(semantic_tar.shape[0]):
            seq_near = obtain_seq_index(i, num_views)
            semantic_tar_near.append(semantic_tar[seq_near].transpose(1,0))
        semantic_tar = np.stack(semantic_tar_near)
    semantics_cdn = np.expand_dims(semantic_tar, 1).repeat(semantic_src.shape[0], axis=1)
    
    imgs_tar, nfs_tar = [], []
    focal_tar, c_tar = [], []
    for i in range(semantic_tar.shape[0]):
        semantics_cdn[i,:, 79:] = pose_part
        
        img = Image.open(img_files_tar[i])
        w, h = img.size
        scale = 1.0
        if img.size != image_size:
            scale = image_size[0] / img.size[0]
            img = img.resize(image_size, Image.BILINEAR)
        img_tensor = transform(img)
        imgs_tar.append(img_tensor)
        fx = torch.tensor(f_tar) * scale
        fy = torch.tensor(f_tar) * scale
        cx = torch.tensor(w / 2) * scale
        cy = torch.tensor(h / 2) * scale
        near, far = nf_tar[0], nf_tar[1]
        focal_tar.append(torch.tensor((fx, fy), dtype=torch.float32))
        c_tar.append(torch.tensor((cx, cy), dtype=torch.float32))
        nfs_tar.append(torch.tensor((near, far), dtype=torch.float32))
    imgs_tar = torch.stack(imgs_tar).float()
    poses_tar = torch.Tensor(poses_tar)
    focal_tar = torch.stack(focal_tar)
    c_tar = torch.stack(c_tar)
    nfs_tar = torch.stack(nfs_tar)
    
    if sem_win>1:
        semantic_src = torch.Tensor(semantic_src)
        semantics_cdn = torch.Tensor(semantics_cdn)
    else:
        semantic_src = torch.Tensor(semantic_src)[:, :, None].expand(-1, -1, 27)
        semantics_cdn = torch.Tensor(semantics_cdn)[:, :, :, None].expand(-1, -1, -1, 27)
    
    result = {
            "images_src":
            imgs_src,
            "poses_src":
            poses_src,
            "focal_src":
            focal,
            "c_src":
            c,
            "nfs_src":
            nfs_src,
            "semantic_src":
            semantic_src,
            "images_tar":
            imgs_tar,
            "poses_tar":
            poses_tar,
            "focal_tar":
            focal_tar,
            "c_tar":
            c_tar,
            "nfs_tar":
            nfs_tar,
            "semantics_cdn":
            semantics_cdn,
            
        }
    return result






set_seed(10)
args, conf = util.args.parse_args(
    extra_args,
    training=True,
    default_ray_batch_size=128,
    default_conf="conf/exp/fp_mixexp_2D_implicit_Indep.conf",
    default_datadir="/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb/test",
    default_expname="000_debug",
    default_gpu_id="2")
device = util.get_cuda(args.gpu_id[0])

# ----------------------------------------------------------------
# args.resume = True
# args.visual_path = 'visual_video_test'
# args.resume_init = True
# args.resume_ckpt_init = 'results/0505_2Dimplicitdeform_indep_video(mixexp_baseFE)/checkpoints/pixel_nerf_latest'
# args.src_name = 'id10283#h87Y8nir1o0#009834#010003'
# args.tar_name = 'id10283#h87Y8nir1o0#009834#010003'
# args.semantic_window=27
# args.ids_src = ['00062','00121','00200']
# args.only_exp_driven = True
# args.method_gen_pose = 'composite'
# ----------------------------------------------------------------


if args.tar_name is None:
    args.tar_name = args.src_name
# Save path
save_path = os.path.join(args.resultdir, args.name, args.visual_path)
os.makedirs(save_path, exist_ok=True)
vid_name = f"video_driven_{args.src_name}_{args.tar_name}"
sub_save_path = os.path.join(save_path, vid_name)
os.makedirs(sub_save_path, exist_ok=True)

# Metric prepare
if args.src_name == args.tar_name and not args.only_exp_driven:
    excel_path = os.path.join(sub_save_path, 'metrics.xlsx')
    sheet_name = 'metric'
    excel_title = [['num_in','PSNR', 'SSIM', 'LPIPS'],]
    util.write_excel_xlsx(excel_path, sheet_name, excel_title)

## load model (NeRF_Net)
net = make_model(conf["model"], sem_win=args.semantic_window).to(device=device)
net.load_weights(args,
                 opt_init=args.resume_init,
                 ckpt_path_init=args.resume_ckpt_init,
                 strict=False)

## set renderer
renderer = NeRFRenderer.from_conf(
    conf["renderer"],
    lindisp=False,
).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id,
                                    simple_output=True).eval()

## Prepare data for rendering
## load datasets
data = load_driven_data(args.datadir, args.src_name, args.tar_name, args.nviews, ids_src=args.ids_src, sem_win=args.semantic_window)
# source inputs
images_in = data["images_src"].to(device=device)  # (NV, 3, H, W)
NV, C, H, W = images_in.shape
if args.semantic_window == 1:
    semantic_src = data["semantic_src"][:, :, 13].to(device=device)
else:
    semantic_src = data["semantic_src"][:, :, :].to(device=device)
poses_src = data["poses_src"].to(device=device)  # (NV, 4, 4)
focal_src = data["focal_src"]  # (1)
c_src = data["c_src"]
nfs_src = data["nfs_src"]

# targets
imgs_tar = data['images_tar']
semantics_cdn = data['semantics_cdn']
poses_tar = data["poses_tar"].to(device=device)  # (NV, 4, 4)
focal_tar = data["focal_tar"]  # (1)
c_tar = data["c_tar"]
nfs_tar = data["nfs_tar"]
NM = imgs_tar.shape[0]
if NM>120:
    NM=120
print("All test frames are:", NM)

if args.only_exp_driven:  # TODO： Modify!!!!!
    method_gen_pose =  args.method_gen_pose  # 'spiral' or 'spherical' or 'composite'
    z_near, z_far = nfs_src[0,0], nfs_src[0,1]
    render_poses = gen_render_poses(z_near, z_far, NM, method=method_gen_pose, poses=data["poses_src"].clone().detach())
    poses_tar = torch.tensor(render_poses, dtype=torch.float32).to(device=device)

if args.fix_tar_pose:  # TODO： Modify!!!!! 并入表情驱动
    poses_tar = poses_tar[:1].repeat(NM, 1, 1).to(device=device)
     
images_in_0to1 = images_in * 0.5 + 0.5  # (NV, 3, H, W)
source_views = (images_in_0to1.permute(
        0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3))
vis_src_list = []
for i in range(NV):
    vis_src_list.append(util.add_color_border(source_views[i],
                            color=[255, 0, 0],
                            linewidth=3))
all_frames, all_depths, all_gts, all_pres, ini_depths = [], [], [], [], []
for i in range(NM):
    semantic={}
    semantic["semantic_src"] = semantic_src
    if args.semantic_window == 1:
        semantic["semantic_cdn"] = semantics_cdn[i, :, :, 13].to(device=device)
    else:
        semantic["semantic_cdn"] = semantics_cdn[i, :, :, :].to(device=device)
    test_rays = util.gen_rays(poses_tar[i:(i+1)],
                                W,
                                H,
                                focal_tar[i:(i+1)],
                                nfs_tar[i:(i+1), 0],
                                nfs_tar[i:(i+1), 1],
                                c=c_tar[i:(i+1)]).to(device=device)  # (NV, H, W, 8)
    
    images_tar_0to1 = imgs_tar[i] * 0.5 + 0.5  # (NV, 3, H, W)
    gt = images_tar_0to1.permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
    renderer.eval()
    with torch.no_grad():
        net.encode(
                images_in.unsqueeze(0),
                poses_src.unsqueeze(0),
                focal_src[None].to(device=device),
                c=c_src[None].to(device=device),
                semantic=semantic
            )
        test_rays = test_rays.reshape(1, H * W, -1)
        chunk_size = args.chunk_size
        alpha_coarse_np, rgb_coarse_np, depth_coarse_np = [], [], []
        alpha_fine_np, rgb_fine_np, depth_fine_np = [], [], []
        reminder_size = H * W % chunk_size
        num_chunk = H * W // chunk_size + int(reminder_size > 0)
        all_rgb, all_depth = [], []
        for chunk_idx in range(num_chunk):
            if chunk_idx == num_chunk - 1:
                rays_chunk = test_rays[:, chunk_idx * chunk_size:, :]
            else:
                rays_chunk = test_rays[:, chunk_idx *
                                        chunk_size:(chunk_idx + 1) *
                                        chunk_size, :]
            rgb, _depth = render_par(rays_chunk)
            all_rgb.append(rgb[0])
            all_depth.append(_depth[0])
        frame = torch.cat(all_rgb).view(H, W, 3).cpu().numpy()
        depth_frame = torch.cat(all_depth).view(H, W).cpu().numpy()
        ini_depths.append(depth_frame)
        depth_cmap = util.cmap(depth_frame) / 255
        img_stack_tar_out = np.hstack(vis_src_list+[gt, frame])
        all_frames.append(img_stack_tar_out)
        all_depths.append(depth_cmap)
        all_gts.append(gt)
        all_pres.append(frame)
        imageio.imwrite(os.path.join(sub_save_path, '%04d_gt.png'%i), (gt * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(sub_save_path, '%04d_pre.png'%i), (frame * 255).astype(np.uint8))
        # Calculate Metrics
        if args.src_name == args.tar_name and not args.only_exp_driven:
            psnr_v, ssim_v, lpips_v = util.metric_function(frame, gt, 1, device)
            line = [[f'{i}',f'{psnr_v}',f'{ssim_v}',f'{lpips_v}',],]
            util.write_excel_xlsx_append(excel_path, sheet_name, line)
    print('Finish frame', i+1)
all_frames = np.stack(all_frames)
all_depths = np.stack(all_depths)
all_gts = np.stack(all_gts)
all_pres = np.stack(all_pres)
ini_depths = np.stack(ini_depths)


## Saving
print("Writing video & Save images!")
vid_path = os.path.join(save_path, "video_" + vid_name + ".mp4")
vid_depth_path = os.path.join(save_path, "video_" + vid_name + "_depth.mp4")
imageio.mimwrite(vid_path, (all_frames * 255).astype(np.uint8),
                    fps=args.fps,
                    quality=8)
imageio.mimwrite(vid_depth_path, (all_depths * 255).astype(np.uint8),
                    fps=args.fps,
                    quality=8)
imageio.mimwrite(os.path.join(sub_save_path, 'video_gt.mp4'), 
                    (all_gts * 255).astype(np.uint8),
                    fps=args.fps,
                    quality=8)
imageio.mimwrite(os.path.join(sub_save_path, 'video_pre.mp4'), 
                    (all_pres * 255).astype(np.uint8),
                    fps=args.fps,
                    quality=8)
# save depth
np.save(os.path.join(sub_save_path, 'depths.npy'), ini_depths)

    
