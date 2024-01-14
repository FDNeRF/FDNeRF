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
        "--datatype",
        type=str,
        default='vox')  # vox / facescape / wild / wild_static
    parser.add_argument(
        "--exps_list",
        default=None,
        nargs='+',
    )
    parser.add_argument(
        "--use_tar_exp",
        action="store_true",
    )
    parser.add_argument("--srctar_exp_nid", type=int, default=999)
    parser.add_argument("--rendertype",
                        type=str,
                        default='std_static',
                        help='std_static or standard_video or spiral_video')
    parser.add_argument(
        "--tar_exps_list",
        default=['neutral', 'smile', 'mouth_stretch', 'eye_closed', 'lip_funneler', 'angle', 'eye_closed_mouse_pout'],
        nargs='+',)
    parser.add_argument(
        "--average_semantic",
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

def load_driven_data(root_path, src_name, nviews_in, ids_src=None, sem_win=1, image_size=(256, 256), datatype='vox', folder_reserve=None):
    use_near_3dmm_src = False
    use_near_3dmm_tar = False
    if sem_win>1:
        use_near_3dmm_src = True
        use_near_3dmm_tar = True
    if datatype=='vox':
        load_img_folder = "images_masked"
        load_para_folder="images_3dmm"
    elif datatype == 'facescape':  # or 'wild':
        load_img_folder = "images_masked"
        load_para_folder = "images_masked"
    else: # datatype == 'wild':
        load_img_folder = "images_3dmm"
        load_para_folder = "images_3dmm"
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
    if datatype == 'wild_static':
        view_src_path = view_src_path.replace('/mixexp/', f'/{folder_reserve}/')
        para_src_path = para_src_path.replace('/mixexp/', f'/{folder_reserve}/')
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

    if sem_win>1:
        semantic_src = torch.Tensor(semantic_src)
    else:
        semantic_src = torch.Tensor(semantic_src)[:, :, None].expand(-1, -1, 27)
    
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
    default_gpu_id="1")
device = util.get_cuda(args.gpu_id[0])
# ----------------------------------------------------------------
# ---- test on vox datasets ---- 
# args.tar_exps_list = ['smile',]
# args.use_tar_exp = True

# args.resume = True
# args.visual_path = 'visual_static_test_Indep'
# args.resume_init = True
# args.resume_ckpt_init = 'results/0504_2Dimplicitdeform_indep(mixexp_baseFE)/checkpoints/pixel_nerf_latest'
# args.src_name = 'id10283#j8UugkSTzzk#001372#002396' #'id10284#EoCPhxtWUOc#002260#002691' 'id10286#FP4TghS5_UQ#003694#004036' 'id10291#uiBjIKX_0l8#000067#000277'
# args.ids_src = ['00152', '00820', '00971'] #'00000','00050','00255'  '00000','00203','00278'  '00006','00049','00122'
# args.srctar_exp_nid = 1
# args.semantic_window=1

# ---- test on wild datasets ---- 
# args.resume = True
# args.visual_path = 'visual_static_test_Indep'
# args.resume_init = True
# args.resume_ckpt_init = 'results/0504_2Dimplicitdeform_indep(mixexp_baseFE)/checkpoints/pixel_nerf_latest'
# args.datatype = 'wild_static'
# args.datadir = '/home/zhangjingbo/Datasets/FaceDatasets/Wild/test'
# args.src_name = 'eckert'
# args.ids_src = ['00016','00012','00037']
# args.exps_list = ['2_eye_closed','3_mouth_stretch','1_neutral']
# args.semantic_window=1
# ----------------------------------------------------------------

save_path = os.path.join(args.resultdir, args.name, args.visual_path, args.src_name+f'({args.ids_src[0]})', args.rendertype)
os.makedirs(save_path, exist_ok=True)
vid_name = args.split + f"_{args.src_name}"

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
if args.datatype == 'wild_static':
    data = {}
    assert len(args.ids_src) == len(args.exps_list)
    images_src, semantic_src, poses_src, focal_src, c_src, nfs_src = [], [], [], [], [], []
    for i in range(len(args.ids_src)):
        data_i = load_driven_data(args.datadir, args.src_name, 1, ids_src=[args.ids_src[i],], sem_win=args.semantic_window, datatype=args.datatype, folder_reserve=args.exps_list[i])
        images_src.append(data_i["images_src"]) 
        semantic_src.append(data_i["semantic_src"]) 
        poses_src.append(data_i["poses_src"]) 
        focal_src.append(data_i["focal_src"]) 
        c_src.append(data_i["c_src"]) 
        nfs_src.append(data_i["nfs_src"]) 
    images_src = torch.cat(images_src)
    semantic_src = torch.cat(semantic_src)
    poses_src = torch.cat(poses_src)
    focal_src = torch.cat(focal_src)
    c_src = torch.cat(c_src)
    nfs_src = torch.cat(nfs_src)
    data = {"images_src": images_src, "poses_src": poses_src, "focal_src": focal_src, \
            "c_src": c_src, "nfs_src": nfs_src, "semantic_src": semantic_src}
else:
    data = load_driven_data(args.datadir, args.src_name, args.nviews, ids_src=args.ids_src, sem_win=args.semantic_window, datatype=args.datatype)

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

# input-exp as targets
input_as_tar = not args.use_tar_exp
if input_as_tar:
    semantic_in = data["semantic_src"].numpy()
    pose_part = semantic_in[:, 79:]
    semantics_cdn = np.expand_dims(semantic_in, 1).repeat(semantic_in.shape[0], axis=1)
    for i in range(semantics_cdn.shape[0]):
        semantics_cdn[i,:, 79:] = pose_part
    semantics_cdn = torch.Tensor(semantics_cdn)
    if args.srctar_exp_nid < 10:
        e_id = args.srctar_exp_nid
        semantics_cdn = semantics_cdn[e_id:e_id+1]
else:
    std_exps = np.load('data/std_exps.npy', allow_pickle=True).item()
    semantic_in = []
    for exp in args.tar_exps_list:
        semantic_in.append(std_exps[exp])
    semantic_in = np.stack(semantic_in)
    semantics_cdn = torch.Tensor(semantic_in)[:, None, :, None].expand(-1, args.nviews, -1, 27)
    semantic_in = data["semantic_src"]
    pose_part = semantic_in[:, 79:]
    pose_part = pose_part[None, :, :, :].expand(semantics_cdn.shape[0], -1, -1, -1)
    semantics_cdn =torch.cat([semantics_cdn,pose_part],dim=2)

dt = 0.75
mean_dz = 1. / (((1. - dt) / nfs_src[0,0] + dt / nfs_src[0,1]))
dz=np.array(mean_dz)
t=0
for i in range(NV):
    t += np.linalg.norm(np.array(data["poses_src"][i, :3, 3]))
t = np.array(t/NV)
if args.rendertype == 'std_static':
    render_poses = util.get_standard_static_pose(focal=t)
elif args.rendertype == 'standard_video':
    render_poses = util.get_standard_poses(focal=t, N_views=120)
elif args.rendertype == 'spiral_video':
    render_poses = util.get_circle_spiral_poses(focal=t, N_views=120)
poses_tar = torch.tensor(render_poses, dtype=torch.float32).to(device=device)
NT = poses_tar.shape[0]
focal_tar = focal_src[0:1].repeat(NT, 1)
c_tar = c_src[0:1].repeat(NT, 1)
nfs_tar = nfs_src[0:1].repeat(NT, 1)
images_in_0to1 = images_in * 0.5 + 0.5  # (NV, 3, H, W)
source_views = (images_in_0to1.permute(
        0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3))
vis_src_list = []
for i in range(NV):
    vis_src_list.append(source_views[i])
    # vis_src_list.append(util.add_color_border(source_views[i],
    #                         color=[255, 0, 0],
    #                         linewidth=3))
    vid_path_in = os.path.join(save_path, vid_name + f"_in{i}.png")
    imageio.imwrite(vid_path_in, (source_views[i]* 255).astype(np.uint8))
multi_frames_in = np.hstack(vis_src_list)
vid_path_in = os.path.join(save_path, vid_name + "_in_all.png")
imageio.imwrite(vid_path_in, (multi_frames_in * 255).astype(np.uint8))

for sem_ord_tar in range(semantics_cdn.shape[0]):
    frames = []
    for i in range(NT):
        semantic={}
        semantic["semantic_src"] = semantic_src
        if args.semantic_window == 1:
            semantic["semantic_cdn"] = semantics_cdn[sem_ord_tar, :, :, 13].to(device=device)
        else:
            semantic["semantic_cdn"] = semantics_cdn[sem_ord_tar, :, :, :].to(device=device)
        test_rays = util.gen_rays(poses_tar[i:(i+1)],
                                    W,
                                    H,
                                    focal_tar[i:(i+1)],
                                    nfs_tar[i:(i+1), 0],
                                    nfs_tar[i:(i+1), 1],
                                    c=c_tar[i:(i+1)]).to(device=device)  # (NV, H, W, 8)
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
            depth_cmap = util.cmap(depth_frame) / 255
            frames.append(frame)
        if args.use_tar_exp:
            vid_path_out = os.path.join(save_path, vid_name + f"_out_{args.tar_exps_list[sem_ord_tar]}_{i}.png")
        else:
            if args.srctar_exp_nid < 10:
                vid_path_out = os.path.join(save_path, vid_name + f"_out{args.srctar_exp_nid}_{i}.png")
            else:
                vid_path_out = os.path.join(save_path, vid_name + f"_out{sem_ord_tar}_{i}.png")
        imageio.imwrite(vid_path_out, (frame * 255).astype(np.uint8))
        print('finish frame ', i)
   

    ## Saving
    if args.rendertype == 'std_static':
        multi_frames = np.hstack(frames)
        if args.use_tar_exp:
            vid_path_out = os.path.join(save_path, vid_name + f"_out_{args.tar_exps_list[sem_ord_tar]}_all.png")
        else:
            vid_path_out = os.path.join(save_path, vid_name + f"_out{sem_ord_tar}_all.png")
        imageio.imwrite(vid_path_out, (multi_frames * 255).astype(np.uint8))
    elif args.rendertype == 'standard_video' or args.rendertype == 'spiral_video':
        frames = np.stack(frames)
        vid_path_out = os.path.join(
            save_path, vid_name + f"_out{args.srctar_exp_nid}_{args.rendertype}_{sem_ord_tar}.mp4")
        imageio.mimwrite(vid_path_out, (frames * 255).astype(np.uint8),
                        fps=30,
                        quality=8)

print(f"Finish {args.src_name}!")
