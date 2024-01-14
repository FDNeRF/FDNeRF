"""
Author: Eckert ZHANG
Date: 2021-11-04 21:49:57
LastEditTime: 2022-03-18 01:14:39
LastEditors: Eckert ZHANG
Description: 
"""
import sys, os
import random

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import tqdm
from dotmap import DotMap
# preprocess-PIRenderer
from preprocess_v2.utils import get_model_pirenderer, load_ckpt_pirenderer


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
    parser.add_argument("--subset",
                        "-S",
                        type=int,
                        default=0,
                        help="Subset in data to use")
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
    parser.add_argument("--scale",
                        type=float,
                        default=1.0,
                        help="Video scale relative to input size")
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
    return parser


set_seed(10)
args, conf = util.args.parse_args(
    extra_args,
    training=True,
    default_ray_batch_size=128,
    default_conf="conf/exp/fs_multiexp_2Dedit_v2.conf",
    default_datadir="/data/zhangjingbo/FaceScape_rendered/3dmm_nocrop",
    default_expname="0318_PIR-NeRF-Concat_video",
    default_gpu_id="0")
args.resume = True
device = util.get_cuda(args.gpu_id[0])

args.visual_path = 'visual_video'
args.ray_batch_size = 8000
args.num_video_frames = 60
args.resume_init = True
args.resume_ckpt_init = 'checkpoints/03_static_rendered/pixel_nerf_latest'

## load model (Preprocess_Net)
ckpt_path = args.ckpt_warp2D
if args.resume:
    if args.resume_init:
        ckpt_path = os.path.join(os.path.dirname(args.resume_ckpt_init),
                                 'latest_ckpt_preprocess.pt')
    else:
        ckpt_path = f'results/{args.name}/checkpoints/latest_ckpt_preprocess.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = args.ckpt_warp2D
stage_net_G = 'gen'
net_G, net_G_ema = get_model_pirenderer(conf["preprocess"], device, EMA=False)
net_G, _ = load_ckpt_pirenderer(net_G, net_G_ema, ckpt_path)

## load model (NeRF_Net)
net = make_model(conf["model"]).to(device=device)
net.load_weights(args,
                 opt_init=args.resume_init,
                 ckpt_path_init=args.resume_ckpt_init,
                 strict=False)

## load datasets
dset = get_split_dataset(args.dataset_format,
                         args.datadir,
                         want_split=args.split,
                         training=False)

## set renderer
renderer = NeRFRenderer.from_conf(
    conf["renderer"],
    lindisp=dset.lindisp,
).to(device=device)
# if args.resume:
#     if args.resume_init:
#         path_re = os.path.dirname(args.resume_ckpt_init) + '/_renderer'
#         if os.path.exists(path_re):
#             renderer.load_state_dict(torch.load(path_re, map_location=device))
#     else:
#         renderer_state_path = "%s/%s/%s/_renderer" % (
#             args.resultdir,
#             args.name,
#             args.checkpoints_path,
#         )
#         if os.path.exists(renderer_state_path):
#             renderer.load_state_dict(
#                 torch.load(renderer_state_path, map_location=device))

render_par = renderer.bind_parallel(net, args.gpu_id,
                                    simple_output=True).eval()

## Prepare data for rendering
data = dset[args.subset]
print("Data instance loaded:", data["scan"] + '/' + data["exp_tar"])

images_in = data["images"].to(device=device)  # (NV, 3, H, W)
images_ref = data["images_ref"].to(device=device)  # (NV, 3, H, W)
NV, C, H, W = images_in.shape
input_img_preprocess = images_in[:args.nviews].reshape(-1, C, H, W)
# gt_img_preprocess = images_ref[:args.nviews].reshape(-1, C, H, W)
len_sem = 70
input_semantic = data["semantic_cdn"][:args.nviews, :len_sem,
                                      0].to(device=device)
output_dict = net_G(input_img_preprocess, input_semantic, stage_net_G)
if stage_net_G == 'gen':
    fake_img = output_dict['fake_image']
    all_images = torch.cat(
        (fake_img.view(-1, C, H, W), images_in[args.nviews:]), dim=0)
else:
    warp_img = output_dict['warp_image']
    all_images = torch.cat(
        (warp_img.view(-1, C, H, W), images_in[args.nviews:]), dim=0)

poses = data["poses"]  # (NV, 4, 4)
focal = data["focal"]
nfs = data["nfs"]
c = data.get("c")

if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".
            format(args.scale, H, W))
    H, W = Ht, Wt

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

print("Generating rays")

get_spiral_poses = True
if get_spiral_poses:
    print("Computes poses that follow a spiral path")

    render_poses = util.get_spiral(poses, [z_near, z_far],
                                   rads_scale=0.5,
                                   N_views=args.num_video_frames)
    render_poses = torch.tensor(render_poses, dtype=torch.float32)
else:
    print("Using default (360 loop) camera trajectory")
    if args.radius == 0.0:
        radius = (z_near + z_far) * 0.5
        print("> Using default camera radius", radius)
    else:
        radius = args.radius
    # Use 360 pose sequence from NeRF
    render_poses = torch.stack(
        [
            util.pose_spherical(angle, args.elevation, radius)
            for angle in np.linspace(-180, 180, args.num_video_frames + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)
render_focal = focal[:1, :].repeat(args.num_video_frames, 1)
render_nfs = nfs[:1, :].repeat(args.num_video_frames, 1)
render_c = c[:1, :].repeat(args.num_video_frames, 1)

# (NV, H, W, 8)
render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    render_focal * args.scale,
    render_nfs[:, 0],
    render_nfs[:, 1],
    c=render_c * args.scale if c is not None else None,
).to(device=device)
tar_rays = util.gen_rays(
    poses[-1:],
    W,
    H,
    focal[-1:] * args.scale,
    nfs[-1:, 0],
    nfs[-1:, 1],
    c=c[-1:] * args.scale if c is not None else None,
).to(device=device)

focal = focal.to(device=device)
src_images = all_images[:args.nviews]
src_poses = poses[:args.nviews].to(device=device)

# Ensure decent sampling resolution
if renderer.n_coarse < 64:
    renderer.n_coarse = 64
    renderer.n_fine = 128

with torch.no_grad():
    print("Encoding source view(s)")
    net.encode(
        src_images.unsqueeze(0),
        src_poses.unsqueeze(0),
        focal[:args.nviews][None].to(device=device),
        c[:args.nviews][None].to(device=device),
    )

    print("Rendering", args.num_video_frames * H * W, "rays")

    all_rgb = []
    for rays in tqdm.tqdm(
            torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)):
        rgb, _depth = render_par(rays[None])
        all_rgb.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb)
    # rgb_fine (V*H*W, 3)
    frames = rgb_fine.view(-1, H, W, 3)

    all_rgb = []
    for rays in tqdm.tqdm(
            torch.split(tar_rays.view(-1, 8), args.ray_batch_size, dim=0)):
        rgb, _depth = render_par(rays[None])
        all_rgb.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb)
    # rgb_fine (V*H*W, 3)
    frame_tar = rgb_fine.view(H, W, 3)

    # NF = render_rays.shape[0]
    # test_rays = render_rays.reshape(NF, H * W, -1)
    # chunk_size = args.ray_batch_size
    # reminder_size = H * W % chunk_size
    # num_chunk = H * W // chunk_size + int(reminder_size > 0)
    # all_rgb_fine = []
    # for ni in range(NF):
    #     rgb_np = []
    #     for chunk_idx in range(num_chunk):
    #         if chunk_idx == num_chunk - 1:
    #             rays_chunk = test_rays[ni:ni + 1, chunk_idx * chunk_size:, :]
    #         else:
    #             rays_chunk = test_rays[ni:ni + 1,
    #                                    chunk_idx * chunk_size:(chunk_idx + 1) *
    #                                    chunk_size, :]
    #         rgb, _depth = render_par(rays_chunk)
    #         rgb_np.append(rgb.cpu())
    #     rgb_np = torch.cat(rgb_np, dim=1)
    #     rgb_np = rgb_np[0].numpy().reshape(H, W, 3)
    #     all_rgb_fine.append(rgb_np)
    # frames = np.stack(all_rgb_fine)

## Saving
save_path = os.path.join(args.resultdir, args.name, args.visual_path)
os.makedirs(save_path, exist_ok=True)
vid_name = args.split + "{:04}".format(args.subset)

print("Writing video")
vid_path = os.path.join(save_path, "video_" + vid_name + ".mp4")
imageio.mimwrite(vid_path, (frames.cpu().numpy() * 255).astype(np.uint8),
                 fps=args.fps,
                 quality=8)

print("Writing Image")
viewimg_in_path = os.path.join(save_path, "video" + vid_name + "_view_in.jpg")
viewimg_out_path = os.path.join(save_path,
                                "video" + vid_name + "_view_out.jpg")
imgs_in = torch.cat([
    input_img_preprocess[0], input_img_preprocess[1], input_img_preprocess[2], \
    images_in[-1]
], dim=2).detach().cpu().permute(1, 2, 0) * 0.5 + 0.5
imgs_warped = torch.cat([src_images[0], src_images[1], src_images[2]],
                        dim=2).detach().cpu().permute(1, 2, 0) * 0.5 + 0.5
imgs_out = torch.cat([imgs_warped, frame_tar.detach().cpu()], dim=1)
img_in_out = torch.cat([imgs_in, imgs_out], dim=0)
imageio.imwrite(viewimg_in_path, (img_in_out * 255).numpy().astype(np.uint8))

imageio.imwrite(viewimg_out_path,
                (frame_tar.detach().cpu().numpy() * 255).astype(np.uint8))

print("Wrote to", vid_path)
