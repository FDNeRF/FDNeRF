"""
Author: Eckert ZHANG
Date: 2021-09-10 00:43:52
LastEditTime: 2022-03-25 10:06:20
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf-portrait/train/train2_PIRender.py
Description: 
"""

import sys, random
import os, pdb, imageio

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from dotmap import DotMap

# preprocess-PIRenderer
# from preprocess.utils import get_model_pirenderer, load_ckpt_pirenderer
# from preprocess.loss.perceptual import PerceptualLoss


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
        "--batch_size",
        "-B",
        type=int,
        default=2,
        help="Object batch size ('SB')",
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=int,
        default=12,
        help=
        "Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--max_nview",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--nview_test",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--semantic_window",
        type=int,
        default=1, # 1 or 27
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--with_metric",
        action="store_true",
    )
    parser.add_argument(
        "--with_mask",
        action="store_true",
    )
    parser.add_argument(
        "--average_semantic",
        action="store_true",
        help="Freeze encoder weights and only train MLP",
    )
    return parser


set_seed(10)
args, conf = util.args.parse_args(
    extra_args,
    training=True,
    default_ray_batch_size=128, # 128
    default_conf=
    'conf/exp/fp_mixexp_2D_implicit_video.conf', 
    default_datadir=
    '/home/zhangjingbo/Datasets/FaceDatasets/VoxCeleb',
    default_expname="000_debug",
    default_gpu_id="1 2 3")
device = util.get_cuda(args.gpu_id[0])

# load models
net = make_model(conf["model"], sem_win=args.semantic_window).to(device=device)

# freeze encoder net or not
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("-----Encoder frozen!")
    net.encoder.eval()
else:
    print("-----Encoder training!")

# load datasets
dset, val_dset, _ = get_split_dataset(args.dataset_format,
                                      args.datadir,
                                      n_view_in=args.nviews,
                                      list_prefix=args.dataset_prefix,
                                      sem_win=args.semantic_window,
                                      with_mask=args.with_mask)

# set renderer
renderer = NeRFRenderer.from_conf(
    conf["renderer"],
    lindisp=dset.lindisp,
).to(device=device)

# Parallize
if args.only_video:
    render_par = renderer.bind_parallel(net, args.gpu_id,
                                        simple_output=True).eval()
else:
    render_par = renderer.bind_parallel(net, args.gpu_id).eval()
max_nview = min(args.nviews, args.max_nview)
nviews = list(range(1, max_nview+1))
nviews_test = [3, 6, 9]


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net,
                         dset,
                         val_dset,
                         args,
                         conf["train"],
                         device=device)
        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print("lambda coarse {} and fine {}".format(self.lambda_coarse,
                                                    self.lambda_fine))
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if net.mlp_fine is not None:
            self.use_fine_mlp = True

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        # write the metric data to excel file
        if self.args.only_test and self.args.with_metric:
            self.excel_path = os.path.join(self.visual_path, 'metrics.xlsx')
            os.makedirs(self.visual_path, exist_ok=True)
            self.sheet_name = 'metric'
            excel_title = [['step', 'model', 'target_id', 'input_ids','num_in','PSNR', 'SSIM', 'LPIPS'],]
            util.write_excel_xlsx(self.excel_path, self.sheet_name, excel_title)

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        all_images = data["images"]  # (SB, NV, 3, H, W)
        all_images = torch.cat([all_images[:,:curr_nviews], all_images[:,-1:]], dim=1).to(device=device)
        SB, NV, C, H, W = all_images.shape
        semantic={}
        len_sem = 85
        if self.args.semantic_window == 1:
            if self.args.average_semantic:
                semantic["semantic_src"] = torch.mean(data["semantic_src"][:,:curr_nviews,:, :], dim=-1).to(device=device)
                semantic["semantic_cdn"] = torch.mean(data["semantic_cdn"][:,:curr_nviews,:, :], dim=-1).to(device=device)
            else:
                semantic["semantic_src"] = data["semantic_src"][:,:curr_nviews,:, 13].to(device=device)
                semantic["semantic_cdn"] = data["semantic_cdn"][:,:curr_nviews,:, 13].to(device=device)
        else:
            s0, s1, s2, s3 = data["semantic_src"][:,:curr_nviews,:, :].shape
            semantic["semantic_src"] = data["semantic_src"][:,:curr_nviews,:, :].reshape(-1, s2, s3).to(device=device)
            semantic["semantic_cdn"] = data["semantic_cdn"][:,:curr_nviews,:, :].reshape(-1, s2, s3).to(device=device)
        
        all_poses = data["poses"]  # (SB, NV, 4, 4)
        all_focals = data["focal"]  # (SB, NV, 2)
        all_c = data["c"]  # (SB, NV, 2)
        all_nfs = data["nfs"]
        all_poses = torch.cat([all_poses[:,:curr_nviews], all_poses[:,-1:]], dim=1).to(device=device)
        all_focals = torch.cat([all_focals[:,:curr_nviews], all_focals[:,-1:]], dim=1)
        all_c = torch.cat([all_c[:,:curr_nviews], all_c[:,-1:]], dim=1)
        all_nfs = torch.cat([all_nfs[:,:curr_nviews], all_nfs[:,-1:]], dim=1)

        all_rgb_gt = []
        all_rays = []

        for obj_idx in range(SB):
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = all_c[obj_idx]
            images_0to1 = images * 0.5 + 0.5

            # Get 8-D camera rays: cam_centers(3), cam_raydir(3), cam_nears(1), cam_fars(1)
            nfs = all_nfs[obj_idx]
            cam_rays = util.gen_rays(poses,
                                        W,
                                        H,
                                        focal,
                                        nfs[:, 0],
                                        nfs[:, 1],
                                        c=c)  # (NV, H, W, 8)
            
            rgb_gt_tar = images_0to1[-1]
            # (3, H, W) --> (H, W, 3)
            rgb_gt_all = (rgb_gt_tar.permute(1, 2,
                                             0).contiguous().reshape(-1, 3))

            pix_inds = torch.randint(0, H * W, (args.ray_batch_size, ))
            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays[-1].view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device)  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        # (SB, NS, 3, H, W)
        src_images = all_images[:, :curr_nviews]
        src_poses = all_poses[:, :curr_nviews]  # (SB, NS, 4, 4)
        all_focals = all_focals[:, :curr_nviews]
        all_c = all_c[:, :curr_nviews]
        all_poses = all_images = None

        net.encode(
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
            semantic=semantic
        )
        render_dict = DotMap(render_par(
            all_rays,
            want_weights=True,
        ))
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        loss = rgb_loss
        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data,
                                  is_train=False,
                                  global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        scan = data['scan'][batch_idx]
        img_ids = [data['img_id'][i][batch_idx] for i in range(len(data['img_id']))]
        if self.args.only_test:
            curr_nviews = args.nview_test
        else:
            curr_nviews = nviews_test[torch.randint(0, len(nviews_test), ()).item()]

        # preprocessing stage
        images = data["images"][batch_idx]
        images = torch.cat([images[:curr_nviews], images[-1:]], dim=0).to(device=device)
        NV, C, H, W = images.shape
        if self.args.with_mask and len(data["masks"]) > 0:
            masks = data["masks"][batch_idx]
            masks = torch.cat([masks[:curr_nviews], masks[-1:]], dim=0)
        else:
            masks = None

        semantic={}
        len_sem = 85
        if self.args.semantic_window == 1:
            if self.args.average_semantic:
                semantic["semantic_src"] = torch.mean(data["semantic_src"][batch_idx,:curr_nviews,:, :], dim=-1).to(device=device)
                semantic["semantic_cdn"] = torch.mean(data["semantic_cdn"][batch_idx,:curr_nviews,:, :], dim=-1).to(device=device)
            else:
                semantic["semantic_src"] = data["semantic_src"][batch_idx,:curr_nviews,:, 13].to(device=device)
                semantic["semantic_cdn"] = data["semantic_cdn"][batch_idx,:curr_nviews,:, 13].to(device=device)
        else:
            s1, s2, s3 = data["semantic_src"][batch_idx,:curr_nviews,:, :].shape
            semantic["semantic_src"] = data["semantic_src"][batch_idx,:curr_nviews,:, :].reshape(-1, s2, s3).to(device=device)
            semantic["semantic_cdn"] = data["semantic_cdn"][batch_idx,:curr_nviews,:, :].reshape(-1, s2, s3).to(device=device)
        
        poses = data["poses"][batch_idx]  # (NV, 4, 4)
        focal = data["focal"][batch_idx]  # (1)
        c = data["c"][batch_idx]
        nfs = data["nfs"][batch_idx]
        poses = torch.cat([poses[:curr_nviews], poses[-1:]], dim=0).to(device=device)
        focal = torch.cat([focal[:curr_nviews], focal[-1:]], dim=0)
        c = torch.cat([c[:curr_nviews], c[-1:]], dim=0)
        nfs = torch.cat([nfs[:curr_nviews], nfs[-1:]], dim=0)

        cam_rays = util.gen_rays(poses[-1:],
                                     W,
                                     H,
                                     focal[-1:],
                                     nfs[-1:, 0],
                                     nfs[-1:, 1],
                                     c=c)  # (NV, H, W, 8)
        del data
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (images_0to1[:curr_nviews].permute(
            0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3))

        gt = images_0to1[-1].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        del images_0to1
        with torch.no_grad():
            test_rays = cam_rays  # (H, W, 8)
            test_images = images[:curr_nviews]  # (NS, 3, H, W)
            del cam_rays, images
            if len(focal.shape) == 2:
                focal = focal[:curr_nviews][None]
                c = c[:curr_nviews][None]
            net.encode(
                test_images.unsqueeze(0),
                poses[:curr_nviews].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
                semantic=semantic
            )
            test_images, poses, focal, c = None, None, None, None
            test_rays = test_rays.reshape(1, H * W, -1)

            chunk_size = self.args.chunk_size
            alpha_coarse_np, rgb_coarse_np, depth_coarse_np = [], [], []
            alpha_fine_np, rgb_fine_np, depth_fine_np = [], [], []
            reminder_size = H * W % chunk_size
            num_chunk = H * W // chunk_size + int(reminder_size > 0)

            for chunk_idx in range(num_chunk):
                if chunk_idx == num_chunk - 1:
                    rays_chunk = test_rays[:, chunk_idx * chunk_size:, :]
                else:
                    rays_chunk = test_rays[:, chunk_idx *
                                           chunk_size:(chunk_idx + 1) *
                                           chunk_size, :]
                render_dict = DotMap(render_par(rays_chunk, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine

                alpha_coarse_np.append(coarse.weights.sum(dim=-1).cpu())
                rgb_coarse_np.append(coarse.rgb.cpu())
                depth_coarse_np.append(coarse.depth.cpu())

                if self.use_fine_mlp:
                    alpha_fine_np.append(fine.weights.sum(dim=-1).cpu())
                    rgb_fine_np.append(fine.rgb.cpu())
                    depth_fine_np.append(fine.depth.cpu())
            alpha_coarse_np = torch.cat(alpha_coarse_np, dim=1)
            rgb_coarse_np = torch.cat(rgb_coarse_np, dim=1)
            depth_coarse_np = torch.cat(depth_coarse_np, dim=1)
            if self.use_fine_mlp:
                alpha_fine_np = torch.cat(alpha_fine_np, dim=1)
                rgb_fine_np = torch.cat(rgb_fine_np, dim=1)
                depth_fine_np = torch.cat(depth_fine_np, dim=1)

            alpha_coarse_np = alpha_coarse_np[0].numpy().reshape(H, W)
            rgb_coarse_np = rgb_coarse_np[0].numpy().reshape(H, W, 3)
            depth_coarse_np = depth_coarse_np[0].numpy().reshape(H, W)
            if self.use_fine_mlp:
                alpha_fine_np = alpha_fine_np[0].numpy().reshape(H, W)
                rgb_fine_np = rgb_fine_np[0].numpy().reshape(H, W, 3)
                depth_fine_np = depth_fine_np[0].numpy().reshape(H, W)

        # claculate PSNR
        if self.use_fine_mlp:
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np
        if masks is not None:
            mask_tar = masks[-1].numpy()
            rgb_psnr = rgb_psnr * np.expand_dims(mask_tar, axis=2)
            rgb_psnr = rgb_psnr.astype(np.float32)
        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)
        if self.args.only_test and self.args.with_metric:
            psnr_v, ssim_v, lpips_v = util.metric_function(rgb_psnr, gt, 1, device)
            line = [[f'{global_step}', f'{scan}',f'{img_ids[-1]}',f'{img_ids[:curr_nviews]}', 
            f'{curr_nviews}',f'{psnr_v}',f'{ssim_v}',f'{lpips_v}',],]
            # excel_title = [['step', 'model', 'target_id', 'input_ids','num_in','PSNR', 'SSIM', 'LPIPS'],]
            util.write_excel_xlsx_append(self.excel_path, self.sheet_name, line)

        # visualization
        print("c rgb min {} max {}".format(rgb_coarse_np.min(),
                                           rgb_coarse_np.max()))
        print("c alpha min {}, max {}".format(alpha_coarse_np.min(),
                                              alpha_coarse_np.max()))
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_src_list = []
        for i in range(curr_nviews):
            # vis_src_list.append(util.add_color_border(source_views[i],
            #                       color=[255, 0, 0],
            #                       linewidth=3))
            vis_src_list.append(source_views[i])
        vis_src = np.hstack(vis_src_list)
        vis_list = vis_src_list + [
            util.add_color_border(gt.copy(), color=[0, 255, 0], linewidth=3),
            depth_coarse_cmap,
            util.add_color_border(rgb_coarse_np.copy(),
                                  color=[0, 255, 255],
                                  linewidth=3),
            # alpha_coarse_cmap,
        ]
        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if self.use_fine_mlp:
            print("f rgb min {} max {}".format(rgb_fine_np.min(),
                                               rgb_fine_np.max()))
            print("f alpha min {}, max {}".format(alpha_fine_np.min(),
                                                  alpha_fine_np.max()))
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            # vis_list = vis_src_list + [
            #     util.add_color_border(gt, color=[0, 255, 0], linewidth=3),
            #     depth_fine_cmap,
            #     util.add_text_psnr(
            #         util.add_color_border(rgb_fine_np,
            #                               color=[0, 255, 255],
            #                               linewidth=3), psnr),
            #     # alpha_fine_cmap,
            # ]
            vis_list = vis_src_list + [
                gt,
                depth_fine_cmap,
                util.add_text_psnr(rgb_fine_np.copy(), psnr),
                # alpha_fine_cmap,
            ]
            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))

        # set the renderer network back to train mode
        renderer.train()

        if self.args.only_test:
            return vis_fine, vals, [vis_src, gt, rgb_psnr, depth_fine_np]
        else:
            return vis_fine, vals

    def vis_video_step(self, data, global_step, idx=None): 
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        curr_nviews = args.nview_test
        scan = data["scan"][batch_idx]
        img_ids = [data['img_id'][i][batch_idx] for i in range(len(data['img_id']))]
        img_id_tar = img_ids[-1]

        # preprocessing stage 
        # (SB, NV, 3, H, W)
        images = data["images"][batch_idx]
        images = torch.cat([images[:curr_nviews], images[-1:]], dim=0).to(device=device)
        NV, C, H, W = images.shape
        semantic={}
        len_sem = 85
        if self.args.semantic_window == 1:
            if self.args.average_semantic:
                semantic["semantic_src"] = torch.mean(data["semantic_src"][batch_idx,:curr_nviews,:, :], dim=-1).to(device=device)
                semantic["semantic_cdn"] = torch.mean(data["semantic_cdn"][batch_idx,:curr_nviews,:, :], dim=-1).to(device=device)
            else:
                semantic["semantic_src"] = data["semantic_src"][batch_idx,:curr_nviews,:, 13].to(device=device)
                semantic["semantic_cdn"] = data["semantic_cdn"][batch_idx,:curr_nviews,:, 13].to(device=device)
        else:
            s1, s2, s3 = data["semantic_src"][batch_idx,:curr_nviews,:, :].shape
            semantic["semantic_src"] = data["semantic_src"][batch_idx,:curr_nviews,:, :].reshape(-1, s2, s3).to(device=device)
            semantic["semantic_cdn"] = data["semantic_cdn"][batch_idx,:curr_nviews,:, :].reshape(-1, s2, s3).to(device=device)

        poses = data["poses"][batch_idx]  # (NV, 4, 4)
        focal = data["focal"][batch_idx]
        c = data["c"][batch_idx]
        nfs = data["nfs"][batch_idx]
        poses = torch.cat([poses[:curr_nviews], poses[-1:]], dim=0)
        focal = torch.cat([focal[:curr_nviews], focal[-1:]], dim=0)
        c = torch.cat([c[:curr_nviews], c[-1:]], dim=0)
        nfs = torch.cat([nfs[:curr_nviews], nfs[-1:]], dim=0)

        print("Generating rays")
        if self.args.pose_traj_video == 'spiral':
            print("Computes poses following a circle spiral path")
            c2w_avg = util.poses_avg(poses)
            # render_poses = util.get_spiral(poses, [z_near, z_far],
            #                                rads_scale=0.8,
            #                                N_views=args.num_video_frames)
            render_poses = util.get_circle_spiral_poses_from_pose(c2w_avg, f_delta=0.1, N_views=args.num_video_frames, n_r=1)
            render_poses = torch.tensor(render_poses, dtype=torch.float32)
        elif self.args.pose_traj_video == 'standard':
            print("Using standard camera trajectory")
            c2w_avg = util.poses_avg(poses)
            render_poses = util.get_standard_poses_from_tar_pose(c2w_avg, N_views=args.num_video_frames)
            render_poses = torch.tensor(render_poses, dtype=torch.float32)
        render_focal = focal[:1, :].repeat(args.num_video_frames, 1)
        render_nfs = nfs[:1, :].repeat(args.num_video_frames, 1)
        render_c = c[:1, :].repeat(args.num_video_frames, 1)
        cam_rays = util.gen_rays(
            render_poses,
            W,
            H,
            render_focal,
            render_nfs[:, 0],
            render_nfs[:, 1],
            c=render_c,
        )

        del data
        images_0to1 = images * 0.5 + 0.5 
        source_views = (images_0to1[:curr_nviews].permute(
            0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3))
        gt = images_0to1[-1].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)

        # set renderer net to eval mode
        renderer.eval()
        with torch.no_grad():
            test_rays = cam_rays.to(device=device)  # (:, H, W, 8)
            test_images = images[:curr_nviews]  # (NS, 3, H, W)
            del cam_rays
            if len(focal.shape) == 2:
                focal = focal[:curr_nviews][None]
                c = c[:curr_nviews][None]
            net.encode(
                test_images.unsqueeze(0),
                poses[:curr_nviews].unsqueeze(0).to(device=device),
                focal.to(device=device),
                c=c.to(device=device),
                semantic=semantic
            )
            test_images, poses, focal, c = None, None, None, None
            test_rays = test_rays.reshape(-1, H * W, 8)
            NF = test_rays.shape[0]

            chunk_size = self.args.chunk_size
            reminder_size = H * W % chunk_size
            num_chunk = H * W // chunk_size + int(reminder_size > 0)

            all_rgb_fine = []
            all_depth = []
            for ni in range(NF):
                # alpha_coarse_np, rgb_coarse_np, depth_coarse_np = [], [], []
                # alpha_fine_np, rgb_fine_np, depth_fine_np = [], [], []
                rgb_np = []
                depth_np = []
                for chunk_idx in range(num_chunk):
                    if chunk_idx == num_chunk - 1:
                        rays_chunk = test_rays[ni:ni + 1,
                                               chunk_idx * chunk_size:, :]
                    else:
                        rays_chunk = test_rays[ni:ni + 1, chunk_idx *
                                               chunk_size:(chunk_idx + 1) *
                                               chunk_size, :]
                    rgb, depth = render_par(rays_chunk)
                    rgb_np.append(rgb.cpu())
                    depth_np.append(depth.cpu())
                rgb_np = torch.cat(rgb_np, dim=1)
                rgb_np = rgb_np[0].numpy().reshape(H, W, 3)
                all_rgb_fine.append(rgb_np)
                depth_np = torch.cat(depth_np, dim=1)
                depth_np = depth_np[0].numpy().reshape(H, W)
                all_depth.append(depth_np)
                print(f'finish frame {ni+1}/{NF}')
        frames = np.stack(all_rgb_fine)
        all_depth = np.stack(all_depth)
        vid_name = f"test_{scan}_{img_id_tar}_nv{curr_nviews}({self.args.pose_traj_video})"
        vis_src_list = []
        for i in range(curr_nviews):
            vis_src_list.append(source_views[i])
        frames_in = np.hstack(vis_src_list)
        return frames, vid_name, [frames_in, gt, all_depth]


trainer = PixelNeRFTrainer()
trainer.start()
