"""
Author: Eckert ZHANG
Date: 2021-11-04 21:49:58
LastEditTime: 2021-12-22 21:11:55
LastEditors: Eckert ZHANG
Description: 
"""
"""
Implements image encoders
"""
import torch, pdb
from torch import nn
import torch.nn.functional as F
import torchvision
import util
from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import imageio, math, glob, os


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_UV_strategy = False
        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            # print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models,
                                 backbone)(pretrained=pretrained,
                                           norm_layer=norm_layer)
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent",
                             torch.empty(1, 1, 1, 1),
                             persistent=False)
        self.register_buffer("latent_scaling",
                             torch.empty(2, dtype=torch.float32),
                             persistent=False)
        # self.latent (B, L, H, W)

    def index(self,
              uv,
              cam_z=None,
              image_size=(),
              z_bounds=None,
              freeze_enc=False):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            if freeze_enc:
                return samples[:, :, :, 0].detach()  # (B, C, N)
            else:
                return samples[:, :, :, 0]  # (B, C, N)

    def index_window(
        self,
        uv,
        cam_z=None,
        image_size=(),
        z_bounds=None,
        strategy='max',
        padding=5,
        attn_net=None,
        freeze_enc=False,
        visual_selection=False,
    ):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (SB, NV, N, 2) image points (u,v), while self.latent is (SB*NV, L, H, W)
                Here: N --> num of points
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
                if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :param padding, size of feature window
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index_window"):
            SB, NV, N, _ = uv.shape
            B, L, H, W = self.latent.shape
            device = uv.device
            assert SB * NV == B
            latent = self.latent.reshape(SB, NV, L, H, W)

            # pdb.set_trace()
            if visual_selection:
                visual_point_idx = [
                    int(N / 5),
                    int(2 * N / 5),
                    int(3 * N / 5),
                    int(4 * N / 5), N - 1
                ]
                images_reduce = self.images.detach().cpu().reshape(
                    SB, NV, 3, H, W)

            N_p = 2 * padding + 1
            ys, xs = torch.meshgrid(torch.linspace(-padding, padding, N_p),
                                    torch.linspace(-padding, padding, N_p))
            grid_padding = torch.stack((ys, xs)).permute(1, 2, 0).to(device)
            grid_padding = grid_padding / torch.tensor([W, H]).to(device)
            del ys, xs

            with profiler.record_function("encoder_index_pre_window"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0
                    grid_padding = grid_padding * self.latent_scaling

            uv = uv.unsqueeze(3)  # (SB, NV, N, 1, 2)
            grid_padding = grid_padding.unsqueeze(2).reshape(-1, 1, 2)
            grid_padding = grid_padding[None, None].repeat(NV - 1, N, 1, 1, 1)
            samples = []
            for i in range(SB):
                feat_ref = F.grid_sample(
                    latent[i, :1],
                    uv[i, :1],
                    align_corners=True,
                    mode=self.index_interp,
                    padding_mode=self.index_padding,
                )
                samples.append(feat_ref)

                # uv_warp =[NV-1, N, N_p*N_p, 1, 2]
                uv_warp = uv[i, 1:].unsqueeze(2).expand(
                    -1, -1, N_p * N_p, -1, -1) + grid_padding
                uv_warp = uv_warp.reshape(NV - 1, -1, 1, 2)
                feat_warp = F.grid_sample(
                    latent[i, 1:],
                    uv_warp,
                    align_corners=True,
                    mode=self.index_interp,
                    padding_mode=self.index_padding,
                )
                feat_warp = feat_warp.reshape(NV - 1, L, N, -1, 1)

                if strategy == 'max':
                    # score_sml = [NV-1, N, N_p*N_p, 1]
                    score_sml = torch.cosine_similarity(feat_ref.unsqueeze(3),
                                                        feat_warp,
                                                        dim=1)
                    idx = torch.argmax(score_sml, dim=2)

                    # for visualizing of selection features
                    if visual_selection:
                        save_path = './results_max_fea'
                        num_d = (SB + 4 * SB *
                                 (NV - 1)) * len(visual_point_idx)
                        num = int(
                            len(glob.glob(os.path.join(save_path, "*"))) /
                            num_d)
                        for point_idx in visual_point_idx:
                            # point_idx = visual_point_idx[0]
                            uvs_point = ((uv[i, :, point_idx, 0, :] + 1) /
                                         self.latent_scaling).detach().cpu()
                            vis_img = []
                            for v_idx in range(NV):
                                img_ref = (images_reduce[i, v_idx].permute(
                                    1, 2, 0) + 1) / 2
                                fea_ref = (latent[i, v_idx, :3].clone().permute( \
                                                      1, 2, 0))
                                H_p = int(uvs_point[v_idx, 1] * H)
                                W_p = int(uvs_point[v_idx, 0] * W)
                                if H_p > 0 and W_p > 0 and H_p < H and W_p < W:
                                    img_ref[H_p, W_p] = 1
                                    fea_ref[H_p, W_p] = 1
                                vis_img.append(img_ref.detach().cpu() * 255)
                                vis_img.append(fea_ref.detach().cpu() * 255)
                                # imageio.imwrite(
                                #     save_path +
                                #     f'/test_{num}_p_{point_idx}_img_in{v_idx+1}.png',
                                #     (img_ref * 255
                                #      ).detach().cpu().numpy().astype('uint8'))
                                # imageio.imwrite(
                                #     save_path +
                                #     f'/test_{num}_p_{point_idx}_fea_in{v_idx+1}.png',
                                #     (fea_ref * 255
                                #      ).detach().cpu().numpy().astype('uint8'))
                            vis_img = torch.cat(vis_img, dim=1)
                            imageio.imwrite(
                                save_path +
                                f'/test_{num}_p_{point_idx}_imgs.png',
                                vis_img.numpy().astype('uint8'))
                            # uv_src = [NV-1, N, N_p*N_p, 1, 2]
                            uv_src = uv_warp.reshape(
                                grid_padding.shape).detach().cpu()
                            for sub_i in range(1, NV):
                                img_src = (images_reduce[i, sub_i].permute(
                                    1, 2, 0) + 1) / 2
                                # fea_src = (latent[i,
                                #                   sub_i, :3].permute(1, 2, 0))
                                img_src_q = img_src.clone()
                                uvs_points = (
                                    uv_src[sub_i - 1, point_idx, :, 0, :] +
                                    1) / (self.latent_scaling).detach().cpu()
                                img_detail = torch.zeros([N_p, N_p,
                                                          3]).reshape(-1, 3)
                                mask_detail = torch.zeros([N_p,
                                                           N_p]).reshape(-1)
                                for jj in range(N_p * N_p):
                                    uvs_point = uvs_points[jj]
                                    H_p = int(torch.round(uvs_point[1] * H))
                                    W_p = int(torch.round(uvs_point[0] * W))
                                    if H_p > 0 and W_p > 0 and H_p < H and W_p < W:
                                        img_detail[jj] = img_src[H_p, W_p]
                                        img_src_q[H_p, W_p] = 1
                                        if jj == idx[sub_i - 1, point_idx, :]:
                                            img_src[H_p, W_p] = 1
                                            mask_detail[jj] = 1
                                img_detail = img_detail.reshape(
                                    N_p, N_p, 3).permute(2, 0, 1)[None]
                                mask_detail = mask_detail.reshape(1, N_p,
                                                                  N_p)[None]
                                img_detail = F.interpolate(
                                    img_detail,
                                    scale_factor=20,
                                    mode="area",
                                    align_corners=None,
                                    recompute_scale_factor=True,
                                )
                                mask_detail = F.interpolate(
                                    mask_detail,
                                    scale_factor=20,
                                    mode="area",
                                    align_corners=None,
                                    recompute_scale_factor=True,
                                )
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}.png',
                                    (img_src_q * 255
                                     ).detach().cpu().numpy().astype('uint8'))
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}_sel.png',
                                    (img_src * 255
                                     ).detach().cpu().numpy().astype('uint8'))
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}_det.png',
                                    (img_detail[0].permute(1, 2, 0) * 255
                                     ).detach().cpu().numpy().astype('uint8'))
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}_mask.png',
                                    (mask_detail[0].permute(1, 2, 0) * 255
                                     ).detach().cpu().numpy().astype('uint8'))

                    # method 1: select feature vector according to index
                    # seq=(torch.arange(0,(NV-1)*N)*N_p * N_p).unsqueeze(1).repeat(1,N_p * N_p).reshape(-1)
                    # idx = idx.repeat(1, 1, N_p * N_p).reshape(-1)+seq.to(device)
                    # feat_warp = feat_warp.permute(0, 2, 3,4, 1).reshape(-1, 1, L)
                    # feat_warp = feat_warp.index_select(0, idx).reshape(NV - 1, N, N_p * N_p, 1, L)
                    # feat_warp = feat_warp[:,:,0].permute(0,3,1,2)

                    # method 2: select feature vector according to index
                    idx = idx.unsqueeze(3).repeat(1, 1, 1, L)
                    feat_warp = feat_warp.permute(0, 2, 3, 4, 1).squeeze()
                    feat_warp = feat_warp.gather(dim=2, index=idx).permute(
                        0, 3, 1, 2)
                    samples.append(feat_warp)
                    del idx

                    # method 3: select feature vector according to index
                    # ff = torch.zeros([NV - 1, L, N, 1]).to(device)
                    # for i in range(NV - 1):
                    #     for j in range(N):
                    #         ff[i, :, j] = feat_warp[i, :, j, idx[i, j, 0]]
                    # samples.append(ff)
                    # del ff
                elif strategy == 'mean':
                    samples.append(torch.mean(feat_warp, dim=3))
                elif strategy == 'attention':
                    # feat_ref = [1, L, N_point, 1], feat_warp=[2, L, N_point, N_p*N_p, 1]
                    if visual_selection:
                        save_path = './results_att_fea_joint2'
                        num_d = (3 * SB * (NV - 1)) * len(visual_point_idx)
                        num = int(
                            len(glob.glob(os.path.join(save_path, "*"))) /
                            num_d)

                    ### To reduce memory usage, adopt type 1
                    # type 1: compute per view
                    feat_warp = feat_warp.squeeze()
                    feat_ref = feat_ref.permute(0, 2, 1, 3).reshape(-1, L, 1)
                    for nv_i in range(NV - 1):
                        # feat_warp=[1, L, N_point, N_p*N_p] --> [1, N_point, L, N_p*N_p]
                        feat_i = feat_warp[nv_i:nv_i + 1].permute(
                            0, 2, 1, 3).reshape(-1, L, N_p * N_p)
                        if freeze_enc:
                            weight = attn_net(feat_ref.detach(),
                                              feat_i.detach())
                        else:
                            weight = attn_net(feat_ref, feat_i)
                        #weight =[N,1,N_p*N_p]
                        weight = weight.unsqueeze(1)
                        feat_i = torch.bmm(weight, feat_i.transpose(1, 2))
                        # [1, N, 1, L] --> [1, L, N, 1]
                        feat_i = feat_i.reshape(1, N, 1, L).permute(0, 3, 1, 2)
                        samples.append(feat_i)

                        if visual_selection:
                            for point_idx in visual_point_idx:
                                # uv_src = [NV-1, N, N_p*N_p, 1, 2]
                                uv_src = uv_warp.reshape(
                                    grid_padding.shape).detach().cpu()
                                sub_i = nv_i + 1
                                img_src = (images_reduce[i, sub_i].permute(
                                    1, 2, 0) + 1) / 2
                                # fea_src = (latent[i,
                                #                   sub_i, :3].permute(1, 2, 0))
                                img_src_q = img_src.clone()
                                uvs_points = (
                                    uv_src[sub_i - 1, point_idx, :, 0, :] +
                                    1) / (self.latent_scaling).detach().cpu()
                                img_detail = torch.zeros([N_p, N_p,
                                                          3]).reshape(-1, 3)
                                mask_detail = torch.zeros([N_p,
                                                           N_p]).reshape(-1)
                                for jj in range(N_p * N_p):
                                    mask_detail[jj] = weight[point_idx, 0, jj]
                                    uvs_point = uvs_points[jj]
                                    H_p = int(torch.round(uvs_point[1] * H))
                                    W_p = int(torch.round(uvs_point[0] * W))
                                    # print(H_p,W_p)
                                    if H_p > 0 and W_p > 0 and H_p < H and W_p < W:
                                        img_detail[jj] = img_src[H_p, W_p]
                                        img_src_q[H_p, W_p] = 1
                                img_detail = img_detail.reshape(
                                    N_p, N_p, 3).permute(2, 0, 1)[None]
                                mask_detail = mask_detail.reshape(1, N_p,
                                                                  N_p)[None]
                                img_detail = F.interpolate(
                                    img_detail,
                                    scale_factor=20,
                                    mode="area",
                                    align_corners=None,
                                    recompute_scale_factor=True,
                                )
                                mask_detail = F.interpolate(
                                    mask_detail,
                                    scale_factor=20,
                                    mode="area",
                                    align_corners=None,
                                    recompute_scale_factor=True,
                                )
                                ma = mask_detail.max()
                                mi = mask_detail.min()
                                mask_detail = (mask_detail - mi) / (ma - mi)
                                # pdb.set_trace()
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}.png',
                                    (img_src_q * 255
                                     ).detach().cpu().numpy().astype('uint8'))
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}_det.png',
                                    (img_detail[0].permute(1, 2, 0) * 255
                                     ).detach().cpu().numpy().astype('uint8'))
                                imageio.imwrite(
                                    save_path +
                                    f'/test_{num}_p_{point_idx}_fea_src{sub_i}_mask.png',
                                    (mask_detail[0].permute(1, 2, 0) * 255
                                     ).detach().cpu().numpy().astype('uint8'))

                    # type 2: compute among views
                    # feat_warp = feat_warp.squeeze().permute(
                    #     0, 2, 1, 3).reshape(-1, L, N_p * N_p)
                    # feat_ref = feat_ref.repeat(NV - 1, 1, 1, \
                    #                 1).permute(0, 2, 1, 3).reshape(-1, L, 1)

                    # if freeze_enc:
                    #     weight = attn_net(feat_ref.detach(),
                    #                     feat_warp.detach())
                    # else:
                    #     weight = attn_net(feat_ref, feat_warp)
                    # weight = weight.unsqueeze(1)
                    # feat_warp = torch.bmm(weight, feat_warp.transpose(1, 2))
                    # feat_warp = feat_warp.reshape(NV - 1, N, 1,
                    #                             L).permute(0, 3, 1, 2)
                    # samples.append(feat_warp)

            del grid_padding, feat_warp
            samples = torch.cat(samples, dim=0)
            if freeze_enc and strategy != 'attention':
                samples = samples[:, :, :, 0].detach()  # (B, C, N)
            else:
                samples = samples[:, :, :, 0]  # (B, C, N)

            return samples  # (B, C, N)

    def forward(self, x, warpingnet=None, smtic=None):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.use_UV_strategy:
            self.images = x.detach()
        if warpingnet is not None:
            imgs = x.detach()
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            latent = torch.cat(latents, dim=1)
            #
            # import imageio
            # ff = latent.detach()
            # fn = [64,90,94]
            # for kk in range(ff.shape[0]):
            #     ff_p = ff[kk, fn].permute(1, 2, 0)
            #     imageio.imwrite(f'fea_o_{kk}.png', ff_p.cpu())
            #
        if warpingnet is not None:
            latent = warpingnet(imgs, smtic, latent)['warp_feamap']
        self.latent = latent
        
        # import imageio
        # ff = latent.detach()
        # fn = [64,90,94]
        # for kk in range(ff.shape[0]):
        #     ff_p = ff[kk, fn].permute(1, 2, 0)
        #     imageio.imwrite(f'fea_w_{kk}.png', ff_p.cpu())

        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling -
                                                     1) * 2.0
        

        if self.use_UV_strategy:
            scale_size = self.latent.shape[-1] / self.images.shape[-1]
            self.images = F.interpolate(
                self.images,
                scale_factor=scale_size,
                mode="area",
                align_corners=None,
                recompute_scale_factor=True,
            )
            return self.latent, self.images
        else:
            return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """
    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models,
                             backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
