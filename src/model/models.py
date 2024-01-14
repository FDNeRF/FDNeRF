"""
Author: Eckert ZHANG
Date: 2021-07-11 22:59:14
LastEditTime: 2022-02-21 18:31:57
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf/src/model/models.py
Description: Main model implementation
"""
import torch
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import os.path as osp
import warnings
from .attention import AttentionNet


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.encoder = make_encoder(conf["encoder"])
        # Image features?
        self.use_encoder = conf.get_bool("use_encoder", True)

        # load a certain range (box) of features of image, 'box_size_half' is half size of box
        # self.use_UVfea_box = conf.get_bool("use_UVfea_box", False)
        # self.UVfea_strategy = conf.get_string("UVfea_strategy", 'max')
        # self.box_size_half = conf.get_int("box_size_half", 3)
        # self.multiview_fea_weighted = False  #
        # if self.UVfea_strategy == "attention":
        #     self.use_attn = True
        # else:
        #     self.use_attn = False

        self.use_xyz = conf.get_bool("use_xyz", False)
        # Must use some feature.
        assert self.use_encoder or self.use_xyz

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool("normalize_z", True)

        # Stop ConvNet gradient (freeze weights)
        self.stop_encoder_grad = (stop_encoder_grad)
        # Positional encoding
        self.use_code = conf.get_bool("use_code", False)
        # Positional encoding applies to viewdirs
        self.use_code_viewdirs = conf.get_bool("use_code_viewdirs", True)

        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)

        # Global image features?
        self.use_global_encoder = conf.get_bool("use_global_encoder", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1
        # if self.use_attn:
        #     self.attention = AttentionNet(D_in=d_latent,
        #                                   D_hidden=256,
        #                                   re_enc=False)
        # else:
        #     self.attention = None

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = ImageEncoder.from_conf(
                conf["global_encoder"])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 4

        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(
            conf["mlp_coarse"],
            d_in,
            d_latent,
            d_out=d_out,
        )
        self.mlp_fine = make_mlp(
            conf["mlp_fine"],
            d_in,
            d_latent,
            d_out=d_out,
            allow_empty=True,
        )
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        [summary]

        Args:
            images ([type]): (NS, 3, H, W), NS is number of input (aka source or reference) views
            poses ([type]): (NS, 4, 4)
            focal ([type]): focal's length () or (2) or (NS) or (NS, 2) [fx, fy]
            z_bounds ([type], optional): ignored argument (used in the past). Defaults to None.
            c ([type], optional): principal point None or () or (2) or (NS) or (NS, 2) [cx, cy]. 
                                default is center of image. Defaults to None.
        """
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1)  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

    def forward(self,
                xyz,
                coarse=True,
                viewdirs=None,
                far=False,
                visual_selection=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param 
            xyz (SB, B, 3)
            SB is batch of objects
            B is batch of points (in rays)
            NS is number of input views
        :return 
            (SB, B, 4) r g b sigma  
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3],
                                   xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1,
                                                             1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs,
                                                 NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(self.poses[:, None, :3, :3],
                                            viewdirs)  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat((z_feature, viewdirs),
                                          dim=1)  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            if self.use_encoder:
                # Grab encoder's latent code.
                uv = -xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-7)  # (SB, B, 2)
                if len(self.focal.shape) == 3:
                    nn = self.focal.shape[-1]
                    uv *= self.focal.reshape(-1, nn).unsqueeze(1)
                    nn = self.c.shape[-1]
                    uv += self.c.reshape(-1, nn).unsqueeze(1)
                else:
                    uv *= repeat_interleave(
                        self.focal.unsqueeze(1),
                        NS if self.focal.shape[0] > 1 else 1)
                    uv += repeat_interleave(
                        self.c.unsqueeze(1),
                        NS if self.c.shape[0] > 1 else 1)  # (SB*NS, B, 2)

                # load a range of features for warpped view, find the nearest feature to first view
                # load it object by object
                # output: latent (SB*NS, latent, B)
                latent = self.encoder.index(uv,
                                            None,
                                            self.image_shape,
                                            freeze_enc=self.stop_encoder_grad)

                # freezing encoder in .index()
                if self.stop_encoder_grad:
                    latent = latent.detach()
                # (SB*NS, latent, B) --> (SB*NS*B, latent)
                latent = latent.transpose(1, 2).reshape(-1, self.latent_size)

                if self.d_in == 0:
                    # z_feature not needed
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)

            if self.use_global_encoder:
                # Concat global latent code if enabled
                global_latent = self.global_encoder.latent
                assert mlp_input.shape[0] % global_latent.shape[0] == 0
                num_repeats = mlp_input.shape[0] // global_latent.shape[0]
                global_latent = repeat_interleave(global_latent, num_repeats)
                mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            # Interpret the output
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output

    def load_weights(self,
                     args,
                     opt_init=False,
                     strict=True,
                     device=None,
                     ckpt_path_init=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at <exp>/checkpoints/pixel_nerf_init to use as initialization.
        param: 
            opt_init - if true, loads from init checkpoint instead of usual even when resuming
        """
        if not args.resume:
            return
        ckpt_name = ("pixel_nerf_init"
                     if opt_init and args.resume else "pixel_nerf_latest")
        model_path = "%s/%s/%s/%s" % (args.resultdir, args.name,
                                      args.checkpoints_path, ckpt_name)

        if ckpt_path_init is not None and opt_init:
            model_path = ckpt_path_init

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(torch.load(model_path, map_location=device),
                                 strict=strict)
        elif args.resume:
            warnings.warn((
                "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                +
                "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                +
                "If training, unless you are startin a new experiment, please remember to pass --resume."
            ).format(model_path))
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        param: 
            opt_init - if true, saves from init checkpoint instead of usual
        """
        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"

        ckpt_path = osp.join(args.resultdir, args.name, args.checkpoints_path,
                             ckpt_name)
        torch.save(self.state_dict(), ckpt_path)
        return self
