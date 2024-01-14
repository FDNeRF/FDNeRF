"""
Author: Eckert ZHANG
Date: 2021-07-11 22:59:14
LastEditTime: 2022-03-17 18:09:10
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf/src/util/util.py
Description: 
"""
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math, lpips
import warnings
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim


# for forward facing scenes
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] -
                                     rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] -
                                     rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def cmap(img, color_map=cv2.COLORMAP_HOT):
    """
    Apply 'HOT' color to a float image
    """
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)


def batched_index_select_nd(t, inds):
    """
    Index select on dim 1 of a n-dimensional batched tensor.
    :param t (batch, n, ...)
    :param inds (batch, k)
    :return (batch, k, ...)
    """
    return t.gather(
        1, inds[(..., ) + (None, ) * (len(t.shape) - 2)].expand(
            -1, -1, *t.shape[2:]))


def batched_index_select_nd_last(t, inds):
    """
    Index select on dim -1 of a >=2D multi-batched tensor. inds assumed
    to have all batch dimensions except one data dimension 'n'
    :param t (batch..., n, m)
    :param inds (batch..., k)
    :return (batch..., n, k)
    """
    dummy = inds.unsqueeze(-2).expand(*inds.shape[:-1], t.size(-2),
                                      inds.size(-1))
    out = t.gather(-1, dummy)
    return out


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.0, ), (1.0, ))])


def homogeneous(points):
    """
    Concat 1 to each point
    :param points (..., 3)
    :return (..., 4)
    """
    return F.pad(points, (0, 1), "constant", 1.0)


def gen_grid(*args, ij_indexing=False):
    """
    Generete len(args)-dimensional grid.
    Each arg should be (lo, hi, sz) so that in that dimension points
    are taken at linspace(lo, hi, sz).
    Example: gen_grid((0,1,10), (-1,1,20))
    :return (prod_i args_i[2], len(args)), len(args)-dimensional grid points
    """
    return torch.from_numpy(
        np.vstack(
            np.meshgrid(*(np.linspace(lo, hi, sz, dtype=np.float32)
                          for lo, hi, sz in args),
                        indexing="ij" if ij_indexing else "xy")).reshape(
                            len(args), -1).T)


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0] + 1e-7)
    Y = Y.to(device=device) / float(f[1] + 1e-7)
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def coord_from_blender(dtype=torch.float32, device="cpu"):
    """
    Blender to standard coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def coord_to_blender(dtype=torch.float32, device="cpu"):
    """
    Standard to Blender coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def look_at(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Get 4x4 camera to world space matrix, for camera looking at target
    """
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world


def get_cuda(gpu_id):
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (torch.device("cuda:%d" % gpu_id)
            if torch.cuda.is_available() else torch.device("cpu"))


def masked_sample(masks, num_pix, prop_inside, thresh=0.5):
    """
    :return (num_pix, 3)
    """
    num_inside = int(num_pix * prop_inside + 0.5)
    num_outside = num_pix - num_inside
    inside = (masks >= thresh).nonzero(as_tuple=False)
    outside = (masks < thresh).nonzero(as_tuple=False)

    pix_inside = inside[torch.randint(0, inside.shape[0], (num_inside, ))]
    pix_outside = outside[torch.randint(0, outside.shape[0], (num_outside, ))]
    pix = torch.cat((pix_inside, pix_outside))
    return pix


def bbox_sample(bboxes, num_pix):
    """
    :return (num_pix, 3)
    """
    image_ids = torch.randint(0, bboxes.shape[0], (num_pix, ))
    pix_bboxes = bboxes[image_ids]
    x = (torch.rand(num_pix) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0]) +
         pix_bboxes[:, 0]).long()
    y = (torch.rand(num_pix) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1]) +
         pix_bboxes[:, 1]).long()
    pix = torch.stack((image_ids, y, x), dim=-1)
    return pix


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    # To Get camera unprojection map for given image size.
    # [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    if len(focal) == num_images:
        cc = []
        for ii in range(num_images):
            cam_map = unproj_map(width,
                                 height,
                                 focal[ii].squeeze(),
                                 c=c[ii],
                                 device=device)
            cc.append(cam_map)
        cam_unproj_map = torch.stack(cc)
    else:
        cam_unproj_map = (unproj_map(width,
                                     height,
                                     focal.squeeze(),
                                     c=c,
                                     device=device).unsqueeze(0).repeat(
                                         num_images, 1, 1, 1))
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(poses[:, None, None, :3, :3],
                              cam_unproj_map.unsqueeze(-1))[:, :, :, :, 0]
    if ndc:
        z_near, z_far = torch.zeros_like(z_near), torch.ones_like(z_far)
        if len(focal) == num_images:
            cc, cr = [], []
            for ii in range(num_images):
                cam_c, ray_d = ndc_rays(height, width, focal[ii, 0], 1.0,
                                        cam_centers[ii], cam_raydir[ii])
                cc.append(cam_c)
                cr.append(ray_d)
            cam_centers = torch.stack(cc)
            cam_raydir = torch.stack(cr)
            del cc, cr
        else:
            cam_centers, cam_raydir = ndc_rays(height, width, focal[0], 1.0,
                                               cam_centers, cam_raydir)
        # cam_centers, cam_raydir = ndc_rays(width, height, focal, 1.0,
        #                                    cam_centers, cam_raydir)

    assert len(z_near) == len(z_far), \
        "Length of 'z_near' and 'z_far' should be same!!!"
    if len(z_near) == num_images:
        cam_nears = z_near.clone().detach().view(-1, 1, 1, 1).expand(
            -1, height, width, -1).to(device)
        cam_fars = z_far.clone().detach().view(-1, 1, 1, 1).expand(
            -1, height, width, -1).to(device)
    else:
        cam_nears = (torch.tensor(z_near,
                                  device=device).view(1, 1, 1, 1).expand(
                                      num_images, height, width, -1))
        cam_fars = (torch.tensor(z_far, device=device).view(1, 1, 1, 1).expand(
            num_images, height, width, -1))
    return torch.cat((cam_centers, cam_raydir, cam_nears, cam_fars),
                     dim=-1)  # (B, H, W, 8)


def trans_t(t):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (torch.tensor(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    ) @ c2w)
    return c2w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d,
                                       affine=True,
                                       track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=False,
                                       track_running_stats=False)
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" %
                                  norm_type)
    return norm_layer


def make_conv_2d(
    dim_in,
    dim_out,
    padding_type="reflect",
    norm_layer=None,
    activation=None,
    kernel_size=3,
    use_bias=False,
    stride=1,
    no_pad=False,
    zero_init=False,
):
    conv_block = []
    amt = kernel_size // 2
    if stride > 1 and not no_pad:
        raise NotImplementedError(
            "Padding with stride > 1 not supported, use same_pad_conv2d")

    if amt > 0 and not no_pad:
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(amt)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(amt)]
        elif padding_type == "zero":
            conv_block += [nn.ZeroPad2d(amt)]
        else:
            raise NotImplementedError("padding [%s] is not implemented" %
                                      padding_type)

    conv_block.append(
        nn.Conv2d(dim_in,
                  dim_out,
                  kernel_size=kernel_size,
                  bias=use_bias,
                  stride=stride))
    if zero_init:
        nn.init.zeros_(conv_block[-1].weight)
    #  else:
    #  nn.init.kaiming_normal_(conv_block[-1].weight)
    if norm_layer is not None:
        conv_block.append(norm_layer(dim_out))

    if activation is not None:
        conv_block.append(activation)
    return nn.Sequential(*conv_block)


def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height,
                           0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def same_pad_conv2d(t,
                    padding_type="reflect",
                    kernel_size=3,
                    stride=1,
                    layer=None):
    """
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(t,
                 calc_same_pad_conv2d(t.shape, kernel_size, stride),
                 mode=padding_type)


def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride)
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]


def combine_interleaved(t, inner_dims=(1, ), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    elif agg_type == "weighted":
        t_ref = t[:, 0]
        t_wrp = torch.mean(t[:, 1:], dim=1)
        t = 0.6 * t_ref + 0.4 * t_wrp
    elif agg_type == "concat":
        # ts = [t[:,i] for i in range(inner_dims[0])]
        # t = torch.cat(ts, dim=-1)
        SB, NV, B, _ = t.shape
        t = t.transpose(1, 2).reshape(SB, B, -1)
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def psnr(pred, target):
    """
    Compute PSNR of two tensors in decibels.
    pred/target should be of same size or broadcastable
    """
    mse = ((pred - target)**2).mean()
    psnr = -10 * math.log10(mse)
    return psnr

def metric_function(pred, target, data_range, device):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    psnr = ski_psnr(target, pred, data_range=data_range)
    ssim = ski_ssim(target,
                    pred,
                    data_range=data_range,
                    multichannel=True)
    # normalize=False: input should be range [-1, +1], normalize=True: [0, 1]
    lpips_v = loss_fn_alex(torch.from_numpy(pred).permute(2,0,1).to(device), torch.from_numpy(target).permute(2,0,1).to(device), normalize=True) 
    lpips_v = np.mean(lpips_v.cpu().detach().numpy())
    return psnr,ssim,lpips_v


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    """
    Rotation matrix to quaternion
    """
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_module(net):
    """
    Shorthand for either net.module (if net is instance of DataParallel) or net
    """
    if isinstance(net, torch.nn.DataParallel):
        return net.module
    else:
        return net


def add_color_border(img, color=[255, 0, 0], linewidth=3):
    for c in range(len(color)):
        img[:linewidth, :, c] = color[c] / 255.0
        img[-linewidth:, :, c] = color[c] / 255.0
        img[:, :linewidth, c] = color[c] / 255.0
        img[:, -linewidth:, c] = color[c] / 255.0
    return img


def add_text_psnr(img, psnr, color=[255, 255, 255]):
    h, w, c = img.shape
    img_back = np.zeros(img.shape, dtype=np.uint8)
    text = 'PSNR: %.3f' % psnr
    cv2.putText(img_back, text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    # cv2.putText(img_back, text, (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #             (255, 255, 255), 3)
    # cv2.imwrite('./test.png', img_back)
    img_back = img_back / 255
    img_tex = img_back
    for i in range(3):
        img_tex[:, :, i] = img_tex[:, :, i] * color[i] / 255.0
    img = img * (1 - img_back) + img_tex
    return img


# Computes poses that follow a spiral path for rendering purpose.
def pose_spherical_dtu(radii,
                       focus_depth,
                       n_poses=120,
                       world_center=np.array([0, 0, 0])):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi,
                         n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center + world_center],
                                  1)]  # (3, 4)

    return np.stack(poses_spiral, 0) @ np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                                 [0, 0, -1, 0], [0, 0, 0, 1]
                                                 ])  # (n_poses, 3, 4)


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) *
            rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_far, rads_scale=0.5, N_views=120):

    # center pose
    c2w = poses_avg(c2ws_all)
    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2

    tt = c2ws_all[:, :3, 3] - c2w[:3, 3][None]
    # tt = np.matmul(c2w[:3, :3].T, (c2ws_all[:, :3, 3] - c2w[:3, 3])[..., np.newaxis])[..., 0]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w,
                                      up,
                                      rads,
                                      focal,
                                      zdelta,
                                      zrate=.5,
                                      N=N_views)
    # render_poses0 = pose_spherical_dtu(rads, focal, n_poses=N_views)
    return np.stack(render_poses)

def render_path_axis(c2w, up, ax, rad, focal, N):
    render_poses = []
    center = c2w[:3,3]
    v = c2w[:3,ax] * rad
    for t in np.linspace(-1.,1.,N+1)[:-1]:
        c = center + t * v
        z = normalize(c - (center - focal * c2w[:3,2]))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spherical_pose(c2ws_all, near_far, rads_scale=0.5, N_views=120):
    # center pose
    c2w = poses_avg(c2ws_all)
    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = np.array(mean_dz)

    # Get spherical pose
    render_poses = []
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    for theta in np.linspace(-15, 15, N_views+1):
        angle = theta/180*np.pi
        # diantou
        # c = center + focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
        c = center + focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)

def get_composite_pose(c2ws_all, near_far, rads_scale=0.5, N_views=120):  # TODO: need to modify (Eckert 22/05)
    # center pose
    c2w = poses_avg(c2ws_all)
    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = 1.3 #np.array(mean_dz) * 0.8

    # Get spherical pose
    render_poses = []
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    NV1 = int(N_views//4)
    NV2 = N_views-2*NV1
    max_angle = 30
    for theta in np.linspace(-max_angle, max_angle, NV1):
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    
    # rad = np.sqrt(np.sum(np.square(render_poses[-1][:3, 3]-center)))
    rad = np.linalg.norm(render_poses[-1][:3, 3]-render_poses[0][:3, 3]) * 0.5
    for theta in np.linspace(np.pi, 2. * np.pi + np.pi, NV2 + 1)[:-1]:
        c = center + rad*(normalize(c2w[:3, 0])*np.cos(theta)+normalize(c2w[:3, 1])*np.sin(theta))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))

    for theta in np.linspace(max_angle, -max_angle, NV1):
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)


def get_circle_spiral_poses_from_pose(c2w, f_delta=0, N_views=120, n_r=2):
    """
    focal distance is the distance between c_cam and origin;
    Here, we let 'focal' value change in the range [focal-f_delta, focal+f_delta],
    when f_delta=0, the focal will be fixed.
    """
    # standard pose
    focal = np.linalg.norm(c2w[:3, 3])
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]

    # Get start pose
    angle_h_start = 10
    angle = -angle_h_start/180*np.pi
    c_s = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
    z = normalize(c_s - origin)
    pose_start = viewmatrix(z, up, c_s)

    render_poses = []
    alpha_list = list(np.linspace(0, 360*n_r, N_views))
    if f_delta > 0:
        # NV1 = int(N_views//4)
        # NV2 = N_views-3*NV1
        # focals = list(np.linspace(focal, focal+f_delta, NV1))+list(np.linspace(focal+f_delta, focal-f_delta, 2*NV1))+list(np.linspace(focal-f_delta, focal, NV2))
        focals = list(np.linspace(focal+f_delta, focal-f_delta, N_views))
    else:
        focals = [focal for i in range(N_views)]

    # for i, alpha in enumerate(alpha_list):
    #     angle = alpha/180*np.pi
    #     f = focals[i]
    #     r = f * np.sin(angle_h_start/180*np.pi)
    #     c = c_s + r*(normalize(c2w[:3, 0])*(1-np.cos(angle))+up*np.sin(angle))
    #     z = normalize(c - origin)
    #     render_poses.append(viewmatrix(z, up, c))

    r = focal * np.sin(angle_h_start/180*np.pi)
    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        f = focals[i]
        b = center - f * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
        c = b + r * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))

    return np.stack(render_poses)

def get_standard_poses_from_tar_pose(c2w, N_views=120):
    focal = np.linalg.norm(c2w[:3, 3])
    # Get average pose
    up = normalize(c2w[:3, 1])
    
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    NV1 = int(N_views//8)
    NV2 = N_views-7*NV1
    max_angle_h = 25
    max_angle_v = 20
    max_angle_v2 = 20

    render_poses = []
    theta_list = list(np.linspace(0, -max_angle_h, NV1)) + list(np.linspace(-max_angle_h, max_angle_h, 2*NV1)) + list(np.linspace(max_angle_h, 0, NV1))
    for theta in theta_list:
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    theta_list = list(np.linspace(0, -max_angle_v, NV1)) + list(np.linspace(-max_angle_v, max_angle_v2, 2*NV1)) + list(np.linspace(max_angle_v2, 0, NV2))
    for theta in theta_list:
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    
    return np.stack(render_poses)

def get_circle_spiral_poses(focal=1.4, f_delta=0, N_views=120, n_r=2):
    """
    focal distance is the distance between c_cam and origin;
    Here, we let 'focal' value change in the range [focal-f_delta, focal+f_delta],
    when f_delta=0, the focal will be fixed.
    """
    # standard pose
    c2w = np.eye(4)
    c2w[1,3] = 0.01
    c2w[2,3] = focal
    up = normalize(c2w[:3, 1])
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]

    # Get start pose
    angle_h_start = 15
    angle = -angle_h_start/180*np.pi
    c_s = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
    z = normalize(c_s - origin)
    pose_start = viewmatrix(z, up, c_s)

    render_poses = []
    alpha_list = list(np.linspace(0, 360*n_r, N_views))
    if f_delta > 0:
        # NV1 = int(N_views//4)
        # NV2 = N_views-3*NV1
        # focals = list(np.linspace(focal, focal+f_delta, NV1))+list(np.linspace(focal+f_delta, focal-f_delta, 2*NV1))+list(np.linspace(focal-f_delta, focal, NV2))
        focals = list(np.linspace(focal+f_delta, focal-f_delta, N_views))
    else:
        focals = [focal for i in range(N_views)]

    # for i, alpha in enumerate(alpha_list):
    #     angle = alpha/180*np.pi
    #     f = focals[i]
    #     r = f * np.sin(angle_h_start/180*np.pi)
    #     c = c_s + r*(normalize(c2w[:3, 0])*(1-np.cos(angle))+up*np.sin(angle))
    #     z = normalize(c - origin)
    #     render_poses.append(viewmatrix(z, up, c))

    
    r = focal * np.sin(angle_h_start/180*np.pi)
    c_s = []
    for i, alpha in enumerate(alpha_list):
        angle = alpha/180*np.pi
        # if np.sin(angle) > 0:
        #     angle_h = (angle_h_start/180*np.pi)*(1-0.5*np.abs(np.sin(angle)))
        # else:
        #     angle_h = 1.0 * (angle_h_start/180*np.pi)
        # r = focal * np.sin(angle_h)
        f = focals[i]
        b = center - f * (1-np.cos(angle_h_start/180*np.pi)) * normalize(c2w[:3, 2])
        c = b + r * (normalize(c2w[:3, 0]) * np.cos(angle) - up * np.sin(angle))
        c_s.append(c)
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    c_s = np.stack(c_s)

    # ----------------------------------------------------------------
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D 
    # x = c_s[:60, 0]
    # y = c_s[:60, 0]
    # z = c_s[:60, 0]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y, z)
    # ax.view_init(elev=90., azim=0)
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.savefig('test_points.png')

    # ----------------------------------------------------------------
    

    return np.stack(render_poses)

def get_standard_poses(focal = 1.4, N_views=120):
    # standard pose
    c2w = np.eye(4)
    c2w[1,3] = 0.01
    c2w[2,3] = focal
    # Get average pose
    up = normalize(c2w[:3, 1])
    
    
    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    NV1 = int(N_views//8)
    NV2 = N_views-7*NV1
    max_angle_h = 25
    max_angle_v = 15
    max_angle_v2 = 10

    render_poses = []
    theta_list = list(np.linspace(0, -max_angle_h, NV1)) + list(np.linspace(-max_angle_h, max_angle_h, 2*NV1)) + list(np.linspace(max_angle_h, 0, NV1))
    for theta in theta_list:
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    theta_list = list(np.linspace(0, -max_angle_v, NV1)) + list(np.linspace(-max_angle_v, max_angle_v2, 2*NV1)) + list(np.linspace(max_angle_v2, 0, NV2))
    for theta in theta_list:
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    
    return np.stack(render_poses)

def get_standard_static_pose(focal = 1.4):
    # standard pose
    c2w = np.eye(4)
    c2w[1,3] = 0.01
    c2w[2,3] = focal
    # Get average pose
    up = normalize(c2w[:3, 1])

    angle = -10/180*np.pi
    c = c2w[:3, 3] - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
    z = normalize(c - c2w[:3, 3] + focal*c2w[:3, 2])
    c2w = viewmatrix(z, up, c)
    up = normalize(c2w[:3, 1])


    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    max_angle_h = 25
    max_angle_v = 20
    max_angle_v2 = 15
    
    render_poses = []
    theta_list = [max_angle_h, 0, -max_angle_h]
    for theta in theta_list:
        angle = theta/180*np.pi
        c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
        z = normalize(c - origin)
        render_poses.append(viewmatrix(z, up, c))
    # theta_list = [-max_angle_v, max_angle_v2]
    # for theta in theta_list:
    #     angle = theta/180*np.pi
    #     c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
    #     z = normalize(c - origin)
    #     render_poses.append(viewmatrix(z, up, c))
    
    return np.stack(render_poses)

def get_standard_static_pose2(focal = 1.4):
    # standard pose
    c2w = np.eye(4)
    c2w[1,3] = 0.01
    c2w[2,3] = focal
    # Get average pose
    up = normalize(c2w[:3, 1])

    angle = -10/180*np.pi
    c = c2w[:3, 3] - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
    z = normalize(c - c2w[:3, 3] + focal*c2w[:3, 2])
    c2w = viewmatrix(z, up, c)
    up = normalize(c2w[:3, 1])


    center = c2w[:3, 3]
    origin = center - focal*c2w[:3, 2]
    max_angle_h = 20
    max_angle_v = 20
    max_angle_v2 = 15
    
    render_poses = []
    theta_list = [0, ]
    theta_y_list = [0, ]
    for theta in theta_list:
        for theta_y in theta_y_list:
            angle = theta/180*np.pi
            angle_y = theta_y/180*np.pi
            c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle)) - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle_y))+normalize(c2w[:3, 1])*np.sin(angle_y))
            z = normalize(c - origin)
            render_poses.append(viewmatrix(z, up, c))

    # for theta in theta_list:
    #     angle = theta/180*np.pi
    #     c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 0])*np.sin(angle))
    #     z = normalize(c - origin)
    #     render_poses.append(viewmatrix(z, up, c))
    # theta_list = [-max_angle_v, max_angle_v2]
    # for theta in theta_list:
    #     angle = theta/180*np.pi
    #     c = center - focal*(normalize(c2w[:3, 2])*(1-np.cos(angle))+normalize(c2w[:3, 1])*np.sin(angle))
    #     z = normalize(c - origin)
    #     render_poses.append(viewmatrix(z, up, c))
    
    return np.stack(render_poses)
