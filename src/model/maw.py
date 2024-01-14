"""
Author: Eckert ZHANG
Date: 2022-01-12 20:28:25
LastEditTime: 2022-01-18 23:32:31
LastEditors: Eckert ZHANG
Description: 
"""
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[
            1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
            2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
        nn.ConvTranspose2d, nn.ConvTranspose3d
    ]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2,
                                                              0::2]


def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec**2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack(
            (1. - 2. * rvec[:, 1]**2 - 2. * rvec[:, 2]**2, 2. *
             (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]), 2. *
             (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]), 2. *
             (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
             1. - 2. * rvec[:, 0]**2 - 2. * rvec[:, 2]**2, 2. *
             (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]), 2. *
             (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]), 2. *
             (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
             1. - 2. * rvec[:, 0]**2 - 2. * rvec[:, 1]**2),
            dim=1).view(-1, 3, 3)


class ConvWarp(nn.Module):
    def __init__(self, d_in, displacementwarp=False):
        super(ConvWarp, self).__init__()
        self.d_in = d_in
        self.displacementwarp = displacementwarp

        self.warp1 = nn.Sequential(nn.Linear(self.d_in, 256),
                                   nn.LeakyReLU(0.2), nn.Linear(256, 1024),
                                   nn.LeakyReLU(0.2))
        self.warp2 = nn.Sequential(nn.ConvTranspose3d(1024, 512, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose3d(512, 512, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose3d(512, 256, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose3d(256, 256, 4, 2, 1),
                                   nn.LeakyReLU(0.2),
                                   nn.ConvTranspose3d(256, 3, 4, 2, 1))
        for m in [self.warp1, self.warp2]:
            initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1.0, 1.0, 32),
                                          np.linspace(-1.0, 1.0, 32),
                                          np.linspace(-1.0, 1.0, 32),
                                          indexing='ij')
        self.register_buffer(
            "grid",
            torch.tensor(
                np.stack((xgrid, ygrid, zgrid),
                         axis=0)[None].astype(np.float32)))

    def forward(self, encoding):
        finalwarp = self.warp2(self.warp1(encoding).view(-1, 1024, 1, 1,
                                                         1)) * (2. / 1024)
        if not self.displacementwarp:
            finalwarp = finalwarp + self.grid
        return finalwarp


class AffineMixWarp(nn.Module):
    def __init__(self, d_in, d_hidden=256, num_sub_warp=8):
        super(AffineMixWarp, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_sub = num_sub_warp
        print('**num_sub_warp=', num_sub_warp)

        self.quat = Quaternion()

        self.trunk = nn.Sequential(
            nn.Linear(self.d_in, self.d_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.LeakyReLU(0.2),
        )

        self.warps = nn.Sequential(nn.Linear(self.d_hidden, 128),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(128, 3 * self.n_sub))
        self.warpr = nn.Sequential(nn.Linear(self.d_hidden, 128),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(128, 4 * self.n_sub))
        self.warpt = nn.Sequential(nn.Linear(self.d_hidden, 128),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(128, 3 * self.n_sub))
        self.weightbranch = nn.Sequential(
            nn.Linear(self.d_hidden, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, self.n_sub * 32 * 32 * 32))
        for m in [
                self.trunk, self.warps, self.warpr, self.warpt,
                self.weightbranch
        ]:
            initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1.0, 1.0, 32),
                                          np.linspace(-1.0, 1.0, 32),
                                          np.linspace(-1.0, 1.0, 32),
                                          indexing='ij')
        self.register_buffer(
            "grid",
            torch.tensor(
                np.stack((xgrid, ygrid, zgrid),
                         axis=-1)[None].astype(np.float32)))

    def forward(self, encoding):
        encoding = self.trunk(encoding)
        warps = self.warps(encoding).view(encoding.size(0), self.n_sub, 3)
        warpr = self.warpr(encoding).view(encoding.size(0), self.n_sub, 4)
        warpt = self.warpt(encoding).view(encoding.size(0), self.n_sub,
                                          3) * 0.1
        warprot = self.quat(warpr.view(-1, 4)).view(encoding.size(0),
                                                    self.n_sub, 3, 3)

        weight = torch.exp(
            self.weightbranch(encoding).view(encoding.size(0), self.n_sub, 32,
                                             32, 32))

        warpedweight = torch.cat([
            F.grid_sample(
                weight[:, i:i + 1, :, :, :],
                torch.sum(
                    ((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :,
                                                                    None, :] *
                     warprot[:, None, None, None, i, :, :]),
                    dim=5) * warps[:, None, None, None, i, :],
                padding_mode='border') for i in range(weight.size(1))
        ],
                                 dim=1)

        warp = torch.sum(torch.stack([
            warpedweight[:, i, :, :, :, None] * (torch.sum(
                ((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :,
                                                                None, :] *
                 warprot[:, None, None, None, i, :, :]),
                dim=5) * warps[:, None, None, None, i, :])
            for i in range(weight.size(1))
        ],
                                     dim=1),
                         dim=1) / torch.sum(warpedweight, dim=1).clamp(
                             min=0.001)[:, :, :, :, None]

        return warp.permute(0, 4, 1, 2, 3)
