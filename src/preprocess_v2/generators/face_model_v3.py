"""
Author: Eckert ZHANG
Date: 2021-11-10 18:15:51
LastEditTime: 2022-03-17 14:07:48
LastEditors: Eckert ZHANG
Description: 
"""
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_v2.util import flow_util
from preprocess_v2.generators.base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder
from preprocess_v2.generators.swin_transformer_nodspl import SwinTransformer3D_Encoder


class FaceGenerator(nn.Module):
    def __init__(self, mapping_net, warpping_net, editing_net, common):
        super(FaceGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net, **common)
        self.editing_net = EditingNet(**editing_net, **common)

    def forward(self, input_image, driving_source, stage=None):
        if stage == 'warp':
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
        else:
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            output['fake_image'] = self.editing_net(input_image,
                                                    output['warp_image'],
                                                    descriptor)
        return output


class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        """
        Mapping Net: Inputs = [:, 70]

        Args:
            coeff_nc (_type_): dimension of input conditioned params
            descriptor_nc (_type_): dimension of output latent code
            layer (_type_): num of layers
        """
        super(MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first = nn.Sequential(
            nn.Linear(coeff_nc, descriptor_nc, bias=True))

        for i in range(layer):
            net = nn.Sequential(
                nonlinearity, nn.Linear(descriptor_nc,
                                        descriptor_nc,
                                        bias=True))
            setattr(self, 'encoder' + str(i), net)

        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out
        return out


class MappingNet1(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        """
        Mapping Net: Inputs = [:, 70, 1]

        Args:
            coeff_nc (_type_): dimension of input conditioned params
            descriptor_nc (_type_): dimension of output latent code
            layer (_type_): num of layers
        """
        super(MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc,
                            descriptor_nc,
                            kernel_size=1,
                            padding=0,
                            bias=True))

        for i in range(layer):
            net = nn.Sequential(
                nonlinearity,
                torch.nn.Conv1d(descriptor_nc,
                                descriptor_nc,
                                kernel_size=1,
                                padding=0))
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out
        out = self.pooling(out)
        return out


class MappingNet0(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        """
        Mapping Net: Inputs = [:, 73, 27]

        Args:
            coeff_nc (_type_): dimension of input conditioned params
            descriptor_nc (_type_): dimension of output latent code
            layer (_type_): num of layers
        """
        super(MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc,
                            descriptor_nc,
                            kernel_size=7,
                            padding=0,
                            bias=True))

        for i in range(layer):
            net = nn.Sequential(
                nonlinearity,
                torch.nn.Conv1d(descriptor_nc,
                                descriptor_nc,
                                kernel_size=3,
                                padding=0,
                                dilation=3))
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:, :, 3:-3]
        out = self.pooling(out)
        return out


class WarpingNet(nn.Module):
    def __init__(self, image_nc, descriptor_nc, base_nc, max_nc, encoder_layer,
                 decoder_layer, use_spect):
        super(WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'nonlinearity': nonlinearity, 'use_spect': use_spect}

        self.descriptor_nc = descriptor_nc
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                        max_nc, encoder_layer, decoder_layer,
                                        **kwargs)

        self.flow_out = nn.Sequential(
            norm_layer(self.hourglass.output_nc), nonlinearity,
            nn.Conv2d(self.hourglass.output_nc,
                      2,
                      kernel_size=7,
                      stride=1,
                      padding=3))

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output = {}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)

        deformation = flow_util.convert_flow_to_deformation(
            final_output['flow_field'])
        final_output['warp_image'] = flow_util.warp_image(
            input_image, deformation)
        return final_output


class EditingNet(nn.Module):
    def __init__(self, image_nc, descriptor_nc, layer, base_nc, max_nc,
                 num_res_blocks, use_spect):
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {
            'norm_layer': norm_layer,
            'nonlinearity': nonlinearity,
            'use_spect': use_spect
        }
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc * 2, base_nc, max_nc, layer,
                                   **kwargs)
        self.swin3D_encoder = SwinTransformer3D_Encoder(
            in_chans=max_nc,  #image_nc * 2,
            embed_dim=max_nc,  #base_nc,
            embed_dim_max=max_nc,
            depths=[2, 2, 2],
            num_heads=[2, 4, 8],
            window_size=(3, 4, 4),  #(nv,w,h)
            patch_norm=True,
            downsample=None)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc,
                                   max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        B, C, W, H = input_image.shape
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        # # --> B, C, D, H, W
        _, c_dep, w_dep, h_dep = x[-1].shape
        x_dep = x[-1].reshape(-1, 3, c_dep, w_dep,
                              h_dep).permute(0, 2, 1, 3, 4)
        x_dep = self.swin3D_encoder(x_dep)
        x[-1] = x_dep.permute(0, 2, 1, 3, 4).reshape(-1, c_dep, w_dep, h_dep)
        gen_image = self.decoder(x, descriptor)
        return gen_image


class EditingNet0(nn.Module):
    def __init__(self, image_nc, descriptor_nc, layer, base_nc, max_nc,
                 num_res_blocks, use_spect):
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {
            'norm_layer': norm_layer,
            'nonlinearity': nonlinearity,
            'use_spect': use_spect
        }
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc * 2, base_nc, max_nc, layer,
                                   **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc,
                                   max_nc, layer, num_res_blocks, **kwargs)
        deug_test = 0

    def forward(self, input_image, warp_image, descriptor):
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image