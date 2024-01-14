
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b,c,h,w = flow.shape
    flow_norm = 2 * torch.cat([flow[:,:1,...]/(w-1),flow[:,1:,...]/(h-1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0,2,3,1)
    return deformation

def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """    
    b,c,h,w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed   

def warp_image(source_image, deformation):
    r"""warp the input image according to the deformation

    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """ 
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=False)
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation, align_corners=False) 

def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

class LayerNorm2d(nn.Module):
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, \
                self.weight.expand(normalized_shape),
                self.bias.expand(normalized_shape))

        else:
            return F.layer_norm(x, normalized_shape)

class ADAINHourglass(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, encoder_layers,
                 decoder_layers, nonlinearity, use_spect):
        super(ADAINHourglass, self).__init__()
        self.encoder = ADAINEncoder(image_nc, pose_nc, ngf, img_f,
                                    encoder_layers, nonlinearity, use_spect)
        self.decoder = ADAINDecoder(pose_nc, ngf, img_f, encoder_layers,
                                    decoder_layers, True, nonlinearity,
                                    use_spect)
        self.output_nc = self.decoder.output_nc

    def forward(self, x, z):
        return self.decoder(self.encoder(x, z), z)

class ADAINEncoder(nn.Module):
    def __init__(self,
                 image_nc,
                 pose_nc,
                 ngf,
                 img_f,
                 layers,
                 nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(ADAINEncoder, self).__init__()
        self.layers = layers
        self.input_layer = nn.Conv2d(image_nc,
                                     ngf,
                                     kernel_size=7,
                                     stride=1,
                                     padding=3)
        for i in range(layers):
            in_channels = min(ngf * (2**i), img_f)
            out_channels = min(ngf * (2**(i + 1)), img_f)
            model = ADAINEncoderBlock(in_channels, out_channels, pose_nc,
                                      nonlinearity, use_spect)
            setattr(self, 'encoder' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x, z):
        out = self.input_layer(x)
        out_list = [out]
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out, z)
            out_list.append(out)
        return out_list

class ADAINDecoder(nn.Module):
    """docstring for ADAINDecoder"""
    def __init__(self,
                 pose_nc,
                 ngf,
                 img_f,
                 encoder_layers,
                 decoder_layers,
                 skip_connect=True,
                 nonlinearity=nn.LeakyReLU(),
                 use_spect=False):

        super(ADAINDecoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.skip_connect = skip_connect
        use_transpose = True

        for i in range(encoder_layers - decoder_layers, encoder_layers)[::-1]:
            in_channels = min(ngf * (2**(i + 1)), img_f)
            in_channels = in_channels * 2 if i != (
                encoder_layers - 1) and self.skip_connect else in_channels
            out_channels = min(ngf * (2**i), img_f)
            model = ADAINDecoderBlock(in_channels, out_channels, out_channels,
                                      pose_nc, use_transpose, nonlinearity,
                                      use_spect)
            setattr(self, 'decoder' + str(i), model)

        self.output_nc = out_channels * 2 if self.skip_connect else out_channels

    def forward(self, x, z):
        out = x.pop() if self.skip_connect else x
        for i in range(self.encoder_layers - self.decoder_layers,
                       self.encoder_layers)[::-1]:
            model = getattr(self, 'decoder' + str(i))
            out = model(out, z)
            out = torch.cat([out, x.pop()], 1) if self.skip_connect else out
        return out

class ADAINEncoderBlock(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 feature_nc,
                 nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(ADAINEncoderBlock, self).__init__()
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv_0 = spectral_norm(
            nn.Conv2d(input_nc, output_nc, **kwargs_down), use_spect)
        self.conv_1 = spectral_norm(
            nn.Conv2d(output_nc, output_nc, **kwargs_fine), use_spect)

        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(output_nc, feature_nc)
        self.actvn = nonlinearity

    def forward(self, x, z):
        x = self.conv_0(self.actvn(self.norm_0(x, z)))
        x = self.conv_1(self.actvn(self.norm_1(x, z)))
        return x

class ADAINDecoderBlock(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 hidden_nc,
                 feature_nc,
                 use_transpose=True,
                 nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(ADAINDecoderBlock, self).__init__()
        # Attributes
        self.actvn = nonlinearity
        hidden_nc = min(input_nc,
                        output_nc) if hidden_nc is None else hidden_nc

        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if use_transpose:
            kwargs_up = {
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'output_padding': 1
            }
        else:
            kwargs_up = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        # create conv layers
        self.conv_0 = spectral_norm(
            nn.Conv2d(input_nc, hidden_nc, **kwargs_fine), use_spect)
        if use_transpose:
            self.conv_1 = spectral_norm(
                nn.ConvTranspose2d(hidden_nc, output_nc, **kwargs_up),
                use_spect)
            self.conv_s = spectral_norm(
                nn.ConvTranspose2d(input_nc, output_nc, **kwargs_up),
                use_spect)
        else:
            self.conv_1 = nn.Sequential(
                spectral_norm(nn.Conv2d(hidden_nc, output_nc, **kwargs_up),
                              use_spect),
                nn.Upsample(scale_factor=2, align_corners=False))
            self.conv_s = nn.Sequential(
                spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs_up),
                              use_spect),
                nn.Upsample(scale_factor=2, align_corners=False))
        # define normalization layers
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        self.norm_s = ADAIN(input_nc, feature_nc)

    def forward(self, x, z):
        x_s = self.shortcut(x, z)
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        x_s = self.conv_s(self.actvn(self.norm_s(x, z)))
        return x_s

class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias), nn.ReLU())
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1, 1)
        beta = beta.view(*beta.size()[:2], 1, 1)
        out = normalized * (1 + gamma) + beta
        return out


class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        """
        Mapping Net: Inputs = [:, len]

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

        for i in range(layer-1):
            net = nn.Sequential(
                nonlinearity, nn.Linear(descriptor_nc,
                                        descriptor_nc,
                                        bias=True))
            setattr(self, 'encoder' + str(i), net)

        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out
        return out

class MappingNet_window(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        """
        Mapping Net: Inputs = [:, len, win]

        Args:
            coeff_nc (_type_): dimension of input conditioned params
            descriptor_nc (_type_): dimension of output latent code
            layer (_type_): num of layers
        """
        super(MappingNet_window, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first_w = nn.Sequential(
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
            setattr(self, 'encoder_w' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc
    
    def forward(self, input_3dmm):
        out = self.first_w(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder_w' + str(i))
            out = model(out) + out[:, :, 3:-3]
        out = self.pooling(out).squeeze(dim=-1)
        return out

class WarpingNet(nn.Module):
    def __init__(self, image_nc, descriptor_nc, base_nc, max_nc, encoder_layer,
                 decoder_layer, use_spect):
        super(WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'nonlinearity': nonlinearity, 'use_spect': use_spect}

        if image_nc == 512:
            self.deform_base_on_fea = True
        else:
            self.deform_base_on_fea = False
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

    def forward(self, input_image, descriptor, feamap):
        final_output = {}
        if self.deform_base_on_fea:
            output = self.hourglass(feamap, descriptor)
        else:
            output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)

        deformation = convert_flow_to_deformation(
            final_output['flow_field'])
        final_output['warp_feamap'] = warp_image(
            feamap, deformation)
        return final_output

