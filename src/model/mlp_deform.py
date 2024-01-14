"""
Author: Eckert ZHANG
Date: 2021-12-20 19:30:04
LastEditTime: 2022-03-09 23:23:19
LastEditors: Eckert ZHANG
Description: 
"""
import torch
from torch import nn
import numpy as np
from typing import Any, Optional, Tuple
import util
from .code import PositionalEncoding
from .deform_util import exp_se3, to_homogenous, from_homogenous


class _MLP(nn.Module):
    """Basic MLP class with hidden layers and an output layer."""
    def __init__(
        self,
        channel_in: int,
        num_layers: int,
        channel_hidden: int,
        channel_out: int,
        use_bias: bool = True,
        skips: Tuple[int] = tuple(),
        activation_hidden: str = 'relu',
        activation_out=None,
        init_type: str = 'const',
        init_para: dict = {
            'bias': 0.0,
            'weight': 0.0
        },
    ):
        """
        Define a basic MLP with an input layer.

        Args:
            channel_in (int): [description]
            num_layers (int): [description]
            channel_hidden (int): [description]
            channel_out (int): [description]
            use_bias (bool, optional): [description]. Defaults to True.
            skips (Tuple[int], optional): [description]. Defaults to tuple().
        """
        super().__init__()
        self.c_in = channel_in
        self.c_out = channel_out
        self.c_hidden = channel_hidden
        self.n_layers = num_layers
        self.skips = skips
        self.use_bias = use_bias
        self.activ_h = activation_hidden
        self.activ_out = activation_out

        self.init_type = init_type
        self.init_para = init_para

        self.linears = nn.ModuleList()
        if self.n_layers > 0:
            self.linears = nn.ModuleList([nn.Linear(self.c_in, self.c_hidden, bias=self.use_bias)] + \
                [nn.Linear(self.c_hidden, self.c_hidden, bias=self.use_bias) if i not in self.skips \
                    else nn.Linear(self.c_hidden + self.c_in, self.c_hidden, bias=self.use_bias)
                for i in range(1, self.n_layers)
            ])
        self.activation_hidden = None
        if self.activ_h == 'relu':
            self.activation_hidden = nn.ReLU()

        for lin in self.linears:
            if self.use_bias:
                nn.init.constant_(lin.bias, self.init_para['bias'])

            if self.init_type == 'const':
                nn.init.constant_(lin.weight, self.init_para['weight'])
            elif self.init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")
            elif self.init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(lin.weight, a=0, mode="fan_in")
            elif self.init_type == 'uniform':
                nn.init.uniform_(lin.weight, a=0, b=1)
            elif self.init_type == 'normal':
                nn.init.normal_(lin.weight, mean=0, std=1)

        self.out_linear = None
        self.activation_out = None
        if self.c_out > 0:
            self.out_linear = nn.Linear(self.c_hidden,
                                        self.c_out,
                                        bias=self.use_bias)
            if self.activ_out == 'relu':
                self.activation_out = nn.ReLU()
            elif self.activ_out == 'sigmoid':
                self.activation_out = nn.Sigmoid()

            if self.init_type == 'const':
                nn.init.constant_(self.out_linear.weight,
                                  self.init_para['weight'])
            elif self.init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(self.out_linear.weight,
                                        a=0,
                                        mode="fan_in")
            elif self.init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(self.out_linear.weight,
                                         a=0,
                                         mode="fan_in")
            elif self.init_type == 'uniform':
                nn.init.uniform_(self.out_linear.weight, a=0, b=1)
            elif self.init_type == 'normal':
                nn.init.normal_(self.out_linear.weight, mean=0, std=1)
            if self.use_bias:
                nn.init.constant_(self.out_linear.bias, self.init_para['bias'])

    def forward(self, x):
        inputs = x
        for i, layer in enumerate(self.linears):
            if i in self.skips:
                x = torch.cat([x, inputs], -1)
            x = layer(x)
            if self.activation_hidden is not None:
                x = self.activation_hidden(x)
        if self.out_linear is not None:
            x = self.out_linear(x)
            if self.activation_out is not None:
                x = self.activation_out(x)
        return x


class mlp_project(nn.Module):
    """
    Project Net conditioned on embed parameters to map (x,y,z) to its position in camera coordinate (UV coordinates) w.r.t. input views
    """
    def __init__(
        self,
        conf,
        d_point_in=3,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf['code'],
                                                        d_in=d_point_in)
        self.channel_embed = conf.get_int("channel_embed", 134)
        self.channel_in = self.pos_encoder.d_out + self.channel_embed
        self.trunk_layer = conf.get_int("trunk_layer", 6)
        self.post_layer = conf.get_int("post_layer", 2)
        self.channel_hidden = conf.get_int("channel_hidden", 256)
        self.use_skip_project = conf.get_bool("use_skip_project", True)
        self.skips_project = (4, ) if self.use_skip_project else ()

        self.use_hyper_net = conf.get_bool("hyper_net", True)
        self.channel_hyper_out = 0
        if self.use_hyper_net:
            self.hyper_layer = conf.get_int("hyper_layer", 6)
            self.channel_hyper_out = conf.get_int("channel_hyper_out", 64)
            self.use_skip_hyper = conf.get_bool("use_skip_hyper", True)
            self.skips_hyper = (4, ) if self.use_skip_hyper else ()

        self.project_trunk = _MLP(
            self.channel_in,
            self.trunk_layer,
            self.channel_hidden,
            channel_out=0,
            use_bias=True,
            skips=self.skips_project,
            activation_hidden='relu',
            init_type='kaiming_normal',
        )
        self.project_post = _MLP(
            self.channel_hidden + self.channel_hyper_out,
            self.post_layer,
            self.channel_hidden,
            channel_out=0,
            use_bias=True,
            skips=(),
            activation_hidden='relu',
            init_type='kaiming_normal',
        )
        self.project_out = nn.ModuleList(
            [nn.Linear(self.channel_hidden, 128, bias=True)] +
            [nn.Linear(128, 64, bias=True)] + [nn.Linear(64, 3, bias=True)])
        self.activation_out = nn.ReLU()
        for layer in self.project_out:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.constant_(layer.weight, 0.0)

        if self.use_hyper_net:
            self.hyper_net = _MLP(
                self.channel_in,
                self.hyper_layer,
                self.channel_hidden,
                channel_out=self.channel_hyper_out,
                use_bias=True,
                skips=self.skips_hyper,
                activation_hidden='relu',
                init_type='kaiming_normal',
            )

    def forward(self, points, params_embed):
        """
        Args:
            points (SB*NS*B, 3): sample points on the rays in world coordinate
            params_embed (SB*NS*B, len): pose & expression parameters

        Returns:
            project_out (SB*NS*B, 3): projected position in cam_coordinate conditioned by params_embed
        """
        points_embed = self.pos_encoder(points)
        inputs = torch.cat([points_embed, params_embed], dim=-1)
        project_out = self.project_trunk(inputs)
        if self.use_hyper_net:
            hyper_out = self.hyper_net(inputs)
            project_out = torch.cat([project_out, hyper_out], dim=-1)
        project_out = self.project_post(project_out)
        for l in range(len(self.project_out)):
            if l < len(self.project_out) - 1:
                project_out = self.activation_out(
                    self.project_out[l](project_out))
            else:
                project_out = self.project_out[l](project_out)

        return project_out


class TansField2_2Dedit(nn.Module):
    """
    Represents a Trans_MLP
    """
    def __init__(
        self,
        conf,
        d_emb_in,
        d_point_in=3,
        d_hidden=128,
        n_layers=6,
        skips=(4, ),
        d_out=3,
        combine_layer=3,
        combine_type="average",
        use_bias=True,
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=d_point_in)
        self.num_layers = n_layers
        self.d_in = d_emb_in + self.pos_encoder.d_out
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.skips = skips
        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_bias = use_bias

        # self.linears = nn.ModuleList([nn.Linear(self.d_in, self.d_hidden, bias=self.use_bias)] + \
        #         [nn.Linear(self.d_hidden, self.d_hidden, bias=self.use_bias) if i not in self.skips \
        #             else nn.Linear(self.d_hidden + self.d_in, self.d_hidden, bias=self.use_bias)
        #         for i in range(1, self.num_layers)
        #     ])
        layers = []
        for i in range(1, self.num_layers):
            if i not in self.skips:
                if i == self.combine_layer:
                    layers.append(
                        nn.Linear(2 * self.d_hidden,
                                  self.d_hidden,
                                  bias=self.use_bias))
                else:
                    layers.append(
                        nn.Linear(self.d_hidden,
                                  self.d_hidden,
                                  bias=self.use_bias))
            else:
                layers.append(
                    nn.Linear(self.d_hidden + self.d_in,
                              self.d_hidden,
                              bias=self.use_bias))
        self.linears = nn.ModuleList(
            [nn.Linear(self.d_in, self.d_hidden, bias=self.use_bias)] + layers)

        for lin in self.linears:
            nn.init.constant_(lin.bias, 0.0)
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()

        self.out_linear = nn.Linear(self.d_hidden,
                                    self.d_out,
                                    bias=self.use_bias)
        nn.init.constant_(self.out_linear.bias, 0.0)
        nn.init.constant_(self.out_linear.weight, 0.0)

    def forward(self, points, params_embed, point_w, combine_inner_dims=(1, )):
        """
        combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer; where combine_inner_dims = (num_view_per_obj, batch of points)
        """
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            x = torch.cat([points_embed, params_embed], dim=-1)
        else:
            x = points_embed
        x_init = x
        for layer in range(0, self.num_layers):
            lin = self.linears[layer]
            if layer == self.combine_layer:
                x_avg = util.combine_interleaved(x, combine_inner_dims,
                                                 self.combine_type)
                x_avg = x_avg.reshape(-1, self.d_hidden)
                x_avg = util.repeat_interleave(x_avg, combine_inner_dims[0])
                x = torch.cat([x, x_avg], -1)
                # x_init = util.combine_interleaved(x_init, combine_inner_dims,
                #                                   self.combine_type)

            if layer in self.skips:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = self.activation(lin(x))

        translation = self.out_linear(x)
        warped_points = point_w + translation.reshape(-1, 3)

        return warped_points


class SE3Field2_2Dedit(nn.Module):
    """
    Represents a Trans_MLP
    """
    def __init__(
        self,
        conf,
        d_emb_in,
        d_point_in=3,
        d_hidden=128,
        n_layers=6,
        skips=(4, ),
        d_out=3,
        combine_layer=3,
        combine_type="average",
        use_bias=True,
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=d_point_in)
        self.num_layers = n_layers
        self.d_in = d_emb_in + self.pos_encoder.d_out
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.skips = skips
        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_bias = use_bias

        # self.linears = nn.ModuleList([nn.Linear(self.d_in, self.d_hidden, bias=self.use_bias)] + \
        #         [nn.Linear(self.d_hidden, self.d_hidden, bias=self.use_bias) if i not in self.skips \
        #             else nn.Linear(self.d_hidden + self.d_in, self.d_hidden, bias=self.use_bias)
        #         for i in range(1, self.num_layers)
        #     ])
        layers = []
        for i in range(1, self.num_layers):
            if i not in self.skips:
                if i == self.combine_layer:
                    layers.append(
                        nn.Linear(2 * self.d_hidden,
                                  self.d_hidden,
                                  bias=self.use_bias))
                else:
                    layers.append(
                        nn.Linear(self.d_hidden,
                                  self.d_hidden,
                                  bias=self.use_bias))
            else:
                layers.append(
                    nn.Linear(self.d_hidden + self.d_in,
                              self.d_hidden,
                              bias=self.use_bias))
        self.linears = nn.ModuleList(
            [nn.Linear(self.d_in, self.d_hidden, bias=self.use_bias)] + layers)

        for lin in self.linears:
            nn.init.constant_(lin.bias, 0.0)
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()

        self.out1_linear = nn.Linear(self.d_hidden,
                                     self.d_out,
                                     bias=self.use_bias)
        nn.init.constant_(self.out1_linear.bias, 0.0)
        nn.init.kaiming_normal_(self.out1_linear.weight, a=0, mode="fan_in")

        self.out2_linear = nn.Linear(self.d_hidden,
                                     self.d_out,
                                     bias=self.use_bias)
        nn.init.constant_(self.out2_linear.bias, 0.0)
        nn.init.kaiming_normal_(self.out2_linear.weight, a=0, mode="fan_in")

    def forward(self, points, params_embed, point_w, combine_inner_dims=(1, )):
        """
        combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer; where combine_inner_dims = (num_view_per_obj, batch of points)
        """
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            x = torch.cat([points_embed, params_embed], dim=-1)
        else:
            x = points_embed
        x_init = x
        for layer in range(0, self.num_layers):
            lin = self.linears[layer]
            if layer == self.combine_layer:
                x_avg = util.combine_interleaved(x, combine_inner_dims,
                                                 self.combine_type)
                x_avg = x_avg.reshape(-1, self.d_hidden)
                x_avg = util.repeat_interleave(x_avg, combine_inner_dims[0])
                x = torch.cat([x, x_avg], -1)
                # x_init = util.combine_interleaved(x_init, combine_inner_dims,
                #                                   self.combine_type)

            if layer in self.skips:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = self.activation(lin(x))

        w = self.out1_linear(x).reshape(-1, 3)
        v = self.out2_linear(x).reshape(-1, 3)
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)

        warped_points = point_w
        warped_points = from_homogenous(
            torch.matmul(transform,
                         to_homogenous(warped_points)[..., None])[..., 0])

        return warped_points


class TansField2_ImplicitNet(nn.Module):
    """
    Represents a Trans_MLP
    """
    def __init__(
        self,
        conf,
        d_emb_in,
        d_point_in=3,
        d_hidden=128,
        n_layers=6,
        skips=(4, ),
        d_out=3,
        combine_layer=3,
        combine_type="average",
        use_bias=True,
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=d_point_in)
        self.num_layers = n_layers
        self.d_in = d_emb_in + self.pos_encoder.d_out
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.skips = skips
        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_bias = use_bias

        self.linears = nn.ModuleList([nn.Linear(self.d_in, self.d_hidden, bias=self.use_bias)] + \
                [nn.Linear(self.d_hidden, self.d_hidden, bias=self.use_bias) if i not in self.skips \
                    else nn.Linear(self.d_hidden + self.d_in, self.d_hidden, bias=self.use_bias)
                for i in range(1, self.num_layers)
            ])

        for lin in self.linears:
            nn.init.constant_(lin.bias, 0.0)
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()

        self.out_linear = nn.Linear(self.d_hidden,
                                    self.d_out,
                                    bias=self.use_bias)
        nn.init.constant_(self.out_linear.bias, 0.0)
        nn.init.constant_(self.out_linear.weight, 0.0)

    def forward(self, points, params_embed, point_w, combine_inner_dims=(1, )):
        """
        combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer; where combine_inner_dims = (num_view_per_obj, batch of points)
        """
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            x = torch.cat([points_embed, params_embed], dim=-1)
        else:
            x = points_embed
        x_init = x
        for layer in range(0, self.num_layers):
            lin = self.linears[layer]
            if layer == self.combine_layer:
                x = util.combine_interleaved(x, combine_inner_dims,
                                             self.combine_type)
                x_init = util.combine_interleaved(x_init, combine_inner_dims,
                                                  self.combine_type)

            if layer in self.skips:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = self.activation(lin(x))

        translation = self.out_linear(x)
        warped_points = point_w + translation.reshape(-1, 3)

        return warped_points


class SE3Field2_ImplicitNet(nn.Module):
    """
    Represents a Trans_MLP
    """
    def __init__(
        self,
        conf,
        d_emb_in,
        d_point_in=3,
        d_hidden=128,
        n_layers=6,
        skips=(4, ),
        d_out=3,
        combine_layer=3,
        combine_type="average",
        use_bias=True,
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=d_point_in)
        self.num_layers = n_layers
        self.d_in = d_emb_in + self.pos_encoder.d_out
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.skips = skips
        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_bias = use_bias

        self.linears = nn.ModuleList([nn.Linear(self.d_in, self.d_hidden, bias=self.use_bias)] + \
                [nn.Linear(self.d_hidden, self.d_hidden, bias=self.use_bias) if i not in self.skips \
                    else nn.Linear(self.d_hidden + self.d_in, self.d_hidden, bias=self.use_bias)
                for i in range(1, self.num_layers)
            ])

        for lin in self.linears:
            nn.init.constant_(lin.bias, 0.0)
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()

        self.out1_linear = nn.Linear(self.d_hidden,
                                     self.d_out,
                                     bias=self.use_bias)
        nn.init.constant_(self.out1_linear.bias, 0.0)
        nn.init.kaiming_normal_(self.out1_linear.weight, a=0, mode="fan_in")

        self.out2_linear = nn.Linear(self.d_hidden,
                                     self.d_out,
                                     bias=self.use_bias)
        nn.init.constant_(self.out2_linear.bias, 0.0)
        nn.init.kaiming_normal_(self.out2_linear.weight, a=0, mode="fan_in")

    def forward(self, points, params_embed, point_w, combine_inner_dims=(1, )):
        """
        combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer; where combine_inner_dims = (num_view_per_obj, batch of points)
        """
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            x = torch.cat([points_embed, params_embed], dim=-1)
        else:
            x = points_embed
        x_init = x
        for layer in range(0, self.num_layers):
            lin = self.linears[layer]
            if layer == self.combine_layer:
                x = util.combine_interleaved(x, combine_inner_dims,
                                             self.combine_type)
                # x_init = util.combine_interleaved(x_init, combine_inner_dims,
                #                                   self.combine_type)

            if layer in self.skips:
                if len(x.shape) != len(x_init.shape):
                    x = util.repeat_interleave(x, combine_inner_dims[0]).reshape(-1, x.shape[-1])
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = self.activation(lin(x))

        w = self.out1_linear(x).reshape(-1, 3)
        v = self.out2_linear(x).reshape(-1, 3)
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)

        warped_points = point_w
        warped_points = from_homogenous(
            torch.matmul(transform,
                         to_homogenous(warped_points)[..., None])[..., 0])

        return warped_points


class TranslationField_mlp(nn.Module):
    def __init__(
            self,
            conf,
            channel_in=3,
            channel_out=3,
            channel_embed=128,
            channel_hidden=128,
            trunk_layer=6,
            skips=(4, ),
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=channel_in)
        self.c_in_net = self.pos_encoder.d_out + channel_embed
        self.c_out = channel_out
        self.c_hidden = channel_hidden
        self.trunk_layer = trunk_layer
        self.skips = skips

        self.mlp = _MLP(
            self.c_in_net,
            self.trunk_layer,
            self.c_hidden,
            channel_out=self.c_out,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='const',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )

    def forward(self, points, params_embed):
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            inputs = torch.cat([points_embed, params_embed], dim=-1)
        else:
            inputs = points_embed
        translation = self.mlp(inputs)
        warped_points = points + translation

        return warped_points


class SE3Field_mlp(nn.Module):
    def __init__(
            self,
            conf,
            channel_in=3,
            channel_embed=128,
            channel_hidden=128,
            trunk_layer=6,
            skips=(4, ),
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=channel_in)
        self.c_in_net = self.pos_encoder.d_out + channel_embed
        self.c_hidden = channel_hidden
        self.trunk_layer = trunk_layer
        self.skips = skips

        self.trunk = _MLP(
            self.c_in_net,
            self.trunk_layer,
            self.c_hidden,
            channel_out=0,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )
        self.branches_w = _MLP(
            self.c_in_net,
            num_layers=0,
            channel_hidden=self.c_hidden,
            channel_out=3,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )
        self.branches_v = _MLP(
            self.c_in_net,
            num_layers=0,
            channel_hidden=self.c_hidden,
            channel_out=3,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )
        print('**Deformation Network is Setted!')

    def warp(self, points, params_embed):
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            inputs = torch.cat([points_embed, params_embed], dim=-1)
        else:
            inputs = points_embed
        trunk_out = self.trunk(inputs)

        w = self.branches_w(trunk_out)
        v = self.branches_v(trunk_out)
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)

        warped_points = points
        warped_points = from_homogenous(
            torch.matmul(transform,
                         to_homogenous(warped_points)[..., None])[..., 0])

        return warped_points

    def unit_vectors(self, length):
        result = []
        for i in range(0, length):
            x = torch.zeros(length)
            x[i] = 1
            result.append(x)
        return result

    def forward(self,
                points,
                params_embed,
                return_jacobian=False,
                batch_size=128):
        warped_points = self.warp(points, params_embed)

        # v_points = torch.ones_like(points)
        # v_params = torch.ones_like(params_embed)
        # aa = torch.autograd.functional.jvp(self.warp, (points, params_embed), v=(v_points,v_params), create_graph=True)[0]
        # jac_loop = []
        # for o in warped_points.view(-1):
        #     # self.warp.zero_grad()
        #     grad = []
        #     o.backward(retain_graph=True)
        #     for param in points.view(-1):
        #         grad.append(param.grad.reshape(-1))
        #     jac_loop.append(torch.cat(grad))
        # jac_loop = torch.stack(jac_loop)

        # jac_loop = []
        # points.requires_grad=True
        # for np in range(points.shape[0]):
        #     grad = []
        #     for i in range(3):
        #         warped_points[np, i].backward(retain_graph=True)
        #         grad.append(points[np].grad)
        #     jac_loop.append(torch.stack(grad))
        # jac_loop = torch.stack(jac_loop)
        # result = [torch.autograd.grad(outputs=[warped_points], inputs=[points], grad_outputs=[unit], retain_graph=True)[0] for unit in self.unit_vectors(warped_points.size(0))]

        if return_jacobian:
            jacs = []
            points = points.reshape(batch_size, -1, 3)
            B, Bp = points.shape[:-1]
            params_embed = params_embed.reshape(B, Bp, -1)

            for i in range(batch_size):
                jac = torch.autograd.functional.jacobian(
                    self.warp, (points[i], params_embed[i]),
                    create_graph=True,
                    vectorize=True)[0].sum(2)
                jacs.append(jac)
            jacs = torch.stack(jacs).reshape(-1, 3, 3)
            return warped_points, jac
        else:
            return warped_points


class SE3Field_mlp0(nn.Module):
    def __init__(
            self,
            conf,
            channel_in=3,
            channel_embed=128,
            channel_hidden=128,
            trunk_layer=6,
            skips=(4, ),
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=channel_in)
        self.c_in_net = self.pos_encoder.d_out + channel_embed
        self.c_hidden = channel_hidden
        self.trunk_layer = trunk_layer
        self.skips = skips

        self.trunk = _MLP(
            self.c_in_net,
            self.trunk_layer,
            self.c_hidden,
            channel_out=0,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )
        self.branches_w = _MLP(
            self.c_in_net,
            num_layers=0,
            channel_hidden=self.c_hidden,
            channel_out=3,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )
        self.branches_v = _MLP(
            self.c_in_net,
            num_layers=0,
            channel_hidden=self.c_hidden,
            channel_out=3,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )
        print('**Deformation Network is Setted!')

    def forward(self, points, params_embed):
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            inputs = torch.cat([points_embed, params_embed], dim=-1)
        else:
            inputs = points_embed
        trunk_out = self.trunk(inputs)

        w = self.branches_w(trunk_out)
        v = self.branches_v(trunk_out)
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)

        warped_points = points
        warped_points = from_homogenous(
            torch.matmul(transform,
                         to_homogenous(warped_points)[..., None])[..., 0])

        return warped_points


class HyperSheetMLP(nn.Module):
    def __init__(
            self,
            conf,
            channel_in=3,
            channel_out=4,
            channel_embed=128,
            channel_hidden=64,
            trunk_layer=6,
            skips=(4, ),
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding.from_conf(conf, d_in=channel_in)
        self.c_in_net = self.pos_encoder.d_out + channel_embed
        self.c_out = channel_out
        self.c_hidden = channel_hidden
        self.trunk_layer = trunk_layer
        self.skips = skips

        self.hyperlin = _MLP(
            self.c_in_net,
            self.trunk_layer,
            self.c_hidden,
            channel_out=self.c_out,
            use_bias=True,
            skips=self.skips,
            activation_hidden='relu',
            init_type='kaiming_normal',
            init_para={
                'bias': 0,
                'weight': 0
            },
        )

    def forward(self, points, params_embed):
        points_embed = self.pos_encoder(points)
        if params_embed is not None:
            inputs = torch.cat([points_embed, params_embed], dim=-1)
        else:
            inputs = points_embed
        x = self.hyperlin(inputs)
        return x


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm1d(1, affine=False)

        nhidden = 128
        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias), nn.ReLU())
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)

        nn.init.constant_(self.mlp_shared[0].bias, 0.0)
        # nn.init.constant_(self.mlp_shared[0].weight, 0.0)
        nn.init.kaiming_normal_(self.mlp_shared[0].weight, a=0, mode="fan_in")
        nn.init.constant_(self.mlp_gamma.bias, 0.0)
        # nn.init.constant_(self.mlp_gamma.weight, 0.0)
        nn.init.kaiming_normal_(self.mlp_gamma.weight, a=0, mode="fan_in")
        nn.init.constant_(self.mlp_beta.bias, 0.0)
        # nn.init.constant_(self.mlp_beta.weight, 0.0)
        nn.init.kaiming_normal_(self.mlp_beta.weight, a=0, mode="fan_in")

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x[:, None])[:, 0]

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        # gamma = gamma.view(*gamma.size()[:2], 1, 1)
        # beta = beta.view(*beta.size()[:2], 1, 1)
        out = normalized * (1 + gamma) + beta
        # out = x * (1 + gamma) + beta
        return out


class MLP(nn.Module):
    """
    """
    def __init__(self,
                 d_in,
                 d_hidden=512,
                 d_cdn=128,
                 skip_in=(),
                 d_out=3,
                 num_layers=8,
                 cdn_type='adain',
                 edit_type='deform'):
        """
        MLP Net for deformation or neural position encoder

        Args:
            d_in (int): dimention of inputs
            d_hidden (int): dimention of hidden layers
            d_cdn (int, optional): dimention of condition. Defaults to 128.
            skip_in (tuple, optional): [description]. Defaults to ().
            d_out (int, optional): [description]. Defaults to 3.
            num_layers (int, optional): [description]. Defaults to 8.
        """
        super().__init__()

        self.num_layers = num_layers
        self.skip_in = skip_in
        self.d_in = d_in
        self.d_cdn = d_cdn
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.cdn_type = cdn_type

        if cdn_type == 'adain':
            self.adain = ADAIN(d_hidden, d_cdn)
        elif cdn_type == 'concat':
            self.d_in = d_in + d_cdn
        else:
            print("Unexpected cdn_type!")

        self.linears = nn.ModuleList([nn.Linear(self.d_in, self.d_hidden)] + \
            [nn.Linear(self.d_hidden, self.d_hidden) if i not in self.skip_in else nn.Linear(self.d_hidden + self.d_in, self.d_hidden)
            for i in range(self.num_layers - 1)
        ])
        # layers = []
        # for i in range(self.num_layers - 1):
        #     if i not in self.skip_in:
        #         layers.append(nn.Linear(self.d_hidden, self.d_hidden))
        #     else:
        #         layers.append(
        #             nn.Linear(self.d_hidden + self.d_in, self.d_hidden))
        # self.linears = nn.ModuleList([nn.Linear(self.d_in, self.d_hidden)] +
        #                              layers)

        for lin in self.linears:
            nn.init.constant_(lin.bias, 0.0)
            if edit_type == 'deform':
                nn.init.constant_(lin.weight, 0.0)
            else:
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()
        self.linear_out = nn.Linear(self.d_hidden, self.d_out)
        nn.init.constant_(self.linear_out.bias, 0.0)
        if edit_type == 'deform':
            nn.init.constant_(self.linear_out.weight, 0.0)
        else:
            nn.init.kaiming_normal_(self.linear_out.weight, a=0, mode="fan_in")

    def forward(self, x):
        input_pts, input_cdns = torch.split(x, [self.d_in, self.d_cdn], dim=-1)
        h = input_pts

        for i, layer in enumerate(self.linears):
            h = self.activation(layer(h))

            if self.cdn_type == 'adain':
                h = self.adain(h, input_cdns)

            if i in self.skip_in:
                h = torch.cat([input_pts, h], -1)

        out = self.linear_out(h)

        return out

    # @classmethod
    # def from_conf(cls, conf, d_in, **kwargs):
    #     return cls(
    #         d_in,
    #         conf.get_list("dims"),
    #         skip_in=conf.get_list("skip_in"),
    #         beta=conf.get_float("beta", 0.0),
    #         dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),
    #         combine_layer=conf.get_int("combine_layer", 1000),
    #         combine_type=conf.get_string("combine_type",
    #                                      "average"),  # average | max
    #         **kwargs)
