"""
Author: Eckert ZHANG
Date: 2021-11-04 21:49:58
LastEditTime: 2021-12-17 00:19:06
LastEditors: Eckert ZHANG
Description: 
"""
from .encoder import SpatialEncoder, ImageEncoder
from .resnetfc import ResnetFC, ResnetFC_Indep
from .resnetfc_edit import ResnetFC_edit
from .mlp import ImplicitNet


def make_mlp(conf,
             d_in,
             d_latent=0,
             allow_empty=False,
             d_exp_param=0,
             d_dir_indep=0,
             **kwargs):
    mlp_type = conf.get_string("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet" and d_exp_param == 0:
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "resnet_Indep":
        net = ResnetFC_Indep.from_conf(conf, d_pos_in=d_in-d_dir_indep, d_dir_in=d_dir_indep, d_latent=d_latent, **kwargs)
    elif mlp_type == "resnet" and d_exp_param > 0:
        net = ResnetFC_edit.from_conf(conf,
                                      d_in,
                                      d_latent=d_latent,
                                      d_exp_param=d_exp_param,
                                      **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net
