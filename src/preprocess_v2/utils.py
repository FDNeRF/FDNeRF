"""
Author: Eckert ZHANG
Date: 2021-11-10 18:56:55
LastEditTime: 2022-03-31 19:40:10
LastEditors: Eckert ZHANG
Description: 
"""
import importlib, pdb
import numpy as np
from preprocess_v2.util.trainer import *
from preprocess_v2.util.trainer import _calculate_model_size


def get_model_pirenderer(opt, device, EMA=False):
    gen_module, gen_network_name = opt.gen.type.split('::')
    lib = importlib.import_module(gen_module)
    network = getattr(lib, gen_network_name)
    net_G = network(**opt.gen.param).to(device)
    init_bias = getattr(opt.trainer.init, 'bias', None)
    net_G.apply(
        weights_init(opt.trainer.init.type, opt.trainer.init.gain, init_bias))

    print('Preprocessing net [{}] parameter count: {:,}'.format(
        'net_G', _calculate_model_size(net_G)))
    print('Initialize net_G weights using '
          'type: {} gain: {}'.format(opt.trainer.init.type,
                                     opt.trainer.init.gain))
    if EMA:
        net_G_ema = network(**opt.gen.param).to(device)
        net_G_ema.eval()
        accumulate(net_G_ema, net_G, 0)
        return net_G, net_G_ema
    else:
        return net_G


def load_ckpt_pirenderer(net_G, ckpt_path, net_G_ema=None, load_EMA=False):
    print('Loading ckpt of Preprocess Net: {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path,
                            map_location=lambda storage, loc: storage)
    if load_EMA:
        try:
            net_G.load_state_dict(checkpoint['net_G_ema'], strict=False)
        except:
            net_G.mapping_net.load_state_dict(checkpoint['net_G_ema'],
                                              strict=False)
            net_G.warpping_net.load_state_dict(checkpoint['net_G_ema'],
                                               strict=False)
            net_G.editing_net.load_state_dict(checkpoint['net_G_ema'],
                                              strict=False)
    else:
        try:
            net_G.load_state_dict(checkpoint['net_G'], strict=False)
        except:
            net_G.mapping_net.load_state_dict(checkpoint['net_G'],
                                              strict=False)
            net_G.warpping_net.load_state_dict(checkpoint['net_G'],
                                               strict=False)
            net_G.editing_net.load_state_dict(checkpoint['net_G'],
                                              strict=False)
    if net_G_ema is not None and 'net_G_ema' in checkpoint:
        net_G_ema.load_state_dict(checkpoint['net_G_ema'], strict=False)
        # pdb.set_trace()
    elif net_G_ema is not None and 'net_G_ema' not in checkpoint:
        net_G_ema.load_state_dict(checkpoint['net_G'], strict=False)

    if net_G_ema is not None:
        return net_G, net_G_ema
    else:
        return net_G
