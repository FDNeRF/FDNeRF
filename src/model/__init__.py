"""
Author: Eckert ZHANG
Date: 2021-11-04 21:49:58
LastEditTime: 2022-03-09 18:54:07
LastEditors: Eckert ZHANG
Description: 
"""
from .models import PixelNeRFNet
from .models_implicit import PixelNeRFNet_implicit_edit


def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "pixelnerf_implicit_edit":
        net = PixelNeRFNet_implicit_edit(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
