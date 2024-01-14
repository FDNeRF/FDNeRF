"""
Author: Eckert ZHANG
Date: 2021-11-10 18:15:52
LastEditTime: 2022-03-15 21:59:56
LastEditors: Eckert ZHANG
Description: 
"""
import torch.backends.cudnn as cudnn

from preprocess_v2.util.distributed import master_only_print as print


def init_cudnn(deterministic, benchmark):
    r"""Initialize the cudnn module. The two things to consider is whether to
    use cudnn benchmark and whether to use cudnn deterministic. If cudnn
    benchmark is set, then the cudnn deterministic is automatically false.

    Args:
        deterministic (bool): Whether to use cudnn deterministic.
        benchmark (bool): Whether to use cudnn benchmark.
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    print('cudnn benchmark: {}'.format(benchmark))
    print('cudnn deterministic: {}'.format(deterministic))
