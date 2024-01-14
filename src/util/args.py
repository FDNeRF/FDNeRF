"""
Author: Eckert ZHANG
Date: 2021-07-11 22:59:14
LastEditTime: 2022-03-25 10:05:58
LastEditors: Eckert ZHANG
FilePath: /pixel-nerf/src/util/args.py
Description: 
"""
import sys
import os, shutil
import argparse
from pyhocon import ConfigFactory


def parse_args(
    callback=None,
    training=False,
    default_conf="conf/default_mv.conf",
    default_expname="example",
    default_data_format="dvr",
    default_num_epochs=10000000,
    default_lr=1e-4,
    default_gamma=1.00,
    default_datadir="data",
    default_ray_batch_size=50000,
    default_gpu_id="0",
):
    parser = argparse.ArgumentParser()

    # configuration file
    parser.add_argument(
        "--conf",
        "-c",
        type=str,
        default=None,
        help=
        "The path of config file, default file is stored in the folder 'conf/'"
    )

    # resuming setting
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="setting: continue training or not, the default is 'False'")
    parser.add_argument(
        "--resume_init",
        action="store_true",
        help="setting: resume from initial ckpt or not, the default is 'False'"
    )
    parser.add_argument(
        "--resume_ckpt_init",
        type=str,
        default=None,
        help="",
    )

    # training mode
    parser.add_argument(
        "--warp_pretrain",
        action="store_true",
        help="Pretrain stage for net_G. If True, pretrain net_G.",
    )
    parser.add_argument(
        "--Joint_Train",
        action="store_true",
        help="setting: jointly train PIrender or not, the default is 'False'")
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        help=
        "setting: visul fixed test views or not during training, the default is 'False'"
    )
    parser.add_argument(
        "--only_test",
        action="store_true",
        help="setting: only test model, no training, the default is 'False'")
    parser.add_argument(
        "--only_video",
        action="store_true",
        help=
        "setting: to generate video for test model, no training, the default is 'False'"
    )
    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=90,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--pose_traj_video",
        type=str,
        default='standard', # 'standard', 'spiral'
    )

    # other items
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=default_gpu_id,
        help=
        "the ids of GPU(s) to use, space delimited, which is a 'str' and splited by spaces"
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=default_expname,
        help="the experiment name, which is also the folder name of results")
    parser.add_argument(
        "--exp_group_name",
        "-G",
        type=str,
        default=None,
        help="if we want to group some experiments together",
    )
    parser.add_argument(
        "--dataset_format",
        "-F",
        type=str,
        default=None,
        help="Dataset format",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default='mixwild',
        help="Prefix of Dataset file",
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        default="logs",
        help="logs output directory",
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="checkpoints output directory",
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="visuals",
        help="visualization output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_num_epochs,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8000,
        help="size of chunks for test rendering",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=default_lr,
        help="learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=default_gamma,
        help="learning rate decay factor",
    )
    parser.add_argument(
        "--datadir",
        "-D",
        type=str,
        default=None,
        help="Dataset directory",
    )
    parser.add_argument(
        "--resultdir",
        type=str,
        default='results',
        help="Results' directory",
    )
    parser.add_argument(
        "--ray_batch_size",
        "-R",
        type=int,
        default=default_ray_batch_size,
        help="Ray batch size",
    )

    if callback is not None:
        parser = callback(parser)
    args = parser.parse_args()

    if args.exp_group_name is not None:
        args.name = os.path.join(args.exp_group_name, args.name)

    os.makedirs(os.path.join(args.resultdir, args.name, args.checkpoints_path),
                exist_ok=True)
    os.makedirs(os.path.join(args.resultdir, args.name, args.visual_path),
                exist_ok=True)

    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    EXPCONF_PATH = os.path.join(PROJECT_ROOT, "expconf.conf")
    expconf = ConfigFactory.parse_file(EXPCONF_PATH)
    if args.conf is None:
        args.conf = expconf.get_string("config." + args.name, default_conf)
    if args.datadir is None:
        args.datadir = expconf.get_string("datadir." + args.name,
                                          default_datadir)

    conf = ConfigFactory.parse_file(args.conf)
    shutil.copyfile(
        args.conf,
        os.path.join(args.resultdir, args.name,
                     args.conf.split('/')[-1]))

    if args.dataset_format is None:
        args.dataset_format = conf.get_string("data.format",
                                              default_data_format)

    args.gpu_id = list(map(int, args.gpu_id.split()))

    # args.resume = True

    print("EXPERIMENT NAME:", args.name)
    if training:
        print("CONTINUE?", "yes" if args.resume else "no")
    print("* Config file:", args.conf)
    print("* Dataset format:", args.dataset_format)
    print("* Dataset location:", args.datadir)
    return args, conf
