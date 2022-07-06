# encoding=utf-8

import os

import argparse
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
from loguru import logger

from yolox.core import Trainer, launch
from yolox.exp import get_exp


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("YOLOX train parser")

    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="",  # ""
                        help="model name")
    parser.add_argument("--train_name",
                        type=str,
                        default="train",  # ""
                        help="model name")
    parser.add_argument("--val_name",
                        type=str,
                        default="test",  # ""
                        help="model name")

    ## ----- data root dir, eg: /mnt/diskd/even/MOT17, /mnt/diskb/maqiao/multiClass
    parser.add_argument("--data_dir",
                        type=str,
                        default="/mnt/diskb/maqiao/multiClass",
                        help="")
    parser.add_argument("--train_root",
                        type=str,
                        default="/mnt/diskb/maqiao/multiClass",
                        help="")
    parser.add_argument("--val_root",
                        type=str,
                        default="/mnt/diskb/maqiao/multiClass",
                        help="")

    ## ---------- experiment file path, eg: ../exps/example/mot/yolox_tiny_det.py
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_det_c5_dark_ssl.py",
                        type=str,
                        help="plz input your experiment description file")

    ## -----Darknet cfg file path
    parser.add_argument("--cfg",
                        type=str,
                        default="../cfg/yolox_darknet_tiny_bb46.cfg",
                        help="")

    ## ---------- checkpoint file path
    ## latest_ckpt.pth.tar, yolox_tiny_32.8.pth ../pretrained/v5.46.weights
    # ../YOLOX_outputs/yolox_tiny_det_c5_dark/latest_ckpt.pth.tar
    parser.add_argument("-c",
                        "--ckpt",
                        default="../pretrained/v5.46.weights",  # None
                        type=str,
                        help="checkpoint file")

    parser.add_argument("--cutoff",
                        type=int,
                        default=44,
                        help="")
    ## ----------

    parser.add_argument("--debug",
                        type=int,
                        default=1,  # False | True
                        help="")

    # distributed
    parser.add_argument("--dist-backend",
                        default="nccl",
                        type=str,
                        help="distributed backend")
    parser.add_argument("--dist-url",
                        default=None,
                        type=str,
                        help="url used to set up distributed training")

    ## ----------
    parser.add_argument("--local_rank",
                        default=0,
                        type=int,
                        help="local rank for dist training")

    parser.add_argument("--resume",
                        default=False,
                        action="store_true",
                        help="resume training")
    parser.add_argument("-e",
                        "--start_epoch",
                        default=None,
                        type=int,
                        help="resume training start epoch")
    parser.add_argument("--num_machines",
                        default=1,
                        type=int,
                        help="num of node for training")
    parser.add_argument("--machine_rank",
                        default=0,
                        type=int,
                        help="node rank for multi-node training")

    ## ---------- data type: float16 or not
    parser.add_argument("--fp16",
                        dest="fp16",
                        type=bool,
                        default=True,
                        # action="store_true",
                        help="Adopting mix precision training.")

    parser.add_argument("-o",
                        "--occupy",
                        dest="occupy",
                        default=False,  # False
                        action="store_true",
                        help="occupy GPU memory first for training.")
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER, )

    ## ---------- batch size and device id
    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=3,  # 4(for debugging), 8, 16, 18, 20, 24, 32, 48, 64
                        help="batch size")

    ## ----- set devices
    parser.add_argument("-nd",
                        "--n_devices",
                        type=int,
                        default=1,  # number of devices(gpus)
                        help="device for training")

    parser.add_argument("-d",
                        "--devices",
                        type=str,
                        default="7",
                        help="The device(GPU) ids.")

    return parser


@logger.catch
def main(exp, opt):
    """
    :param exp:
    :param opt:
    :return:
    """
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably!"
                      " You may see unexpected behavior "
                      "when restarting from checkpoints.")

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, opt)
    trainer.train()


if __name__ == "__main__":
    ## ----- parse args
    opt = make_parser().parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.devices

    if len(opt.devices.split(",")) != opt.n_devices:
        opt.n_devices = len(opt.devices.split(","))

    ## ----- parse experiment file
    exp = get_exp(opt.exp_file, opt.name)
    exp.merge(opt.opts)

    ## ----- modify number of workers
    if opt.debug != 0:
        exp.data_num_workers = 0

    ## ----- Using cfg file from opt
    if hasattr(exp, "cfg_file_path"):
        exp.cfg_file_path = os.path.abspath(opt.cfg)

    if not opt.experiment_name:
        opt.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if opt.n_devices is None \
        else opt.n_devices
    assert num_gpu <= torch.cuda.device_count()

    launch(main,
           num_gpu,
           opt.num_machines,
           opt.machine_rank,
           backend=opt.dist_backend,
           dist_url=opt.dist_url,
           args=(exp, opt), )
