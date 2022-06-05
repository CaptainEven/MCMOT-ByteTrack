# encoding=utf-8

import argparse
import os
import cv2
import numpy as np
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, post_process


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("Detect video Demo!")

    ## ----- object classes
    parser.add_argument("--class_names",
                        type=str,
                        default="car, bicycle, person, cyclist, tricycle",
                        help="")

    ## ----- exp file, eg: yolox_x_ablation.py
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_tiny_det_c5_dark.py",
                        type=str,
                        help="pls input your experiment description file")

    ## ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
    parser.add_argument("-c",
                        "--ckpt",
                        default="../pretrained/latest_ckpt.pth.tar",
                        type=str,
                        help="ckpt for eval")

    ## "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
    parser.add_argument("--vid_path",
                        default="../videos/test_13.mp4",
                        help="path to images or video")


def detect_onnx(opt):
    """
    :param opt: options
    """


def run(opt):
    """
    :param opt:
    """
    opt = make_parser().parse_args()
    exp = get_exp(opt.exp_file, opt.name)

    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    exp.class_names = class_names
    exp.n_classes = len(exp.class_names)
    print("Number of classes: ", exp.n_classes)

    ## ----- run the tracking
    detect_onnx(exp, opt)

