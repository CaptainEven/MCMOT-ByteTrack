# encoding=utf-8

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F


class FeatureMatcher(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ## ----- exp file, eg: yolox_x_ablation.py
        parser.add_argument("-f",
                            "--exp_file",
                            default="../exps/example/mot/yolox_tiny_track_c5_darknet.py",
                            type=str,
                            help="pls input your experiment description file")

        ## ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
        parser.add_argument("-c",
                            "--ckpt",
                            default="../YOLOX_outputs/yolox_tiny_track_c5_darknet/latest_ckpt.pth.tar",
                            type=str,
                            help="ckpt for eval")

        # input seq videos
        self.parser.add_argument('--videos',
                                 type=str,
                                 default='/mnt/diskb/even/dataset/MCMOT_Evaluate',
                                 help='')  # 'data/samples/videos/'

        self.parser.add_argument('--bin-step',
                                 type=int,
                                 default=5,
                                 help='number of bins for cosine similarity statistics'
                                      '(10 or 5).')

        
