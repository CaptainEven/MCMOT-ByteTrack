#!/usr/bin/env python3
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil

import torch
from loguru import logger


def load_ckpt(model, ckpt):
    """
    :param model:
    :param ckpt:
    :return:
    """
    model_state_dict = model.state_dict()
    load_dict = {}

    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning("{} is not in the ckpt. Please double check and see if this is desired."
                           .format(key_model))
            continue

        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning("Shape of {} in checkpoint is {}, while shape of {} in model is {}."
                           .format(key_model, v_ckpt.shape, key_model, v.shape))
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)

    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    """
    :param state:
    :param is_best:
    :param save_dir:
    :param model_name:
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, model_name + "_ckpt.pth.tar")
    torch.save(state, file_path)
    print("{:s} saved.".format(file_path))

    if is_best:
        best_file_path = os.path.join(save_dir, "best_ckpt.pth.tar")
        shutil.copyfile(file_path, best_file_path)
