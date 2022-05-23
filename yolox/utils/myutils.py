# encoding=utf-8

import os
import random
import shutil

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_darknet_cfg(path):
    """
    :param path:
    :return:
    """
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    m_defs = []  # module definitions

    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            m_defs.append({})
            m_defs[-1]['type'] = line[1:-1].rstrip()
            if m_defs[-1]['type'] == 'convolutional':
                m_defs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return np-array
                m_defs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                m_defs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    m_defs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    m_defs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride',
                 'pad', 'activation', 'layers', 'groups', 'from',
                 'mask', 'anchors', 'classes', 'num', 'jitter',
                 'ignore_thresh', 'truth_thresh', 'random', 'stride_x', 'stride_y',
                 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'groups',
                 'group_id', 'probability']

    f = []  # fields
    for x in m_defs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields

    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return m_defs


def save_max_id_dict(max_id_dict, dict_path):
    """
    序列化max_id_dict到磁盘
    """
    with open(dict_path, "wb") as f:
        np.savez(dict_path, max_id_dict=max_id_dict)