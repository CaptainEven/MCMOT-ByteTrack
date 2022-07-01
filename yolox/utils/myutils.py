# encoding=utf-8

import os

import numpy as np


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


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, axis=0)
    bboxes1 = np.expand_dims(bboxes1, axis=1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)

    return o


def filter_bbox_by_ious(bboxes_1, bboxes_2, iou_thresh):
    """
    filter bboxes_1 by ious with bboxes_2
    :param bboxes_1:
    :param bboxes_2:
    """
    ious = iou_batch(bboxes_1, bboxes_2)
    mask2d = ious < iou_thresh
    inds = np.where(np.sum(mask2d, axis=1) == bboxes_2.shape[0])
    bboxes_1 = bboxes_1[inds]
    return bboxes_1

def filter_by_shape(bboxes, low_thresh=0.5, high_thresh=2.0):
    """
    filter the bboxes by aspect ratio
    """
