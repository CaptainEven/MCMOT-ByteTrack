#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import math
import numpy as np

__all__ = ["mkdir", "nms", "multiclass_nms", "multiclass_NMS", "demo_postprocess"]


def mkdir(path):
    """
    :param: path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def cos_sim(vect_1, vect_2):
    """
    :param vect_1:
    :param vect_2:
    :return:
    """
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), vect_1))))
    norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), vect_2))))
    return sum([vect_1[i] * vect_2[i] for i in range(0, len(vect_1))]) / (norm1 * norm2)


def overlap(x1, w1, x2, w2):
    """
    :param x1:  center_x
    :param w1:  bbox_w
    :param x2:  center_x
    :param w2:  bbox_w
    :return:
    """
    l1 = x1 - w1 / 2.0
    l2 = x2 - w2 / 2.0
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2.0
    r2 = x2 + w2 / 2.0
    right = r1 if r1 < r2 else r2
    return right - left

def tlbr2xywh(tlbr):
    """
    @param tlbr: x1y1x2y2
    @return: center_x, center_y, box_w, box_h
    """
    tlbr = np.squeeze(tlbr)
    xywh = tlbr.copy()
    xywh[:2] = (tlbr[:2] + tlbr[2:]) * 0.5
    xywh[2:] = (tlbr[2:] - tlbr[:2])
    return xywh

def box_intersection(box1, box2, box_type="x1y1x2y2"):
    """             0         1         2      3
    :param box1: center_x, center_y, bbox_w, bbox_h
    :param box2: center_x, center_y, bbox_w, bbox_h
    :return:
    """
    if box_type == "x1y1x2y2":
        box1 = tlbr2xywh(box1)
        box2 = tlbr2xywh(box2)
    elif box_type == "xywh":
        pass
    else:
        print("[Err]: wrong box type, should be x1y1x2y2 or xywh, exit!")
        exit(-1)

    w = overlap(box1[0], box1[2], box2[0], box2[2])
    h = overlap(box1[1], box1[3], box2[1], box2[3])

    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


def box_union(box1, box2, box_type="x1y1x2y2"):
    """
    @param box1: center_x, center_y, box_w, box_h
    @param box2:
    @param box_type:
    @return:
    """
    if box_type == "x1y1x2y2":
        box1 = tlbr2xywh(box1)
        box2 = tlbr2xywh(box2)
    elif box_type == "xywh":
        pass  # do nothing
    else:
        print("[Err]: wrong box type, should be x1y1x2y2 or xywh, exit!")
        exit(-1)

    i = box_intersection(box1, box2, box_type="xywh")
    u = box1[2] * box1[3] + box2[2] * box2[3] - i
    return u


def box_iou(box1, box2, box_type="x1y1x2y2"):
    """
    @param box1:
    @param box2:
    @return:
    """
    return box_intersection(box1, box2, box_type) / box_union(box1, box2, box_type)


def nms(boxes, scores, nms_thr):
    """
    Single class NMS implemented in Numpy.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_NMS(bboxes_dict, scores_dict, nms_thr):
    """
    Multiclass NMS implemented in Numpy
    """
    all_class_dets = []
    for cls_id, bboxes in bboxes_dict.items():
        bboxes = np.array(bboxes)
        scores = np.array(scores_dict[cls_id])
        keep = nms(bboxes, scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_id
            dets = np.concatenate([bboxes[keep],
                                   scores[keep, None],
                                   cls_inds], 1)
            all_class_dets.append(dets)

    if len(all_class_dets) == 0:
        return None

    return np.concatenate(all_class_dets, 0)


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """
    Multiclass NMS implemented in Numpy
    """
    final_dets = []
    num_class = scores.shape[1]
    for cls_id in range(num_class):
        cls_scores = scores[:, cls_id]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_id
                dets = np.concatenate([valid_boxes[keep],
                                       valid_scores[keep, None],
                                       cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs
