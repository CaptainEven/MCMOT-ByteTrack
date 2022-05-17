#!/usr/bin/env python3
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """
    :param img:
    :param boxes:
    :param scores:
    :param cls_ids:
    :param conf:
    :param class_names:
    :return:
    """
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    """
    :param idx:
    :return:
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking_mc(image,
                     tlwhs_dict,
                     obj_ids_dict,
                     num_classes,
                     scores=None,
                     frame_id=0,
                     fps=0.0,
                     id2cls=None):
    """
    :param image:
    :param tlwhs_dict:
    :param obj_ids_dict:
    :param num_classes:
    :param scores:
    :param frame_id:
    :param fps:
    :param id2cls:
    :return:
    """
    img = np.ascontiguousarray(np.copy(image))
    im_h, im_w = img.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1.0, image.shape[1] / 1000.0)  # 1600.
    # text_thickness = 1 if text_scale > 1.1 else 1
    text_thickness = 2  # 自定义ID文本线宽
    line_thickness = max(1, int(image.shape[1] / 500.0))

    radius = max(5, int(im_w / 140.))

    ## ----- draw fps
    cv2.putText(img,
                'frame: %d fps: %.2f' % (frame_id, fps),
                (0, int(15 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 255),
                thickness=2)

    for cls_id in range(num_classes):
        cls_tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]

        for i, tlwh in enumerate(cls_tlwhs):
            tlwh = np.squeeze(tlwh)
            x1, y1, w, h = tlwh
            int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))  # x1, y1, x2, y2
            obj_id = int(obj_ids[i])
            tr_id_text = '{}'.format(int(obj_id))

            _line_thickness = 1 if obj_id <= 0 else line_thickness
            color = get_color(abs(obj_id))
            # cls_color = cls_color_dict[id2cls[cls_id]]

            # draw bbox
            cv2.rectangle(img=img,
                          pt1=int_box[0:2],  # (x1, y1)
                          pt2=int_box[2:4],  # (x2, y2)
                          color=color,
                          thickness=line_thickness)

            ## draw class name
            cv2.putText(img,
                        id2cls[cls_id],
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 255, 255),  # cls_id: yellow
                        thickness=text_thickness)

            txt_w, txt_h = cv2.getTextSize(id2cls[cls_id],
                                           fontFace=cv2.FONT_HERSHEY_PLAIN,
                                           fontScale=text_scale, thickness=text_thickness)

            ## draw track id
            cv2.putText(img,
                        tr_id_text,
                        (int(x1), int(y1) - txt_h),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale * 1.2,
                        (0, 255, 255),  # cls_id: yellow
                        thickness=text_thickness)

    return img


def plot_tracking_sc(image, tlwhs, obj_ids,
                     scores=None, frame_id=0, fps=0.0, ids2=None):
    """
    :param image:
    :param tlwhs:
    :param obj_ids:
    :param scores:
    :param frame_id:
    :param fps:
    :param ids2:
    :return:
    """
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        int_bbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im,
                      int_bbox[0:2],
                      int_bbox[2:4],
                      color=color,
                      thickness=line_thickness)
        cv2.putText(im,
                    id_text,
                    (int_bbox[0], int_bbox[1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 255),
                    thickness=text_thickness)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
