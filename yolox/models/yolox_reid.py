#!/usr/bin/env python
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOXReID(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, backbone=None, head=None):
        """
        :param backbone:
        :param head:
        """
        super().__init__()

        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        """
        :param x:
        :param targets:
        :return:
        """
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            
            loss, iou_loss, conf_loss, cls_loss, l1_loss, loss_reid, num_fg = \
                self.head.forward(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "reid_loss": loss_reid,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
