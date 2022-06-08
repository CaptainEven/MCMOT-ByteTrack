#!/usr/bin/env python
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

# from yolox.models.yolo_head import YOLOXHead
# from yolox.models.yolo_pafpn import YOLOPAFPN

from yolox.models.darknet_backbone import DarknetBackbone
from yolox.models.darknet_head import DarknetHead


## ----- Decoupled backbone and head Using
class YOLOXDark(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self,
                 cfg_path,
                 backbone=None,
                 head=None,
                 n_classes=5):
        """
        :param cfg_path: configure file path for DarknetBackbone
        :param backbone:
        :param head:
        :param n_classes: number of object classes for detection
        """
        super().__init__()

        if backbone is None:
            backbone = DarknetBackbone(cfg_path=cfg_path)
        if head is None:
            head = DarknetHead(num_classes=n_classes)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        """
        :param x:
        :param targets:
        :return:
        """
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone.forward(x)  # 1/8, 1/16, 1/32

        if self.training:
            assert targets is not None

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = \
                self.head.forward(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head.forward(fpn_outs)

        return outputs
