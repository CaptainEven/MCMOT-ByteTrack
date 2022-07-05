#!/usr/bin/env python
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from yolox.models.darknet_backbone import DarknetBackbone
from yolox.models.darknet_head import DarknetHeadSSL


class YOLOXDarkSSL(nn.Module):
    """
    YOLOX model module. The module list is defined
    by create_yolov3_modules function.
    The network returns loss values from three YOLO layers
    during training and detection results during test.
    """

    def __init__(self,
                 cfg_path,
                 backbone=None,
                 head=None,
                 n_classes=5,
                 T=0.2):
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
            head = DarknetHeadSSL(num_classes=n_classes)

        self.backbone = backbone
        self.head = head

        self.T = T  # temperature

    def forward(self, inps, targets=None, q=None, k=None, n=None):
        """
        :param inps:
        :param targets:
        :return:
        """
        # fpn output content features of all(default: 3) scales
        fpn_outs = self.backbone.forward(inps)  # 1/8, 1/16, 1/32

        if self.training:
            assert targets is not None
            assert not (q is None or k is None or n is None)

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = \
                self.head.forward(fpn_outs, targets, inps)

            ## ---------- TODO: Calculate SSL loss
            # ----- reshape the q,k,n
            q = q.reshape(q.shape[0]*q.shape[1], q.shape[2], q.shape[3], q.shape[4])
            k = k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3], k.shape[4])
            n = n.reshape(n.shape[0]*n.shape[1], n.shape[2], n.shape[3], n.shape[4])

            # ----- inference
            q_vectors = self.backbone.forward(q)[0]  # 200×96×28×28
            k_vectors = self.backbone.forward(k)[0]
            n_vectors = self.backbone.forward(n)[0]

            q_vectors = self.head.reid_convs(q_vectors)
            q_vectors = self.head.reid_preds(q_vectors)
            q_vectors = q_vectors.reshape(q_vectors.shape[0], -1)  # n×128
            q_vectors = nn.functional.normalize(q_vectors, dim=1)

            k_vectors = self.head.reid_convs(k_vectors)
            k_vectors = self.head.reid_preds(k_vectors)
            k_vectors = k_vectors.reshape(k_vectors.shape[0], -1)  # n×128
            k_vectors = nn.functional.normalize(k_vectors, dim=1)

            n_vectors = self.head.reid_convs(n_vectors)
            n_vectors = self.head.reid_preds(n_vectors)
            n_vectors = n_vectors.reshape(n_vectors.shape[0], -1)  # k×128
            n_vectors = nn.functional.normalize(n_vectors, dim=1)

            # ---------- SSL loss calculation
            # ---number of objects
            valid_lb_inds = targets.sum(dim=2) > 0  # batch_size×50(True | False)
            n_objs = valid_lb_inds.sum(dim=1)  # batch_size×n_lb_valid
            # print(n_objs)

            ## ---------- processing each sample(image) of the batch
            ssl_loss = 0.0
            for batch_idx in range(targets.shape[0]):
                num_gt = int(n_objs[batch_idx])
                if num_gt == 0:
                    continue

                q_vectors = q_vectors[:num_gt]
                k_vectors = k_vectors[:num_gt]
                # print(q_vectors.shape)  # 17×128

                ## --- compute logits
                # Einstein sum is more intuitive
                # dot product, positive logits: Nx1
                l_pos = torch.einsum('nc,nc->n', [q_vectors, k_vectors]).unsqueeze(-1)

                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', [q_vectors, n_vectors.T])

                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                l_ssl = self.head.softmax_loss(logits, labels)
                ssl_loss = ssl_loss + l_ssl
            loss = loss + ssl_loss

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "ssl_loss": ssl_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head.forward(fpn_outs)

        return outputs


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

    def forward(self, inps, targets=None):
        """
        :param inps:
        :param targets:
        :return:
        """
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone.forward(inps)  # 1/8, 1/16, 1/32

        if self.training:
            assert targets is not None

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = \
                self.head.forward(fpn_outs, targets, inps)
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
