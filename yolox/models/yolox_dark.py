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

            ## ---------- Calculate SSL loss
            # ---number of objects
            valid_lb_inds = targets.sum(dim=2) > 0  # batch_size×50(True | False)
            num_gts = valid_lb_inds.sum(dim=1)  # batch_size×n_lb_valid

            ssl_loss = 0.0
            tri_loss = 0.0
            for batch_idx, num_gt in enumerate(num_gts):
                num_gt = int(num_gt)
                if num_gt == 0:
                    continue

                q_vectors = q[batch_idx]
                k_vectors = k[batch_idx]
                n_vectors = n[batch_idx]

                # ----- inference
                q_vectors = self.backbone.forward(q_vectors)[0]  # 200×96×28×28
                k_vectors = self.backbone.forward(k_vectors)[0]
                n_vectors = self.backbone.forward(n_vectors)[0]

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
                ## ---------- processing each sample(image) of the batch
                q_vectors = q_vectors[:num_gt]  # num_gt×128
                k_vectors = k_vectors[:num_gt]

                ## ----- Calculate similarity matrix loss
                sm_output = torch.mm(q_vectors, k_vectors.T)
                sm_diff = sm_output - torch.eye(num_gt).cuda()
                sm_diff = torch.pow(sm_diff, 2)
                l_ssl_sm = sm_diff.sum()
                ssl_loss += l_ssl_sm / (num_gt * num_gt)

                ## ----- TODO: Calculate similarity loss
                ## of feature map and patch feature vector difference

                ## ----- Calculate Triplet loss
                tri_cnt = 0
                for i in range(q_vectors.shape[0]):
                    anc = q_vectors[i]
                    pos = k_vectors[i]
                    for j in range(k_vectors.shape[0]):
                        if j != i:
                            neg = k_vectors[j]
                            tri_loss += self.head.triplet_loss.forward(anc, pos, neg)
                            tri_cnt += 1
                for i in range(k_vectors.shape[0]):
                    anc = k_vectors[i]
                    pos = q_vectors[i]
                    for j in range(q_vectors.shape[0]):
                        if j != i:
                            neg = q_vectors[j]
                            tri_loss += self.head.triplet_loss.forward(anc, pos, neg)
                            tri_cnt += 1

                if tri_cnt > 0:
                    ssl_loss += tri_loss / tri_cnt

                ## ----- calculate Cycle loss
                cycle_loss = 0.0
                cyc_cnt = 0
                for i in range(q_vectors.shape[0]):
                    for j in range(k_vectors.shape[0]):
                        if j != i:
                            sim_qi_kj = torch.dot(q_vectors[i], k_vectors[j])
                            sim_qj_ki = torch.dot(k_vectors[i], q_vectors[j])
                            cycle_loss += abs(sim_qi_kj - sim_qj_ki)
                            cyc_cnt += 1

                if cyc_cnt > 0:
                    ssl_loss += cycle_loss / cyc_cnt

                ## ----- Calculate contrastive loss
                # Einstein sum is more intuitive
                # dot product, positive logits: nx1
                l_pos = torch.einsum('nc,nc->n', [q_vectors, k_vectors]).unsqueeze(-1)

                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', [q_vectors, n_vectors.T])

                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                ssl_loss += self.head.softmax_loss(logits, labels) / num_gt

            loss += ssl_loss

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
