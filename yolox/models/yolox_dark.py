#!/usr/bin/env python
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import numpy as np
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
                 T=0.5):
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

        # temperature
        self.T = T

        ## ----- version info
        self.__name__ = "Darknet"
        self.version = np.array([1, 0, 0], dtype=np.int32)
        self.seen = np.array([0], dtype=np.int64)

    def forward(self, inps, targets=None, p0=None, p1=None, p2=None):
        """
        :param inps:
        :param targets:
        :return:
        """
        ## ----- pass through backbone fpn output content features
        # of all(default: 3) scales: 1/8, 1/16, 1/32
        layer_outs = self.backbone.forward(inps)
        fpn_outs = self.backbone.fpn_outs

        ## ----- pass through the head
        if self.training:
            assert targets is not None
            assert not (p0 is None or p1 is None or p2 is None)

            ## ----- Get object detection losses and feature map
            losses, feature_map = self.head.forward(fpn_outs, targets, inps)
            total_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = losses

            ## ----- Get size
            net_h, net_w = inps.shape[2], inps.shape[3]
            map_h, map_w = feature_map.shape[2], feature_map.shape[3]

            ## ----- Get number of objects(ground truth)
            valid_lb_inds = targets.sum(dim=2) > 0  # batch_size×50(True | False)
            num_gts = valid_lb_inds.sum(dim=1)  # batch_size×n_lb_valid

            ## ---------- Calculate SSL loss of the batch
            ssl_loss = 0.0
            cycle_loss = 0.0
            # sim_mat_loss = 0.0
            scale_consistent_loss = 0.0

            for batch_idx, num_gt in enumerate(num_gts):
                num_gt = int(num_gt)
                if num_gt == 0:
                    continue

                ## ----- Get feature vectors of the image
                p0_vectors = p0[batch_idx]
                p1_vectors = p1[batch_idx]
                p2_vectors = p2[batch_idx]

                # ----- inference
                p0_layers = self.backbone.forward(p0_vectors)
                p1_layers = self.backbone.forward(p1_vectors)
                p2_layers = self.backbone.forward(p2_vectors)

                p0_fpn_layers = [p0_layers[i] for i in self.backbone.out_inds]
                p1_fpn_layers = [p1_layers[i] for i in self.backbone.out_inds]
                p2_fpn_layers = [p2_layers[i] for i in self.backbone.out_inds]

                p0_feature_map = self.head.get_feature_map(p0_fpn_layers)  # 1/8
                p1_feature_map = self.head.get_feature_map(p1_fpn_layers)  # 1/8
                p2_feature_map = self.head.get_feature_map(p2_fpn_layers)  # 1/8

                p0_vectors = self.head.reid_convs(p0_feature_map)
                p0_vectors = self.head.reid_preds(p0_vectors)
                p0_vectors = p0_vectors.reshape(p0_vectors.shape[0], -1)
                p0_vectors = nn.functional.normalize(p0_vectors, dim=1)

                p1_vectors = self.head.reid_convs(p1_feature_map)
                p1_vectors = self.head.reid_preds(p1_vectors)
                p1_vectors = p1_vectors.reshape(p1_vectors.shape[0], -1)
                p1_vectors = nn.functional.normalize(p1_vectors, dim=1)

                p2_vectors = self.head.reid_convs(p2_feature_map)
                p2_vectors = self.head.reid_preds(p2_vectors)
                p2_vectors = p2_vectors.reshape(p2_vectors.shape[0], -1)
                p2_vectors = nn.functional.normalize(p2_vectors, dim=1)

                # ---------- SSL loss calculations
                ## ---------- processing each sample(image) of the batch
                p0_vectors = p0_vectors[:num_gt]  # num_gt×128
                p1_vectors = p1_vectors[:num_gt]
                p2_vectors = p2_vectors[:num_gt]

                # ## ----- Calculate similarity matrix loss
                # sm_output = torch.mm(p1_vectors, p2_vectors.T)
                # sm_diff = sm_output - torch.eye(num_gt).cuda()
                # sm_diff = torch.pow(sm_diff, 2)
                # sim_mat_loss = sm_diff.sum() / (num_gt * num_gt)

                ## ----- Calculate feature scale-consistency loss
                ## of feature map and patch feature vector difference
                for i, (p0_vector, p1_vector, p2_vector) in enumerate(zip(p0_vectors, p1_vectors, p2_vectors)):
                    cls_id, cx, cy, w, h = targets[batch_idx][i]  # in net_size

                    ## ----- get center_x, center_y in feature map
                    center_x = int(cx / net_w * map_w + 0.5)
                    center_y = int(cy / net_h * map_h + 0.5)
                    center_x = center_x if center_x < map_w else map_w - 1
                    center_y = center_y if center_y < map_h else map_h - 1

                    feature_vector = feature_map[batch_idx, :, center_y, center_x]
                    feature_vector = nn.functional.normalize(feature_vector.view(1, -1), dim=1)
                    feature_vector = torch.squeeze(feature_vector)

                    scale_consistent_loss += 1.0 - torch.dot(p0_vector, feature_vector)
                    # scale_consistent_loss += 1.0 - torch.dot(p1_vector, feature_vector)
                    # scale_consistent_loss += 1.0 - torch.dot(p2_vector, feature_vector)

                if targets.shape[0] > 0:
                    scale_consistent_loss /= targets.shape[0]

                ## ----- calculate Cycle consistency loss
                cyc_cnt = 0
                for i in range(p1_vectors.shape[0]):
                    for j in range(p2_vectors.shape[0]):
                        if j != i:
                            sim_p1i_p2j = torch.dot(p1_vectors[i], p2_vectors[j])
                            sim_p1j_p2i = torch.dot(p1_vectors[j], p2_vectors[i])
                            cycle_loss += abs(sim_p1i_p2j - sim_p1j_p2i)
                            cyc_cnt += 1

                if cyc_cnt > 0:
                    cycle_loss /= cyc_cnt

                ## ----- Calculate intra-positive SSL loss
                logits_intra_pos = torch.einsum('nc,ck->nk', [p1_vectors, p2_vectors.T])
                # logits_intra_pos = torc.mm(p1_vectors, p2_vectors.T)
                logits_intra_pos /= self.T
                labels_intra = torch.arange(logits_intra_pos.shape[0]).cuda()
                ssl_intra_loss = self.head.softmax_loss(logits_intra_pos, labels_intra) / num_gt
                ssl_loss += ssl_intra_loss

            # total_loss += sim_mat_loss
            total_loss += scale_consistent_loss
            total_loss += cycle_loss
            total_loss += ssl_loss

            outputs = {
                "total_loss": total_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "ssl_loss": ssl_loss,
                # "sim_mat_loss": sim_mat_loss,
                "cycle_loss": cycle_loss,
                "scale_consistent_loss": scale_consistent_loss,
                "num_fg": num_fg,
            }
        else:  # testing
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
