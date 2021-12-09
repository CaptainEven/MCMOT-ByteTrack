#!/usr/bin/env python3
# encoding=utf-8
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        """
        :param reduction:
        :param loss_type:
        """
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        :param pred:
        :param target:
        :return:
        """
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2),
                       (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2),
                       (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # return loss.mean(0).sum() / num_boxes

    return loss.sum() / num_boxes


def _expand_binary_labels(labels, label_weights, label_channels):
    """
    :param labels:
    :param label_weights:
    :param label_channels:
    :return:
    """
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMC(nn.Module):
    """
    GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        """
        :param bins:
        :param momentum:
        :param use_sigmoid:
        :param loss_weight:
        """
        super(GHMC, self).__init__()

        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6

        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()

        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred, dtype=torch.float32).cuda()

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        try:
            tot = max(valid.float().sum().item(), 1.0)
        except Exception as e:
            print(e)

        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot

        return loss * self.loss_weight
