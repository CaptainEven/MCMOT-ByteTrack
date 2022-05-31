# encoding=utf-8

import os
import math
import torch
import torch.nn as nn
from yolox.utils.myutils import parse_darknet_cfg
from .darknet_modules import build_modules

import torch.nn.functional as F
from loguru import logger

from yolox.utils import bboxes_iou
from .losses import IOUloss, GHMC, UncertaintyLoss


class YOLOXDarknet(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self,
                 cfg,
                 net_size=(768, 448),
                 strides=[8, 16, 32],
                 num_classes=5,
                 init_weights=True):
        """
        Darknet based
        :param cfg:
        :param net_size:
        :param strides:
        :param num_classes:
        :return
        """
        super().__init__()

        if not os.path.isfile(cfg):
            logger.error("Invalid cfg file path: {:s}.".format(cfg))
            exit(-1)

        ## ----- build the network
        self.module_defs = parse_darknet_cfg(cfg)
        logger.info("Network config file parsed.")
        self.module_list, self.routs = build_modules(self.module_defs, net_size, cfg, 3)
        if init_weights:
            self.init_weights()
            logger.info("Network weights initialized.")

        ## ----- define some modules
        self.n_anchors = 1
        self.num_classes = num_classes
        logger.info("Number of object classes: {:d}.".format(self.num_classes))
        self.decode_in_inference = True  # for deploy, set to False
        # self.scale_1st = 0.125  # 1/8

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)
        self.expanded_strides = [None] * len(strides)

    def init_layer_weights(self, m):
        """
        :param m:
        :return:
        """
        import torch.nn.init as init
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def init_weights(self):
        """
        Init weights of the net work
        """
        for m in self.module_list:
            self.init_layer_weights(m)

    def forward_once(self, x):
        """
        :param x:
        :return:
        """
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []  # 3(or 2) yolo layers correspond to 3(or 2) reid feature map layers

        # ---------- traverse the network(by traversing the module_list)
        use_output_layers = ['WeightedFeatureFusion',  # Shortcut(add)
                             'FeatureConcat',  # Route(concatenate)
                             'FeatureConcat_l',
                             'RouteGroup',
                             'ScaleChannel',
                             'ScaleChannels',  # my own implementation
                             'SAM']

        # ----- traverse forward
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in use_output_layers:  # sum, concat
                x = module.forward(x, out)

            elif name == 'YOLOLayer':  # x: current layer, out: previous layers output
                yolo_out.append(module.forward(x, out))

            # We need to process a shortcut layer combined with a activation layer
            # followed by a activation layer
            elif name == 'Sequential':
                for j, layer in enumerate(module):  # for debugging...
                    layer_name = layer.__class__.__name__

                    if layer_name in use_output_layers:
                        x = layer.forward(x, out)
                    else:
                        x = layer.forward(x)

            # run module directly, i.e. mtype = 'upsample', 'maxpool', 'batchnorm2d' etc.
            else:
                x = module.forward(x)

            # ----------- record previous output layers
            out.append(x if self.routs[i] else [])
            # out.append(x)  # for debugging: output every layer
        # ----------

        return x, out

    def forward(self, x, targets=None):
        """
        :param x:
        :param targets:
        """
        ## ----- out: final output, outs: outputs of each layer
        out, layer_outs = self.forward_once(x)
        imgs = x  # a batch of imgs data

        if self.training:
            assert targets is not None

        ## ----- build feature maps: 1/8, 1/16, 1/32
        self.fpn_outs = [layer_outs[58], layer_outs[52], layer_outs[46]]

        ## ----- build outputs
        self.reg_outputs = [layer_outs[93], layer_outs[95], layer_outs[97]]  # regression
        self.cls_outputs = [layer_outs[75], layer_outs[77], layer_outs[79]]  # classification
        self.obj_outputs = [layer_outs[99], layer_outs[101], layer_outs[103]]  # object-ness
        self.feature_map = layer_outs[61]  # feature vector map

        # traverse each scale
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (reg_output, obj_output, cls_output, stride_this_level) in enumerate(
                zip(self.reg_outputs, self.obj_outputs, self.cls_outputs, self.strides)
        ):
            if self.training:
                ## ----- concatenate different branch of outputs
                output = torch.cat([reg_output, obj_output, cls_output], 1)

                ## ----- grading and reshaping
                output, grid = self.get_output_and_grid(output, k, stride_this_level, x.type())

                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1])
                                        .fill_(stride_this_level)
                                        .type_as(self.fpn_outs[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                ## ----- concatenate different branch of outputs
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        if self.training:
            ## ---------- compute losses in the head
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                targets,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=self.fpn_outs[0].dtype,
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            self.hw = [x.shape[-2:] for x in outputs]

            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                outputs = self.decode_outputs(outputs, dtype=self.fpn_outs[0].type())

        return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        :param output:
        :param k:
        :param stride:
        :param dtype:
        :return:
        """
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )  ## --- reshape here

        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    def decode_outputs(self, outputs, dtype):
        """
        :param outputs:
        :param dtype:
        :return:
        """
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            # strides.append(torch.full((*shape, 1), stride)) # 2021.11.3
            strides.append(torch.full((*shape, 1), stride, dtype=torch.float))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

        return outputs

    def get_losses(self,
                   imgs,
                   x_shifts,
                   y_shifts,
                   expanded_strides,
                   labels,
                   outputs,
                   origin_preds,
                   dtype, ):
        """
        :param imgs:
        :param x_shifts:
        :param y_shifts:
        :param expanded_strides:
        :param labels:
        :param outputs:
        :param origin_preds:
        :param dtype:
        :return:
        """
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        ## ----- calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        ## ---------- processing each sample(image) of the batch
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                ## ----- Get ground truths
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # reg
                gt_classes = labels[batch_idx, :num_gt, 0]  # cls
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:  # noqa
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(batch_idx,
                                             num_gt,
                                             total_num_anchors,
                                             gt_bboxes_per_image,
                                             gt_classes,
                                             bboxes_preds_per_image,
                                             expanded_strides,
                                             x_shifts,
                                             y_shifts,
                                             cls_preds,
                                             bbox_preds,
                                             obj_preds,
                                             labels,
                                             imgs, )
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                ## ---------- build targets by matched GT inds
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) \
                             * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                ## ----------

                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)),
                                                   gt_bboxes_per_image[matched_gt_inds],
                                                   expanded_strides[0][fg_mask],
                                                   x_shifts=x_shifts[0][fg_mask],
                                                   y_shifts=y_shifts[0][fg_mask], )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() \
                   / num_fg
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        """
        return L1 targets
        """
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(self,
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        mode="gpu", ):
        """
        :param batch_idx:
        :param num_gt:
        :param total_num_anchors:
        :param gt_bboxes_per_image:
        :param gt_classes:
        :param bboxes_preds_per_image:
        :param expanded_strides:
        :param x_shifts:
        :param y_shifts:
        :param cls_preds:
        :param bbox_preds:
        :param obj_preds:
        :param labels:
        :param imgs:
        :param mode:
        :return:
        """

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        img_size = imgs.shape[2:]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image,
                                                                 expanded_strides,
                                                                 x_shifts,
                                                                 y_shifts,
                                                                 total_num_anchors,
                                                                 num_gt,
                                                                 img_size)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        ## ----- cxcywh
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                          * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(),
                                                        gt_cls_per_image,
                                                        reduction="none").sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(self,
                          gt_bboxes_per_image,
                          expanded_strides,
                          x_shifts,
                          y_shifts,
                          total_num_anchors,
                          num_gt,
                          img_size):
        """
        :param gt_bboxes_per_image:
        :param expanded_strides:
        :param x_shifts:
        :param y_shifts:
        :param total_num_anchors:
        :param num_gt:
        :param img_size:
        :return:
        """
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image

        # [n_anchor] -> [n_gt, n_anchor]
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        # clip center inside image
        gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
        gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
        gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])

        gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        del gt_bboxes_per_image_clip
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        :param cost:
        :param pair_wise_ious:
        :param gt_classes:
        :param num_gt:
        :param fg_mask:
        :return:
        """
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


class YOLOXDarknetReID(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self,
                 cfg,
                 net_size=(768, 448),
                 strides=[8, 16, 32],
                 num_classes=5,
                 init_weights=False,
                 reid=False,
                 max_id_dict=None,
                 use_momentum=True,
                 use_mtl=False):
        """
        Darknet based
        :param cfg:
        :param net_size:
        :param strides:
        :param num_classes:
        :param use_momentum:
        :param use_mtl:
        :return
        """
        super().__init__()

        if not os.path.isfile(cfg):
            logger.error("Invalid cfg file path: {:s}.".format(cfg))
            exit(-1)

        ## ----- build the network
        self.module_defs = parse_darknet_cfg(cfg)
        logger.info("Network config file parsed.")

        self.module_list, self.routs = build_modules(self.module_defs, net_size, cfg, 3, use_momentum)
        logger.info("Network modules built.")

        if init_weights:
            self.init_weights()
            logger.info("Network weights initialized.")
        else:
            logger.info("Network weights not initialized.")

        self.n_anchors = 1
        self.num_classes = num_classes
        logger.info("Number of object classes: {:d}.".format(self.num_classes))

        self.net_size = net_size
        print("Net size: ", self.net_size)

        self.reid = reid
        if self.reid:
            logger.info("ReID: True")

            if self.reid:
                # ----- Define ReID classifiers
                if max_id_dict is not None:
                    self.max_id_dict = max_id_dict
                    self.reid_classifiers = nn.ModuleList()  # num_classes layers of FC
                    for cls_id, nID in self.max_id_dict.items():
                        self.reid_classifiers.append(nn.Linear(128, nID + 5))  # normal FC layers
        else:
            logger.info("ReID: False")

        ## ----- define some modules
        self.decode_in_inference = False  # for deploy, set to False
        self.use_l1 = False
        self.scale_1st = 0.125  # 1/8

        ## ---------- Define loss functions
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.reid_loss = nn.CrossEntropyLoss()
        self.ghm_c = GHMC(bins=100)

        ## --- # whether to use MTL loss
        self.use_mtl = use_mtl
        if self.use_mtl:
            self.tasks = ["iou", "obj", "cls", "l1", "reid"]
            self.loss_dict = dict()
            for task in self.tasks:
                self.loss_dict[task] = 0.0
            self.mtl_loss = UncertaintyLoss(self.tasks)

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(self.strides)
        self.expanded_strides = [None] * len(self.strides)

    def init_layer_weights(self, m):
        """
        :param m:
        :return:
        """
        import torch.nn.init as init
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def init_weights(self):
        """
        Init weights of the net work
        """
        for m in self.module_list:
            self.init_layer_weights(m)

    def forward_once(self, x):
        """
        :param x:
        :return:
        """
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []  # 3(or 2) yolo layers correspond to 3(or 2) reid feature map layers

        # ---------- traverse the network(by traversing the module_list)
        use_output_layers = ['WeightedFeatureFusion',  # Shortcut(add)
                             'FeatureConcat',  # Route(concatenate)
                             'FeatureConcat_l',
                             'RouteGroup',
                             'ScaleChannel',
                             'ScaleChannels',  # my own implementation
                             'SAM']

        # ----- traverse forward
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in use_output_layers:  # sum, concat
                x = module.forward(x, out)

            elif name == 'YOLOLayer':  # x: current layer, out: previous layers output
                yolo_out.append(module.forward(x, out))

            # We need to process a shortcut layer combined with a activation layer
            # followed by a activation layer
            elif name == 'Sequential':
                for j, layer in enumerate(module):  # for debugging...
                    layer_name = layer.__class__.__name__

                    if layer_name in use_output_layers:
                        x = layer.forward(x, out)
                    else:
                        x = layer.forward(x)

            # run module directly, i.e. mtype = 'upsample', 'maxpool', 'batchnorm2d' etc.
            else:
                x = module.forward(x)

            # ----------- record previous output layers
            out.append(x if self.routs[i] else [])
            # out.append(x)  # for debugging: output every layer
        # ----------

        return x, out

    def forward(self, x, targets=None):
        """
        :param x:
        :param targets:
        """
        ## ----- out: final output, outs: outputs of each layer
        out, layer_outs = self.forward_once(x)
        imgs = x  # a batch of imgs data

        if self.training:
            assert targets is not None

        ## ----- build feature maps: 1/8, 1/16, 1/32
        self.fpn_outs = [layer_outs[58], layer_outs[52], layer_outs[46]]

        ## ----- build outputs
        self.reg_outputs = [layer_outs[93], layer_outs[95], layer_outs[97]]  # regression
        self.cls_outputs = [layer_outs[75], layer_outs[77], layer_outs[79]]  # classification
        self.obj_outputs = [layer_outs[99], layer_outs[101], layer_outs[103]]  # object-ness
        self.feature_map = layer_outs[61]  # feature vector map

        ## @even: ----- feature map output
        if self.reid:  # 20×128×56×96
            feature_output = self.feature_map

        # traverse each scale
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (reg_output, obj_output, cls_output, stride_this_level) in enumerate(
                zip(self.reg_outputs, self.obj_outputs, self.cls_outputs, self.strides)
        ):
            if self.training:
                ## ----- concatenate different branch of outputs
                output = torch.cat([reg_output, obj_output, cls_output], 1)

                ## ----- grading and reshaping
                output, grid = self.get_output_and_grid(output, k, stride_this_level, x.type())

                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1])
                                        .fill_(stride_this_level)
                                        .type_as(self.fpn_outs[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                ## ----- concatenate different branch of outputs
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        if self.training:
            ## ---------- compute losses in the head
            if self.reid:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, reid_loss, num_fg = \
                    self.get_losses_with_reid(imgs,
                                              x_shifts,
                                              y_shifts,
                                              expanded_strides,
                                              targets,
                                              torch.cat(outputs, 1), feature_output,
                                              origin_preds,
                                              dtype=self.fpn_outs[0].dtype, )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "reid_loss": reid_loss,
                    "num_fg": num_fg,
                }
            else:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg \
                    = self.get_losses(imgs,
                                      x_shifts,
                                      y_shifts,
                                      expanded_strides,
                                      targets,
                                      torch.cat(outputs, 1),
                                      origin_preds,
                                      dtype=self.fpn_outs[0].dtype, )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
        else:  # testing
            self.hw = [x.shape[-2:] for x in outputs]

            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                if self.reid:
                    outputs = self.decode_outputs(outputs, dtype=self.fpn_outs[0].type()), feature_output
                else:
                    outputs = self.decode_outputs(outputs, dtype=self.fpn_outs[0].type())

        return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        :param output:
        :param k:
        :param stride:
        :param dtype:
        :return:
        """
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )  ## --- reshape here

        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    def decode_outputs(self, outputs, dtype):
        """
        :param outputs:
        :param dtype:
        :return:
        """
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            # strides.append(torch.full((*shape, 1), stride)) # 2021.11.3
            strides.append(torch.full((*shape, 1), stride, dtype=torch.float))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

        return outputs

    def get_losses_with_reid(self,
                             imgs,
                             x_shifts,
                             y_shifts,
                             expanded_strides,
                             labels,
                             outputs, feature_output,
                             origin_preds,
                             dtype,
                             use_mtl=False):
        """
        :param imgs:
        :param x_shifts:
        :param y_shifts:
        :param expanded_strides:
        :param labels:
        :param outputs:
        :param feature_output:
        :param origin_preds:
        :param dtype:
        :param use_mtl:
        :return:
        """
        ## ---------- Get net outputs
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        ## ----- calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        n_label = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:  # False
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        reid_id_targets = []  # track ids, using for ReID loss
        reid_feature_targets = []
        gt_cls_id_targets = []
        l1_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        ## ---------- processing each sample(image) of the batch
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(n_label[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                ## ----- Get ground truths
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # reg
                gt_classes = labels[batch_idx, :num_gt, 0]  # class ids
                gt_ids = labels[batch_idx, :num_gt, 5]  # track ids

                ## ----- Get bbox(reg) predictions
                bboxes_preds_per_image = bbox_preds[batch_idx]  #

                try:  # noqa
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(batch_idx,
                                             num_gt,
                                             total_num_anchors,
                                             gt_bboxes_per_image,
                                             gt_classes,
                                             bboxes_preds_per_image,
                                             expanded_strides,
                                             x_shifts,
                                             y_shifts,
                                             cls_preds,
                                             bbox_preds,
                                             obj_preds,
                                             labels,
                                             imgs, )
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                              CPU mode is applied in this batch. If you want to avoid this issue, \
                              try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                ## ---------- build targets by matched GT inds
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) \
                             * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                # ----- ReID targets
                reid_target = gt_ids[matched_gt_inds]
                gt_cls_id_targets.append(gt_matched_classes.to(torch.int64))

                ## ----- Get feature vector for each GT bbox
                YXs = reg_target[:, :2] * self.scale_1st + 0.5
                YXs = YXs.long()

                Ys = YXs[:, 0]
                Xs = YXs[:, 1]

                ## ----- avoid exceed reid feature map's range
                Xs.clamp_(min=0, max=feature_output.shape[3] - 1)
                Ys.clamp_(min=0, max=feature_output.shape[2] - 1)

                feature = feature_output[batch_idx, :, Ys, Xs]
                feature.transpose_(0, 1)
                reid_feature_targets.append(feature)  # N×128
                ## ----------

                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)),
                                                   gt_bboxes_per_image[matched_gt_inds],
                                                   expanded_strides[0][fg_mask],
                                                   x_shifts=x_shifts[0][fg_mask],
                                                   y_shifts=y_shifts[0][fg_mask], )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            reid_id_targets.append(reid_target)
            fg_masks.append(fg_mask)

            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        reid_id_targets = torch.cat(reid_id_targets, 0)
        reid_feature_targets = torch.cat(reid_feature_targets, 0)
        gt_cls_id_targets = torch.cat(gt_cls_id_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_l1:  # False
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() \
                   / num_fg

        ## ----- compute ReID loss
        loss_reid = 0.0
        for cls_id, id_num in self.max_id_dict.items():
            inds = torch.where(gt_cls_id_targets == cls_id)
            if inds[0].shape[0] == 0:
                # print('skip class id', cls_id)
                continue

            cls_features = reid_feature_targets[inds]

            ## ----- L2 normalize the feature vector
            cls_features = F.normalize(cls_features, dim=1)

            ## ----- pass through the FC layer:
            cls_fc_preds = self.reid_classifiers[cls_id].forward(cls_features).contiguous()

            ## ----- compute loss
            cls_reid_id_target = reid_id_targets[inds]
            cls_reid_id_target = cls_reid_id_target.to(torch.int64)
            # loss_reid += self.reid_loss(cls_fc_preds, cls_reid_id_target)

            # --- GHM-C loss
            target = torch.zeros_like(cls_fc_preds)
            target.scatter_(1, cls_reid_id_target.view(-1, 1).to(torch.int64), 1)
            label_weight = torch.ones_like(cls_fc_preds)
            loss_reid += self.ghm_c.forward(cls_fc_preds, target, label_weight)

        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0

        if use_mtl:
            self.loss_dict["iou_loss"] = loss_iou * reg_weight
            self.loss_dict["obj_loss"] = loss_obj
            self.loss_dict["cls_loss"] = loss_cls
            self.loss_dict["l1_loss"] = loss_l1
            self.loss_dict["reid_loss"] = loss_reid
            loss_sum = self.mtl_loss.forward(self.loss_dict)
            return (
                loss_sum,
                self.loss_dict["iou_loss"],
                self.loss_dict["obj_loss"],
                self.loss_dict["cls_loss"],
                self.loss_dict["l1_loss"],
                self.loss_dict["reid_loss"],
                num_fg / max(num_gts, 1),
            )
        else:
            loss_sum = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + loss_reid
            return (
                loss_sum,
                loss_iou,
                loss_obj,
                loss_cls,
                loss_l1,
                loss_reid,
                num_fg / max(num_gts, 1),
            )

    def get_losses(self,
                   imgs,
                   x_shifts,
                   y_shifts,
                   expanded_strides,
                   labels,
                   outputs,
                   origin_preds,
                   dtype,):
        """
        :param imgs:
        :param x_shifts:
        :param y_shifts:
        :param expanded_strides:
        :param labels:
        :param outputs:
        :param origin_preds:
        :param dtype:
        :return:
        """
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        ## ----- calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        ## ---------- processing each sample(image) of the batch
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                ## ----- Get ground truths
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # reg
                gt_classes = labels[batch_idx, :num_gt, 0]  # cls
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:  # noqa
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(batch_idx,
                                             num_gt,
                                             total_num_anchors,
                                             gt_bboxes_per_image,
                                             gt_classes,
                                             bboxes_preds_per_image,
                                             expanded_strides,
                                             x_shifts,
                                             y_shifts,
                                             cls_preds,
                                             bbox_preds,
                                             obj_preds,
                                             labels,
                                             imgs, )
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                ## ---------- build targets by matched GT inds
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) \
                             * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                ## ----------

                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)),
                                                   gt_bboxes_per_image[matched_gt_inds],
                                                   expanded_strides[0][fg_mask],
                                                   x_shifts=x_shifts[0][fg_mask],
                                                   y_shifts=y_shifts[0][fg_mask], )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() \
                   / num_fg
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        """
        return L1 targets
        """
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(self,
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        mode="gpu", ):
        """
        :param batch_idx:
        :param num_gt:
        :param total_num_anchors:
        :param gt_bboxes_per_image:
        :param gt_classes:
        :param bboxes_preds_per_image:
        :param expanded_strides:
        :param x_shifts:
        :param y_shifts:
        :param cls_preds:
        :param bbox_preds:
        :param obj_preds:
        :param labels:
        :param imgs:
        :param mode:
        :return:
        """

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        img_size = imgs.shape[2:]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image,
                                                                 expanded_strides,
                                                                 x_shifts,
                                                                 y_shifts,
                                                                 total_num_anchors,
                                                                 num_gt,
                                                                 img_size)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        ## ----- cxcywh
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                          * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(),
                                                        gt_cls_per_image,
                                                        reduction="none").sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(self,
                          gt_bboxes_per_image,
                          expanded_strides,
                          x_shifts,
                          y_shifts,
                          total_num_anchors,
                          num_gt,
                          img_size):
        """
        :param gt_bboxes_per_image:
        :param expanded_strides:
        :param x_shifts:
        :param y_shifts:
        :param total_num_anchors:
        :param num_gt:
        :param img_size:
        :return:
        """
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image

        # [n_anchor] -> [n_gt, n_anchor]
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5
        # clip center inside image
        gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
        gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
        gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])

        gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        del gt_bboxes_per_image_clip
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        :param cost:
        :param pair_wise_ious:
        :param gt_classes:
        :param num_gt:
        :param fg_mask:
        :return:
        """
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
