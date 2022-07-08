#!/usr/bin/env python3
# encoding=utf-8
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

import os
import os.path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger

from yolox.data.data_augment import GaussianBlur
from yolox.evaluators.voc_eval import voc_eval
from yolox.utils.myutils import filter_bbox_by_ious
from .datasets_wrapper import Dataset
from .voc_classes import C5_CLASSES
from ..data_augment import PatchTransform


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        """
        :param class_to_ind:
        :param keep_difficult:
        """
        self.class_to_ind = class_to_ind or dict(zip(C5_CLASSES, range(len(C5_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.find("markNode").iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.keep_difficult and difficult:
            #     continue
            # name = obj.find("name").text.lower().strip()
            name = obj.find("targettype").text.lower().strip()
            if name not in ["car", "bicycle", "person", "cyclist", "tricycle"]:
                continue
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetSSL(Dataset):
    """
    VOC Detection Dataset Object with SSL, do not use mix_up
    input is image, target is annotation
    Args:
        root (string): filepath to VOC devkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                 data_dir,
                 f_list_path,
                 img_size=(768, 448),
                 preproc=None,
                 target_transform=AnnotationTransform(),
                 max_patches=50,
                 num_negatives=30,
                 neg_pos_iou_thresh=0.1,
                 patch_size=(224, 224),
                 max_sample_times=5):
        """
        :param data_dir:
        :param img_size:
        :param preproc:
        :param target_transform:
        :param patch_size: h, w
        """
        super().__init__(img_size)

        self.root = data_dir
        self.f_list_path = f_list_path

        ## -----image size: (width, height)
        self.img_size = img_size

        ## ----- The patch size for training
        self.patch_size = patch_size  # h, w
        self.max_pos_patches = max_patches  # number of max patches
        self.max_neg_patches = num_negatives
        self.np_iou_thresh = neg_pos_iou_thresh

        self.preproc = preproc
        self.target_transform = target_transform

        self.max_sample_times = max_sample_times

        # self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        # self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._annopath = os.path.join(self.root + "/", "%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root + "/", "%s", "JPEGImages", "%s.jpg")
        self._classes = C5_CLASSES
        logger.info("Classes: " + ",".join(self._classes))
        self.ids = list()

        if not os.path.isfile(self.f_list_path):
            logger.error("invalid file list path!")
            exit(-1)

        with open(self.f_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder_name = line.split('/')[5]
                img_name = line.split('/')[-1].replace('.jpg', '').replace('\n', '')
                self.ids.append((folder_name, img_name))

        logger.info("Total {:d} VOC detection samples to be trained."
                    .format(len(self.ids)))

        ## ----- Define the positive sample transformations
        self.pos_patch_transform = PatchTransform(patch_size=self.patch_size)

        ## ----- Define the negative sample transformations
        # self.neg_patch_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225], )
        #     ]
        # )

        self.neg_patch_transform = transforms.Compose(
            [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        """
        :return:
        """
        return len(self.ids)

    def load_anno(self, index):
        """
        :param index:
        :return:
        """
        img_id = self.ids[index]
        label_f_path = self._annopath % img_id
        et_root = ET.parse(label_f_path).getroot()
        if self.target_transform is not None:
            target = self.target_transform(et_root)

        return target

    def random_shape_crops(self, W, H, num=100, max_sample_times=10):
        """
        :param W:
        :param H:
        :param num:
        :param max_sample_times:
        """
        valid_count = 0
        sample_num = num * 5
        sample_n_times = 0

        x1y1x2y2_final = np.empty((0, 4), dtype=np.int64)
        while valid_count < num:
            if sample_n_times > max_sample_times:
                break

            ## ----- batch generating x1, y1, x2, y2
            x1x2 = np.random.randint(0, W, size=(sample_num, 2))
            y1y2 = np.random.randint(0, H, size=(sample_num, 2))

            inds = np.where(x1x2[:, 1] > x1x2[:, 0])
            x1x2 = x1x2[inds]

            inds = np.where(y1y2[:, 1] > y1y2[:, 0])
            y1y2 = y1y2[inds]

            x1y1x2y2 = np.empty((0, 4), dtype=np.int64)
            if x1x2.shape[0] >= y1y2.shape[0]:
                x1y1x2y2 = np.concatenate((np.expand_dims(x1x2[:y1y2.shape[0], 0], axis=1),
                                           np.expand_dims(y1y2[:, 0], axis=1),
                                           np.expand_dims(x1x2[:y1y2.shape[0], 1], axis=1),
                                           np.expand_dims(y1y2[:, 1], axis=1)), axis=1)
            else:
                x1y1x2y2 = np.concatenate((np.expand_dims(x1x2[:, 0], axis=1),
                                           np.expand_dims(y1y2[:x1x2.shape[0], 0], axis=1),
                                           np.expand_dims(x1x2[:, 1], axis=1),
                                           np.expand_dims(y1y2[:x1x2.shape[0], 1], axis=1)), axis=1)

            if sample_n_times == 0:
                x1y1x2y2_final = x1y1x2y2.copy()
            else:
                x1y1x2y2_final = np.concatenate([x1y1x2y2_final, x1y1x2y2], axis=1)

            valid_count = x1y1x2y2_final.shape[0]
            if valid_count >= num:
                break

        if x1y1x2y2_final.shape[0] >= num:
            return x1y1x2y2_final[:num]
        else:
            return x1y1x2y2_final

    def random_crops(self,
                     W, H,
                     crop_size=(320, 320),
                     num=100,
                     max_sample_times=30):
        """
        :param W: img width
        :param H: img height
        :param crop_size:
        :param num:
        :param max_sample_times:
        """
        valid_count = 0
        sample_num = num * 5
        sample_n_times = 0

        x1y1x2y2_final = np.empty((0, 4), dtype=np.int64)
        while valid_count < num:
            if sample_n_times > max_sample_times:
                break

            x1 = np.random.randint(0, W - crop_size[1], size=(sample_num))
            y1 = np.random.randint(0, H - crop_size[0], size=(sample_num))

            x2 = x1 + crop_size[1]
            y2 = y1 + crop_size[0]

            x2 = x2[x2 < W]
            y2 = y2[y2 < H]

            if x2.size > y2.size:
                x1 = x1[:y2.size]
                y1 = y1[:y2.size]
                x2 = x2[:y2.size]
            else:
                x1 = x1[:x2.size]
                y1 = y1[:x2.size]
                y2 = y2[:x2.size]

            x1 = np.expand_dims(x1, axis=1)
            y1 = np.expand_dims(y1, axis=1)
            x2 = np.expand_dims(x2, axis=1)
            y2 = np.expand_dims(y2, axis=1)

            x1y1x2y2 = np.concatenate([x1, y1, x2, y2], axis=1)
            if sample_n_times == 0:
                x1y1x2y2_final = x1y1x2y2.copy()
            else:
                x1y1x2y2_final = np.concatenate([x1y1x2y2_final, x1y1x2y2], axis=0)

            valid_count = x1y1x2y2_final.shape[0]
            if valid_count >= num:
                break

        if x1y1x2y2_final.shape[0] >= num:
            return x1y1x2y2_final[:num]
        else:
            return x1y1x2y2_final

    def gen_negatives(self,
                      img,
                      pos_bboxs,
                      max_sample_times=30):
        """
        Sample negative proposals
        :param img:
        :param pos_bboxs:
        :param max_sample_times:
        """
        if img is None:
            print("[Err]: empty image, exit now.")
            exit(-1)
        if pos_bboxs.size == 0:  # 不能返回empty negative和positive, 无法保证batch对齐
            # print("[Warning]: empty positive bboxes, return random negative bboxes.")
            return self.random_crops(img.shape[1],
                                     img.shape[0],
                                     self.patch_size,
                                     self.max_neg_patches,
                                     self.max_sample_times)

        H, W = img.shape[:2]

        ## ----- Get the max positive bbox
        # areas = (pos_bboxs[:, 2] - pos_bboxs[:, 0]) \
        #         * (pos_bboxs[:, 3] - pos_bboxs[:, 1])
        # max_area_id = np.argmax(areas)
        # max_bbox_w = int((pos_bboxs[max_area_id][2] - pos_bboxs[max_area_id][0]) * 0.5)
        # max_bbox_h = int((pos_bboxs[max_area_id][3] - pos_bboxs[max_area_id][1]) * 0.5)
        max_w_id = np.argmax(pos_bboxs[:, 2] - pos_bboxs[:, 0])  # x2 - x1
        max_h_id = np.argmax(pos_bboxs[:, 3] - pos_bboxs[:, 1])  # y2 - y1
        max_bbox_w = int((pos_bboxs[max_w_id][2] - pos_bboxs[max_w_id][0]) * 0.5)
        max_bbox_h = int((pos_bboxs[max_w_id][3] - pos_bboxs[max_w_id][1]) * 0.5)

        neg_patches = []
        valid_neg_count = 0
        sample_num = self.max_neg_patches * 5
        sample_n_times = 1
        rand_sample_negatives = False

        neg_bboxes_final = np.empty((0, 4), dtype=np.int64)
        while valid_neg_count < self.max_neg_patches:
            if sample_n_times > max_sample_times:
                break

            if sample_num == 0:
                # print("[Warning]: can not get enough negative samples, do sample again.")
                sample_num = self.max_neg_patches * 5
                self.np_iou_thresh += 0.1
                if self.np_iou_thresh >= 0.5:
                    # print("[Warning]: start random sampling for negative samples.")
                    rand_sample_negatives = True

            if not rand_sample_negatives:
                neg_bboxes = self.random_shape_crops(W,
                                                     H,
                                                     self.max_neg_patches * 5,
                                                     self.max_sample_times)
            else:
                neg_bboxes = self.random_crops(W,
                                               H,
                                               self.patch_size,
                                               self.max_neg_patches,
                                               self.max_sample_times)

            if sample_n_times == 1:
                if not rand_sample_negatives:
                    ## ----- filtering by IOU check
                    neg_bboxes = filter_bbox_by_ious(neg_bboxes, pos_bboxs, self.np_iou_thresh)

                    ## ----- filter by aspect ratio check
                    neg_bboxes = filter(lambda x: (x[2] - x[0]) /
                                                  (x[3] - x[1]) > 0.5 and
                                                  (x[2] - x[0])
                                                  / (x[3] - x[1]) < 2.0,
                                        neg_bboxes)
                    neg_bboxes = filter(lambda x: (x[2] - x[0]) < max_bbox_w and
                                                  (x[3] - x[1]) < max_bbox_h and
                                                  (x[2] - x[0]) > 0.02 * W and
                                                  (x[3] - x[1]) > 0.02 * H,
                                        neg_bboxes)
                    neg_bboxes = np.array(list(neg_bboxes))

                ## --- first time: copy
                neg_bboxes_final = neg_bboxes.copy()

            else:
                if not rand_sample_negatives:
                    ## ----- filtering by IOU check
                    neg_bboxes = filter_bbox_by_ious(neg_bboxes, pos_bboxs, self.np_iou_thresh)

                    ## ----- filter by aspect ratio check
                    neg_bboxes = filter(lambda x: (x[2] - x[0]) /
                                                  (x[3] - x[1]) > 0.5 and
                                                  (x[2] - x[0])
                                                  / (x[3] - x[1]) < 2.0,
                                        neg_bboxes)
                    neg_bboxes = filter(lambda x: (x[2] - x[0]) < max_bbox_w and
                                                  (x[3] - x[1]) < max_bbox_h and
                                                  (x[2] - x[0]) > 0.02 * W and
                                                  (x[3] - x[1]) > 0.02 * H,
                                        neg_bboxes)
                    neg_bboxes = np.array(list(neg_bboxes))

                ## --- not the first time: cat
                if neg_bboxes_final.size > 0:
                    if neg_bboxes.size > 0:
                        neg_bboxes_final = np.concatenate([neg_bboxes_final, neg_bboxes],
                                                          axis=0)
                else:
                    neg_bboxes_final = neg_bboxes.copy()

            valid_neg_count = neg_bboxes_final.shape[0]
            if valid_neg_count >= self.max_neg_patches:
                # print("[Info]: negative samples got.")
                neg_bboxes_final = neg_bboxes_final[:self.max_neg_patches]
                break

            # sample_num -= 10  # sample_num decrease the next sampling
            sample_n_times += 1

        # ## ----- visualize
        # cv2.imwrite("/mnt/diskd/tmp/orgin.jpg", img)
        # for i, neg_bboxes in enumerate(neg_bboxes_final):
        #     ## ----- Get a patch and visualize
        #     x1, y1, x2, y2 = neg_bboxes
        #     patch = img[y1:y2, x1:x2, :]
        #
        #     save_path = "/mnt/diskd/tmp/patch_{:d}.jpg".format(i + 1)
        #     cv2.imwrite(save_path, patch)
        #     print("{:s} saved.".format(save_path))

        ## ----- reset the IOU threshold
        self.np_iou_thresh = 0.1

        if neg_bboxes_final.shape[0] < self.max_neg_patches:
            more_neg_bboxes = self.random_crops(W,
                                                H,
                                                self.patch_size,
                                                self.max_neg_patches - neg_bboxes_final.shape[0],
                                                self.max_sample_times)
            if neg_bboxes_final.size == 0:
                neg_bboxes_final = more_neg_bboxes.copy()
            else:
                neg_bboxes_final = np.concatenate([neg_bboxes_final, more_neg_bboxes],
                                                  axis=0)

        return neg_bboxes_final

    def pull_item(self, idx):
        """
        Returns the original image and target at an index for mix_up
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        ## ----- load image
        img_id = self.ids[idx]
        img_path = self._imgpath % img_id

        ## ----- Read in original image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # HWC
        height, width, _ = img.shape

        ## ----- load label, target: x1,y1,x2,y2,cls_id
        targets = self.load_anno(idx)  # n_objs×5
        # print(target.shape)

        ## ----- Get positive bboxes
        pos_bboxes = np.empty((0, 4), dtype=np.int64)
        pos_patches = []

        if targets.size == 0:  # empty objs
            pos_bboxes = self.random_crops(img.shape[1],
                                           img.shape[0],
                                           self.patch_size,
                                           self.max_pos_patches,
                                           self.max_sample_times)
            for i, bbox in enumerate(pos_bboxes):
                x1, y1, x2, y2 = bbox

                ## ----- Get patch
                patch = img[int(y1):int(y2), int(x1):int(x2), :]
                pos_patches.append(patch)
        else:
            pos_bboxes = np.zeros((targets.shape[0], 4), dtype=np.float64)

            ## ----- record positive bboxes
            for i, bbox_clsid in enumerate(targets):
                if i >= self.max_pos_patches:
                    break

                x1, y1, x2, y2, cls_id = bbox_clsid

                x1 = x1 if x1 >= 0 else 0
                x1 = x1 if x1 < width else width - 1

                y1 = y1 if y1 >= 0 else 0
                y1 = y1 if y1 < height else height - 1

                x2 = x2 if x2 >= 0 else 0
                x2 = x2 if x2 < width else width - 1

                y2 = y2 if y2 >= 0 else 0
                y2 = y2 if y2 < height else height - 1

                ## ----- Get patch
                patch = img[int(y1):int(y2), int(x1):int(x2), :]
                pos_patches.append(patch)
                pos_bboxes[i] = np.array([x1, y1, x2, y2], dtype=np.float64)

        ## ---------- load patches for query and key
        # ----- Get positive pairs of patches
        q = torch.zeros((self.max_pos_patches, 3, self.patch_size[0], self.patch_size[1]))
        k = torch.zeros((self.max_pos_patches, 3, self.patch_size[0], self.patch_size[1]))
        n = torch.zeros((self.max_neg_patches, 3, self.patch_size[0], self.patch_size[1]))
        for i, patch in enumerate(pos_patches):
            ## ----- Resize
            if patch.size == 0:  # empty positive patch
                continue

            try:
                patch = cv2.resize(patch, self.patch_size, cv2.INTER_AREA)
            except Exception as e:
                print(e)
            try:
                q_item, k_item = self.pos_patch_transform(Image.fromarray(patch))
                q[i] = q_item
                k[i] = k_item
            except Exception as e:
                print(e)
                print(img_path, q_item, k_item)

        # img_hw = (height, width)

        ## ----- Generate negative samples
        neg_bboxes = self.gen_negatives(img, pos_bboxes, self.max_sample_times)
        for i, neg_bbox in enumerate(neg_bboxes):
            x1, y1, x2, y2 = neg_bbox
            patch = img[int(y1):int(y2), int(x1):int(x2), :]
            patch = cv2.resize(patch, self.patch_size, cv2.INTER_AREA)
            n_item = self.neg_patch_transform(Image.fromarray(patch))
            n[i] = n_item

        return img, targets, q, k, n

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img, target, q, k, n = self.pull_item(idx)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, q, k, n

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        # IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        # mAPs = []
        # for iou in IouTh:
        #     mAP = self._do_python_eval(output_dir, iou)
        #     mAPs.append(mAP)

        # print("--------------------------------------------------------------")
        # print("map_5095:", np.mean(mAPs))
        # print("map_50:", mAPs[0])
        # print("--------------------------------------------------------------")
        return 0, 0  # np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):  # not use.
        """
        :return:
        """
        filename = "comp4_det_test" + "_{:s}.txt"
        # filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        filedir = os.path.join("/users/duanyou/c5/experiments/YOLOX-main-duan/YOLOX_outputs/yolox-tiny")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        """
        :param all_boxes:
        :return:
        """
        for im_ind, index in enumerate(self.ids):
            index0 = index[0]  # test_3000
            index1 = index[1]  # 10_2_XGTCTX00014_TX020127_20201222092014_866_0
            filename = '/mnt/diskc/even/ByteTrack/YOLOX_outputs/yolox_nano_det_c5/results_3000/' + index1 + '.txt'
            # filename = self.results_path + index1 + '.txt'
            with open(filename, "wt") as f:
                f.write("class scores x y w h total= \n")
                for cls_ind, cls in enumerate(C5_CLASSES):
                    if cls == "__background__":
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        if dets[k, -1] < 0.2:
                            continue
                        f.write(
                            "{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                                cls_ind,
                                dets[k, -1],
                                ((dets[k, 0] + 1) + (dets[k, 2] + 1)) / 3840,
                                ((dets[k, 1] + 1) + (dets[k, 3] + 1)) / 2160,
                                ((dets[k, 2] + 1) - (dets[k, 0] + 1)) / 1920,
                                ((dets[k, 3] + 1) - (dets[k, 1] + 1)) / 1080,
                            )
                        )
        # for cls_ind, cls in enumerate(VOC_CLASSES):
        #     cls_ind = cls_ind
        #     if cls == "__background__":
        #         continue
        #     print("Writing {} VOC results file".format(cls))
        #     filename = self._get_voc_results_file_template().format(cls)
        #     with open(filename, "wt") as f:
        #         for im_ind, index in enumerate(self.ids):
        #             index0 = index[0] # test_3000
        #             index1 = index[1] # 10_2_XGTCTX00014_TX020127_20201222092014_866_0
        #             dets = all_boxes[cls_ind][im_ind]
        #             if dets == []:
        #                 continue
        # for k in range(dets.shape[0]):
        #     f.write(
        #         "{:s} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
        #             index0,
        #             index1,
        #             dets[k, -1],
        #             dets[k, 0] + 1,
        #             dets[k, 1] + 1,
        #             dets[k, 2] + 1,
        #             dets[k, 3] + 1,
        #         )
        #     )

    def _write_voc_results_file_ori(self, all_boxes):
        """
        :param all_boxes:
        :return:
        """
        for cls_ind, cls in enumerate(C5_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index0 = index[0]
                    index1 = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index0,
                                index1,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    # TO DO.
    def _do_python_eval(self, output_dir="output", iou=0.5):
        # rootpath = os.path.join(self.root, "VOC" + self._year)
        # name = self.image_set[0][1]
        # annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        # imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        # cachedir = os.path.join(
        #     self.root, "annotations_cache", "VOC" + self._year, name
        # )
        # if not os.path.exists(cachedir):
        #     os.makedirs(cachedir)
        # aps = []
        # # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        # print("Eval IoU : {:.2f}".format(iou))
        # if output_dir is not None and not os.path.isdir(output_dir):
        #     os.mkdir(output_dir)
        cachedir = "/users/duanyou/c5/experiments/YOLOX-main-duan/YOLOX_outputs/yolox-tiny"
        annopath = []
        imagesetfile = []
        detall = [['name', 'obj_type', 'score', 0, 0, 0, 0]]
        f1 = open(self.root, 'r')
        lines = f1.readlines()
        for line in lines:
            image_name = line.replace('\n', '')
            xml_name = image_name.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
            annopath.append(xml_name)
            imagesetfile.append(image_name)
        for i, cls in enumerate(C5_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            print('filename: ', filename)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=False,
            )
            # detall = np.delete(detall, 0, axis = 0)
            # det_objtype = [obj for obj in detall if obj[1] == cls]
            # rec, prec, ap = voc_eval(det_objtype, annopath, imagesetfile, classname=cls, ovthresh=iou)
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            # if output_dir is not None:
            #     with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
            #         pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return 0  # np.mean(aps)


class VOCDetection(Dataset):
    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                 data_dir,
                 f_list_path,
                 img_size=(768, 448),
                 preproc=None,
                 target_transform=AnnotationTransform(), ):
        """
        :param data_dir:
        :param img_size:
        :param preproc:
        :param target_transform:
        """
        super().__init__(img_size)

        self.root = data_dir
        self.f_list_path = f_list_path

        ## -----image size: (width, height)
        self.img_size = img_size

        self.preproc = preproc
        self.target_transform = target_transform

        # self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        # self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._annopath = os.path.join(self.root + "/", "%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root + "/", "%s", "JPEGImages", "%s.jpg")
        self._classes = C5_CLASSES
        logger.info("Classes: " + ",".join(self._classes))
        self.ids = list()

        if not os.path.isfile(self.f_list_path):
            print("[Err]: invalid file list path!")
            exit(-1)

        with open(self.f_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder_name = line.split('/')[5]
                img_name = line.split('/')[-1].replace('.jpg', '').replace('\n', '')
                self.ids.append((folder_name, img_name))

        logger.info("Total {:d} VOC detection samples to be trained."
                    .format(len(self.ids)))

    def __len__(self):
        """
        :return:
        """
        return len(self.ids)

    def load_anno(self, index):
        """
        :param index:
        :return:
        """
        img_id = self.ids[index]
        label_f_path = self._annopath % img_id
        target = ET.parse(label_f_path).getroot()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target

    def pull_item(self, idx):
        """
        Returns the original image and target at an index for mixup
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        ## ----- load image
        img_id = self.ids[idx]
        img_path = self._imgpath % img_id
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # HWC
        height, width, _ = img.shape

        ## ----- load label
        target = self.load_anno(idx)
        # print(target.shape)

        img_info = (height, width)

        return img, target, img_info, idx

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img, target, img_info, img_id = self.pull_item(idx)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        # IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        # mAPs = []
        # for iou in IouTh:
        #     mAP = self._do_python_eval(output_dir, iou)
        #     mAPs.append(mAP)

        # print("--------------------------------------------------------------")
        # print("map_5095:", np.mean(mAPs))
        # print("map_50:", mAPs[0])
        # print("--------------------------------------------------------------")
        return 0, 0  # np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):  # not use.
        """
        :return:
        """
        filename = "comp4_det_test" + "_{:s}.txt"
        # filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        filedir = os.path.join("/users/duanyou/c5/experiments/YOLOX-main-duan/YOLOX_outputs/yolox-tiny")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        """
        :param all_boxes:
        :return:
        """
        for im_ind, index in enumerate(self.ids):
            index0 = index[0]  # test_3000
            index1 = index[1]  # 10_2_XGTCTX00014_TX020127_20201222092014_866_0
            filename = '/mnt/diskc/even/ByteTrack/YOLOX_outputs/yolox_nano_det_c5/results_3000/' + index1 + '.txt'
            # filename = self.results_path + index1 + '.txt'
            with open(filename, "wt") as f:
                f.write("class scores x y w h total= \n")
                for cls_ind, cls in enumerate(C5_CLASSES):
                    if cls == "__background__":
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        if dets[k, -1] < 0.2:
                            continue
                        f.write(
                            "{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                                cls_ind,
                                dets[k, -1],
                                ((dets[k, 0] + 1) + (dets[k, 2] + 1)) / 3840,
                                ((dets[k, 1] + 1) + (dets[k, 3] + 1)) / 2160,
                                ((dets[k, 2] + 1) - (dets[k, 0] + 1)) / 1920,
                                ((dets[k, 3] + 1) - (dets[k, 1] + 1)) / 1080,
                            )
                        )
        # for cls_ind, cls in enumerate(VOC_CLASSES):
        #     cls_ind = cls_ind
        #     if cls == "__background__":
        #         continue
        #     print("Writing {} VOC results file".format(cls))
        #     filename = self._get_voc_results_file_template().format(cls)
        #     with open(filename, "wt") as f:
        #         for im_ind, index in enumerate(self.ids):
        #             index0 = index[0] # test_3000
        #             index1 = index[1] # 10_2_XGTCTX00014_TX020127_20201222092014_866_0
        #             dets = all_boxes[cls_ind][im_ind]
        #             if dets == []:
        #                 continue
        # for k in range(dets.shape[0]):
        #     f.write(
        #         "{:s} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
        #             index0,
        #             index1,
        #             dets[k, -1],
        #             dets[k, 0] + 1,
        #             dets[k, 1] + 1,
        #             dets[k, 2] + 1,
        #             dets[k, 3] + 1,
        #         )
        #     )

    def _write_voc_results_file_ori(self, all_boxes):
        """
        :param all_boxes:
        :return:
        """
        for cls_ind, cls in enumerate(C5_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index0 = index[0]
                    index1 = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index0,
                                index1,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    # TO DO.
    def _do_python_eval(self, output_dir="output", iou=0.5):
        # rootpath = os.path.join(self.root, "VOC" + self._year)
        # name = self.image_set[0][1]
        # annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        # imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        # cachedir = os.path.join(
        #     self.root, "annotations_cache", "VOC" + self._year, name
        # )
        # if not os.path.exists(cachedir):
        #     os.makedirs(cachedir)
        # aps = []
        # # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        # print("Eval IoU : {:.2f}".format(iou))
        # if output_dir is not None and not os.path.isdir(output_dir):
        #     os.mkdir(output_dir)
        cachedir = "/users/duanyou/c5/experiments/YOLOX-main-duan/YOLOX_outputs/yolox-tiny"
        annopath = []
        imagesetfile = []
        detall = [['name', 'obj_type', 'score', 0, 0, 0, 0]]
        f1 = open(self.root, 'r')
        lines = f1.readlines()
        for line in lines:
            image_name = line.replace('\n', '')
            xml_name = image_name.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
            annopath.append(xml_name)
            imagesetfile.append(image_name)
        for i, cls in enumerate(C5_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            print('filename: ', filename)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=False,
            )
            # detall = np.delete(detall, 0, axis = 0)
            # det_objtype = [obj for obj in detall if obj[1] == cls]
            # rec, prec, ap = voc_eval(det_objtype, annopath, imagesetfile, classname=cls, ovthresh=iou)
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            # if output_dir is not None:
            #     with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
            #         pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return 0  # np.mean(aps)
