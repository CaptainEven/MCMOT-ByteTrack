#!/usr/bin/env python3
# encoding=utf-8
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

import os
import os.path

import cv2
import numpy as np

from .datasets_wrapper import Dataset
from .voc_classes import C5_CLASSES


class MCMOTDataset(Dataset):
    def __init__(self,
                 data_dir,
                 img_size=(768, 448),
                 preproc=None):
        """
        :param data_dir:
        :param img_size:
        :param preproc:
        """
        super().__init__(img_size)

        ## ----- image size: (width, height)
        self.img_size = img_size

        ## ----- transforms
        self.preproc = preproc

        ## ----- object class names
        self._classes = C5_CLASSES

        ## ----- Get images root and labels root
        self.root = data_dir
        if not os.path.isdir(self.root):
            print("[Err]: invalid root.")
            exit(-1)

        self.img_root = self.root + "/JPEGImages"
        self.txt_root = self.root + "/labels_with_ids"
        if not (os.path.isdir(self.img_root) and os.path.isdir(self.txt_root)):
            print("[Err]: invalid img root or txt root!")
            exit(-1)

        self.img_dirs = [self.img_root + "/" + x for x in os.listdir(self.img_root)
                         if os.path.isdir(self.img_root + "/" + x)]
        self.txt_dirs = [self.txt_root + "/" + x for x in os.listdir(self.txt_root)
                         if os.path.isdir(self.txt_root + "/" + x)]

        assert len(self.img_dirs) == len(self.txt_dirs)

        self.img_paths = []
        self.txt_paths = []
        for img_dir in self.img_dirs:
            for img_name in os.listdir(img_dir):
                if img_name.endswith(".jpg"):
                    img_path = img_dir + "/" + img_name
                    txt_path = img_path.replace("JPEGImages", "labels_with_ids") \
                        .replace(".jpg", ".txt")
                    if os.path.isfile(img_path) and os.path.isfile(txt_path):
                        self.img_paths.append(img_path)
                        self.txt_paths.append(txt_path)
        print("Total {:d} samples.".format(len(self.img_paths)))

    def load_label(self, idx, img_info):
        """
        :param idx:
        :param img_info: img_height, img_width
        :return:
        """
        H, W = img_info
        label_path = self.txt_paths[idx]
        label = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                items = line.split(" ")
                items = list(map(lambda x: float(x), items))

                ## ----- parsing
                cls_id = items[0]
                tr_id = items[1] - 1  # track id should start with 0

                cx = items[2] * W
                cy = items[3] * H
                bw = items[4] * W
                bh = items[5] * H

                x1 = cx - bw * 0.5
                x2 = cx + bw * 0.5
                y1 = cy - bh * 0.5
                y2 = cy + bh * 0.5

                ## -----
                items = [x1, y1, x2, y2, cls_id, tr_id]
                label.append(items)
                target = np.array(label, dtype=np.float32)

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
        img_path = self.img_paths[idx]
        img_dir_path, img_name = os.path.split(img_path)
        img_dir_name = img_dir_path.split("/")[-1]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        img_info = (height, width)

        ## ----- load label
        target = self.load_label(idx, img_info)

        return img, target, img_info, (img_dir_name, img_name)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def __len__(self):
        """
        :return:
        """
        return len(self.img_paths)
