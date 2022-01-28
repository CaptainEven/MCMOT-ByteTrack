#!/usr/bin/env python3
# encoding=utf-8
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

import os
import os.path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from loguru import logger
from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
from .voc_classes import C5_CLASSES


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
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        ## ----- load label
        target = self.load_anno(idx)
        # print(target.shape)

        img_info = (height, width)

        return img, target, img_info, idx

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
