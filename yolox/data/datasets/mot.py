# encoding=utf-8

import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from .datasets_wrapper import Dataset
from ..dataloading import get_yolox_datadir


class MOTDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self,
                 data_dir=None,
                 json_file="train_half.json",
                 name="",
                 img_size=(608, 1088),
                 preproc=None):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
        @:param data_dir (str): dataset root directory
        @:param json_file (str): COCO json file name
        @:param name (str): COCO data name (e.g. 'train2017' or 'val2017')
        @:param img_size (int): target image size after pre-processing
        @:param preproc: data augmentation strategy
        """
        super().__init__(img_size)

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        print("\ndata_dir: {:s}\n".format(data_dir))

        self.data_dir = data_dir
        self.json_file_name = json_file

        json_f_path = os.path.join(self.data_dir, "annotations", self.json_file_name)
        if not os.path.isfile(json_f_path):
            print("[Err]: invalid json file path: {:s}.".format(json_f_path))
            exit(-1)

        self.coco = COCO(json_f_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        """
        :return:
        """
        return len(self.ids)

    def _load_coco_annotations(self):
        """
        :return:
        """
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        """
        :param id_:
        :return:
        """
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        """
        :param index:
        :return:
        """
        return self.annotations[index][0]

    def pull_item(self, idx):
        """
        :param idx:
        :return:
        """
        id_ = self.ids[idx]

        res, img_info, file_name = self.annotations[idx]

        # load image and preprocess
        img_path = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_path)
        assert img is not None

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        """
        One image/label pair for the given index is picked up and pre-processed.
        :param idx: idx (int): data index
        :return: img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(idx)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
