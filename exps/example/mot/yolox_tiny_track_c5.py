# encoding=utf-8
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self, n_workers=8, debug=False, reid=True):
        """
        YOLOX Tiny
        :param n_workers:
        :param debug:
        :param reid: Do reid or not
        """
        super(Exp, self).__init__()

        ## ----- number of classes
        self.num_classes = 5  # 1, 5, 80
        ## -----

        ## ----- net size?
        self.depth = 0.33
        self.width = 0.5

        # ## ----- Define file list path(imgs and txts(labels) path)
        # self.train_f_list_path = "/users/duanyou/c5/data_all/train_all.txt"
        # self.test_f_list_path = "/users/duanyou/c5/data_all/test5000.txt"
        # ## -----

        ## ----- Define max id dict
        self.max_id_dict_f_path = "/mnt/diskb/even/dataset/MCMOT/max_id_dict.npz"
        if os.path.isfile(self.max_id_dict_f_path):
            load_dict = np.load(self.max_id_dict_f_path, allow_pickle=True)
            self.max_id_dict = load_dict['max_id_dict'][()]
            print(self.max_id_dict)

        if debug:
            self.data_num_workers = 0
        else:
            self.data_num_workers = n_workers

        self.depth = 0.33
        self.width = 0.375
        self.scale = (0.5, 1.5)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.val_ann = "train.json"
        self.input_size = (448, 768)  # (608, 1088)
        self.test_size = (448, 768)  # (608, 1088)
        self.random_size = (12, 26)
        self.max_epoch = 100
        self.print_interval = 30
        self.save_ckpt_batch_interval = 100  # save ckpt every 100 iters
        self.eval_interval = 0  # 5
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        self.reid = reid

    def get_model(self):
        """
        :return:
        """
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXReID, YOLOXHead, YOLOXHeadReID

        def init_yolo(M):
            """
            :param M:
            :return:
            """
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]

            ## ----- backbone and head
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            if self.reid:
                head = YOLOXHeadReID(self.num_classes,
                                     self.width,
                                     in_channels=in_channels,
                                     reid=True,
                                     max_id_dict=self.max_id_dict,
                                     net_size=(448, 768))
            else:
                head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)

            ## ----- combine backbone abd head
            if self.reid:
                self.model = YOLOXReID(backbone, head)
            else:
                self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model

    def get_data_loader(self,
                        batch_size,
                        is_distributed,
                        data_dir=None,
                        no_aug=False):
        """
        :param batch_size:
        :param is_distributed:
        :param data_dir:
        :param no_aug:
        :return:
        """
        from yolox.data import (
            MCMOTDataset,
            TrainTransformTrack,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
        )

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mix_det")

        dataset = MCMOTDataset(data_dir=data_dir,
                               img_size=(768, 448),
                               preproc=TrainTransformTrack(
                                   rgb_means=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225),
                                   max_labels=50,
                               ), )
        self.max_id_dict = dataset.max_id_dict

        # dataset = MosaicDetection(
        #     dataset,
        #     mosaic=not no_aug,
        #     img_size=self.input_size,
        #     preproc=TrainTransform(
        #         rgb_means=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225),
        #         max_labels=1000,
        #     ),
        #     degrees=self.degrees,
        #     translate=self.translate,
        #     scale=self.scale,
        #     shear=self.shear,
        #     perspective=self.perspective,
        #     enable_mixup=self.enable_mixup,
        # )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self,
                        batch_size,
                        is_distributed,
                        data_dir=None,
                        testdev=False):
        """
        :param batch_size:
        :param is_distributed:
        :param data_dir:
        :param testdev:
        :return:
        """
        from yolox.data import MCMOTDataset, ValTransform

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")

        valdataset = MCMOTDataset(data_dir=data_dir,
                                  img_size=(768, 448),
                                  preproc=ValTransform(
                                      rgb_means=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225),
                                  ), )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self,
                      batch_size,
                      is_distributed,
                      data_dir=None,
                      testdev=False):
        """
        :param batch_size:
        :param is_distributed:
        :param data_dir:
        :param testdev:
        :return:
        """
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, data_dir, testdev=testdev)
        evaluator = COCOEvaluator(dataloader=val_loader,
                                  img_size=self.test_size,
                                  confthre=self.test_conf,
                                  nmsthre=self.nmsthre,
                                  num_classes=self.num_classes,
                                  testdev=testdev, )

        return evaluator