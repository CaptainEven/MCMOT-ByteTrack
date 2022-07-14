# encoding=utf-8
import os

import torch
import torch.distributed as dist
from loguru import logger

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


## ----- Decoupled backbone and head for C5(5 classes) detection
class Exp(MyExp):
    def __init__(self, n_workers=4, debug=False):
        """
        YOLOX Tiny
        :param n_workers:
        :param debug:
        """
        super(Exp, self).__init__()

        ## ----- number of object classes
        self.model = None
        self.num_classes = 5
        ## -----

        ## ----- net scale
        self.depth = 0.33
        self.width = 0.5

        ## ----- Define file list path(imgs and txts(labels) path)
        self.train_f_list_path = "/mnt/diskb/even/ByteTrack/datasets/train_all.txt"
        self.test_f_list_path = "/mnt/diskb/even/ByteTrack/datasets/test3000.txt"
        self.train_f_list_path = os.path.abspath(self.train_f_list_path)
        self.test_f_list_path = os.path.abspath(self.test_f_list_path)
        if not os.path.isfile(self.train_f_list_path):
            logger.error("invalid train file list path: {:s}"
                         .format(self.train_f_list_path))
            exit(-1)
        if not os.path.isfile(self.test_f_list_path):
            logger.error("invalid test file list path: {:s}"
                         .format(self.test_f_list_path))
            exit(-1)
        ## -----

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
        self.max_epoch = 10  # 100
        self.print_interval = 1  # 10
        self.eval_interval = 0  # 100
        self.save_ckpt_batch_interval = 30  # 300
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        self.n_classes = 5
        self.cfg_file_path = "../cfg/yolox_darknet_tiny_bb46.cfg"
        self.max_labels = 30

    def get_model(self):
        """
        :return:
        """
        from yolox.models.darknet_backbone import DarknetBackbone
        from yolox.models.darknet_head import DarknetHeadSSL
        from yolox.models.yolox_dark import YOLOXDarkSSL

        if getattr(self, "model", None) is None:
            self.cfg_file_path = os.path.abspath(self.cfg_file_path)
            if not os.path.isfile(self.cfg_file_path):
                logger.error("invalid cfg file path: {:s}, exit now!"
                             .format(self.cfg_file_path))
                exit(-1)
            logger.info("Cfg file path: {:s}.".format(self.cfg_file_path))

            backbone = DarknetBackbone(cfg_path=self.cfg_file_path,
                                       net_size=(768, 448),
                                       in_chans=3,
                                       out_inds=[20, 26, 45],
                                       init_weights=True,
                                       use_momentum=True)
            head = DarknetHeadSSL(num_classes=self.n_classes,
                                  width=self.width,
                                  strides=[8, 16, 32],
                                  in_channels=[256, 256, 512],
                                  act="lrelu",  # leaky relu
                                  depth_wise=False)  # 156 -> 96, 512 -> 192
            self.model = YOLOXDarkSSL(cfg_path=self.cfg_file_path,
                                      backbone=backbone,
                                      head=head,
                                      n_classes=self.n_classes)

        return self.model

    def get_data_loader(self,
                        batch_size,
                        is_distributed,
                        data_dir=None,
                        name="",
                        no_aug=False):
        """
        :param batch_size:
        :param is_distributed:
        :param data_dir:
        :param name:
        :param no_aug:
        :return:
        """
        from yolox.data import (
            VOCDetSSL,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
        )

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mix_det")

        dataset = VOCDetSSL(data_dir=data_dir,
                            f_list_path=self.train_f_list_path,
                            img_size=(768, 448),
                            preproc=TrainTransform(
                                rgb_means=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225),
                                max_labels=self.max_labels,
                            ),
                            max_patches=self.max_labels,
                            patch_size=(224, 224), )

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

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self,
                        batch_size,
                        is_distributed,
                        data_dir=None,
                        name="",
                        testdev=False):
        """
        :param batch_size:
        :param is_distributed:
        :param data_dir:
        :param name:
        :param testdev:
        :return:
        """
        from yolox.data import VOCDetection, ValTransform

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")

        # valdataset = MOTDataset(
        #     data_dir=data_dir,
        #     json_file=self.val_ann,
        #     img_size=self.test_size,
        #     name=name,
        #     preproc=ValTransform(
        #         rgb_means=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225),
        #     ),
        # )

        valdataset = VOCDetection(data_dir=data_dir,
                                  f_list_path=self.test_f_list_path,
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
                      name="",
                      testdev=False):
        """
        :param batch_size:
        :param is_distributed:
        :param data_dir:
        :param name:
        :param testdev:
        :return:
        """
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size,
                                          is_distributed,
                                          data_dir,
                                          name,
                                          testdev=testdev)
        evaluator = COCOEvaluator(dataloader=val_loader,
                                  img_size=self.test_size,
                                  confthre=self.test_conf,
                                  nmsthre=self.nmsthre,
                                  num_classes=self.num_classes,
                                  testdev=testdev, )

        return evaluator
