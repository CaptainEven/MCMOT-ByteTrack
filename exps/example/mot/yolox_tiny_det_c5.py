# encoding=utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self, n_workers=4, debug=False):
        """
        YOLOX Tiny
        :param n_workers:
        :param debug:
        """
        super(Exp, self).__init__()

        ## ----- number of classes
        self.num_classes = 5  # 1, 5, 80
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
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.random_size = (12, 26)
        self.max_epoch = 100
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

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
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mix_det")

        dataset = MOTDataset(
            data_dir=data_dir,
            json_file=self.train_ann,
            name=name,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

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
        from yolox.data import MOTDataset, ValTransform

        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        valdataset = MOTDataset(
            data_dir=data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name=name,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
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

        val_loader = self.get_eval_loader(batch_size, is_distributed, data_dir, name, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

        return evaluator
