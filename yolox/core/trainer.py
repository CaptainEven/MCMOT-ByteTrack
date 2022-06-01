#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time

import torch
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.models.darknet_modules import load_darknet_weights
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp, opt):
        # init function only defines some basic attr,
        # other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.opt = opt

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = opt.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = opt.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # ---------- data/dataloader related attr
        self.data_type = torch.float16 if opt.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.dir_path = os.path.abspath(os.path.join(exp.output_dir,
                                                     opt.experiment_name))
        logger.info("Dir path: {:s}".format(self.dir_path))

        if self.rank == 0:
            os.makedirs(self.dir_path, exist_ok=True)

        setup_logger(self.dir_path,
                     distributed_rank=self.rank,
                     filename="train_log.txt",
                     mode="a", )

    def train(self):
        """
        :return:
        """
        self.before_train()

        try:
            self.train_in_epoch()
        except Exception as e:
            print(e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        """
        :return:
        """
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        """
        :return:
        """
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        """
        :return:
        """
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model.forward(inps, targets)
        loss = outputs["total_loss"]
        # loss = outputs["mtl_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(iter_time=iter_end_time - iter_start_time,
                          data_time=data_end_time - iter_start_time,
                          lr=lr,
                          **outputs, )

    def before_train(self):
        """
        The preparations before training
        :return:
        """
        logger.info("args: {}".format(self.opt))
        logger.info("exp value:\n{}".format(self.exp))

        ## ----- Set device
        torch.cuda.set_device(self.local_rank)

        ## ----- Get the network
        net = self.exp.get_model()

        if not self.opt.debug:
            logger.info("Model Summary: {}"
                        .format(get_model_info(net, self.exp.test_size)))
        net.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.opt.batch_size)

        # value of epoch will be set in `resume_train`
        ## ----- load from checkpoint
        net = self.resume_train(net)

        ## ---------- data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(batch_size=self.opt.batch_size,
                                                     is_distributed=self.is_distributed,
                                                     data_dir=self.opt.train_root,
                                                     no_aug=self.no_aug)

        logger.info("Init pre-fetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)

        # max_iter means iterations per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.opt.batch_size,
                                                      self.max_iter)
        if self.opt.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            net = DDP(net, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(net, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = net
        self.model.train()  # train mode

        self.evaluator = self.exp.get_evaluator(batch_size=self.opt.batch_size,
                                                is_distributed=self.is_distributed,
                                                data_dir=self.opt.val_root)

        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.dir_path)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        """
        :return:
        """
        logger.info("Training of experiment is done and the best AP is {:.2f}"
                    .format(self.best_ap * 100))

    def before_epoch(self):
        """
        :return:
        """
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")

            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        """
        :return:
        """
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest")

        if self.exp.eval_interval != 0:
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                all_reduce_norm(self.model)
                self.evaluate_and_save_model()

    def before_iter(self):
        """
        :return:
        """
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            ## ----- TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(self.epoch + 1,
                                                              self.max_epoch,
                                                              self.iter + 1,
                                                              self.max_iter)
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(["{}: {:.3f}".format(k, v.latest)
                                  for k, v in loss_meter.items()])

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(["{}: {:.3f}s"
                                 .format(k, v.avg) for k, v in time_meter.items()])

            logger.info("{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}"
                        .format(progress_str,
                                gpu_mem_usage(),
                                time_str,
                                loss_str,
                                self.meter["lr"].latest, )
                        + (", size: {:d}, {}".format(self.input_size[0], eta_str)))
            self.meter.clear_meters()

        ## ----- @even: save ckpt during an epoch
        if self.exp.save_ckpt_batch_interval != 0 \
                and (self.iter + 1) % self.exp.save_ckpt_batch_interval == 0:
            self.save_ckpt(ckpt_name="latest")

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(self.train_loader,
                                                     self.epoch,
                                                     self.rank,
                                                     self.is_distributed)

    @property
    def progress_in_iter(self):
        """
        :return:
        """
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, net):
        """
        :param net:
        :return:
        """
        if self.opt.resume:
            logger.info("resume training")
            if self.opt.ckpt is None:
                ckpt_path = os.path.join(self.dir_path, "latest" + "_ckpt.pth.tar")
            else:
                ckpt_path = self.opt.ckpt

            ckpt_path = os.path.abspath(ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)

            ## ---- resume the model/optimizer state dict
            net.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (self.opt.start_epoch - 1
                           if self.opt.start_epoch is not None
                           else ckpt["start_epoch"])
            self.start_epoch = start_epoch
            logger.info("loaded checkpoint '{}' (epoch {})"
                        .format(self.opt.resume, self.start_epoch))  # noqa
        else:
            if self.opt.ckpt is not None:
                ckpt_path = os.path.abspath(self.opt.ckpt)
                if not os.path.isfile(ckpt_path):
                    logger.error("invalid ckpt file path: {:s}"
                                 .format(ckpt_path))
                else:
                    logger.info("Loading ckpt {:s}...".format(ckpt_path))

                if self.opt.ckpt.endswith(".tar") \
                        or self.opt.ckpt.endswith(".pth"):
                    ckpt = torch.load(ckpt_path, map_location=self.device)["model"]
                    net = load_ckpt(net, ckpt)
                elif self.opt.ckpt.endswith(".weights"):
                    if hasattr(net, "module_list"):
                        load_darknet_weights(net, ckpt_path, self.opt.cutoff)
                    elif hasattr(net, "backbone") and \
                            hasattr(net.backbone, "module_list"):
                        load_darknet_weights(net.backbone, ckpt_path, self.opt.cutoff)
                logger.info("{:s} loaded!".format(ckpt_path))

            self.start_epoch = 0

        return net

    def evaluate_and_save_model(self):
        """
        :return:
        """
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        ap50_95, ap50, summary = self.exp.eval(evalmodel, self.evaluator, self.is_distributed)
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        # self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        """
        :param ckpt_name:
        :param update_best_ckpt:
        :return:
        """
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model

            logger.info("Save weights to {}".format(self.dir_path))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(ckpt_state, update_best_ckpt, self.dir_path, ckpt_name, )


class Trainer_det:  # line 115. loss = outputs["total_loss"]
    def __init__(self, exp, opt):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.opt = opt

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = opt.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = opt.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # ---------- data/dataloader related attr
        self.data_type = torch.float16 if opt.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, opt.experiment_name)
        print("file path: ", self.file_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        """
        :return:
        """
        self.before_train()

        try:
            self.train_in_epoch()
        except Exception as e:
            print(e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        """
        :return:
        """
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        """
        :return:
        """
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        """
        :return:
        """
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model.forward(inps, targets)
        loss = outputs["total_loss"]
        # loss = outputs["mtl_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(iter_time=iter_end_time - iter_start_time,
                          data_time=data_end_time - iter_start_time,
                          lr=lr,
                          **outputs, )

    def before_train(self):
        """
        The preparations before training
        :return:
        """
        logger.info("args: {}".format(self.opt))
        logger.info("exp value:\n{}".format(self.exp))

        ## ----- model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        if not self.opt.debug:
            logger.info("Model Summary: {}"
                        .format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.opt.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(batch_size=self.opt.batch_size,
                                                     is_distributed=self.is_distributed,
                                                     data_dir=self.opt.train_root,
                                                     no_aug=self.no_aug)

        logger.info("Init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.opt.batch_size,
                                                      self.max_iter)
        if self.opt.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()  # train mode

        self.evaluator = self.exp.get_evaluator(batch_size=self.opt.batch_size,
                                                is_distributed=self.is_distributed,
                                                data_dir=self.opt.val_root)

        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        """
        :return:
        """
        logger.info("Training of experiment is done and the best AP is {:.2f}"
                    .format(self.best_ap * 100))

    def before_epoch(self):
        """
        :return:
        """
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")

            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        """
        :return:
        """
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest")

        if self.exp.eval_interval != 0:
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                all_reduce_norm(self.model)
                self.evaluate_and_save_model()

    def before_iter(self):
        """
        :return:
        """
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            ## ----- TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(self.epoch + 1,
                                                              self.max_epoch,
                                                              self.iter + 1,
                                                              self.max_iter)
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(["{}: {:.3f}".format(k, v.latest)
                                  for k, v in loss_meter.items()])

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info("{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}"
                        .format(progress_str,
                                gpu_mem_usage(),
                                time_str,
                                loss_str,
                                self.meter["lr"].latest, )
                        + (", size: {:d}, {}".format(self.input_size[0], eta_str)))
            self.meter.clear_meters()

        ## ----- @even: save ckpt during an epoch
        if self.exp.save_ckpt_batch_interval != 0 \
                and (self.iter + 1) % self.exp.save_ckpt_batch_interval == 0:
            self.save_ckpt(ckpt_name="latest")

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(self.train_loader,
                                                     self.epoch,
                                                     self.rank,
                                                     self.is_distributed)

    @property
    def progress_in_iter(self):
        """
        :return:
        """
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        """
        :param model:
        :return:
        """
        if self.opt.resume:
            logger.info("resume training")
            if self.opt.ckpt is None:
                ckpt_path = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
            else:
                ckpt_path = self.opt.ckpt

            ckpt = torch.load(ckpt_path, map_location=self.device)

            ## ---- resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (self.opt.start_epoch - 1
                           if self.opt.start_epoch is not None
                           else ckpt["start_epoch"])
            self.start_epoch = start_epoch
            logger.info("loaded checkpoint '{}' (epoch {})"
                        .format(self.opt.resume, self.start_epoch))  # noqa
        else:
            if self.opt.ckpt is not None:
                logger.info("loading checkpoint for fine tuning...")
                ckpt_path = self.opt.ckpt
                ckpt = torch.load(ckpt_path, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
                print("{:s} loaded!".format(ckpt_path))
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        """
        :return:
        """
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        ap50_95, ap50, summary = self.exp.eval(evalmodel, self.evaluator, self.is_distributed)
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        # self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        """
        :param ckpt_name:
        :param update_best_ckpt:
        :return:
        """
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model

            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(ckpt_state, update_best_ckpt, self.file_name, ckpt_name, )
