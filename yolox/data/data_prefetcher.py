#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import torch
import torch.distributed as dist

from yolox.utils import synchronize


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speed-up your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        """
        :param loader:
        """
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        """
        :return:
        """
        try:  # return the items from dataset
            self.next_input, self.next_target, \
            self.next_q, self.next_k, self.next_n = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            # self.input_cuda()  # put input to cuda
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

            ## ----- TODO: put q, k, n to cuda
            self.next_q = self.next_q.cuda(non_blocking=True)
            self.next_k = self.next_k.cuda(non_blocking=True)
            self.next_n = self.next_n.cuda(non_blocking=True)

    def next(self):
        """
        :return:
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target

        ## ----- get q,k,n
        q = self.next_q
        k = self.next_k
        n = self.next_n

        if input is not None:
            self.record_stream(input)
            # input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())

        if q is not None:
            q.record_stream(torch.cuda.current_stream())
        if k is not None:
            k.record_stream(torch.cuda.current_stream())
        if n is not None:
            n.record_stream(torch.cuda.current_stream())

        # get input and target
        self.preload()

        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def random_resize(data_loader, exp, epoch, rank, is_distributed):
    tensor = torch.LongTensor(1).cuda()
    if is_distributed:
        synchronize()

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor.fill_(size)

    if is_distributed:
        synchronize()
        dist.broadcast(tensor, 0)

    input_size = data_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    return input_size
