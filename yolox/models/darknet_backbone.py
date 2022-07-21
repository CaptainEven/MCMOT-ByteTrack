# encoding=utf-8

import os

import torch.nn as nn
from loguru import logger

from yolox.utils.myutils import parse_darknet_cfg
from .darknet_modules import build_modules


class DarknetBackbone(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self,
                 cfg_path,
                 net_size=(768, 448),
                 in_chans=3,
                 out_inds=[],
                 init_weights=False,
                 use_momentum=True):
        """
        Darknet(cfg file defined) based backbone
        :param cfg_path:
        :param net_size: w, h
        :param in_chans:
        :param out_inds: output layer inds
        :param init_weights:
        :param use_momentum:
        """
        super().__init__()

        if not os.path.isfile(cfg_path):
            logger.error("Invalid cfg file path: {:s}, exit now!".format(cfg_path))
            exit(-1)

        self.out_inds = out_inds
        assert len(out_inds) == 3
        self.id0, self.id1, self.id2 = self.out_inds
        logger.info("id0: {:d}, id1: {:d}, id2: {:d}"
                    .format(self.id0, self.id1, self.id2))

        ## ----- build the network
        self.module_defs = parse_darknet_cfg(cfg_path)
        logger.info("Network config file parsed.")

        self.module_list, self.routs = build_modules(self.module_defs,
                                                     net_size,
                                                     cfg_path,
                                                     in_chans=in_chans,
                                                     use_momentum=use_momentum)
        logger.info("Network modules built.")

        if init_weights:
            self.init_weights()
            logger.info("Network weights initialized.")
        else:
            logger.info("Network weights not initialized.")

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

    def forward(self, x):
        """
        :param x:
        """
        ## ----- out: final output, outs: outputs of each layer
        out, layer_outs = self.forward_once(x)

        ## ----- build feature maps of 3 scales: 1/8, 1/16, 1/32
        self.fpn_outs = (layer_outs[self.id0],
                         layer_outs[self.id1],
                         layer_outs[self.id2])

        shallow_layer = layer_outs[13]

        # return self.fpn_outs
        return self.fpn_outs, shallow_layer
