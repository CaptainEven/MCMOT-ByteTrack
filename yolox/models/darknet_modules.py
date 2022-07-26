# encoding=utf-8

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# import random
# from torch.nn import init
# from torchvision.models import resnet


def load_darknet_weights(model, weights, cutoff=0):
    """
    :param model:
    :param weights:
    :param cutoff:
    :return:
    """
    print("Loading weights from {:s}...".format(os.path.abspath(weights)))
    print('Cutoff: ', cutoff)

    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        model.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        model.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    # for i, (mod_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
    for i, (mod_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        if cutoff != 0 and i > cutoff:
            break

        if mod_def['type'] == 'convolutional' or mod_def[
            'type'] == 'deconvolutional':  # how to load 'deconvolutional' layer
            conv = module[0]
            if mod_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases

                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb

                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb

                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb

                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)

                conv.bias.data.copy_(conv_b)
                ptr += nb

            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_darknet_weights(net, path='model.weights', cutoff=-1):
    """
    :param net:
    :param path:
    :param cutoff:
    :return:
    """
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        multi_gpu = type(net) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        print("[Info]: multi-gpu: ", multi_gpu)
        if multi_gpu:
            net.module.version.tofile(f)  # (int32) version Info: major, minor, revision
            net.module.seen.tofile(f)  # (int64) number of images seen during training

            # Iterate through layers
            # for i, (m_def, module) in enumerate(zip(self.module.module_defs[:cutoff], self.module.module_list[:cutoff])):
            for i, (m_def, module) in enumerate(zip(net.module.module_defs, net.module.module_list)):

                if m_def['type'] == 'convolutional' or m_def['type'] == 'deconvolutional':
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if m_def['batch_normalize']:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.data.cpu().numpy().tofile(f)
                        bn_layer.running_var.data.cpu().numpy().tofile(f)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)
        else:
            net.version.tofile(f)  # (int32) version Info: major, minor, revision
            net.seen.tofile(f)  # (int64) number of images seen during training

            # Iterate through layers
            # for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            for i, (m_def, module) in enumerate(zip(net.module_defs, net.module_list)):
                if m_def['type'] == 'convolutional' or m_def['type'] == 'deconvolutional':
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if m_def['batch_normalize']:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.data.cpu().numpy().tofile(f)
                        bn_layer.running_var.data.cpu().numpy().tofile(f)
                    # Save conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    # Save conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_idx, layers, stride):
        """
        :param anchors:
        :param nc:
        :param img_size:
        :param yolo_idx:
        :param layers:
        :param stride:
        """
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.index = yolo_idx  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y grid points
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        """
        :param ng:
        :param device:
        :return:
        """
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, pred, out):
        """
        :param pred:
        :param out:
        :return:
        """
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            pred = out[self.layers[i]]
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), pred.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(pred[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            pred = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    pred += w[:, j:j + 1] * \
                            F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear',
                                          align_corners=False)

        else:
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids(ng=(nx, ny), device=pred.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, na, ny, nx, no(classes + xywh))
        pred = pred.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return pred

        else:  # inference
            io = pred.clone()  # inference output

            # ---------- process pred to io
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh YOLO method
            io[..., :4] *= self.stride  # map from YOLO layer's scale to net input's scale
            torch.sigmoid_(io[..., 4:])  # sigmoid for confidence score and cls pred

            # gathered pred output: io: view [1, 3, 13, 13, 85] as [1, 507, 85]
            io = io.view(bs, -1, self.no)

            # return io, pred
            return io, pred


## route_lhalf
class FeatureConcat_l(nn.Module):
    def __init__(self, layers):
        """
        :param layers:
        """
        super(FeatureConcat_l, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        return torch.cat([outputs[i][:, :outputs[i].shape[1] // 2, :, :] for i in self.layers], 1) if self.multiple else \
            outputs[self.layers[0]][:, :outputs[self.layers[0]].shape[1] // 2, :, :]


## shortcut
class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        """
        :param layers:
        :param weight:
        """
        super(WeightedFeatureFusion, self).__init__()

        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = torch.tensor(x.shape[1])  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = torch.tensor(a.shape[1])  # feature channels

            # Adjust channels
            if torch.equal(nx, na):  # same shape
                x = x + a
            elif torch.gt(nx, na):  # nx > na, slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


## route layer
class FeatureConcat(nn.Module):
    def __init__(self, layers):
        """
        :param layers:
        """
        super(FeatureConcat, self).__init__()
        self.layer_inds = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        return torch.cat([outputs[i] for i in self.layer_inds], 1) \
            if self.multiple else outputs[self.layer_inds[0]]


class GAP(nn.Module):
    def __init__(self, dimension=1):
        """
        :param dimension:
        """
        super(GAP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(dimension)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return self.avg_pool(x)


class AttentionGAP(nn.Module):
    def __init__(self, in_channels):
        """
        parameterized self-attention GAP
        @param in_channels:
        """
        super(AttentionGAP, self).__init__()
        self.gap = GAP()

        # self.conv = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=1,
        #                       kernel_size=3,
        #                       stride=1,
        #                       padding=1)

        self.conv = nn.Sequential(*[nn.Conv2d(in_channels=in_channels,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(in_channels=256,
                                              out_channels=1,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)])

    def forward(self, x):
        """
        @param x:
        """
        attention_map = torch.sigmoid(self.conv(x))  # n×c×h×w
        out = self.gap(attention_map * x) / self.gap(attention_map)
        out = torch.squeeze(out)
        return out


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        @param in_channels:
        @param out_channels:
        """
        super(UpSampleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(self.out_channels)])

    def forward(self, x):
        """
        @param x:
        """
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # up-sample
        x = self.conv(x)  # conv
        return x


class UpSampleFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        UpSample and concatenate
        @param in_channels:
        @param out_channels:
        """
        super(UpSampleFuse, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(self.out_channels)])

    def forward(self, x, fuse_layer):
        """
        @param x:
        """
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # up-sample
        x = torch.cat([x, fuse_layer], dim=1)  # fuse
        x = self.conv(x)  # conv
        return x

# weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
class ScaleChannel(nn.Module):
    def __init__(self, layers):
        """
        :param layers:
        """
        super(ScaleChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a
        # return torch.mul(a, x)


# SAM layer: ScaleSpatial
# weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
class SAM(nn.Module):
    def __init__(self, layers):
        super(SAM, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):  # using x as point-wise spacial attention[0, 1]
        a = outputs[self.layers[0]]  # using a as input feature
        return x * a  # point-wise multiplication


class MixDeConv2d(nn.Module):  # MixDeConv: Mixed Depthwise DeConvolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self,
                 in_ch,
                 out_ch,
                 k=(3, 5, 7),
                 stride=1,
                 dilation=1,
                 bias=True,
                 method='equal_params'):
        """
        :param in_ch:
        :param out_ch:
        :param k:
        :param stride:
        :param dilation:
        :param bias:
        :param method:
        """
        super(MixDeConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.ConvTranspose2d(in_channels=in_ch,
                                                   out_channels=ch[g],
                                                   kernel_size=k[g],
                                                   stride=stride,
                                                   padding=k[g] // 2,  # 'same' pad
                                                   dilation=dilation,
                                                   bias=bias) for g in range(groups)])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return torch.cat([m(x) for m in self.m], 1)


# Dropout layer
class Dropout(nn.Module):
    def __init__(self, prob):
        """
        :param prob:
        """
        super(Dropout, self).__init__()
        self.prob = float(prob)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return F.dropout(x, p=self.prob)


class RouteGroup(nn.Module):
    def __init__(self, layers, groups, group_id):
        """
        :param layers:
        :param groups:
        :param group_id:
        """
        super(RouteGroup, self).__init__()
        self.layers = layers
        self.multi = len(layers) > 1
        self.groups = groups
        self.group_id = group_id

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        if self.multi:
            outs = []
            for layer in self.layers:
                out = torch.chunk(outputs[layer], self.groups, dim=1)
                outs.append(out[self.group_id])
            return torch.cat(outs, dim=1)
        else:
            out = torch.chunk(outputs[self.layers[0]], self.groups, dim=1)
            return out[self.group_id]


# Parse cfg file, create every layer
def build_modules(module_defs, img_size, cfg, in_chans=3, use_momentum=True):
    """
    :param module_defs:
    :param img_size:
    :param cfg:
    :param in_chans:
    :param use_momentum:
    :return:
    """
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)

    if in_chans == 3:  # input 3 channels: RGB(or BGR)
        output_filters = [3]
    elif in_chans == 1:  # input 1 channels: gray
        output_filters = [1]

    # define modules to register
    module_list = nn.ModuleList()

    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mod_def in enumerate(module_defs):
        modules = nn.Sequential()

        if mod_def['type'] == 'convolutional':
            bn = mod_def['batch_normalize']
            filters = mod_def['filters']
            k = mod_def['size']  # kernel size
            stride = mod_def['stride'] if 'stride' in mod_def else (mod_def['stride_y'], mod_def['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d',
                                   nn.Conv2d(in_channels=output_filters[-1],
                                             out_channels=filters,
                                             kernel_size=k,
                                             stride=stride,
                                             padding=k // 2 if 'pad' in mod_def else 0,
                                             groups=mod_def['groups'] if 'groups' in mod_def else 1,
                                             bias=not bn))

            else:  # multiple-size conv
                modules.add_module('MixConv2d',
                                   MixConv2d(in_ch=output_filters[-1],
                                             out_ch=filters,
                                             k=k,
                                             stride=stride,
                                             bias=not bn))

            if bn:
                if use_momentum:
                    modules.add_module('BatchNorm2d',
                                       nn.BatchNorm2d(filters, momentum=0.1, eps=1E-5))
                else:
                    modules.add_module('BatchNorm2d',
                                       nn.BatchNorm2d(filters, momentum=0.00, eps=1E-5))

                # @even: add BN to route too.
                routs.append(i)
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mod_def['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mod_def['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
            elif mod_def['activation'] == 'logistic':  # Add logistic activation support
                modules.add_module('activation', nn.Sigmoid())
            elif mod_def['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mod_def['activation'] == 'mish':
                modules.add_module('activation', Mish())

        # To parse deconvolution for learnable up-sampling
        elif mod_def['type'] == 'deconvolutional':
            bn = mod_def['batch_normalize']
            filters = mod_def['filters']
            k = mod_def['size']  # kernel size
            stride = mod_def['stride'] if 'stride' in mod_def else (mod_def['stride_y'], mod_def['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('DeConv2d', nn.ConvTranspose2d(in_channels=output_filters[-1],
                                                                  out_channels=filters,
                                                                  kernel_size=k,
                                                                  stride=stride,
                                                                  padding=k // 2 if mod_def['pad'] else 0,
                                                                  groups=mod_def[
                                                                      'groups'] if 'groups' in mod_def else 1,
                                                                  bias=not bn))
            else:  # multiple-size conv
                modules.add_module('MixDeConv2d', MixDeConv2d(in_ch=output_filters[-1],
                                                              out_ch=filters,
                                                              k=k,
                                                              stride=stride,
                                                              bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.00, eps=1E-5))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mod_def['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mod_def['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
            elif mod_def['activation'] == 'logistic':  # Add logistic activation support
                modules.add_module('activation', nn.Sigmoid())
            elif mod_def['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mod_def['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mod_def['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-5)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mod_def['type'] == 'maxpool':
            k = mod_def['size']  # kernel size
            stride = mod_def['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        # Add support for global average pooling layer
        elif mod_def['type'] == 'avgpool':
            modules = GAP()  # GlobalAvgPool()

        # Add support for local average pooling layer
        elif mod_def['type'] == 'local_avgpool':
            k = mod_def['size']  # kernel size
            stride = mod_def['stride']
            local_avgpool = nn.AvgPool2d(kernel_size=k, stride=stride, count_include_pad=False)
            modules = local_avgpool

        # Add support for dropout layer
        elif mod_def['type'] == 'dropout':
            prob = mod_def['probability']
            modules = Dropout(prob=prob)

        # Add support for scale channels
        elif mod_def['type'] == 'scale_channels':
            layers = mod_def['from']
            filters = output_filters[-1]  #
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleChannel(layers=layers)  # ScaleChannels

        # Add support for SAM: point-wise attention module
        elif mod_def['type'] == 'sam':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mod_def['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            routs.extend([i])  # using sam as feature vector output
            modules = SAM(layers=layers)

        elif mod_def['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=mod_def['stride'])

        # Add support for group route
        elif mod_def['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mod_def['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])

            if 'groups' in mod_def:
                groups = mod_def['groups']
                group_id = mod_def['group_id']
                modules = RouteGroup(layers, groups, group_id)
                filters //= groups
            else:
                modules = FeatureConcat(layers=layers)

        elif mod_def['type'] == 'route_lhalf':  # nn.Sequential() placeholder for 'route' layer
            layers = mod_def['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers]) // 2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mod_def['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mod_def['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])

            # modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

            # ----- to combine a shortcut layer and an activation layer to one layer
            modules.add_module('WeightedFeatureFusion',
                               WeightedFeatureFusion(layers=layers, weight='weights_type' in mod_def))

            # ----- add activation layer after a shortcut layer
            if mod_def['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        elif mod_def['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mod_def['type'] == 'yolo':  # do not parse yolo layer here
            yolo_index += 1
            stride = [8, 16, 32]  # P5, P4, P3 strides
            if any(x in cfg for x in ['tiny',
                                      'mobile', 'Mobile',
                                      'enet', 'Enet']):
                stride = [32, 16, 8]  # stride order reversed

            layers = mod_def['from'] if 'from' in mod_def else []
            modules = YOLOLayer(anchors=mod_def['anchors'][mod_def['mask']],  # anchor list
                                nc=mod_def['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_idx=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mod_def else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3, 85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mod_def['type'])

        # ---------- Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
        # ----------

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True

    return module_list, routs_binary
