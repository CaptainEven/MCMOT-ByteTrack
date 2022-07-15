#!/usr/bin/env python3
# encoding=utf-8
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.
The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import random

import cv2
import math
import numpy as np
import torchvision.transforms as transforms
from loguru import logger

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """
    :param img:
    :param hgain:
    :param sgain:
    :param vgain:
    :return:
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    """
    :param box1:
    :param box2:
    :param wh_thr:
    :param ar_thr:
    :param area_thr:
    :return:
    """
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates


def random_perspective(img,
                       targets=(),
                       degrees=10,
                       translate=0.1,
                       scale=0.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0), ):
    """
    :param img:
    :param targets:
    :param degrees:
    :param translate:
    :param scale:
    :param shear:
    :param perspective:
    :param border:
    :return:
    """
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        # xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        # xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

        targets = targets[targets[:, 0] < width]
        targets = targets[targets[:, 2] > 0]
        targets = targets[targets[:, 1] < height]
        targets = targets[targets[:, 3] > 0]

    return img, targets


def random_distort(image):
    """
    :param image:
    :return:
    """

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_mirror(image, boxes):
    """
    :param image:
    :param boxes:
    :return:
    """
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(image, net_size, mean, std, swap=(2, 0, 1)):
    """
    :param image:
    :param net_size: (H, W)
    :param mean:
    :param std:
    :param swap:
    :return:
    """
    if len(image.shape) == 3:
        padded_img = np.ones((net_size[0], net_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(net_size) * 114.0

    img = np.array(image)

    ## ----- Resize
    r = min(net_size[0] / img.shape[0], net_size[1] / img.shape[1])
    resized_img = cv2.resize(img,
                             (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ## ----- Padding
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    ## ----- BGR to RGB
    padded_img = padded_img[:, :, ::-1]

    ## ----- Normalize to [0, 1]
    padded_img /= 255.0

    ## ----- Standardization
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std

    ## ----- HWC ——> CHW
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r



## TODO: random blur kernel filtering
####################
# isotropic gaussian kernels, identical to 'fspecial('gaussian',hsize,sigma)' in matlab
####################

def isotropic_gaussian_kernel_matlab(l, sigma, tensor=False):
    """
    :param l:
    :param sigma:
    :param tensor:
    :return:
    """
    center = [(l - 1.0) / 2.0, (l - 1.0) / 2.0]
    [x, y] = np.meshgrid(np.arange(-center[1], center[1] + 1), np.arange(-center[0], center[0] + 1))
    arg = -(x * x + y * y) / (2 * sigma * sigma)
    k = np.exp(arg)

    k[k < scipy.finfo(float).eps * k.max()] = 0
    sum_k = k.sum()
    if sum_k != 0:
        k = k / sum_k

    return torch.FloatTensor(k) if tensor else k

def random_isotropic_gaussian_kernel(l=21,
                                     sig_min=0.2,
                                     sig_max=4.0,
                                     scale=3,
                                     tensor=False):
    """
    :param l:
    :param sig_min:
    :param sig_max:
    :param scale:
    :param tensor:
    :return:
    """
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel_matlab(l, x, tensor=tensor)
    return k, np.array([x, x, 0])


####################
# random/stable ani/isotropic gaussian kernel batch generation
####################

####################
# anisotropic gaussian kernels, identical to 'mvnpdf(X,mu,sigma)' in matlab
# due to /np.sqrt((2*np.pi)**2 * sig1*sig2), `sig1=sig2=8` != `sigma=8` in matlab
# rotation matrix [[cos, -sin],[sin, cos]]
####################

def anisotropic_gaussian_kernel_matlab(l,
                                       sig1,
                                       sig2,
                                       theta,
                                       tensor=False):
    """
    :param l:
    :param sig1:
    :param sig2:
    :param theta:
    :param tensor:
    :return:
    """
    # mean = [0, 0]
    # v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    # V = np.array([[v[0], v[1]], [v[1], -v[0]]]) # [[cos, sin], [sin, -cos]]
    # D = np.array([[sig1, 0], [0, sig2]])
    # cov = np.dot(np.dot(V, D), V) # VD(V^-1), V=V^-1

    cov11 = sig1 * np.cos(theta) ** 2 + sig2 * np.sin(theta) ** 2
    cov22 = sig1 * np.sin(theta) ** 2 + sig2 * np.cos(theta) ** 2
    cov21 = (sig1 - sig2) * np.cos(theta) * np.sin(theta)
    cov = np.array([[cov11, cov21], [cov21, cov22]])

    center = l / 2.0 - 0.5
    x, y = np.mgrid[-center:-center + l:1, -center:-center + l:1]
    pos = np.dstack((y, x))
    k = scipy.stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov)

    k[k < scipy.finfo(float).eps * k.max()] = 0

    # normalize the kernel
    sum_k = k.sum()
    if sum_k != 0:
        k = k / sum_k

    return torch.FloatTensor(k) if tensor else k

def random_anisotropic_gaussian_kernel(l=15,
                                       sig_min=0.2,
                                       sig_max=4.0,
                                       scale=3,
                                       tensor=False):
    """
    :param l:
    :param sig_min:
    :param sig_max:
    :param scale:
    :param tensor:
    :return:
    """
    sig1 = sig_min + (sig_max - sig_min) * np.random.rand()
    sig2 = sig_min + (sig1 - sig_min) * np.random.rand()
    theta = np.pi * np.random.rand()

    k = anisotropic_gaussian_kernel_matlab(l=l,
                                           sig1=sig1,
                                           sig2=sig2,
                                           theta=theta,
                                           tensor=tensor)

    return k, np.array([sig1, sig2, theta])

def random_gaussian_kernel(l=21,
                           sig_min=0.2,
                           sig_max=4.0,
                           rate_iso=1.0,
                           scale=3,
                           tensor=False):
    """
    :param l:
    :param sig_min:
    :param sig_max:
    :param rate_iso: iso gauss kernel rate
    :param scale:
    :param tensor:
    :return:
    """
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l,
                                                sig_min=sig_min,
                                                sig_max=sig_max,
                                                scale=scale,
                                                tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l,
                                                  sig_min=sig_min,
                                                  sig_max=sig_max,
                                                  scale=scale,
                                                  tensor=tensor)

class RandomBlur(object):
    """
    Random shaped kernel blurring
    """
    def __init__(self, iso_rate=0.2):
        self.iso_rate = iso_rate



from PIL import Image, ImageFilter


class GaussianBlur(object):
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    """
    def __init__(self, sigma=[0.1, 2.0]):
        """
        :param sigma:
        """
        self.sigma = sigma

    def __call__(self, x):
        """
        :param x: PIL Image
        """
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

import PIL
class RandomLightOrShadow(object):
    """
    Randomly add light or shadow
    """
    def __init__(self, base=200):
        """
        @param base:
        """
        self.base = base

    def __call__(self, x):
        """
        @param x: PIL Image
        """
        if isinstance(x, PIL.Image.Image):
            x = np.array(x)  # PIL Image to numpy array

        x = random_light_or_shadow(x)
        x = Image.fromarray(x)
        return x

## ----- TODO: random colorful light(not only white light)
def random_light_or_shadow(img, base=200, low=10, high=255):
    """
    @param img:
    @param base:
    """
    h, w, c = img.shape  # BGR or RGB

    ## ----- Randomly Generate Gauss Center
    center_x = np.random.randint(- w * 3, w * 3)
    center_y = np.random.randint(- w * 3, w * 3)

    radius_x = np.random.randint(int(h * 0.5), w * 2)
    radius_y = np.random.randint(int(h * 0.5), w * 2)

    delta_x = np.power((radius_x / 4), 2)
    delta_y = np.power((radius_y / 4), 2)

    x_arr, y_arr, c_arr = np.meshgrid(np.arange(w), np.arange(h), np.arange(c))
    weight = np.array(
        base * np.exp(-np.power((center_x - x_arr), 2) / (2 * delta_x))
        * np.exp(-np.power((center_y - y_arr), 2) / (2 * delta_y))
    )

    light_mode = np.random.randint(0, 2)
    if light_mode == 1:  # shadow
        img = img - weight
        img[img < 0] = low  # clipping
    else:  # light
        img = img + weight
        img[img > 255] = high  # clipping

    return img.astype(np.uint8)


class TwoCropsTransform:
    """
    Take two random crops of one image as the query and key.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


## Patch transform to return q and q
class PatchTransform():
    def __init__(self, patch_size=(320, 320)):
        """
        :param patch_size
        """
        self.augmentation = [
            # transforms.RandomResizedCrop(patch_size[0], scale=(0.2, 1.)),
            transforms.RandomApply([RandomLightOrShadow(base=200)], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.transform = TwoCropsTransform(transforms.Compose(self.augmentation))

    def __call__(self, patch):
        """
        :param patch
        :return a list of q and k
        """
        return self.transform(patch)


class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50):
        """
        :param p:
        :param rgb_means:
        :param std:
        :param max_labels:
        """
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels
        logger.info("max_labels: {:d}.".format(self.max_labels))

    def __call__(self, image, targets, net_dim):
        """
        :param image:
        :param targets:
        :param net_dim: input net size: net_h, net_w
        :return:
        """
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, net_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]

        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = random_distort(image)
        image_t, boxes = random_mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, net_dim, self.means, self.std)

        # boxes [xyxy] to [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        ## ----- minimum bbox threshold
        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, net_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((labels_t, boxes_t))

        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

        image_t = np.ascontiguousarray(image_t, dtype=np.float32)

        return image_t, padded_labels


class TrainTransformTrack:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=100):
        """
        :param p:
        :param rgb_means:
        :param std:
        :param max_labels:
        """
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim):
        """
        :param image:
        :param targets:
        :param input_dim:
        :return:
        """
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]

        ids_o = targets_o[:, 5]

        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = random_distort(image)
        image_t, boxes = random_mirror(image_t, boxes)
        height, width, _ = image_t.shape

        ## ----- resize, pad, BGR2RGB, normalize
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)

        ## ----- Resize box
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        ids_t = ids[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

            ids_t = ids_o

        labels_t = np.expand_dims(labels_t, 1)

        ids_t = np.expand_dims(ids_t, 1)

        ## ----- concatenate class, bbox, id
        targets_t = np.hstack((labels_t, boxes_t, ids_t))
        padded_labels = np.zeros((self.max_labels, 6))

        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)

        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network
    dimension -> tensorize -> color adj
    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        """
        :param rgb_means:
        :param std:
        :param swap:
        """
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        """
        :param img:
        :param res:
        :param input_size:
        :return:
        """
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5))
