# encoding=utf-8
import torch

from yolox.models.yolox_darknet import YOLOXDarknet

if __name__ == "__main__":
    net = YOLOXDarknet("../cfg/yolox_darknet_tiny.cfg")
    input = torch.zeros((1, 3, 448, 768))
    net.forward(input)
