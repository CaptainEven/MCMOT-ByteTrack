# encoding=utf-8

import argparse
import os
import shutil

import tensorrt as trt
import torch
from loguru import logger
from torch2trt import torch2trt

from yolox.exp import get_exp


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("YOLOX TensorRT deploy")

    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default="",
                        help="")
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="C5Det",
                        help="model name")

    ## yolox_tiny_det_c5.py yolox_s_mix_det.py
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_tiny_det_c5.py",
                        type=str,
                        help="pls input your experiment description file", )
    parser.add_argument("-c",
                        "--ckpt",
                        default="../pretrained/latest_ckpt.pth.tar",
                        type=str,
                        help="ckpt path")

    return parser


@logger.catch
def main():
    """
    :return:
    """
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    dir_path = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(dir_path, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(dir_path, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt

    if not os.path.isfile(ckpt_file):
        print("[Err]: invalid ckpt file path.".format(ckpt_file))
        exit(-1)
    ckpt = torch.load(ckpt_file, map_location="cpu")

    ## ---------- load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint done.")

    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()

    print("Start converting....")
    model_trt = torch2trt(model,
                          [x],
                          fp16_mode=False,  # True | False
                          log_level=trt.Logger.INFO,
                          max_workspace_size=(1 << 32), )  # 1 << 32

    trt_save_path = os.path.abspath(dir_path + "/" + exp.exp_name + "_trt.pth")
    print("Saving trt file to {:s}...".format(trt_save_path))
    torch.save(model_trt.state_dict(), trt_save_path)
    print("{:s} saved.".format(trt_save_path))
    logger.info("Converted TensorRT model done.")

    engine_file_path = os.path.abspath(dir_path + "/" + exp.exp_name + "_trt.engine")
    print("Engine file path: {:s}".format(engine_file_path))

    engine_file_demo_path = os.path.join("../deploy", "TensorRT", "cpp", "model_trt.engine")
    engine_file_demo_path = os.path.abspath(engine_file_demo_path)
    with open(engine_file_path, "wb") as f:
        f.write(model_trt.engine.serialize())
    print("{:s} saved.".format(engine_file_path))

    shutil.copyfile(engine_file_path, engine_file_demo_path)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
