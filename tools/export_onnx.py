# encoding=utf-8

import argparse
import os

import torch
from loguru import logger
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")

    parser.add_argument("--output-name",
                        type=str,
                        default="bytetrack_s.onnx",
                        help="output name of models")
    parser.add_argument("--input",
                        default="images",
                        type=str,
                        help="input node name of onnx model")
    parser.add_argument("--output",
                        default="output",
                        type=str,
                        help="output node name of onnx model")
    parser.add_argument("-o",
                        "--opset",
                        default=11,
                        type=int,
                        help="onnx opset version")
    parser.add_argument("--no-onnxsim",
                        action="store_true",
                        help="use onnxsim or not")
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_tiny_det_c5_dark.py",
                        type=str,
                        help="expriment description file", )
    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")
    parser.add_argument("-c",
                        "--ckpt",
                        default="../YOLOX_outputs/yolox_tiny_det_c5_dark/latest_ckpt.pth.tar",
                        type=str,
                        help="ckpt path")
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER, )

    return parser


@logger.catch
def run():
    """
    Run the exportation
    """
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    net = exp.get_model()
    if args.ckpt is None:
        ckpt_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_path = os.path.join(ckpt_name, "best_ckpt.pth.tar")
    else:
        ckpt_path = args.ckpt
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isfile(ckpt_path):
        logger.info("loading ckpt {:s}...".format(ckpt_path))
    else:
        logger.error("invalid ckpt path: {:s}".format(ckpt_path))

    # load the model state dict
    ckpt = torch.load(ckpt_path, map_location="cpu")

    net.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    net.load_state_dict(ckpt)
    # net = replace_module(net, nn.SiLU, SiLU)
    net.head.decode_in_inference = False
    logger.info("loading checkpoint done.")

    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    torch.onnx._export(net,
                       dummy_input,
                       args.output_name,
                       input_names=[args.input],
                       output_names=[args.output],
                       opset_version=args.opset, )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    run()
