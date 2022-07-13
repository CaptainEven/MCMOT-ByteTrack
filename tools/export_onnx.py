# encoding=utf-8

import argparse
import os

import torch
from loguru import logger

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")

    parser.add_argument("--output_onnx_path",
                        type=str,
                        default="bytetrack.onnx",
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
                        type=bool,
                        default=True,
                        help="use onnxsim or not")
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_det_c5_dark_ssl.py",
                        type=str,
                        help="expriment description file", )

    ## -----Darknet cfg file path
    parser.add_argument("--cfg",
                        type=str,
                        default="../cfg/yolox_darknet_tiny_bb46.cfg",
                        help="")
    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="ssl",
                        help="model name")
    parser.add_argument("-c",
                        "--ckpt",
                        default="../YOLOX_outputs/yolox_det_c5_dark_ssl/ssl_ckpt.pth.tar",
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
    opt = make_parser().parse_args()
    logger.info("args value: {}".format(opt))

    exp = get_exp(opt.exp_file, opt.name)
    exp.merge(opt.opts)

    ## ----- Using cfg file from opt
    # if hasattr(exp, "cfg_file_path"):
    #     exp.cfg_file_path = os.path.abspath(opt.cfg)
    #
    #     cfg_name = os.path.split(opt.cfg)[-1]
    #     if "." in cfg_name:
    #         cfg_name = cfg_name.split(".")[0]

    opt.output_onnx_path = os.path.abspath("../" + opt.name + ".onnx")

    if not opt.experiment_name:
        opt.experiment_name = exp.exp_name

    net = exp.get_model()
    if opt.ckpt is None:
        ckpt_name = os.path.join(exp.output_dir, opt.experiment_name)
        ckpt_path = os.path.join(ckpt_name, "best_ckpt.pth.tar")
    else:
        ckpt_path = opt.ckpt
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isfile(ckpt_path):
        logger.info("loading ckpt {:s}...".format(ckpt_path))
    else:
        logger.error("invalid ckpt path: {:s}".format(ckpt_path))

    ## ----- load the model state dict
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    net.load_state_dict(ckpt)
    net.eval()  # switch to eval mode

    # net = replace_module(net, nn.SiLU, SiLU)
    net.head.decode_in_inference = False
    logger.info("loading checkpoint done.")

    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    torch.onnx._export(net,
                       dummy_input,
                       opt.output_onnx_path,
                       input_names=[opt.input],
                       output_names=[opt.output],
                       opset_version=opt.opset, )
    logger.info("generated onnx model named {}".format(opt.output_onnx_path))

    if not opt.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnx-simplify to reduce redundant model.
        onnx_model = onnx.load(opt.output_onnx_path)
        model_simp, check = simplify(onnx_model)

        assert check, "Simplified ONNX model could not be validated"

        onnx.save(model_simp, opt.output_onnx_path)
        logger.info("generated simplified onnx model named {}"
                    .format(opt.output_onnx_path))


if __name__ == "__main__":
    run()
