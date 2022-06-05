# encoding=utf-8

import argparse
import os

import cv2
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.tracking_utils.timer import Timer
from yolox.utils import post_process


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("Detect video Demo!")

    parser.add_argument("--vid_path",
                        type=str,
                        default="../videos/test_13.mp4",
                        help="The input video path.")

    parser.add_argument("--output_dir",
                        type=str,
                        default="../output",
                        help="")

    ## ----- object classes
    parser.add_argument("--class_names",
                        type=str,
                        default="car, bicycle, person, cyclist, tricycle",
                        help="")

    ## ----- exp file, eg: yolox_x_ablation.py
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_tiny_det_c5_dark.py",
                        type=str,
                        help="pls input your experiment description file")

    parser.add_argument("--onnx_path",
                        type=str,
                        default="../yolox_darknet_tiny_bb46.onnx",
                        help="")

    ## "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
    parser.add_argument("--vid_path",
                        default="../videos/test_13.mp4",
                        help="path to images or video")

    parser.add_argument("--time_type",
                        type=str,
                        default="",
                        help="latest | current")

    parser.add_argument("--dev",
                        type=str,
                        default="cuda_fp16",
                        help="cpu | cuda | cuda_fp16")
    return parser


def inference(net,
              img,
              test_size, mean, std,
              num_classes, conf_thresh, nms_thresh,
              timer):
    """
    :param net:
    :param img:
    :param test_size:
    :param mean:
    :param std:
    :param num_classes:
    :param conf_thresh:
    :param nms_thresh:
    :param timer:
    :return:
    """
    img_info = {"id": 0}

    if isinstance(img, str):
        img_info["file_name"] = os.path.basename(img)
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    img, ratio = preproc(img, test_size, mean, std)
    img_info["ratio"] = ratio
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    with torch.no_grad():
        timer.tic()

        ## ----- forward
        outputs = net.forward(img)
        ## -----

        if isinstance(outputs, tuple):
            outputs, feature_map = outputs[0], outputs[1]
        outputs = post_process(outputs, num_classes, conf_thresh, nms_thresh)
        # logger.info("Infer time: {:.4f}s".format(time.time() - t0))

    return outputs, img_info


def detect_onnx(opt):
    """
    :param opt: options
    """
    ## ----- Read the onnx model by OpenCV
    onnx_path = os.path.abspath(opt.onnx_path)
    if not os.path.isfile(opt.onnx_path):
        logger.error("invalid onnx file path: {:s}, exit now!"
                     .format(onnx_path))
        exit(-1)

    ## ----- Define net and read in weights
    net = cv2.dnn.readNet(onnx_path)
    if opt.dev == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    elif opt.dev == "cuda_fp16":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    elif opt.dev == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ## ----- Set output dir
    if not os.path.isdir(opt.output_dir):
        logger.warning("invalid output dir: {:s}.".format(opt.output_dir))
        os.makedirs(opt.output_dir)
        logger.info("{:s} made.".format(opt.output_dir))

    ## ----- Read the video
    video_path = os.path.abspath(opt.vid_path)
    if not os.path.isfile(video_path):
        logger.error("invalid input video path: {:s}, exit now!"
                     .format(video_path))
        exit(-1)
    cap = cv2.VideoCapture(video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # int

    vid_save_path = ""
    if opt.time_type == "latest":
        vid_save_path = opt.output_dir + "/detect_onnx.mp4"
    elif opt.time_type == "current":
        pass

    vid_save_path = os.path.abspath(vid_save_path)
    logger.info("Writing results to {:s}...".format(vid_save_path))
    vid_writer = cv2.VideoWriter(vid_save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,
                                 (int(width), int(height)))

    timer = Timer()

    frame_id = 0

    while True:
        if frame_id % 30 == 0:  # logging per 30 frames
            if frame_id != 0:
                logger.info('Processing frame {:03d}/{:03d} | fps {:.2f}'
                            .format(frame_id,
                                    n_frames,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {:03d}/{:03d} | fps {:.2f}'
                            .format(frame_id,
                                    n_frames,
                                    30.0))

        ## ----- read the video
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = inference(frame, timer)

            dets = outputs[0]


def run():
    """
    run the detection
    """
    opt = make_parser().parse_args()

    ## ----- run the tracking
    detect_onnx(opt)


if __name__ == "__main__":
    run()
