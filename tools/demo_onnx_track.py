# encoding=utf-8

import argparse
import copy
import os
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger

from yolox.tracker.byte_tracker import ByteTracker
from yolox.tracking_utils.timer import Timer
from yolox.utils.demo_utils import multiclass_nms, multiclass_NMS
from yolox.utils.visualize import draw_mcmot


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

    parser.add_argument("--track_buffer",
                        type=int,
                        default=240,
                        help="")

    parser.add_argument("--track_thresh",
                        type=float,
                        default=0.5,
                        help="")

    parser.add_argument("--conf",
                        type=float,
                        default=0.4,
                        help="")

    parser.add_argument("--match_thresh",
                        type=float,
                        default=0.8,
                        help="")

    parser.add_argument("--time_type",
                        type=str,
                        default="latest",
                        help="latest | current")

    parser.add_argument("--mode",
                        type=str,
                        default="show",
                        help="save | show")

    parser.add_argument("--log_interval",
                        type=int,
                        default=30,
                        help="")

    parser.add_argument("--dev",
                        type=str,
                        default="cuda",
                        help="cpu | cuda | cuda_fp16")

    parser.add_argument("--gpu_id",
                        type=int,
                        default=0,
                        help="")

    return parser


def pre_process(image, net_size, mean, std):
    """
    :param image:
    :param net_size:
    :param std:
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
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

    ## ----- BGR to RGB
    padded_img = padded_img[:, :, ::-1]

    ## ----- Normalize to [0, 1]
    padded_img /= 255.0

    ## ----- Standardization
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std

    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r


def _pre_process(image, net_size, mean, std):
    """
    :param image:
    """
    image_info = {}
    image_info['raw_img'] = copy.deepcopy(image)
    image_info['width'] = image.shape[1]
    image_info['height'] = image.shape[0]
    preprocessed_image, scale = pre_process(image,
                                            net_size,
                                            mean,
                                            std)
    image_info['scale'] = scale
    return preprocessed_image, image_info


def post_process(outputs, net_size):
    """
    :param outputs:
    :param net_size:
    """
    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    h_sizes = [net_size[0] // stride for stride in strides]
    w_sizes = [net_size[1] // stride for stride in strides]

    for h_size, w_size, stride in zip(h_sizes, w_sizes, strides):
        x_range, y_range = np.arange(w_size), np.arange(h_size)
        xv, yv = np.meshgrid(x_range, y_range)
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    ## ----- center_x, center_y
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides

    ## ----- bbox_w, bbox_h
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def gen_proposals_dict(outputs, net_size, scale, score_th):
    """
    :param outputs:
    :param net_size:
    :param scale:
    :param score_th:
    """
    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    h_sizes = [net_size[0] // stride for stride in strides]
    w_sizes = [net_size[1] // stride for stride in strides]

    for h_size, w_size, stride in zip(h_sizes, w_sizes, strides):
        x_range, y_range = np.arange(w_size), np.arange(h_size)
        xv, yv = np.meshgrid(x_range, y_range)
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    ## ----- center_x, center_y
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides

    ## ----- bbox_w, bbox_h
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    predictions = outputs[0]

    boxes = predictions[:, :4]  # 7056×10 -> 7056×4
    scores = predictions[:, 4:5] * predictions[:, 5:]  # 7056×5

    ## ---- turn boxes to bboxes
    bboxes = np.ones_like(boxes)
    bboxes[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5  # x_center to x1
    bboxes[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5  # y_center to y1
    bboxes[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5  # get x2
    bboxes[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5  # get y2
    bboxes /= scale  # scale back to image size

    proposal_count = 0
    bboxes_dict = defaultdict(list)
    scores_dict = defaultdict(list)
    for i, (bbox, score_arr) in enumerate(zip(bboxes, scores)):
        for cls_id in range(predictions.shape[1] - 5):
            score = score_arr[cls_id]
            if score < score_th:
                continue
            bboxes_dict[cls_id].append(bbox)
            scores_dict[cls_id].append(score)
            proposal_count += 1

    return bboxes_dict, scores_dict, proposal_count


def generate_proposals(outputs, net_size, scale, score_th):
    """
    :param outputs:
    :param net_size:
    :param scale:
    :param score_th:
    """
    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    h_sizes = [net_size[0] // stride for stride in strides]
    w_sizes = [net_size[1] // stride for stride in strides]

    for h_size, w_size, stride in zip(h_sizes, w_sizes, strides):
        x_range, y_range = np.arange(w_size), np.arange(h_size)
        xv, yv = np.meshgrid(x_range, y_range)
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    ## ----- center_x, center_y
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides

    ## ----- bbox_w, bbox_h
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    predictions = outputs[0]

    boxes = predictions[:, :4]  # 7056×10 -> 7056×4
    scores = predictions[:, 4:5] * predictions[:, 5:]

    ## ---- turn boxes to bboxes
    bboxes = np.ones_like(boxes)
    bboxes[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5  # x_center to x1
    bboxes[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5  # y_center to y1
    bboxes[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5  # get x2
    bboxes[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5  # get y2
    bboxes /= scale  # scale back to image size

    return bboxes, scores


def post_processing(outputs, net_size, scale, nms_th, score_th):
    """
    :param outputs:
    :param net_size:
    :param scale:
    :param nms_th:
    :param score_th:
    """
    bboxes_dict, scores_dict, count = gen_proposals_dict(outputs,
                                                         net_size,
                                                         scale,
                                                         score_th)

    # print("Dets number before NMS: ", count)
    dets = multiclass_NMS(bboxes_dict,
                          scores_dict,
                          nms_thr=nms_th)
    # print("Dets number after NMS: ", len(dets))

    return dets


def _post_process(output, net_size, scale, nms_th, score_th):
    """
    :param output:
    :param net_size: net_h, net_w
    """
    predictions = post_process(output, net_size)

    predictions = predictions[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]  # cls_score * obj_score

    bboxes = np.ones_like(boxes)  # bbox: x1y1x2y2
    bboxes[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    bboxes[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    bboxes[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    bboxes[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    bboxes /= scale  # scale back to image size

    # print("Dets number before NMS: ", len(boxes_xyxy))
    dets = multiclass_nms(bboxes,
                          scores,
                          nms_thr=nms_th,
                          score_thr=score_th)
    # print("Dets number after NMS: ", len(dets))

    return dets


def inference(net,
              img,
              net_size,
              mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225),
              conf_thresh=0.1,
              nms_thresh=0.65):
    """
    :param net:
    :param img:
    :param net_size:
    :param mean:
    :param std:
    :param conf_thresh:
    :param nms_thresh:
    :return:
    """

    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    ## ----- pre-process
    img, img_info = _pre_process(img, net_size, mean, std)

    ## ----- forward
    blob = cv2.dnn.blobFromImage(img)
    net.setInput(blob)
    outputs = net.forward()
    # print(outputs.shape)  # (1, 7056, 10)
    ## -----

    ## ----- post process
    dets = _post_process(output=outputs,
                         net_size=net_size,
                         scale=img_info["scale"],
                         nms_th=nms_thresh,
                         score_th=conf_thresh)

    # dets = post_processing(outputs,
    #                        net_size,
    #                        img_info["scale"],
    #                        nms_thresh,
    #                        conf_thresh)

    return dets, img_info


def track_onnx(opt):
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
    net = cv2.dnn.readNetFromONNX(onnx_path)

    logger.info("Device: {:s}.".format(opt.dev))
    if opt.dev == "cuda":
        cv2.cuda.setDevice(opt.gpu_id)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    elif opt.dev == "cuda_fp16":
        cv2.cuda.setDevice(opt.gpu_id)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    elif opt.dev == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    logger.info("Confidence threshold: {:.3f}.".format(opt.conf))

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
        logger.warning(" not implemented yet!")
        pass

    if opt.mode == "save":
        vid_save_path = os.path.abspath(vid_save_path)
        logger.info("Writing results to {:s}...".format(vid_save_path))
        vid_writer = cv2.VideoWriter(vid_save_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps,
                                     (int(width), int(height)))

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    opt.n_classes = len(class_names)
    for cls_id, cls_name in enumerate(opt.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    net_size = (448, 768)
    # net_h, net_w = net_size

    tracker = ByteTracker(opt, frame_rate=30)

    timer = Timer()

    frame_id = 0

    while True:
        fps = 1.0 / max(1e-5, timer.average_time)

        if frame_id % opt.log_interval == 0:  # logging per 30 frames
            if frame_id != 0:
                logger.info("frame {:03d}/{:03d} | fps {:.2f}"
                            .format(frame_id,
                                    n_frames,
                                    fps))
            else:
                logger.info("frame {:03d}/{:03d} | fps {:.2f}"
                            .format(frame_id,
                                    n_frames,
                                    30.0))

        ## ----- read the video
        ret_val, frame = cap.read()

        if ret_val:
            timer.tic()

            ## ---------- inference:
            # include: pre-process, inference, post-process
            dets, img_info = inference(net, frame, net_size)
            ## ----------

            if dets is None:
                continue

            if dets.shape[0] > 0:
                dets = dets[np.where(dets[:, 4] > opt.conf)]

                ## ---------- update the tracking results
                tracks_dict = tracker.update_tracks(dets)
                ## ----------

                timer.toc()
                online_img = draw_mcmot(img=img_info["raw_img"],
                                        tracks_dict=tracks_dict,
                                        id2cls=id2cls,
                                        frame_id=frame_id + 1,
                                        fps=1.0 / timer.average_time)
            else:  # do nothing
                # timer.toc()
                online_img = img_info['raw_img']

            if opt.mode == "save":
                vid_writer.write(online_img)
            elif opt.mode == "show":
                cv2.namedWindow("Track", 0)
                cv2.imshow("Track", online_img)

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):  # quit
                    break
                elif ch == ord("p"):  # pause
                    cv2.waitKey(0)
        else:
            logger.warning("Read frame {:d} failed!".format(frame_id))
            break

        ## ----- update frame id
        frame_id += 1

    if opt.mode == "save":
        logger.info("{:s} saved.".format(vid_save_path))
    elif opt.mode == "show":
        logger.info("visualizing done.")


def run():
    """
    run the detection
    """
    opt = make_parser().parse_args()

    ## ----- run the tracking
    track_onnx(opt)


if __name__ == "__main__":
    run()
