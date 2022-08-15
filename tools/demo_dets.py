# encoding=utf-8

import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info
from yolox.utils import select_device, find_free_gpu
from yolox.utils.demo_utils import multiclass_nms
from yolox.utils.visualize import plot_detection

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("--demo",
                        default="video",  # image
                        help="demo type,"
                             "eg. image, video, videos, and webcam")
    parser.add_argument("--output_dir",
                        type=str,
                        default="../YOLOX_outputs",
                        help="")
    parser.add_argument("--conf_thresh",
                        default=0.3,
                        type=float,
                        help="test conf")
    parser.add_argument("--nms_thresh",
                        type=float,
                        default=0.6,
                        help="")
    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")
    parser.add_argument("--reid",
                        type=bool,
                        default=False,  # True | False
                        help="")

    ## ----- object classes
    parser.add_argument("--n_classes",
                        type=int,
                        default=5,
                        help="")  # number of object classes
    parser.add_argument("--class_names",
                        type=str,
                        default="car, bicycle, person, cyclist, tricycle",
                        help="")

    ## ----- exp file, eg: yolox_x_ablation.py
    ## yolox_tiny_det_c5_dark.py
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_det_c5_dark_ssl.py",
                        type=str,
                        help="pls input your experiment description file")

    ## -----Darknet cfg file path
    parser.add_argument("--cfg",
                        type=str,
                        default="../cfg/yolox_darknet_tiny_bb46.cfg",
                        help="")

    ## ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
    # yolox_tiny_det_c5_dark
    parser.add_argument("-c",
                        "--ckpt",
                        default="../YOLOX_outputs/yolox_det_c5_dark_ssl/ssl_ckpt.pth.tar",
                        type=str,
                        help="ckpt for eval")

    ## ----- videos dir path
    parser.add_argument("--video_dir",
                        type=str,
                        default="../videos",
                        help="")

    ## "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
    parser.add_argument("--path",
                        default="../videos/test_13.mp4",
                        help="path to images or video")

    ## ----- Web camera's id
    parser.add_argument("--camid",
                        type=int,
                        default=0,
                        help="webcam demo camera id")
    parser.add_argument("--save_result",
                        type=bool,
                        default=True,
                        help="whether to save the inference result of image/video")

    parser.add_argument("--device",
                        default="gpu",
                        type=str,
                        help="device to run our model, can either be cpu or gpu")

    parser.add_argument("--nms",
                        default=None,
                        type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize",
                        default=None,
                        type=int,
                        help="test img size")
    parser.add_argument("--fp16",
                        dest="fp16",
                        default=False,  # False
                        action="store_true",
                        help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse",
                        dest="fuse",
                        default=False,
                        action="store_true",
                        help="Fuse conv and bn for testing.")
    parser.add_argument("--trt",
                        dest="trt",
                        default=False,
                        action="store_true",
                        help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_thresh",
                        type=float,
                        default=0.5,
                        help="detection confidence threshold")
    parser.add_argument("--iou_thresh",
                        type=float,
                        default=0.3,
                        help="the iou threshold in Sort for matching")
    parser.add_argument("--match_thresh",
                        type=int,
                        default=0.8,
                        help="matching threshold for tracking")
    parser.add_argument("--track_buffer",
                        type=int,
                        default=240,  # 30
                        help="the frames for keep lost tracks")
    parser.add_argument('--min-box-area',
                        type=float,
                        default=10,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20",
                        dest="mot20",
                        default=False,
                        action="store_true",
                        help="test mot20.")
    parser.add_argument("--debug",
                        type=bool,
                        default=False,  # True
                        help="")

    return parser


def get_image_list(path):
    """
    :param path:
    :return:
    """
    image_names = []
    for main_dir, sub_dir, file_name_list in os.walk(path):
        for file_name in file_name_list:
            apath = os.path.join(main_dir, file_name)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results_dict(f_path,
                       results_dict,
                       data_type,
                       num_classes=5):
    """
    :param f_path:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls_id},1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(f_path, "w", encoding="utf-8") as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]
            for fr_id, tlwhs, track_ids in cls_results:  # fr_id starts from 1
                if data_type == 'kitti':
                    fr_id -= 1

                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=fr_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              cls_id=cls_id)
                    # if fr_id == 1:
                    #     print(line)

                    f.write(line)
                    # f.flush()

    logger.info('Save results to {}.\n'.format(f_path))


def write_results(file_path, results):
    """
    :param file_path:
    :param results:
    :return:
    """
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(file_path, "w", encoding="utf-8") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):

                if track_id < 0:
                    continue

                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id,
                                          id=track_id,
                                          x1=round(x1, 1),
                                          y1=round(y1, 1),
                                          w=round(w, 1),
                                          h=round(h, 1),
                                          s=round(score, 2))
                f.write(line)

    logger.info('save results to {}'.format(file_path))


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
        xv, yv = np.meshgrid(np.arange(w_size), np.arange(h_size))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def _post_process(outputs, net_size, scale, nms_th, score_th):
    """
    :param outputs:
    :param net_size: net_h, net_w
    """
    ## ----- Get xywh in net_size
    predictions = post_process(outputs, net_size)

    predictions = predictions[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= scale  # scale from net size back to image size

    dets = multiclass_nms(boxes_xyxy,
                          scores,
                          nms_thr=nms_th,
                          score_thr=score_th, )
    return dets


class Predictor(object):
    def __init__(self,
                 net,
                 exp,
                 trt_file=None,
                 decoder=None,
                 device="cpu",
                 fp16=False):
        """
        :param net:
        :param exp:
        :param trt_file:
        :param decoder:
        :param device:
        :param fp16:
        """
        self.net = net
        self.net.head.decode_in_inference = False
        self.decoder = decoder
        self.num_classes = exp.n_classes
        self.conf_thresh = exp.test_conf
        logger.info("confidence thresh: ", self.conf_thresh)

        self.nms_thresh = exp.nms_thresh
        logger.info("NMS thresh: {:.3f}".format(self.nms_thresh))

        self.conf_thresh = exp.conf_thresh
        logger.info("Conf thresh: {:.3f}".format(self.conf_thresh))

        self.test_size = exp.test_size  # H, W
        # self.net_size = self.test_size
        self.device = device
        self.fp16 = fp16

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        """
        :param img:
        :param timer:
        :return:
        """
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.mean, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)

        with torch.no_grad():
            timer.tic()

            ## ----- forward
            outputs = self.net.forward(img)
            ## -----

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            if isinstance(outputs, tuple):
                outputs, feature_map = outputs[0], outputs[1]

            ## ----- post process
            outputs = outputs.cpu().numpy()
            outputs = _post_process(outputs=outputs,
                                    net_size=self.test_size,
                                    scale=img_info["ratio"],
                                    nms_th=self.nms_thresh,
                                    score_th=self.conf_thresh)

        return outputs, img_info


def image_demo(predictor, vis_folder, path, current_time, save_result):
    """
    :param predictor:
    :param vis_folder:
    :param path:
    :param current_time:
    :param save_result:
    :return:
    """
    if os.path.isdir(path):
        file_path_list = get_image_list(path)
    else:
        file_path_list = [path]
    file_path_list.sort()

    net_size = exp.test_size
    net_h, net_w = net_size

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(opt.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    logger.info("Confidence threshold: {:.3f}".format(opt.conf))
    timer = Timer()

    frame_id = 0
    results = []
    for image_name in file_path_list:
        if frame_id % 30 == 0:
            if frame_id != 0:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    30.0))

        with torch.no_grad():
            outputs, img_info = predictor.inference(image_name, timer)
            dets = outputs[0]
            dets = dets.cpu().numpy()
            dets = dets[np.where(dets[:, 4] > opt.conf)]

        ## turn x1,y1,x2,y2,score1,score2,cls_id  (7)
        ## to x1,y1,x2,y2,score,cls_id  (6)
        if dets.shape[1] == 7:
            dets[:, 4] *= dets[:, 5]
            dets[:, 5] = dets[:, 6]
            dets = dets[:, :6]

        if dets.shape[0] > 0:
            ## ----- update the frame
            img_size = [img_info['height'], img_info['width']]

            ## ----- scale back the bbox
            img_h, img_w = img_size
            scale = min(net_h / float(img_h), net_w / float(img_w))
            dets[:, :4] /= scale  # scale x1, y1, x2, y2
            timer.toc()
            online_img = plot_detection(img=img_info['raw_img'],
                                        dets=dets,
                                        frame_id=frame_id + 1,
                                        fps=1.0 / timer.average_time,
                                        id2cls=id2cls)

            timer.toc()
            online_im = None
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(vis_folder,
                                       time.strftime("%Y_%m_%d_%H_%M_%S",
                                                     current_time))
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, online_im)
        ch = cv2.waitKey(0)

        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    # write_results(result_filename, results)


def detect_video(predictor, cap, vid_save_path, opt):
    """
    online or offline tracking
    :param predictor:
    :param cap:
    :param vid_save_path:
    :param opt:
    :return:
    """
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # int

    vid_save_path = os.path.abspath(vid_save_path)
    vid_writer = cv2.VideoWriter(vid_save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,
                                 (int(width), int(height)))

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(opt.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    net_size = exp.test_size

    timer = Timer()

    frame_id = 0
    # results = []
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
        net_h, net_w = net_size

        if ret_val:
            with torch.no_grad():
                outputs, img_info = predictor.inference(frame, timer)
                dets = outputs
                # dets = dets.cpu().numpy()
                # dets = dets[np.where(dets[:, 4] > opt.conf)]

            if dets.size > 0:
                ## turn x1,y1,x2,y2,score1,score2,cls_id  (7)
                ## to x1,y1,x2,y2,score,cls_id  (6)
                if dets.shape[1] == 7:
                    dets[:, 4] *= dets[:, 5]
                    dets[:, 5] = dets[:, 6]
                    dets = dets[:, :6]

                timer.toc()
                online_img = plot_detection(img=img_info['raw_img'],
                                            dets=dets,
                                            frame_id=frame_id + 1,
                                            fps=1.0 / timer.average_time,
                                            id2cls=id2cls)

            else:
                timer.toc()
                online_img = img_info['raw_img']

            if opt.save_result and not opt.debug:
                vid_writer.write(online_img)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            logger.warning("Read frame {:d} failed!".format(frame_id))
            break

        ## ----- update frame id
        frame_id += 1

    logger.info("{:s} saved.".format(vid_save_path))


def imageflow_demo(predictor, vis_dir, current_time, opt):
    """
    :param predictor:
    :param vis_dir:
    :param current_time:
    :param opt:
    :return:
    """
    if opt.demo == "videos":
        if os.path.isdir(opt.video_dir):
            mp4_path_list = [opt.video_dir + "/" + x
                             for x in os.listdir(opt.video_dir)
                             if x.endswith(".mp4")]
            mp4_path_list.sort()
            if len(mp4_path_list) == 0:
                logger.error("empty mp4 video list, exit now!")
                exit(-1)

            for video_path in mp4_path_list:
                if os.path.isfile(video_path):
                    video_name = os.path.split(video_path)[-1][:-4]
                    logger.info("\nstart detecting video {:s} offline..."
                                .format(video_name))

                    ## ----- video capture
                    cap = cv2.VideoCapture(video_path)
                    ## -----

                    save_dir = os.path.join(vis_dir, video_name)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    current_time = time.localtime()
                    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    save_path = os.path.join(save_dir, current_time + ".mp4")

                    ## ---------- Get tracking results
                    detect_video(predictor, cap, save_path, opt)
                    ## ----------

                    print("{:s} tracking offline done.".format(video_name))

    elif opt.demo == "video":
        if os.path.isfile(opt.path):
            video_name = opt.path.split("/")[-1][:-4]
            logger.info("start detecting video {:s} offline...".format(video_name))

            opt.path = os.path.abspath(opt.path)

            ## ----- video capture
            cap = cv2.VideoCapture(opt.path)
            ## -----

            save_dir = os.path.join(vis_dir, video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            current_time = time.localtime()
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_path = os.path.join(save_dir, current_time + ".mp4")

            ## ---------- Get tracking results
            detect_video(predictor, cap, save_path, opt)
            ## ----------

            print("{:s} tracking done offline.".format(video_name))

    elif opt.demo == "camera":
        if os.path.isfile(opt.path):
            cap = cv2.VideoCapture(opt.camid)
            video_name = opt.path.split("/")[-1][:-4]
            save_dir = os.path.join(vis_dir, video_name)
            save_path = os.path.join(save_dir, "camera.mp4")


def run(exp, opt):
    """
    :param exp:
    :param opt:
    :return:
    """
    if not opt.experiment_name:
        opt.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, opt.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if opt.save_result:
        vis_dir = os.path.join(file_name, "track_vis")
        os.makedirs(vis_dir, exist_ok=True)

    ## ----- Set device
    opt.device = str(find_free_gpu())
    logger.info("=> using gpu: {:s}".format(opt.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    device = select_device(device="cpu"
    if not torch.cuda.is_available() else opt.device)
    opt.device = device

    logger.info("Args: {}".format(opt))
    if opt.tsize is not None:
        exp.test_size = (opt.tsize, opt.tsize)

    ## ----- Define the network
    net = exp.get_model()
    net.eval().to(device)
    ## -----

    ## ----- load weights
    if not opt.trt:
        if opt.ckpt is None:
            ckpt_path = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_path = opt.ckpt
        ckpt_path = os.path.abspath(ckpt_path)

        logger.info("Loading checkpoint from {:s}...".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # load the model state dict
        net.load_state_dict(ckpt["model"])
        logger.info("Checkpoint {:s} loaded done.".format(ckpt_path))

    if opt.fuse:
        logger.info("\tFusing model...")
        net = fuse_model(net)

    ## ---------- Define the predictor
    net.head.decode_in_inference = False
    predictor = Predictor(net,
                          exp,
                          None,
                          None,
                          opt.device,
                          False)
    ## ----------

    current_time = time.localtime()
    if opt.demo == "image":
        image_demo(predictor, vis_dir, opt.path, current_time, opt.save_result)
    elif opt.demo == "video" or opt.demo == "videos" or opt.demo == "webcam":
        imageflow_demo(predictor, vis_dir, current_time, opt)


if __name__ == "__main__":
    opt = make_parser().parse_args()
    exp = get_exp(opt.exp_file, opt.name)

    if hasattr(exp, "cfg_file_path"):
        exp.cfg_file_path = os.path.abspath(opt.cfg)
    if hasattr(exp, "output_dir"):
        exp.output_dir = opt.output_dir

    if opt.conf_thresh > 0.0:
        exp.conf_thresh = opt.conf_thresh
        # logger.info("Conf thresh: {:.3f}".format(exp.conf_thresh))
    if opt.nms_thresh is not None:
        exp.nms_thresh = opt.nms_thresh
        # logger.info("NMS thresh: {:.3f}".format(exp.nms_thresh))

    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    exp.class_names = class_names
    exp.n_classes = len(exp.class_names)
    logger.info("Number of classes: {:d}".format(exp.n_classes))

    ## ----- run the tracking
    run(exp, opt)
