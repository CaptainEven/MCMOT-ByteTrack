# encoding=utf-8

import argparse
import os
import time
from collections import defaultdict

import cv2
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info, post_process
from yolox.utils.visualize import plot_tracking_sc, plot_tracking_mc

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("--demo",
                        default="video",  # image
                        help="demo type, eg. image, video, videos, and webcam")
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
    parser.add_argument("-debug",
                        type=bool,
                        default=True,  # True
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
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_tiny_track_c5_darknet.py",
                        type=str,
                        help="pls input your experiment description file")

    ## ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
    parser.add_argument("-c",
                        "--ckpt",
                        default="../YOLOX_outputs/yolox_tiny_track_c5_darknet/latest_ckpt.pth.tar",
                        type=str,
                        help="ckpt for eval")

    parser.add_argument("--task",
                        type=str,
                        default="track",
                        help="Task mode: track or detect")

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
    parser.add_argument("--conf",
                        default=None,
                        type=float,
                        help="test conf")
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
                        help="tracking confidence threshold")
    parser.add_argument("--track_buffer",
                        type=int,
                        default=240,  # 30
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh",
                        type=int,
                        default=0.8,
                        help="matching threshold for tracking")
    parser.add_argument('--min-box-area',
                        type=float,
                        default=10,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20",
                        dest="mot20",
                        default=False,
                        action="store_true",
                        help="test mot20.")

    return parser


def get_image_list(path):
    """
    :param path:
    :return:
    """
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    """
    :param filename:
    :param results:
    :return:
    """
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
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

    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(self,
                 model,
                 exp,
                 trt_file=None,
                 decoder=None,
                 device="cpu",
                 fp16=False,
                 reid=False):
        """
        :param model:
        :param exp:
        :param trt_file:
        :param decoder:
        :param device:
        :param fp16:
        :param reid:
        """
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.conf_thresh = exp.test_conf
        self.nms_thresh = exp.nms_thresh
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.reid = reid

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

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
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
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

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()

            ## ----- forward
            outputs = self.model.forward(img)
            ## -----

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            if self.reid:
                outputs, feature_map = outputs[0], outputs[1]
                outputs = post_process(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            else:
                if isinstance(outputs, tuple):
                    outputs, feature_map = outputs[0], outputs[1]
                outputs = post_process(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        if self.reid:
            return outputs, feature_map, img_info
        else:
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
        files = get_image_list(path)
    else:
        files = [path]

    files.sort()
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    for image_name in files:
        if frame_id % 30 == 0:
            if frame_id != 0:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    30.0))

        outputs, img_info = predictor.inference(image_name, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            online_im = plot_tracking_sc(img_info['raw_img'],
                                         online_tlwhs,
                                         online_ids,
                                         frame_id=frame_id + 1,
                                         fps=1.0 / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(vis_folder,
                                       time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, online_im)
        ch = cv2.waitKey(0)

        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    # write_results(result_filename, results)


def video_tracking(predictor, cap, save_path, args):
    """
    online or offline tracking
    :param predictor:
    :param cap:
    :param save_path:
    :param args:
    :return:
    """
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # int

    save_path = os.path.abspath(save_path)
    vid_writer = cv2.VideoWriter(save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,
                                 (int(width), int(height)))

    ## ---------- define the tracker
    tracker = BYTETracker(args, frame_rate=30)
    ## ----------

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    timer = Timer()

    frame_id = 0
    results = []

    while True:
        if frame_id % 30 == 0:  ## logging per 30 frames
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
            if args.reid:
                outputs, feature_map, img_info = predictor.inference(frame, timer)
            else:
                outputs, img_info = predictor.inference(frame, timer)

            dets = outputs[0]

            if dets is not None:
                ## ----- update the frame
                img_size = [img_info['height'], img_info['width']]
                # online_targets = tracker.update(dets, img_size, exp.test_size)

                if args.reid:
                    online_dict = tracker.update_mcmot_emb(dets, feature_map, img_size, exp.test_size)
                else:
                    online_dict = tracker.update_mcmot_byte(dets, img_size, exp.test_size)

                ## ----- plot single-class multi-object tracking results
                if tracker.num_classes == 1:
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        # vertical = tlwh[2] / tlwh[3] > 1.6
                        # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        if tlwh[2] * tlwh[3] > args.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)

                    results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

                    timer.toc()
                    online_im = plot_tracking_sc(img_info['raw_img'],
                                                 online_tlwhs,
                                                 online_ids,
                                                 frame_id=frame_id + 1,
                                                 fps=1.0 / timer.average_time)

                ## ----- plot multi-class multi-object tracking results
                elif tracker.num_classes > 1:
                    ## ---------- aggregate current frame's results for each object class
                    online_tlwhs_dict = defaultdict(list)
                    online_ids_dict = defaultdict(list)
                    for cls_id in range(tracker.num_classes):  # process each object class
                        online_targets = online_dict[cls_id]
                        for track in online_targets:
                            online_tlwhs_dict[cls_id].append(track.tlwh)
                            online_ids_dict[cls_id].append(track.track_id)

                    timer.toc()
                    online_im = plot_tracking_mc(image=img_info['raw_img'],
                                                 tlwhs_dict=online_tlwhs_dict,
                                                 obj_ids_dict=online_ids_dict,
                                                 num_classes=tracker.num_classes,
                                                 frame_id=frame_id + 1,
                                                 fps=1.0 / timer.average_time,
                                                 id2cls=id2cls)
            else:
                timer.toc()
                online_im = img_info['raw_img']

            if args.save_result:
                vid_writer.write(online_im)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            print("Read frame {:d} failed!".format(frame_id))
            break

        ## ----- update frame id
        frame_id += 1

    print("{:s} saved.".format(save_path))


def imageflow_demo(predictor, vis_dir, current_time, args):
    """
    :param predictor:
    :param vis_dir:
    :param current_time:
    :param args:
    :return:
    """
    if args.demo == "videos":
        if os.path.isdir(args.video_dir):
            mp4_path_list = [args.video_dir + "/" + x for x in os.listdir(args.video_dir)
                             if x.endswith(".mp4")]
            mp4_path_list.sort()
            for video_path in mp4_path_list:
                if os.path.isfile(video_path):
                    video_name = os.path.split(video_path)[-1][:-4]
                    print("\nStart tracking video {:s} offline...".format(video_name))

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
                    video_tracking(predictor, cap, save_path, args)
                    ## ----------

                    print("{:s} tracking offline done.".format(video_name))

    elif args.demo == "video":
        if os.path.isfile(args.path):
            video_name = args.path.split("/")[-1][:-4]
            print("Start tracking video {:s} offline...".format(video_name))

            args.path = os.path.abspath(args.path)

            ## ----- video capture
            cap = cv2.VideoCapture(args.path)
            ## -----

            save_dir = os.path.join(vis_dir, video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            current_time = time.localtime()
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_path = os.path.join(save_dir, current_time + ".mp4")

            ## ---------- Get tracking results
            video_tracking(predictor, cap, save_path, args)
            ## ----------

            print("{:s} tracking done offline.".format(video_name))

    elif args.demo == "camera":
        if os.path.isfile(args.path):
            cap = cv2.VideoCapture(args.camid)
            video_name = args.path.split("/")[-1][:-4]
            save_dir = os.path.join(vis_dir, video_name)
            save_path = os.path.join(save_dir, "camera.mp4")


def run(exp, args):
    """
    :param exp:
    :param args:
    :return:
    """
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_dir = os.path.join(file_name, "track_vis")
        os.makedirs(vis_dir, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nms_thresh = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    ## ---------- whether to do ReID
    if hasattr(exp, "reid"):
        exp.reid = args.reid

    ## ----- Define the network
    model = exp.get_model()
    if not args.debug:
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if args.device == "gpu":
        model.cuda()
    model.eval()
    ## -----

    if not args.trt:
        if args.ckpt is None:
            ckpt_file_path = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file_path = args.ckpt
        ckpt_file_path = os.path.abspath(ckpt_file_path)

        logger.info("Loading checkpoint...")
        ckpt = torch.load(ckpt_file_path, map_location="cpu")

        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("Checkpoint {:s} loaded done.".format(ckpt_file_path))

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), \
            "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    ## ---------- Define the predictor
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16, args.reid)
    ## ----------

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_dir, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "videos" or args.demo == "webcam":
        imageflow_demo(predictor, vis_dir, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    class_names = args.class_names.split(",")
    args.class_names = class_names
    exp.class_names = class_names
    exp.num_classes = len(exp.class_names)
    print("Number of classes: ", exp.num_classes)

    ## ----- run the tracking
    run(exp, args)
