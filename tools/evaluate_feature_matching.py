# encoding=utf-8

import argparse
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from yolox.exp import get_exp
from yolox.utils import fuse_model, post_process, select_device, find_free_gpu
from yolox.utils.demo_utils import cos_sim, box_iou


class ReIDEvaluator(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--class_names",
                                 type=list,
                                 default=["car",
                                          "bicycle",
                                          "person",
                                          "cyclist",
                                          "tricycle"],
                                 help='*.names path')

        self.parser.add_argument("-n",
                                 "--name",
                                 type=str,
                                 default=None,
                                 help="model name")

        self.parser.add_argument("--output_dir",
                                 type=str,
                                 default="../YOLOX_outputs",
                                 help="")

        ## ----- exp file, eg: yolox_x_ablation.py
        self.parser.add_argument("-f",
                                 "--exp_file",
                                 default="../exps/example/mot/yolox_det_c5_dark_ssl.py",
                                 type=str,
                                 help="pls input your experiment description file")

        ## -----Darknet cfg file path
        self.parser.add_argument("--cfg",
                                 type=str,
                                 default="../cfg/yolox_darknet_tiny_bb46.cfg",
                                 help="")

        ## ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
        # yolox_tiny_det_c5_dark
        self.parser.add_argument("-c",
                                 "--ckpt",
                                 default="../YOLOX_outputs/yolox_det_c5_dark_ssl/ssl_ckpt.pth.tar",
                                 type=str,
                                 help="ckpt for eval")

        self.parser.add_argument("--fuse",
                                 dest="fuse",
                                 default=False,
                                 action="store_true",
                                 help="Fuse conv and bn for testing.")
        # ----------

        # input seq videos
        self.parser.add_argument('--videos',
                                 type=str,
                                 default='/mnt/diskb/even/dataset/MCMOT_Evaluate',
                                 help='')  # 'data/samples/videos/'

        # task mode
        self.parser.add_argument('--task',
                                 type=str,
                                 default='track',
                                 help='task mode: track or detect')

        self.parser.add_argument('--input-type',
                                 type=str,
                                 default='videos',
                                 help='videos or txt')

        # ----- Set net input image width and height
        self.parser.add_argument('--img-size',
                                 type=int,
                                 default=768,
                                 help='Image size')

        self.parser.add_argument('--net_w',
                                 type=int,
                                 default=768,
                                 help='inference size (pixels)')

        self.parser.add_argument('--net_h',
                                 type=int,
                                 default=448,
                                 help='inference size (pixels)')

        # -----
        self.parser.add_argument('--cutoff',
                                 type=int,
                                 default=0,  # 0 or 44, 47
                                 help='cutoff layer index, 0 means all layers loaded.')

        self.parser.add_argument('--dim',
                                 type=int,
                                 default=128,  # 64, 128, 256, 384, 512
                                 help='reid feature map output embedding dimension')

        self.parser.add_argument('--bin-step',
                                 type=int,
                                 default=5,
                                 help='number of bins '
                                      'for cosine similarity statistics(10 or 5).')

        # -----
        self.parser.add_argument('--conf_thresh',
                                 type=float,
                                 default=0.4,
                                 help='object confidence threshold')

        self.parser.add_argument("--nms_thresh",
                                 type=float,
                                 default=0.65,
                                 help="")

        self.parser.add_argument("--conf",
                                 type=float,
                                 default=0.2,
                                 help='object confidence threshold')

        self.parser.add_argument("--iou",
                                 type=float,
                                 default=0.45,
                                 help='IOU threshold for NMS')
        # ----------
        self.parser.add_argument('--classes',
                                 nargs='+',
                                 type=int,
                                 help='filter by class')

        opt = self.parser.parse_args()
        self.opt = opt
        exp = get_exp(self.opt.exp_file, self.opt.name)
        self.exp = exp

        if hasattr(self.exp, "cfg_file_path"):
            self.exp.cfg_file_path = os.path.abspath(self.opt.cfg)
        if hasattr(self.exp, "output_dir"):
            self.exp.output_dir = self.opt.output_dir

        if isinstance(opt.class_names, str):
            class_names = opt.class_names.split(",")
        elif isinstance(opt.class_names, list):
            class_names = opt.class_names

        self.class_names = class_names
        self.opt.class_names = class_names
        self.exp.class_names = class_names
        self.exp.n_classes = len(class_names)
        self.num_classes = len(class_names)
        logger.info("Number of classes: {:d}".format(exp.n_classes))

        # class name to class id and class id to class name
        self.id2cls = defaultdict(str)
        self.cls2id = defaultdict(int)
        for cls_id, cls_name in enumerate(self.class_names):
            self.id2cls[cls_id] = cls_name
            self.cls2id[cls_name] = cls_id

        # video GT
        if not os.path.isdir(self.opt.videos):
            print('[Err]: invalid videos dir.')
            return

        self.videos = [self.opt.videos + '/' + x
                       for x in os.listdir(self.opt.videos)
                       if x.endswith('.mp4')]
        # ----------

        # set device
        self.opt.device = str(find_free_gpu())
        print('Using gpu: {:s}'.format(self.opt.device))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.device
        device = select_device(device='cpu' if not torch.cuda.is_available() else self.opt.device)
        self.opt.device = device
        self.device = device

        ## ----- Define the network
        self.net = self.exp.get_model()
        self.net.to(self.device).eval()
        ## -----

        ## ----- load weights
        self.ckpt = os.path.abspath(self.opt.ckpt)
        logger.info("Loading checkpoint...")
        ckpt_states = torch.load(self.ckpt, map_location="cpu")

        # load the model state dict
        self.net.load_state_dict(ckpt_states["model"])
        logger.info("Checkpoint {:s} loaded done.".format(self.ckpt))

        if opt.fuse:
            logger.info("\tFusing model...")
            self.net = fuse_model(self.net)

        ## ----- Input image parameters
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        ## ----- statistics
        self.correct_sim_bins_dict = defaultdict(int)
        self.wrong_sim_bins_dict = defaultdict(int)
        self.sim_bins_dict = defaultdict(int)
        for edge in range(0, 100, self.opt.bin_step):
            self.correct_sim_bins_dict[edge] = 0
            self.wrong_sim_bins_dict[edge] = 0
            self.sim_bins_dict[edge] = 0

        # gap of the same object class and different object class
        self.min_same_id_sim = 1.0  # init to the max
        self.max_diff_id_sim = -1.0  # init to the min

        self.mean_same_id_sim = 0.0
        self.mean_diff_id_sim = 0.0

        self.num_total_match = 0
        self.num_sim_compute = 0

        print('Feature matcher init done.')

    def reset(self):
        """
        @return:
        """
        # statistics
        self.correct_sim_bins_dict = defaultdict(int)
        self.wrong_sim_bins_dict = defaultdict(int)
        self.sim_bins_dict = defaultdict(int)
        for edge in range(0, 100, self.opt.bin_step):
            self.correct_sim_bins_dict[edge] = 0
            self.wrong_sim_bins_dict[edge] = 0
            self.sim_bins_dict[edge] = 0

        # gap of the same object class and different object class
        self.min_same_id_sim = 1.0  # init to the max
        self.max_diff_id_sim = -1.0  # init to the min

        self.num_total_match = 0
        self.num_sim_compute = 0

    def run(self, cls_id=0, img_w=1920, img_h=1080, viz_dir=None):
        """
        :param cls_id:
        :param img_w:
        :param img_h:
        :param viz_dir:
        :return:
        """
        # create viz dir
        if viz_dir != None:
            if not os.path.isdir(viz_dir):
                os.makedirs(viz_dir)
            else:
                shutil.rmtree(viz_dir)
                os.makedirs(viz_dir)

        # traverse each video seq
        mean_precision = 0.0
        valid_seq_cnt = 0
        num_tps_total = 0
        for video_path in self.videos:  # .mp4
            if not os.path.isfile(video_path):
                print("[Warning]: {:s} not exists.".format(video_path))
                continue

            # get video seq name
            seq_name = os.path.split(video_path)[-1][:-4]

            # current video seq's gt label
            self.darklabel_txt_path = video_path[:-4] + '_gt.txt'
            if not os.path.isfile(self.darklabel_txt_path):
                print('[Warning]: {:s} not exists.'.format(self.darklabel_txt_path))
                continue

            ## ---------- run a video seq
            print("Evaluating seq {:s}...".format(video_path))
            precision, num_tps = self.run_a_seq(video_path, cls_id, viz_dir)
            mean_precision += precision
            num_tps_total += num_tps
            # print('Seq {:s} done.\n'.format(video_path))
            ## ----------

            valid_seq_cnt += 1

        mean_precision /= float(valid_seq_cnt)

        # histogram statistics
        num_correct_list = [self.correct_sim_bins_dict[x] for x in self.correct_sim_bins_dict]
        num_wrong_list = [self.wrong_sim_bins_dict[x] for x in self.wrong_sim_bins_dict]
        num_total_correct = sum(num_correct_list)
        num_total_wrong = sum(num_wrong_list)
        num_total = num_total_correct + num_total_wrong
        # print(num_total_wrong / num_total)
        # print(self.correct_sim_bins_dict)
        # print(self.wrong_sim_bins_dict)

        # detailed statistics
        for edge in range(0, 100, self.opt.bin_step):
            wrong_ratio = self.wrong_sim_bins_dict[edge] / num_total * 100.0
            print('Wrong distribution   [{:3d}, {:3d}]: {:.3f}'
                  .format(edge, edge + self.opt.bin_step, wrong_ratio))

        for edge in range(0, 100, self.opt.bin_step):
            correct_ratio = self.correct_sim_bins_dict[edge] / num_total * 100.0
            print('Correct distribution [{:3d}, {:3d}]: {:.3f}'
                  .format(edge, edge + self.opt.bin_step, correct_ratio))

        for edge in range(0, 100, self.opt.bin_step):
            ratio = self.sim_bins_dict[edge] / self.num_sim_compute * 100.0
            print('Total distribution   [{:3d}, {:3d}]: {:.3f}'
                  .format(edge, edge + self.opt.bin_step, ratio))

        print('\nTotal {:d} true positives detected.'.format(num_tps_total))
        print('Total {:d} matches tested.'.format(num_total))
        print('Num total match: {:d}'.format(self.num_total_match))
        print('Correct matched number: {:d}'.format(num_total_correct))
        print('Wrong matched number:   {:d}'.format(num_total_wrong))
        print('Mean precision:    {:.3f}%'.format(mean_precision * 100.0))
        print('Average precision: {:.3f}%'.format(num_total_correct / self.num_total_match * 100.0))
        print('Min same ID similarity:  {:.3f}'.format(self.min_same_id_sim))
        print('Max diff ID similarity:  {:.3f}'.format(self.max_diff_id_sim))
        print('Mean same ID similarity: {:.3f}'.format(self.mean_same_id_sim / num_total_correct))
        print('Mean diff ID similarity: {:.3f}'.format(self.mean_diff_id_sim / num_total_wrong))

    def load_gt(self, img_w, img_h, one_plus=True, cls_id=0):
        """
        Convert to x1, y1, x2, y2, tr_id(start from 1), cls_id format
        :param img_w: image width
        :param img_h: image height
        :param cls_id: specified object class id
        :return:
        """
        # each frame contains a list
        objs_gt = []

        with open(self.darklabel_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # traverse each frame
            fr_idx = 0
            for fr_i, line in enumerate(lines):
                line = line.strip().split(',')
                fr_id = int(line[0])
                n_objs = int(line[1])

                # traverse each object of the frame
                fr_objs = []
                for cur in range(2, len(line), 6):
                    # read object class id
                    class_type = line[cur + 5].strip()
                    class_id = self.cls2id[class_type]  # class type => class id

                    # read track id
                    if one_plus:
                        track_id = int(line[cur]) + 1  # track_id从1开始统计
                    else:
                        track_id = int(line[cur])

                    # read bbox
                    x1, y1 = int(line[cur + 1]), int(line[cur + 2])
                    x2, y2 = int(line[cur + 3]), int(line[cur + 4])

                    # clip bbox
                    x1 = x1 if x1 >= 0 else 0
                    x1 = x1 if x1 < img_w else img_w - 1
                    y1 = y1 if y1 >= 0 else 0
                    y1 = y1 if y1 < img_h else img_h - 1
                    x2 = x2 if x2 >= 0 else 0
                    x2 = x2 if x2 < img_w else img_w - 1
                    y2 = y2 if y2 >= 0 else 0
                    y2 = y2 if y2 < img_h else img_h - 1

                    fr_objs.append([x1, y1, x2, y2, track_id, class_id])

                objs_gt.append(fr_objs)

        return objs_gt

    def clip_bbox(self, bbox, w, h):
        """
        :param bbox: x1, y1, x2, y2
        :param w: max x
        :param h: max y
        :return:
        """
        x1, y1, x2, y2 = bbox

        if x1 >= x2 or y1 >= y2:
            print('[Err]: wrong bbox.')
            return

        x1 = x1 if x1 < w else w - 1
        x2 = x2 if x2 < w else w - 1
        y1 = y1 if y1 < h else h - 1
        y2 = y2 if y2 < h else h - 1

        bbox = x1, y1, x2, y2
        return bbox

    def get_true_positive(self, fr_id, dets, cls_id=0):
        """
        Compute true positives for the current frame and specified object class
        :param fr_id:
        :param dets: x1, y1, x2, y2, score, cls_id
        :param cls_id:
        :return:
        """
        assert len(self.objs_gt) == self.n_frames

        # get GT objs of current frame for specified object class
        fr_objs_gt = self.objs_gt[fr_id]
        objs_gt = [obj for obj in fr_objs_gt if obj[-1] == cls_id]
        objs_gt = np.array(objs_gt)

        # get predicted objs of current frame for specified object class
        objs_pred = [det for det in dets if det[-1] == cls_id]

        # compute TPs
        pred_match_flag = [False for n in range(len(objs_pred))]
        correct = 0
        TPs = []
        GT_tr_ids = []  # GT ids for each TP
        for i, obj_gt in enumerate(objs_gt):  # each gt obj
            best_iou = 0
            best_pred_id = -1
            for j, obj_pred in enumerate(objs_pred):  # each pred obj
                box_gt = obj_gt[:4]
                box_gt = self.clip_bbox(box_gt, self.img_w, self.img_h)
                box_pred = obj_pred[:4]
                iou = box_iou(box_gt, box_pred, box_type="x1y1x2y2")  # compute iou
                if obj_pred[4] > self.opt.conf and iou > best_iou:  # meet the conf thresh
                    best_pred_id = j
                    best_iou = iou

            # meet the iou thresh and not matched yet
            if best_iou > self.opt.iou and not pred_match_flag[best_pred_id]:
                correct += 1
                pred_match_flag[best_pred_id] = True  # set flag true for matched prediction

                # clipping predicted bbox
                objs_pred[best_pred_id][:4] = self.clip_bbox(objs_pred[best_pred_id][:4], self.img_w, self.img_h)

                TPs.append(objs_pred[best_pred_id])
                GT_tr_ids.append(obj_gt[4])

        return TPs, GT_tr_ids

    def get_feature(self, feature_map,
                    map_w, map_h,
                    img_w, img_h,
                    x1, y1, x2, y2):
        """
        Get feature vector
        :param feature_map:
        :param map_w:
        :param map_h:
        :param img_w:
        :param img_h:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """
        # get center point
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5

        # map center point from net scale to feature map scale(1/4 of net input size)
        center_x = center_x / float(img_w)
        center_x = center_x * float(map_w)
        center_y = center_y / float(img_h)
        center_y = center_y * float(map_h)

        # convert to int64 for indexing
        center_x = int(center_x + 0.5)
        center_y = int(center_y + 0.5)

        # to avoid the object center out of reid feature map's range
        center_x = np.clip(center_x, 0, map_w - 1)
        center_y = np.clip(center_y, 0, map_h - 1)

        # get reid feature vector and put into a dict
        feature_vector = feature_map[0, :, center_y, center_x]

        return feature_vector

    def preproc(self, image, net_size, mean, std, swap=(2, 0, 1)):
        """
        :param image:
        :param net_size: (net_h, net_w)
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

    def inference(self, img):
        """
        @param img:
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

        img, ratio = self.preproc(img, (self.net_h, self.net_w), self.mean, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)  # put input image to device

        with torch.no_grad():
            ## ----- forward, return a tuple
            outputs, feature_map = self.net.forward(img)
            ## -----

            ## ----- Post process for detection
            outputs = post_process(outputs,
                                   self.num_classes,
                                   self.opt.conf_thresh,
                                   self.opt.nms_thresh)

        return outputs, feature_map, img_info

    def run_a_seq(self,
                  video_path,
                  cls_id=0,
                  viz_dir=None):
        """
        :param seq_name:
        :param cls_id:
        :param img_w:
        :param img_h:
        :param viz_dir:
        :return:
        """
        ## ----- video capture
        cap = cv2.VideoCapture(video_path)
        ## -----

        self.img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # read net input width and height
        self.net_h, self.net_w = self.opt.net_h, self.opt.net_w

        # ---------- load GT for all frames
        self.objs_gt = self.load_gt(self.img_w, self.img_h, cls_id=cls_id)

        # ---------- iterate tracking results of each frame
        total = 0
        num_correct = 0
        num_wrong = 0
        sim_sum = 0.0
        num_tps = 0
        fr_id = 0

        with tqdm(total=self.n_frames) as progress_bar:
            while True:
                ## ----- read the video
                ret_val, frame = cap.read()

                if not ret_val:
                    break

                with torch.no_grad():
                    outputs, feature_map, img_info = self.inference(frame)
                    self.img_h, self.img_w = img_info['height'], img_info['width']
                    scale = img_info['ratio']

                    ## ----- Get dets
                    dets = outputs[0]
                    dets = dets.cpu().numpy()
                    dets = dets[np.where(dets[:, 4] > self.opt.conf)]

                    ## ----- Get feature map and normalize
                    feature_map = F.normalize(feature_map, p=2, dim=1)  # N×C×H×W
                    feature_map = feature_map.detach().cpu().numpy()

                ## ----- Get feature map size
                map_n, map_c, self.map_h, self.map_w = feature_map.shape
                # print("Feature map size: {:d}×{:d}".format(w, h))

                ## turn x1,y1,x2,y2,score1,score2,cls_id  (7)
                ## to x1,y1,x2,y2,score,cls_id  (6)
                if dets.shape[1] == 7:
                    dets[:, 4] *= dets[:, 5]
                    dets[:, 5] = dets[:, 6]
                    dets = dets[:, :6]

                if dets.shape[0] > 0:
                    ## ----- scale back the bbox
                    dets[:, :4] /= scale  # scale x1, y1, x2, y2
                else:
                    continue

                # only for car(cls_id == 0) for now
                TPs, GT_tr_ids = self.get_true_positive(fr_id, dets, cls_id=cls_id)

                num_tps += len(TPs)

                # ----- build mapping from TP id to GT track id
                tpid_to_gttrid = [GT_tr_ids[x] for x in range(len(TPs))]

                # ---------- matching statistics
                if fr_id >= 1:  # start from the second image
                    # ----- get GT for the last frame
                    objs_pre_gt = self.objs_gt[fr_id - 1]
                    self.gt_pre = [obj for obj in objs_pre_gt if obj[-1] == cls_id]

                    # ----- get intersection of pre and cur GT for the specified object class
                    # filtering
                    # tr_ids_cur = [x[4] for x in self.gt_cur]
                    # tr_ids_pre = [x[4] for x in self.gt_pre]
                    # tr_ids_gt_common = set(tr_ids_cur) & set(tr_ids_pre)  # GTs intersection
                    # gt_pre_tmp = [x for x in self.gt_pre if x[4] in tr_ids_gt_common]
                    # gt_cur_tmp = [x for x in self.gt_cur if x[4] in tr_ids_gt_common]
                    # self.gt_pre = gt_pre_tmp
                    # self.gt_cur = gt_cur_tmp

                    # ----- get intersection between pre and cur TPs for the specified object class
                    tr_ids_tp_common = set(self.GT_tr_ids_pre) & set(GT_tr_ids)
                    TPs_ids_pre = [self.GT_tr_ids_pre.index(x) for x in self.GT_tr_ids_pre if x in tr_ids_tp_common]
                    TPs_ids_cur = [GT_tr_ids.index(x) for x in GT_tr_ids if x in tr_ids_tp_common]
                    TPs_pre = [self.TPs_pre[x] for x in TPs_ids_pre]
                    TPs_cur = [TPs[x] for x in TPs_ids_cur]
                    # if len(TPs_pre) != len(TPs_cur):
                    #     print("Current frame's TPs not equal to previous' frame.")

                    # ----- update total pairs
                    total += len(TPs_cur)

                    # ----- greedy matching...
                    # print('Frame {:d} start matching for {:d} TP pairs.'.format(fr_id, len(TPs_cur)))
                    for tpid_cur, det_cur in zip(TPs_ids_cur, TPs_cur):  # current frame as row
                        x1_cur, y1_cur, x2_cur, y2_cur = det_cur[:4]
                        reid_feat_vect_cur = self.get_feature(feature_map,
                                                              self.map_w, self.map_h,
                                                              self.img_w, self.img_h,
                                                              x1_cur, y1_cur, x2_cur, y2_cur)
                        best_sim = -1.0
                        best_tp_id_pre = -1
                        for tpid_pre, det_pre in zip(TPs_ids_pre, TPs_pre):  # previous frame as col
                            x1_pre, y1_pre, x2_pre, y2_pre = det_pre[:4]
                            reid_feat_vect_pre = self.get_feature(self.feature_map_pre,
                                                                  self.map_w, self.map_h,
                                                                  self.img_w, self.img_h,
                                                                  x1_pre, y1_pre, x2_pre, y2_pre)

                            # --- compute cosine of cur and pre corresponding feature vector
                            sim = cos_sim(reid_feat_vect_cur, reid_feat_vect_pre)
                            # sim = euclidean(reid_feat_vect_cur, reid_feat_vect_pre)
                            # sim = SSIM(reid_feat_vect_cur, reid_feat_vect_pre)

                            # do cosine similarity statistics
                            sim_tmp = sim * 100.0
                            edge = int(sim_tmp / self.opt.bin_step) * self.opt.bin_step
                            self.sim_bins_dict[edge] += 1

                            # statistics of sim computation number
                            self.num_sim_compute += 1
                            if sim > best_sim:
                                best_sim = sim
                                best_tp_id_pre = tpid_pre

                        # determine matched right or not
                        gt_tr_id_pre = self.tpid_to_gttrid_pre[best_tp_id_pre]
                        gt_tr_id_cur = tpid_to_gttrid[tpid_cur]

                        # update match counting
                        self.num_total_match += 1

                        # if matching correct
                        if gt_tr_id_pre == gt_tr_id_cur:
                            # update correct number
                            num_correct += 1
                            self.mean_same_id_sim += best_sim
                            sim_sum += best_sim

                            ## -----  do visualization for correct and wrong match
                            if viz_dir != None:
                                save_path = viz_dir + '/' \
                                            + 'correct_match_{:s}_fr{:d}id{:d}-fr{:d}id{:d}-sim{:.3f}.jpg' \
                                                .format(seq_name, fr_id - 1, gt_tr_id_pre, fr_id, gt_tr_id_cur,
                                                        best_sim)

                            # do min similarity statistics of same object class
                            if best_sim < self.min_same_id_sim:
                                self.min_same_id_sim = best_sim

                            # do cosine similarity statistics
                            best_sim *= 100.0
                            edge = int(best_sim / self.opt.bin_step) * self.opt.bin_step
                            self.correct_sim_bins_dict[edge] += 1

                        else:  # visualize the wrong match:
                            num_wrong += 1
                            self.mean_diff_id_sim += best_sim

                            # wrong match img saving path
                            if viz_dir != None:
                                save_path = viz_dir + '/' \
                                            + 'wrong_match_{:s}_fr{:d}id{:d}-fr{:d}id{:d}-sim{:.3f}.jpg' \
                                                .format(seq_name, fr_id - 1, gt_tr_id_pre, fr_id, gt_tr_id_cur,
                                                        best_sim)
                            # do max similarity statistics of the different object class
                            if best_sim > self.max_diff_id_sim:
                                self.max_diff_id_sim = best_sim
                            # do cosine similarity statistics
                            best_sim *= 100.0
                            edge = int(best_sim / self.opt.bin_step) * self.opt.bin_step
                            self.wrong_sim_bins_dict[edge] += 1

                        if viz_dir != None:
                            # ----- plot
                            # text and line format
                            text_scale = max(1.0, img_w / 500.0)  # 1600.
                            text_thickness = 2
                            line_thickness = max(1, int(img_w / 500.0))
                            img0_pre = self.img_pre.copy()
                            x1_pre, y1_pre, x2_pre, y2_pre = self.TPs_pre[best_tp_id_pre][:4]  # best match bbox
                            cv2.rectangle(img0_pre,
                                          (int(x1_pre), int(y1_pre)),
                                          (int(x2_pre), int(y2_pre)),
                                          [0, 0, 255],
                                          thickness=line_thickness)
                            cv2.putText(img0_pre,
                                        'id{:d}'.format(gt_tr_id_pre),
                                        (int(x1_pre), int(y1_pre)),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=text_scale,
                                        color=[0, 255, 0],
                                        thickness=text_thickness)
                            img0_cur = img0.copy()
                            cv2.rectangle(img0_cur,
                                          (int(x1_cur), int(y1_cur)),
                                          (int(x2_cur), int(y2_cur)),
                                          [0, 0, 255],
                                          thickness=line_thickness)
                            cv2.putText(img0_cur,
                                        'id{:d}'.format(gt_tr_id_cur),
                                        (int(x1_cur), int(y1_cur)),
                                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                                        fontScale=text_scale,
                                        color=[0, 255, 0],
                                        thickness=text_thickness)
                            img_save = np.zeros((2 * img_h, img_w, 3), dtype=np.uint8)
                            img_save[:img_h, :, :] = img0_pre
                            img_save[img_h:2 * img_h, :, :] = img0_cur
                            cv2.imwrite(save_path, img_save)

                # ---------- update
                self.TPs_pre = TPs
                self.GT_tr_ids_pre = GT_tr_ids
                self.tpid_to_gttrid_pre = tpid_to_gttrid

                self.feature_map_pre = feature_map  # contains 1 feature map
                self.img_pre = frame

                ## ----- Update frame idx
                fr_id += 1

                progress_bar.update()
                ## ------------------------- Iter a frame done

        # compute precision of this seq
        if total > 0:
            precision = num_correct / total
            print("Precision: {:.4f}% | num_TPs: {:d} | num_correct: {:d} | num_wrong: {:d}"
                  " | mean cos sim: {:.5f}\n"
                  .format(precision * 100.0, num_tps, num_correct, num_wrong, sim_sum / num_correct))

        return precision, num_tps


def run_test():
    """
    :return:
    """
    evaluator = ReIDEvaluator()
    evaluator.run(cls_id=0,
                  img_w=1920,
                  img_h=1080,
                  viz_dir=None)  # '/mnt/diskc/even/viz_one_feat'


if __name__ == '__main__':
    run_test()
