# encoding=utf-8

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import select_device, find_free_gpu
from yolox.utils.demo_utils import multiclass_nms
from yolox.utils.visualize import plot_detection

def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("Test mAP")

    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")

    ## ----- object classes
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

    ## "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
    parser.add_argument("--test_path",
                        default="/users/duanyou/c5/data_all/test3000.txt",
                        help="path to images or video")

    ## /mnt/diske/viz_results/det
    parser.add_argument("--viz_dir",
                        type=str,
                        default="",
                        help="")

    parser.add_argument("--net_w",
                        type=int,
                        default=768,
                        help="")
    parser.add_argument("--net_h",
                        type=int,
                        default=448,
                        help="")

    parser.add_argument("--mean",
                        type=tuple,
                        default=(0.485, 0.456, 0.406),
                        help="")
    parser.add_argument("--std",
                        type=tuple,
                        default=(0.229, 0.224, 0.225),
                        help="")

    ## ----- mAP thresh
    parser.add_argument("--iou_thresh",
                        type=float,
                        default=0.5,
                        help="")

    parser.add_argument("--conf_thresh",
                        type=float,
                        default=0.3,
                        help="")

    parser.add_argument("--nms_thresh",
                        type=float,
                        default=0.6,
                        help="")

    return parser


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


def _post_process(outputs, net_size, scale, nms_th, conf_th):
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
                          score_thr=conf_th, )
    return dets


def inference(img_path, net, opt):
    """
    :param img_path:
    :return:
    """
    img_info = {"id": 0}
    if isinstance(img_path, str):
        img_info["file_name"] = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    net_size = (opt.net_h, opt.net_w)  ## Todo
    img, ratio = preproc(img, net_size, opt.mean, opt.std)
    img_info["ratio"] = ratio
    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    img = img.to(opt.device)

    with torch.no_grad():
        ## ----- forward
        outputs = net.forward(img)
        ## -----

        if isinstance(outputs, tuple):
            outputs, feature_map = outputs[0], outputs[1]

        ## ----- post process
        outputs = outputs.cpu().numpy()
        outputs = _post_process(outputs=outputs,
                                net_size=net_size,
                                scale=img_info["ratio"],
                                nms_th=opt.nms_thresh,
                                conf_th=opt.conf_thresh)

    return outputs, img_info


def convert2fraction(size, box):
    """
    :param size: w, h
    :param box: box=x1,y1,x2,y2
    :return:
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_min = box[0] * dw
    y_min = box[1] * dh
    x_max = box[2] * dw
    y_max = box[3] * dh
    return (x_min, y_min, x_max, y_max)


def parse_xml(xml_path):  # 读取标注的xml文件
    """
    Parse a PASCAL VOC xml file
    """
    if not os.path.isfile(xml_path):
        print("[Err]: invalid xml path: {:s}".format(xml_path))
        return None

    in_file = open(xml_path)
    xml_info = in_file.read()
    try:
        root = ET.fromstring(xml_info)
    except Exception as e:
        print("Error: cannot parse file")

    objects = []
    if root.find('markNode') != None:
        obj = root.find('markNode').find('object')
        if obj != None:
            img_w = int(root.find('width').text)
            img_h = int(root.find('height').text)
            for obj in root.iter('object'):
                if 'non_interest' in str(obj.find('targettype').text):
                    continue

                obj_struct = {}
                if obj.find('targettype').text == 'car_rear' \
                        or obj.find('targettype').text == 'car_front':
                    obj_struct['name'] = 'fr'
                else:
                    obj_struct['name'] = obj.find('targettype').text

                obj_struct['pose'] = 0  # obj.find('pose').text
                obj_struct['truncated'] = 0  # int(obj.find('truncated').text)
                obj_struct['difficult'] = 0  # int(obj.find('difficult').text)
                # bbox = obj.find('bndbox')
                b = [float(obj.find('bndbox').find('xmin').text),
                     float(obj.find('bndbox').find('ymin').text),
                     float(obj.find('bndbox').find('xmax').text),
                     float(obj.find('bndbox').find('ymax').text)]

                ## ----- normalize the coordinates(x1, y1, x2, y2) to [0,1]
                bb = convert2fraction((img_w, img_h), b)
                if bb is None:
                    continue

                obj_struct['bbox'] = [bb[0], bb[1], bb[2], bb[3]]
                objects.append(obj_struct)

    return objects


def get_img_name(img_path):
    """
    @param img_path
    """
    img_path = img_path.split('/')[-1]
    name = img_path.replace('.jpg', '')
    return name


def voc_ap(recall, precision):
    """
    @param recall:
    @param precision:
    """
    # 采用更为精确的逐点积分方法
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(dets_pred_cls,
             gt_dict_all,
             img_name_list,
             class_name,
             iou_thresh=0.5):
    """
    :param dets_pred_cls:
    :param xml_path_list:
    :param img_name_list:
    :param class_name:
    :param iou_thresh:
    :return:
    """
    # extract gt objects for this class
    # #按类别获取标注文件，recall和precision都是针对不同类别而言的，
    # AP也是对各个类别分别算的。
    n_gt = 0  # 标记的目标数量
    gt_cls_dict = {}
    for img_name in img_name_list:
        ## ----- Get labels of the whole image
        gts_of_img = gt_dict_all[img_name]

        ## ----- Get labels of the whole image for current class
        gts_cls = [x for x in gts_of_img if x["name"].strip() == class_name]
        gt_bboxes_cls = np.array([x["bbox"] for x in gts_cls])  # 抽取bbox
        difficult = np.array([x["difficult"] for x in gts_cls]).astype(np.bool)  # 如果数据集没有difficult,所有项都是0.

        gt_matched = [False] * len(gts_cls)  # len(img_cls_gts)就是当前类别的gt目标个数，det表示是否检测到，初始化为false。

        # 自增, 非difficult样本数量，如果数据集没有difficult, n_gt数量就是gt数量。
        n_gt += sum(~difficult)
        gt_cls_dict[img_name] = {"bbox": gt_bboxes_cls.copy(),
                                 "difficult": difficult,
                                 "matched": gt_matched}

    # read dets 读取检测结果
    split_lines = dets_pred_cls  # img_name cls_name confidence x1 y1 x2 y2
    pred_img_names = [x[0] for x in split_lines]  # 检测结果中的图像名，image_ids长度20000，但实际图像只有1000张，因为一张图像上可以有多个目标检测结果
    confidence = np.array([float(x[2]) for x in split_lines])  # 检测结果置信度
    BB_preds = np.array([[float(z) for z in x[3:]] for x in split_lines])  # 变为浮点型的bbox。

    n_gt = len(pred_img_names)  # TODO:???

    # sort by confidence 将20000各检测结果按置信度排序
    sorted_ind = np.argsort(-confidence)  # 对confidence的index根据值大小进行降序排列。
    sorted_scores = np.sort(-confidence)  # 降序排列。
    BB_preds = BB_preds[sorted_ind, :]  # 重排bbox，由大概率到小概率。
    pred_img_names = [pred_img_names[x] for x in sorted_ind]

    ## ----- go down dets and mark TPs and FPs
    n_pred_dets = len(pred_img_names)  # 注意这里是20000，不是1000
    TPs = np.zeros(n_pred_dets)  # true positive，长度20000
    FPs = np.zeros(n_pred_dets)  # false positive，长度20000

    # 遍历所有推理检测结果(一个bbox对应一个检测结果)，
    # 因为已经排序，所以这里是从置信度最高到最低遍历
    for i, img_name in enumerate(pred_img_names):
        bbox_pred = BB_preds[i]

        # 当前检测结果所在图像的所有同类别gt
        gts_cls = gt_cls_dict[img_name]

        # 当前检测结果所在图像的所有同类别gt的bbox坐标
        bbox_gt = gts_cls["bbox"].astype(float)
        max_iou = -np.inf

        if bbox_gt.size > 0:
            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
            # intersection
            i_x_min = np.maximum(bbox_gt[:, 0], bbox_pred[0])
            i_y_min = np.maximum(bbox_gt[:, 1], bbox_pred[1])
            i_x_max = np.minimum(bbox_gt[:, 2], bbox_pred[2])
            i_y_max = np.minimum(bbox_gt[:, 3], bbox_pred[3])
            iw = np.maximum(i_x_max - i_x_min + 1.0, 0.0)
            ih = np.maximum(i_y_max - i_y_min + 1.0, 0.0)
            inters = iw * ih
            unions = ((bbox_pred[2] - bbox_pred[0] + 1.0) * (bbox_pred[3] - bbox_pred[1] + 1.0) +
                      (bbox_gt[:, 2] - bbox_gt[:, 0] + 1.0) *
                      (bbox_gt[:, 3] - bbox_gt[:, 1] + 1.0) - inters)
            ious = inters / unions
            max_iou = np.max(ious)
            max_iou_gt_idx = np.argmax(ious)

        if max_iou > iou_thresh:  # 如果当前检测结果与真实标注最大重合率满足阈值
            # if not img_cls_gts["difficult"][j_max]:
            if not gts_cls["matched"][max_iou_gt_idx]:
                TPs[i] = 1.0  # 正检数目+1
                gts_cls["matched"][max_iou_gt_idx] = True  # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
            else:  # 相反，认为检测到一个虚警
                FPs[i] = 1.0
        else:  # 不满足阈值，肯定是虚警
            FPs[i] = 1.0

    # compute precision and recall
    FPs = np.cumsum(FPs)  # 积分图，在当前节点前的虚警数量，fp长度
    TPs = np.cumsum(TPs)  # 积分图，在当前节点前的正检数量
    recall = TPs / float(n_gt)  # 召回率，长度20000，从0到1

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth 准确率，长度20000，长度20000，从1到0
    precision = TPs / np.maximum(TPs + FPs, np.finfo(np.float64).eps)
    ap = voc_ap(recall, precision)

    return ap


def evaluate(exp, opt):
    """
    @param exp:
    @param opt:
    """
    test_list_file_path = os.path.abspath(opt.test_path)
    if not os.path.isfile(test_list_file_path):
        logger.error("invalid test list file path: {:s}, exit now!"
                     .format(test_list_file_path))
        exit(-1)

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(opt.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    ## ----- Set device
    opt.device = str(find_free_gpu())
    logger.info("using gpu: {:s}".format(opt.device))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    device = select_device(device="cpu"
    if not torch.cuda.is_available() else opt.device)
    opt.device = device

    ## ----- Define the network
    net = exp.get_model()
    net.eval().to(device)
    net.head.decode_in_inference = False
    ## -----

    ## ----- load weights
    ckpt_path = os.path.abspath(opt.ckpt)
    if not os.path.isfile(ckpt_path):
        logger.error("invalid ckpt path: {:s}, exit now!".format(ckpt_path))
    logger.info("Loading checkpoint from {:s}...".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["model"])
    logger.info("Checkpoint {:s} loaded done.".format(ckpt_path))

    ## ----- Set images and labels
    img_paths = []
    xml_path_list = []
    with open(test_list_file_path, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            img_path = line.strip()
            if not os.path.isfile(img_path):
                logger.warning("invalid file path: {:s}".format(img_path))
                continue

            xml_path = img_path.replace("JPEGImages", "Annotations") \
                .replace(".jpg", ".xml")
            if not os.path.isfile(xml_path):
                logger.warning("invalid file path: {:s}"
                               .format(xml_pathxml_path))
                continue
            img_paths.append(img_path)

    img_paths.sort()
    xml_path_list = [img_path.replace("JPEGImages", "Annotations")
                     .replace(".jpg", ".xml")
                 for img_path in img_paths]
    logger.info("total {:d} samples to be evaluated.".format(len(img_paths)))

    ## ----- Get predicts
    print("=> Predicting...")
    dets_pred_all = [["img_name", "cls_name", "confidence", 0, 0, 0, 0]]
    img_name_list = []
    with tqdm(total=len(img_paths)) as progress_bar:
        for i, (img_path, xml_path) in enumerate(zip(img_paths, xml_path_list)):
            ## ----- Get image name
            img_name = get_img_name(img_path)
            img_name_list.append(img_name)

            ## ----- Inference
            outputs, img_info = inference(img_path, net, opt)
            if outputs is None:
                continue

            with torch.no_grad():
                dets_predicted = outputs  # [0]
                if dets_predicted is None:
                    continue

            if os.path.isdir(opt.viz_dir):
                ## ----- draw detections
                img_plot = plot_detection(img_info["raw_img"],
                                          dets_predicted,
                                          frame_id=i,
                                          fps=0.0,
                                          id2cls=id2cls)
                save_vis_path = opt.viz_dir + "/" + img_name + ".jpg"
                cv2.imwrite(save_vis_path, img_plot)
                print("{:s} saved".format(save_vis_path))

            ## -----
            dets_pred = []
            for det in dets_predicted:
                x1, y1, x2, y2, confidence, cls_id = det

                ## ----- Clipping the predicted coordinates
                x1 = x1 if x1 >= 0 else 0
                x1 = x1 if x1 < img_info["width"] else img_info["width"] - 1
                x2 = x2 if x2 >= 0 else 0
                x2 = x2 if x2 < img_info["width"] else img_info["width"] - 1
                y1 = y1 if y1 >= 0 else 0
                y1 = y1 if y1 < img_info["height"] else img_info["height"] - 1
                y2 = y2 if y2 >= 0 else 0
                y2 = y2 if y2 < img_info["height"] else img_info["height"] - 1

                ## ----- Normalize the bbox to [0, 1]
                x1 /= img_info["width"]
                x2 /= img_info["width"]
                y1 /= img_info["height"]
                y2 /= img_info["height"]

                cls_name = id2cls[cls_id]
                det_pred = [img_name, cls_name, confidence, x1, y1, x2, y2]
                dets_pred.append(det_pred)

            if len(dets_pred) > 0:
                dets_pred_all = np.vstack((dets_pred_all, dets_pred))

            progress_bar.update()

    # parse_rec函数读取当前图像标注文件，返回当前图像标注，存于recs字典（key是图像名，values是gt）
    print("=> Parsing XMLs...")
    gt_dict_all = {}
    with tqdm(total=len(img_name_list)) as progress_bar:  # 遍历大图
        for img_name, xml_path in zip(img_name_list, xml_path_list):
            parsed_list = parse_xml(xml_path)
            if parsed_list is None:
                continue
            gt_dict_all[img_name] = parsed_list
            progress_bar.update()

    ## ----- Calculate APs for each object class
    dets_pred_all = np.delete(dets_pred_all, 0, axis=0)
    APs = []
    for cls_i, cls_name in enumerate(opt.class_names):
        cls_name = cls_name.strip()

        # if cls_i > 0:
        #     print("pause")
        # elif cls_i == 0:
        #     continue

        print("=> Processing {:s}...".format(cls_name))
        dets_pred_cls = [obj for obj in dets_pred_all if obj[1].strip() == cls_name]
        if len(dets_pred_cls) == 0:
            cls_ap = 0
        else:
            cls_ap = voc_eval(dets_pred_cls,
                              gt_dict_all,
                              img_name_list,
                              cls_name,
                              opt.iou_thresh)
        APs.append(cls_ap)
        print("=> Processing {:s} done.".format(cls_name))

    print("APs: ", APs)
    APs = np.array(APs, dtype=np.float32)
    mAP = np.mean(APs)
    print("mAP of C5: {:.3f}".format(mAP))


if __name__ == "__main__":
    opt = make_parser().parse_args()
    exp = get_exp(opt.exp_file, opt.name)

    if hasattr(exp, "cfg_file_path"):
        exp.cfg_file_path = os.path.abspath(opt.cfg)
    if hasattr(exp, "overlap_thresh"):
        exp.overlap_thresh = opt.overlap_thresh

    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    exp.class_names = class_names
    logger.info("object class names: " + " ".join(opt.class_names))
    exp.n_classes = len(exp.class_names)
    opt.num_classes = len(exp.class_names)
    logger.info("Number of object classes: {:d}".format(exp.n_classes))

    ## ----- run the tracking
    evaluate(exp, opt)
