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
from yolox.utils import post_process, select_device, find_free_gpu


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
                        default=0.55,
                        help="")

    return parser


def inference(img_path, net, opt):
    """
    :param img_path:
    :return:
    """
    img_info = {"id": 0}
    if isinstance(img_path, str):
        img_info["file_name"] = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    net_size = (opt.net_w, opt.net_h)
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

        ## ----- get bbox(x1y1x2y2) and do NMS
        outputs = post_process(outputs,
                               opt.num_classes,
                               opt.conf_thresh,
                               opt.nms_thresh)

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


# def LoadLabel(xml_path, class_names):
#     """
#     @param xml_path:
#     @param class_names:
#     """
#     fl = open(xml_path)
#
#     cn = 0
#     num = 0
#     gt_dets = []
#     label_info = fl.read()
#     if label_info.find('dataroot') < 0:
#         print("Can not find dataroot")
#         fl.close()
#         return gt_dets
#
#     try:
#         root = ET.fromstring(label_info)
#     except(Exception, e):
#         print("Error: cannot parse file")
#         # n = raw_input()
#         fl.close()
#         return gt_dets
#
#     if root.find('markNode') != None:
#         obj = root.find('markNode').find('object')
#         if obj != None:
#             w = int(root.find('width').text)
#             h = int(root.find('height').text)
#             # print("w:%d,h%d" % (w, h))
#             for obj in root.iter('object'):
#                 targettype = obj.find('targettype').text
#                 cartype = obj.find('cartype').text
#                 if targettype == 'car_front' or targettype == 'car_rear':
#                     targettype = 'fr'
#                 if targettype not in class_names and cartype not in class_names:
#                     # print("********************************* "+str(targettype) + "is not in class list *************************")
#                     continue
#
#                 # classes_c9
#                 # if targettype == "car":
#                 #     cartype = obj.find('cartype').text
#                 #     # print(cartype)
#                 #     if cartype == 'motorcycle':
#                 #         targettype = "bicycle"
#                 #     elif cartype == 'truck':
#                 #         targettype = "truck"
#                 #     elif cartype == 'waggon':
#                 #         targettype = 'waggon'
#                 #     elif cartype == 'passenger_car':
#                 #         targettype = 'passenger_car'
#                 #     elif cartype == 'unkonwn' or cartype == "shop_truck":
#                 #         targettype = "other"
#
#                 # classes_c5
#                 if targettype == 'car':
#                     cartype = obj.find('cartype').text
#                     if cartype == 'motorcycle':
#                         targettype = 'bicycle'
#                 if targettype == "motorcycle":
#                     targettype = "bicycle"
#
#                 xmlbox = obj.find('bndbox')
#                 b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#                      float(xmlbox.find('ymax').text))
#                 bb = Convert((w, h), b)
#                 obj = [targettype, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
#                 # print(obj)
#                 gt_dets.append(obj)
#
#     fl.close()
#     return gt_dets

def voc_ap(rec, prec):
    # 采用更为精确的逐点积分方法
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(dets_cls,
             xml_path_list,
             img_name_list,
             class_name,
             iou_thresh=0.5):
    """
    :param dets_cls:
    :param xml_path_list:
    :param img_name_list:
    :param class_name:
    :param iou_thresh:
    :return:
    """
    # 主函数，计算当前类别的recall和precision
    # #det_path检测结果txt文件，路径VOCdevkit/results/VOC20xx/Main/<comp_id>_det_test_aeroplane.txt。
    # 该文件格式：image_name1 type confidence xmin ymin xmax ymax  (图像1的第一个结果)
    #           image_name1 type confidence xmin ymin xmax ymax  (图像1的第二个结果)
    #           image_name2 type confidence xmin ymin xmax ymax  (图像2的第一个结果)
    #           ......
    # 每个结果占一行，检测到多少个BBox就有多少行，这里假设有20000个检测结果

    # det_path: Path to detections
    #     detpath.format(classname) should produce the detection results file.

    # anno_path: Path to annotations
    #     annopath.format(imagename) should be the xml annotations file. #xml 标注文件。

    # img_name_list: Text file containing the list of images, one image per line.
    # #数据集划分txt文件，
    # 路径VOCdevkit/VOC20xx/ImageSets/Main/test.txt这里假设测试图像1000张，那么该txt文件1000行。

    # class_name: Category name (duh) #种类的名字，即类别，假设类别2（一类目标+背景）。

    # cachedir: Directory for caching the annotations
    # #缓存标注的目录路径VOCdevkit/annotation_cache,图像数据只读文件，为了避免每次都要重新读数据集原始数据。

    # [ovthresh]: Overlap threshold (default = 0.5) #重叠的多少大小。
    # [use_07_metric]: Whether to use VOC07's 11 point AP computation
    #     (default False) #是否使用VOC07的AP计算方法，voc07是11个点采样。

    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    img_names_list = [x.strip() for x in img_name_list]

    # parse_rec函数读取当前图像标注文件，返回当前图像标注，存于recs字典（key是图像名，values是gt）
    print("=> Parsing XMLs...")
    recs = {}
    with tqdm(total=len(img_names_list)) as progress_bar:
        for img_name, xml_path in zip(img_names_list, xml_path_list):
            # recs[img_name] = parse_rec(annopath.format(img_name))
            parse_res = parse_xml(xml_path)
            if parse_res is None:
                continue
            recs[img_name] = parse_res
            progress_bar.update()

    # extract gt objects for this class #按类别获取标注文件，recall和precision都是针对不同类别而言的，AP也是对各个类别分别算的。
    class_recs = {}  # 当前类别的标注
    npos = 0  # npos标记的目标数量
    for img_name in img_names_list:
        img_gts = recs[img_name]
        img_cls_gts = [obj for obj in img_gts if obj['name'] == class_name]  # 过滤，只保留recs中指定类别的项，存为R。
        bbox = np.array([x["bbox"] for x in img_cls_gts])  # 抽取bbox
        difficult = np.array([x["difficult"] for x in img_cls_gts]).astype(np.bool)  # 如果数据集没有difficult,所有项都是0.

        det = [False] * len(img_cls_gts)  # len(img_cls_gts)就是当前类别的gt目标个数，det表示是否检测到，初始化为false。
        npos += sum(~difficult)  # 自增，非difficult样本数量，如果数据集没有difficult，npos数量就是gt数量。
        class_recs[img_name] = {'bbox': bbox,
                                'difficult': difficult,
                                'det': det}

    # read dets 读取检测结果
    splitlines = dets_cls  # 该文件格式：imagename1 type confidence xmin ymin xmax ymax
    # splitlines = [x.strip().split(' ') for x in detpath]  # 假设检测结果有20000个，则splitlines长度20000

    img_names = [x[0] for x in splitlines]  # 检测结果中的图像名，image_ids长度20000，但实际图像只有1000张，因为一张图像上可以有多个目标检测结果
    confidence = np.array([float(x[2]) for x in splitlines])  # 检测结果置信度
    BB_pred = np.array([[float(z) for z in x[3:]] for x in splitlines])  # 变为浮点型的bbox。

    npos = len(img_names)

    # sort by confidence 将20000各检测结果按置信度排序
    sorted_ind = np.argsort(-confidence)  # 对confidence的index根据值大小进行降序排列。
    sorted_scores = np.sort(-confidence)  # 降序排列。
    BB_pred = BB_pred[sorted_ind, :]  # 重排bbox，由大概率到小概率。
    img_names = [img_names[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    n_predict_dets = len(img_names)  # 注意这里是20000，不是1000
    tp = np.zeros(n_predict_dets)  # true positive，长度20000
    fp = np.zeros(n_predict_dets)  # false positive，长度20000
    for pred_det_i, img_name in enumerate(img_names):  # 遍历所有检测结果，因为已经排序，所以这里是从置信度最高到最低遍历
        img_cls_gts = class_recs[img_names[pred_det_i]]  # 当前检测结果所在图像的所有同类别gt
        bb = BB_pred[pred_det_i, :].astype(float)  # 当前检测结果bbox坐标
        max_iou = -np.inf
        BB_GT = img_cls_gts['bbox'].astype(float)  # 当前检测结果所在图像的所有同类别gt的bbox坐标

        if BB_GT.size > 0:
            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
            # intersection
            i_x_min = np.maximum(BB_GT[:, 0], bb[0])
            i_y_min = np.maximum(BB_GT[:, 1], bb[1])
            i_x_max = np.minimum(BB_GT[:, 2], bb[2])
            i_y_max = np.minimum(BB_GT[:, 3], bb[3])
            iw = np.maximum(i_x_max - i_x_min + 1.0, 0.0)
            ih = np.maximum(i_y_max - i_y_min + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) +
                   (BB_GT[:, 2] - BB_GT[:, 0] + 1.0) *
                   (BB_GT[:, 3] - BB_GT[:, 1] + 1.0) - inters)

            ious = inters / uni
            max_iou = np.max(ious)  # 最大重合率
            j_max = np.argmax(ious)  # 最大重合率对应的gt, 返回最大索引数

        if max_iou > iou_thresh:  # 如果当前检测结果与真实标注最大重合率满足阈值
            # if not img_cls_gts['difficult'][j_max]:
            if not img_cls_gts['det'][j_max]:
                tp[pred_det_i] = 1.0  # 正检数目+1
                img_cls_gts['det'][j_max] = True  # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
            else:  # 相反，认为检测到一个虚警
                fp[pred_det_i] = 1.0
        else:  # 不满足阈值，肯定是虚警
            fp[pred_det_i] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)  # 积分图，在当前节点前的虚警数量，fp长度
    tp = np.cumsum(tp)  # 积分图，在当前节点前的正检数量
    recall = tp / float(npos)  # 召回率，长度20000，从0到1

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth 准确率，长度20000，长度20000，从1到0
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
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
    xml_paths = []
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
            # xml_paths.append(xml_path)
    img_paths.sort()
    xml_paths = [img_path.replace("JPEGImages", "Annotations")
                     .replace(".jpg", ".xml")
                 for img_path in img_paths]
    logger.info("total {:d} samples to be evaluated.".format(len(img_paths)))

    ## ----- Get predicts
    print("=> Predicting...")
    dets_pred_all = [["img_name", "cls_name", "confidence", 0, 0, 0, 0]]
    img_name_list = []
    with tqdm(total=len(img_paths)) as progress_bar:
        for img_path, xml_path in zip(img_paths, xml_paths):
            ## ----- Get image name
            img_name = get_img_name(img_path)
            img_name_list.append(img_name)

            ## ----- Inference
            outputs, img_info = inference(img_path, net, opt)
            if outputs is None:
                continue

            with torch.no_grad():
                dets_pred = outputs[0]
                if dets_pred is None:
                    continue
                dets_pred = dets_pred.cpu().numpy()

            ## -----
            dets_pred = []
            for det in dets_pred:
                # print(det)
                x1, y1, x2, y2, score_1, score_2, cls_id = det

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
                confidence = score_1 * score_2
                det_pred = [img_name, cls_name, confidence, x1, y1, x2, y2]
                dets_pred.append(det_pred)
            if len(dets_pred) > 0:
                dets_pred_all = np.vstack((dets_pred_all, dets_pred))
            progress_bar.update()

    ## ----- Calculate APs for each object class
    dets_pred_all = np.delete(dets_pred_all, 0, axis=0)
    APs = []
    for cls_name in opt.class_names:
        print("=> processing {:s}...".format(cls_name))
        dets_cls = [obj for obj in dets_pred_all if obj[1] == cls_name]
        if len(dets_cls) == 0:
            cls_ap = 0
        else:
            cls_ap = voc_eval(dets_cls,
                              xml_paths,
                              img_name_list,
                              cls_name,
                              opt.iou_thresh)
        APs.append(cls_ap)
    print("APs: ", APs)


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
    exp.n_classes = len(exp.class_names)
    opt.num_classes = len(exp.class_names)
    logger.info("Number of classes: {:d}".format(exp.n_classes))

    ## ----- run the tracking
    evaluate(exp, opt)
