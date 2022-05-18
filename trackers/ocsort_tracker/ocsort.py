"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import torch
from loguru import logger

from .association import *


def k_previous_obs(observations_dict, cur_age, k):
    """
    @param observations_dict:
    @param cur_age:
    @param k:
    """
    if len(observations_dict) == 0:
        return [-1, -1, -1, -1, -1]

    ## ----- if found observation from k previous time steps
    for i in range(k):
        dt = k - i  # 3, 2, 1
        pre_age = cur_age - dt  # -3, -2, -1
        if pre_age in observations_dict:
            return observations_dict[cur_age - dt]

    ## ----- if k previous observations do not exist
    ## then, use max-aged previous observation: the latest observation
    max_age = max(observations_dict.keys())
    return observations_dict[max_age]


def convert_bbox_to_z(bbox):
    """
    bbox -> z = Hx
    convert to Kalman state: act as measure matrix H of Kalman
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = bbox[0] + w / 2.0  # center x
    y = bbox[1] + h / 2.0  # center y

    s = w * h  # scale is just area
    r = w / float(h + 1e-6)

    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form observation: [x,y,s,r]
    and returns it in the form [x1,y1,x2,y2]
    where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    center_x, center_y = x[0], x[1]
    x1, y1 = center_x - w * 0.5, center_y - h * 0.5
    x2, y2 = center_x + w * 0.5, center_y + h * 0.5

    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


def get_velocity_direction(bbox1, bbox2):
    """
    @param bbox1
    @param bbox2
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state
    of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.
        @param bbox
        @param delta_t
        @param orig
        """
        # define constant velocity model
        if not orig:
            from . import oc_kalmanfilter
            self.kf = oc_kalmanfilter.KalmanFilterNew(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter
            # 此模型没有控制向量输入dim_u=0
            self.kf = KalmanFilter(dim_x=7, dim_z=4, dim_u=0)

        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])  # constant velocity model
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.R[2:, 2:] *= 10.0

        # states: z: center_x, center_y, s(area), r(aspect ratio)
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_last_update = 0  # 距离上次更新的时间(帧数)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []

        ## what's the difference?
        self.hits = 0
        self.hit_streak = 0

        self.age = 0  # what dose age mean?

        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for [non-observation status], 
        the same for the return of function k_previous_obs. 
        It is ugly and I do not like it. 
        But to support generate observation array in a 
        fast and unified way, 
        which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations_dict = dict()  # key: age
        self.history_observations = []
        self.vel_dir = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        @param bbox: x1, y1, x2, y2
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # if previous observation exist
                previous_box = None

                for i in range(self.delta_t):
                    dt = self.delta_t - i  # eg: 3, 2, 1
                    if self.age - dt in self.observations_dict:  # from little age to large age
                        previous_box = self.observations_dict[self.age - dt]  # -1, 0, 1
                        break

                if previous_box is None:
                    previous_box = self.last_observation

                """
                Estimate the track velocity direction
                with observations delta t steps away
                a 2d vector
                """
                self.vel_dir = get_velocity_direction(previous_box, bbox)

            """
            Insert new observations. 
            This is a ugly way to maintain both self.observations
            and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations_dict[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_last_update = 0
            self.history = []

            self.hits += 1
            self.hit_streak += 1  # 连胜

            self.kf.update(convert_bbox_to_z(bbox))

        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and
        returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1  # 每predict一次, 生命周期+1

        if self.time_since_last_update > 0:  # 如果丢失了一次更新, 连胜(连续跟踪)被终止
            self.hit_streak = 0

        self.time_since_last_update += 1  # 每predict一次, 未更新时间(帧数)+1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


from collections import defaultdict


class MCTrackBase(object):
    _id_dict = defaultdict(int)  # the MCBaseTrack class owns this dict

    def __init__(self):
        pass

    @staticmethod
    def init_id_dict(n_classes):
        """
        Initiate _count for all object classes
        @param n_classes:
        """
        for cls_id in range(n_classes):
            MCTrackBase._id_dict[cls_id] = 0
        logger.info("Id dict initialized.")

    @staticmethod
    def next_id(cls_id):
        """
        :param cls_id:
        :return:
        """
        MCTrackBase._id_dict[cls_id] += 1
        return MCTrackBase._id_dict[cls_id]

    @staticmethod
    def reset_cls_track_id(cls_id):
        """
        :param cls_id:
        :return:
        """
        MCTrackBase._id_dict[cls_id] = 0

    @staticmethod
    def reset_track_id(n_classes):
        """
        @param n_classes:
        :return:
        """
        MCTrackBase.init_id_dict()


class MCKalmanTrack(MCTrackBase):
    """
    This class represents the internal state
    of individual tracked objects observed as bbox.
    """

    def __init__(self, bbox, cls_id, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.
        @param bbox:
        @param cls_id: class id
        @param delta_t:
        @param orig:
        """
        super(MCKalmanTrack, self).__init__()

        ## ----- record
        self.cls_id = cls_id

        self.x1y1x2y2, self.tlwh = None, None

        ## ----- update track id
        self.track_id = MCTrackBase.next_id(self.cls_id)

        # define constant velocity model
        if not orig:
            from . import oc_kalmanfilter
            self.kf = oc_kalmanfilter.KalmanFilterNew(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter
            # 此模型没有控制向量输入dim_u=0
            self.kf = KalmanFilter(dim_x=7, dim_z=4, dim_u=0)

        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])  # constant velocity model
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.R[2:, 2:] *= 10.0

        # states: z: center_x, center_y, s(area), r(aspect ratio)
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_last_update = 0  # 距离上次更新的时间(帧数)

        self.history = []

        ## what's the difference?
        self.hits = 0
        self.hit_streak = 0

        self.age = 0  # what dose age mean?

        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for [non-observation status], 
        the same for the return of function k_previous_obs. 
        It is ugly and I do not like it. 
        But to support generate observation array in a 
        fast and unified way, 
        which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations_dict = dict()  # key: age
        self.history_observations = []
        self.vel_dir = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        @param bbox: x1, y1, x2, y2
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # if previous observation exist
                previous_box = None

                for i in range(self.delta_t):
                    dt = self.delta_t - i  # eg: 3, 2, 1
                    if self.age - dt in self.observations_dict:  # from little age to large age
                        previous_box = self.observations_dict[self.age - dt]  # -1, 0, 1
                        break

                if previous_box is None:
                    previous_box = self.last_observation

                """
                Estimate the track velocity direction
                with observations delta t steps away
                a 2d vector
                """
                self.vel_dir = get_velocity_direction(previous_box, bbox)

            """
            Insert new observations. 
            This is a ugly way to maintain both self.observations
            and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations_dict[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_last_update = 0
            self.history = []

            self.hits += 1
            self.hit_streak += 1  # 连胜

            self.kf.update(convert_bbox_to_z(bbox))

        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and
        returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1  # 每predict一次, 生命周期+1

        if self.time_since_last_update > 0:  # 如果丢失了一次更新, 连胜(连续跟踪)被终止
            self.hit_streak = 0

        self.time_since_last_update += 1  # 每predict一次, 未更新时间(帧数)+1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        """
        :param tlbr:
        :return:
        """
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        self.x1y1x2y2 = np.squeeze(convert_x_to_bbox(self.kf.x))
        self.get_tlwh()
        return self.x1y1x2y2

    # @jit(nopython=True)
    def get_tlwh(self):
        """
        :return tlwh
        """
        if self.tlwh is None:
            self.tlwh = MCKalmanTrack.tlbr_to_tlwh(self.x1y1x2y2)
        return self.tlwh


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist
}


class MCOCSort(object):
    def __init__(self,
                 class_names,
                 det_thresh,
                 low_det_thresh=0.1,
                 iou_thresh=0.3,
                 max_age=30,
                 min_hits=3,
                 delta_t=3,
                 associate_func="iou",
                 vel_dir_weight=0.2):
        """
        Sets key parameters for SORT
        @param n_classes: number of object classes
        """
        self.class_names = class_names
        self.n_classes = len(self.class_names)
        logger.info("number of object classes: {:d}".format(self.n_classes))

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_thresh
        logger.info("iou_threshold: {:.3f}".format(self.iou_threshold))

        self.tracks_dict = {}
        for i in range(self.n_classes):
            self.tracks_dict[i] = []

        self.frame_id = 0

        self.det_thresh = det_thresh
        logger.info("det_thresh: {:.3f}".format(self.det_thresh))
        self.low_det_thresh = low_det_thresh
        logger.info("low_det_thresh: {:.3f}".format(self.low_det_thresh))

        self.delta_t = delta_t

        self.associate_func = ASSO_FUNCS[associate_func]  # iou functions
        logger.info("using IOU function: {:s}.".format(associate_func))

        self.vel_dir_weight = vel_dir_weight
        logger.info("vel_dir_weight: ", self.vel_dir_weight)

        ## ----- Initialize the id dict
        # KalmanBoxTracker.count = 0
        MCKalmanTrack.init_id_dict(self.n_classes)

    def update_frame(self, dets, img_size, net_size):
        """
        @param dets: - a numpy array of detections in the format [[x1,y1,x2,y2,score],
        [x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections(use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        @param img_size: img_h, img_w
        @param net_size: net_h, net_w
        """
        if dets is None:
            return np.empty((0, 5))

        ## ----- gpu ——> cpu
        with torch.no_grad():
            dets = dets.cpu().numpy()

        self.frame_id += 1
        if self.frame_id == 1:
            MCKalmanTrack.init_id_dict(self.n_classes)

        ## ----- the track dict to be returned
        ret_dict = defaultdict(list)

        ## ----- image width, height and net width, height
        img_h, img_w = img_size
        net_h, net_w = net_size
        scale = min(net_h / float(img_h), net_w / float(img_w))

        #################### Even: Start MCMOT
        ## ----- Fill box dict and score dict
        boxes_dict = defaultdict(list)
        scores_dict = defaultdict(list)

        for det in dets:
            if det.size == 7:
                x1, y1, x2, y2, score1, score2, cls_id = det  # 7
                score = score1 * score2
            elif det.size == 6:
                x1, y1, x2, y2, score, cls_id = det  # 6
            else:
                logger.error("invalid dets' dimensions: should b 6 or 7.")
                exit(-1)

            box = np.array([x1, y1, x2, y2])
            box /= scale  # convert box to image size

            boxes_dict[int(cls_id)].append(box)
            scores_dict[int(cls_id)].append(score)

        ## ---------- Process each object class
        for cls_id in range(self.n_classes):
            ## ----- class boxes
            bboxes = boxes_dict[cls_id]
            bboxes = np.array(bboxes)

            ## ----- class scores
            scores = scores_dict[cls_id]
            scores = np.array(scores)

            ## ----- build cls_dets
            scores_2d = np.expand_dims(scores, axis=1)
            if bboxes.shape[0] == 0:
                bboxes = np.empty((0, 4), dtype=float)
                scores_2d = np.empty((0, 1), dtype=float)
            dets = np.concatenate((bboxes, scores_2d), axis=1)

            inds_low = scores > self.low_det_thresh
            inds_high = scores < self.det_thresh
            inds_2nd = np.logical_and(inds_low, inds_high)
            remain_inds = scores > self.det_thresh

            bboxes_1st = bboxes[remain_inds]
            bboxes_2nd = bboxes[inds_2nd]

            scores_1st = scores[remain_inds]
            scores_2nd = scores[inds_2nd]

            ## ----- Build dets for the object class
            if remain_inds.size > 0:
                dets = dets[remain_inds]
            else:
                dets = np.empty((0, 5), dtype=float)

            ## ----- object class tracks
            tracks = self.tracks_dict[cls_id]

            # get predicted locations from existing trackers.
            trks = np.zeros((len(tracks), 5))
            to_del = []
            for i, trk in enumerate(trks):
                ## ----- prediction
                pos = tracks[i].predict()[0]

                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(i)

            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for i in reversed(to_del):
                tracks.pop(i)

            ## ----- Compute velocity directions
            velocities = np.array([trk.vel_dir  # velocity direction
                                   if trk.vel_dir is not None else np.array((0, 0))
                                   for trk in tracks])

            ## ----- Get thee last observations for current tracks
            last_boxes = np.array([trk.last_observation for trk in tracks])

            ## ----- Get current tracks' previous observations
            k_observations = [k_previous_obs(trk.observations_dict, trk.age, self.delta_t)
                              for trk in tracks]
            k_observations = np.array(k_observations)

            """
            First round of association
            using high confidence dets and existed trks
            """

            matched, unmatched_dets, unmatched_trks = associate(dets,
                                                                k_observations,
                                                                trks,
                                                                velocities,
                                                                self.iou_threshold,
                                                                self.vel_dir_weight)
            for m in matched:
                tracks[m[1]].update(dets[m[0], :])

            """
            Second round of association by OCR
            """
            ## ----- 处理第一轮匹配中未匹配的dets和trks
            if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
                left_dets = dets[unmatched_dets]
                left_trks = last_boxes[unmatched_trks]
                iou_left = self.associate_func(left_dets, left_trks)  # calculate iou
                iou_left = np.array(iou_left)

                if iou_left.max() > self.iou_threshold:
                    """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                    """
                    rematched_inds = linear_assignment(-iou_left)

                    to_remove_from_unmatch_det_inds = []
                    to_remove_from_unmatch_trk_inds = []
                    for m in rematched_inds:
                        det_idx, trk_idx = unmatched_dets[m[0]], unmatched_trks[m[1]]

                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue

                        tracks[trk_idx].update(dets[det_idx, :])

                        # record matched det inds and trk inds
                        to_remove_from_unmatch_det_inds.append(det_idx)
                        to_remove_from_unmatch_trk_inds.append(trk_idx)

                    unmatched_dets = np.setdiff1d(unmatched_dets,
                                                  np.array(to_remove_from_unmatch_det_inds))
                    unmatched_trks = np.setdiff1d(unmatched_trks,
                                                  np.array(to_remove_from_unmatch_trk_inds))

            for m in unmatched_trks:
                tracks[m].update(None)

            ## ---------- new track and initialization
            # build new trackers for unmatched detections
            for i in unmatched_dets:
                trk = MCKalmanTrack(bbox=dets[i, :],
                                    cls_id=cls_id,
                                    delta_t=self.delta_t)
                tracks.append(trk)

            ## ----- determine the tracks to be returned of this frame
            i = len(tracks)
            for trk in reversed(tracks):
                if trk.last_observation.sum() < 0:
                    d = trk.get_state()[0]  # get current state, not previous observation
                else:
                    """
                    this is optional to [use the recent observation]
                    or the kalman filter prediction,
                    we didn't notice significant difference here
                    """
                    d = trk.last_observation[:4]  # use last observation

                if (trk.time_since_last_update < 1) \
                        and (trk.hit_streak >= self.min_hits
                             or self.frame_id <= self.min_hits):
                    # +1 as MOT benchmark requires positive

                    # ## ----- only return bbox and track id
                    # ret_dict[cls_id].append(np.concatenate((d, [trk.track_id + 1])).reshape(1, -1))

                    # ## ------ return the track object
                    ret_dict[cls_id].append(trk)

                i -= 1

                # remove the dead track
                if trk.time_since_last_update > self.max_age:
                    tracks.pop(i)

            # if len(ret_dict[cls_id]) > 0:  # turn 2d list to 2d array
            #     ret_dict[cls_id] = np.concatenate(ret_dict[cls_id])

        for k, v in ret_dict.items():
            if len(v) == 0:
                v = np.empty((0, 5))

        return ret_dict


class OCSort(object):
    def __init__(self,
                 det_thresh,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 asso_func="iou",
                 inertia=0.2):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_id = 0
        self.det_thresh = det_thresh
        logger.info("det_thresh: {:.3f}".format(self.det_thresh))
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]  # iou functions
        logger.info("using IOU function: {:s}.".format(asso_func))
        self.inertia = inertia
        KalmanBoxTracker.count = 0

    def update_frame(self, dets, img_size, net_size):
        """
        @param dets: - a numpy array of detections in the format [[x1,y1,x2,y2,score],
        [x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections(use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        @param img_size: img_h, img_w
        @param net_size: net_h, net_w
        """
        if dets is None:
            return np.empty((0, 5))

        self.frame_id += 1

        # post_process detections
        if dets.shape[1] == 5:
            scores = dets[:, 4]
            bboxes = dets[:, :4]
        else:
            dets = dets.cpu().numpy()
            scores = dets[:, 4] * dets[:, 5]
            bboxes = dets[:, :4]  # x1y1x2y2

        img_h, img_w = img_size
        net_h, net_w = net_size
        scale = min(net_h / float(img_h), net_w / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]  # high confidence detections

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            ## ----- prediction
            pos = self.tracks[t].predict()[0]

            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        velocities = np.array([trk.vel_dir  # velocity direction
                               if trk.vel_dir is not None else np.array((0, 0))
                               for trk in self.tracks])
        last_boxes = np.array([trk.last_observation for trk in self.tracks])
        k_observations = [k_previous_obs(trk.observations_dict, trk.age, self.delta_t)
                          for trk in self.tracks]
        k_observations = np.array(k_observations)

        """
        First round of association
        using high confidence dets and existed trks
        """
        matched, unmatched_dets, unmatched_trks = associate(dets,
                                                            trks,
                                                            self.iou_threshold,
                                                            velocities,
                                                            k_observations,
                                                            self.inertia)
        for m in matched:
            self.tracks[m[1]].update(dets[m[0], :])

        """
        Second round of association by OCR
        """
        ## ----- 处理第一轮匹配中未匹配的dets和trks
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)  # calculate iou
            iou_left = np.array(iou_left)

            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)

                to_remove_from_unmatch_det_indices = []
                to_remove_from_unmatch_trk_indices = []
                for m in rematched_indices:
                    det_idx, trk_idx = unmatched_dets[m[0]], unmatched_trks[m[1]]

                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue

                    self.tracks[trk_idx].update(dets[det_idx, :])

                    # record matched det inds and trk inds
                    to_remove_from_unmatch_det_indices.append(det_idx)
                    to_remove_from_unmatch_trk_indices.append(trk_idx)

                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_from_unmatch_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_from_unmatch_trk_indices))

        for m in unmatched_trks:
            self.tracks[m].update(None)

        ## ---------- create and initialized
        # new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            self.tracks.append(trk)

        ## ----- determine the tracks to be returned of this frame
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]  # get current state, not previous observation
            else:
                """
                this is optional to [use the recent observation]
                or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]  # use last observation

            if (trk.time_since_last_update < 1) \
                    and (trk.hit_streak >= self.min_hits
                         or self.frame_id <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1

            # remove the dead track
            if trk.time_since_last_update > self.max_age:
                self.tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        """
        @param dets:
        @param cates:
        @param scores:
        """
        self.frame_id += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh

        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict()[0]
            cat = self.tracks[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        velocities = np.array([trk.vel_dir if trk.vel_dir is not None else np.array((0, 0)) for trk in self.tracks])
        last_boxes = np.array([trk.last_observation for trk in self.tracks])
        k_observations = np.array([k_previous_obs(trk.observations_dict, trk.age, self.delta_t) for trk in self.tracks])

        matched, unmatched_dets, unmatched_trks = associate_kitti \
            (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)

        for m in matched:
            self.tracks[m[1]].update(dets[m[0], :])

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:, 4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                        """
                            For some datasets, such as KITTI, there are different categories,
                            we have to avoid associate them together.
                        """
                        cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                        continue
                    self.tracks[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trk.cate = cates[i]
            self.tracks.append(trk)
        i = len(self.tracks)

        for trk in reversed(self.tracks):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if trk.time_since_last_update < 1:
                if (self.frame_id <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id + 1], [trk.cate], [0])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id + 1], [trk.cate],
                                                    [-(prev_i + 1)]))).reshape(1, -1))
            i -= 1
            if trk.time_since_last_update > self.max_age:
                self.tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 7))
