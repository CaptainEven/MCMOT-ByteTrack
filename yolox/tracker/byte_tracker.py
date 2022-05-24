# encoding=utf-8

from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

from trackers.ocsort_tracker.association import associate
from trackers.ocsort_tracker.ocsort import k_previous_obs
from yolox.tracker import matching
from .basetrack import BaseTrack, MCBaseTrack, TrackState
from .kalman_filter import KalmanFilter


class MCTrackFeat(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, cls_id, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param cls_id:
        :param buff_size:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        ## ----- init is_activated to be False
        self.is_activated = False

        self.score = score
        self.track_len = 0

        # fusion factor
        self.alpha = 0.9

        ## ----- features
        self.smooth_feat = temp_feat
        self.update_features(temp_feat)

        # buffered features
        self.features = deque([], maxlen=buff_size)

    def reset_track_id(self):
        """
        :return:
        """
        self.reset_track_id(self.cls_id)

    def update_features(self, feat):
        """
        :param feat:
        :return:
        """
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """
        :return:
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        """
        :param tracks:
        :return:
        """
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        Start a new track: the initial activation
        :param kalman_filter:
        :param frame_id:
        :return:
        """
        self.kalman_filter = kalman_filter  # assign a filter to each track?

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        :param new_track:
        :param frame_id:
        :param new_id:
        :return:
        """
        # kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)

        self.track_len = 0
        self.frame_id = frame_id

        # set flag 'tracked'
        self.state = TrackState.Tracked

        # set flag 'activated'
        self.is_activated = True

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        # set flag 'tracked'
        self.state = TrackState.Tracked

        # set flag 'activated'
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `
        (top left x, top left y, width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] * 0.5

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        :return:
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]

        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        :param tlwh:
        :return:
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """
        :return:
        """
        return self.tlwh_to_xyah(self.tlwh)

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

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        """
        :param tlwh:
        :return:
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """
        返回一个对象的 string 格式。
        :return:
        """
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


# Multi-class Track class with embedding(feature vector)
class MCTrackEmb(MCBaseTrack):
    def __init__(self, tlwh, score, feat, cls_id, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param feat:
        :param cls_id:
        :param buff_size:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)

        self.KF = None
        self.mean, self.covariance = None, None

        ## ----- init is_activated to be False
        self.is_activated = False

        self.score = score
        self.track_len = 0

        ## ----- features
        self.smooth_feat = None
        self.update_features(feat)

        # buffered features
        self.features = deque([], maxlen=buff_size)

        # fusion factor
        self.alpha = 0.9

    def reset_track_id(self):
        """
        :return:
        """
        self.reset_track_id(self.cls_id)

    def update_features(self, feat):
        """
        :param feat:
        :return:
        """
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """
        :return:
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.KF.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        """
        :param tracks:
        :return:
        """
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        Start a new track-let: the initial activation
        :param kalman_filter:
        :param frame_id:
        :return:
        """
        self.KF = kalman_filter

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.KF.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0

        ## ----- Set track states
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        :param new_track:
        :param frame_id:
        :param new_id:
        :return:
        """
        # kalman update
        self.mean, self.covariance = self.KF.update(self.mean,
                                                    self.covariance,
                                                    self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)

        self.track_len = 0
        self.frame_id = frame_id

        ## ----- Update states
        self.state = TrackState.Tracked
        self.is_activated = True
        ## -----

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.KF.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        ## ----- Update the states
        self.state = TrackState.Tracked
        self.is_activated = True
        ## -----

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def box(self):
        return self.box

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """
        :return:
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        """
        :param tlbr:
        :return:
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        """
        :param tlwh:
        :return:
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """
        the str
        :return:
        """
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


# Multi-class Track class without embedding(feature vector)
class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls_id, delta_t=3):
        """
        :param tlwh:
        :param score:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        ## ----- init is_activated to be False
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.age = 0
        self.delta_t = delta_t

        ## ----- record history observations: bbox
        self.observations_dict = dict()  # key: age

        ## ----- record the last observation: bbox
        self.last_observation = np.array([-1, -1, -1, -1, -1])

        ## ----- record velocity direction
        self.vel_dir = None

    def reset_track_id(self):
        """
        :return:
        """
        self.reset_track_id(self.cls_id)

    def predict(self):
        """
        :return:
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        """
        :param tracks:
        :return:
        """
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

                # 每predict一次, 生命周期数+1
                tracks[i].age += 1

    def activate(self, kalman_filter, frame_id):
        """
        Start a new track-let: the initial activation
        :param kalman_filter:
        :param frame_id:
        :return:
        """
        self.kalman_filter = kalman_filter

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        :param new_track:
        :param frame_id:
        :param new_id:
        :return:
        """
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.track_len = 0
        self.frame_id = frame_id
        self.score = new_track.score

        ## ----- Update states
        self.state = TrackState.Tracked
        self.is_activated = True
        ## -----

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, using_delta_t=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type using_delta_t: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        self.score = new_track.score
        new_tlwh = new_track.tlwh
        bbox = self.tlwh2tlbr(new_tlwh)

        """
        Estimate the track velocity direction with observations delta_t steps away
        """
        if using_delta_t:
            if self.last_observation.sum() >= 0:  # if previous observation exist
                previous_box = None

                for i in range(self.delta_t):
                    dt = self.delta_t - i  # eg: 3, 2, 1
                    if self.age - dt in self.observations_dict:  # from little age to large age
                        previous_box = self.observations_dict[self.age - dt]  # -1, 0, 1
                        break

                if previous_box is None:
                    previous_box = self.last_observation

                self.vel_dir = self.get_velocity_direction(previous_box, bbox)
        else:

            """
            Using last observation to calculate vel_dir
            vel_dir: a 2d vector
            """
            if self.last_observation.sum() >= 0:
                self.vel_dir = self.get_velocity_direction(self.last_observation, bbox)
            else:
                self.vel_dir = np.array([0.0, 0.0], dtype=np.float64)
        # print("vel_dir: {:.3f}, {:.3f}".format(self.vel_dir[0], self.vel_dir[1]))

        ## update last observations
        self.observations_dict[self.age] = bbox
        self.last_observation = bbox

        self.mean, self.covariance = \
            self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        ## ----- Update the states
        self.state = TrackState.Tracked
        self.is_activated = True
        ## -----

    @staticmethod
    def get_velocity_direction(bbox1, bbox2):
        """
        @param bbox1
        @param bbox2
        """
        if (bbox2 == bbox1).all():
            return np.array([0.0, 0.0], dtype=np.float64)

        dx1, dy1 = (bbox1[0] + bbox1[2]) * 0.5, (bbox1[1] + bbox1[3]) * 0.5
        dx2, dy2 = (bbox2[0] + bbox2[2]) * 0.5, (bbox2[1] + bbox2[3]) * 0.5
        speed = np.array([dy2 - dy1, dx2 - dx1])  # dy, dx
        # norm = np.sqrt((dy2 - dy1) ** 2 + (dx2 - dx1) ** 2) + 1e-6
        norm = np.linalg.norm(speed, ord=2)
        return speed / (norm + 1e-8)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """
        Get current position in bounding box format
        `(top left x, top left y, width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] * 0.5

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh2tlbr(tlwh):
        """
        :param tlwh:
        """
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """
        :return:
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        """
        :param tlbr:
        :return:
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        """
        :param tlwh:
        :return:
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """
        返回一个对象的 string 格式。
        :return:
        """
        return "OT_({}-{})_({}-{})" \
            .format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        """
        :param tlwh:
        :param score:
        """
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        """
        :return:
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """
        :param stracks:
        :return:
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        Start a new tracklet
        :param kalman_filter:
        :param frame_id:
        :return:
        """
        self.kalman_filter = kalman_filter

        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        :param new_track:
        :param frame_id:
        :param new_id:
        :return:
        """
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.tracklet_len = 0
        self.frame_id = frame_id

        self.state = TrackState.Tracked
        self.is_activated = True

        if new_id:
            self.track_id = self.next_id()

        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """
        :return:
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        """
        :param tlbr:
        :return:
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        """
        :param tlwh:
        :return:
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """
        :return:
        """
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class ByteTracker(object):
    def __init__(self, opt, frame_rate=30, delta_t=3):
        """
        :param opt:
        :param frame_rate:
        :param delta_t:
        """
        self.frame_id = 0
        self.opt = opt
        print("opt:\n", self.opt)

        # self.det_thresh = args.track_thresh
        self.low_det_thresh = 0.1
        self.high_det_thresh = self.opt.track_thresh  # 0.5
        self.high_match_thresh = self.opt.match_thresh  # 0.8
        self.low_match_thresh = 0.5
        self.unconfirmed_match_thresh = 0.7
        self.new_track_thresh = self.high_det_thresh + 0.1  # 0.6
        self.iou_threshold = 0.3
        self.vel_dir_weight = 0.2

        print("Tracker's low det thresh: ", self.low_det_thresh)
        print("Tracker's high det thresh: ", self.high_det_thresh)
        print("Tracker's high match thresh: ", self.high_match_thresh)
        print("Tracker's low match thresh: ", self.low_match_thresh)
        print("Tracker's unconfirmed match thresh: ", self.unconfirmed_match_thresh)
        print("Tracker's new track thresh: ", self.new_track_thresh)

        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        print("Tracker's buffer size: ", self.buffer_size)

        self.kalman_filter = KalmanFilter()

        # Get number of tracking object classes
        self.class_names = opt.class_names
        self.n_classes = opt.n_classes

        # Define track lists for single object class
        self.tracked_tracks = []  # type: list[Track]
        self.lost_tracks = []  # type: list[Track]
        self.removed_tracks = []  # type: list[Track]

        # Define tracks dict for multi-class objects
        self.tracked_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])
        self.lost_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])
        self.removed_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])

        self.tracks = []
        self.delta_t = delta_t

    def update_byte_enhance(self, dets, img_size, net_size):
        """
        enhanced byte track
        :param dets:
        :param img_size:
        :param net_size:
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        ## ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_id_dict(self.n_classes)
        ## -----

        ## ----- image width, height and net width, height
        img_h, img_w = img_size
        net_h, net_w = net_size
        scale = min(net_h / float(img_h), net_w / float(img_w))

        ## ----- gpu ——> cpu
        with torch.no_grad():
            dets = dets.cpu().numpy()

        ## ----- The current frame 8 tracking states recording
        unconfirmed_tracks_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        retrieve_tracks_dict = defaultdict(list)  # re-find
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

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

            remain_inds = scores > self.high_det_thresh
            inds_low = scores > self.low_det_thresh
            inds_high = scores < self.high_det_thresh

            ## class second indices
            inds_2nd = np.logical_and(inds_low, inds_high)

            bboxes_1st = bboxes[remain_inds]
            bboxes_2nd = bboxes[inds_2nd]

            scores_1st = scores[remain_inds]
            scores_2nd = scores[inds_2nd]

            if len(bboxes_1st) > 0:
                '''Build Tracks from Detections'''
                detections = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                              (tlbr, s) in zip(bboxes_1st, scores_1st)]

                scores_1st_ = np.expand_dims(scores_1st, axis=1)
                if bboxes_1st.shape[0] == 0:
                    bboxes_1st = np.empty((0, 4), dtype=float)
                    scores_1st_ = np.empty((0, 1), dtype=float)
                dets = np.concatenate((bboxes_1st, scores_1st_), axis=1)
                # print(dets.shape)
            else:
                detections = []
            # print(detections)

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_tracks_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with high score detection boxes'''
            ## ----- build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # ---------- Predict the current location with KF
            # self.tracks = tracked_tracks_dict[cls_id]
            # MCTrack.multi_predict(tracked_tracks_dict[cls_id])  # only predict tracked tracks

            self.tracks = track_pool_dict[cls_id]
            MCTrack.multi_predict(track_pool_dict[cls_id])
            # ----------

            ## ---------- TODO: using vel_dir enhanced matching...
            ## ----- build dets(x1y1x2y2score) and trks(x1y1x2y2score) for matching
            trks = np.zeros((len(self.tracks), 5))
            to_del = []
            for i, track in enumerate(self.tracks):
                x1, y1, x2, y2 = track.tlbr
                trks[i] = [x1, y1, x2, y2, track.score]
                if np.any(np.isnan([x1, y1, x2, y2])):
                    to_del.append(i)

            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for i in reversed(to_del):
                self.tracks.pop(i)

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
            matches, u_detection, u_track = associate(dets,
                                                      k_observations,
                                                      trks,
                                                      velocities,
                                                      self.iou_threshold,
                                                      self.vel_dir_weight)

            # for m in matches:
            #     self.tracks[m[1]].update(detections[m[0], :], self.frame_id)

            # ## ----- Matching with Hungarian Algorithm
            # dists = matching.iou_distance(track_pool_dict[cls_id], detections)
            # dists = matching.fuse_score(dists, detections)
            # matches, u_track, u_detection = matching.linear_assignment(dists,
            #                                                            thresh=self.high_match_thresh)

            # --- process matched pairs between track pool and current frame detection
            for i_track, i_det in matches:
                track = track_pool_dict[cls_id][i_track]
                det = detections[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    retrieve_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with low score detection boxes'''
            # association the un-track to the low score detections
            if len(bboxes_2nd) > 0:
                '''Detections'''
                cls_detections_2nd = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                                      (tlbr, s) in zip(bboxes_2nd, scores_2nd)]
            else:
                cls_detections_2nd = []
            # print(cls_detections_second)

            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]

            dists = matching.iou_distance(r_tracked_tracks, cls_detections_2nd)
            matches, u_track, u_detection_2nd = matching.linear_assignment(dists,
                                                                           thresh=self.low_match_thresh)  # thresh=0.5

            for i_track, i_det in matches:
                track = r_tracked_tracks[i_track]
                det = cls_detections_2nd[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    retrieve_tracks_dict[cls_id].append(track)

            # process unmatched tracks for two rounds
            for i_track in u_track:
                track = r_tracked_tracks[i_track]
                if not track.state == TrackState.Lost:
                    # mark unmatched track as lost track
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            # current frame's unmatched detection
            detections = [detections[i] for i in u_detection]

            # iou matching
            dists = matching.iou_distance(unconfirmed_tracks_dict[cls_id], detections)

            if not self.opt.mot20:
                dists = matching.fuse_score(dists, detections)

            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists,
                                                                             thresh=self.unconfirmed_match_thresh)  # 0.7

            for i_track, i_det in matches:
                track = unconfirmed_tracks_dict[cls_id][i_track]
                det = detections[i_det]
                track.update(det, self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_tracks_dict[cls_id][i_track])

            for i_track in u_unconfirmed:  # process unconfirmed tracks
                track = unconfirmed_tracks_dict[cls_id][i_track]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """Step 4: Init new tracks"""
            for i_new in u_detection:  # current frame's unmatched detection
                track = detections[i_new]
                if track.score < self.new_track_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                # if fr_id > 1, tracked but not activated
                track.activate(self.kalman_filter, self.frame_id)

                # activated_tracks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           retrieve_tracks_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

        ## ---------- Return final online targets of the frame
        return output_tracks_dict
        #################### MCMOT end

    def update_mcmot_byte(self, dets, img_size, net_size):
        """
        Original byte tracking
        :param dets:
        :param img_size:
        :param net_size:
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        ## ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_id_dict(self.n_classes)
        ## -----

        ## ----- image width, height and net width, height
        img_h, img_w = img_size
        net_h, net_w = net_size
        scale = min(net_h / float(img_h), net_w / float(img_w))

        ## ----- gpu ——> cpu
        with torch.no_grad():
            dets = dets.cpu().numpy()

        ## ----- The current frame 8 tracking states recording
        unconfirmed_tracks_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        retrieve_tracks_dict = defaultdict(list)  # re-find
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

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

            remain_inds = scores > self.high_det_thresh
            inds_low = scores > self.low_det_thresh
            inds_high = scores < self.high_det_thresh

            ## class second indices
            inds_2nd = np.logical_and(inds_low, inds_high)

            bboxes_1st = bboxes[remain_inds]
            bboxes_2nd = bboxes[inds_2nd]

            scores_1st = scores[remain_inds]
            scores_2nd = scores[inds_2nd]

            if len(bboxes_1st) > 0:
                '''Detections'''
                detections = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                              (tlbr, s) in zip(bboxes_1st, scores_1st)]
            else:
                detections = []
            # print(detections)

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_tracks_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with high score detection boxes'''
            ## ----- build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # ---------- Predict the current location with KF
            # MCTrack.multi_predict(track_pool_dict[cls_id])
            MCTrack.multi_predict(tracked_tracks_dict[cls_id])  # only predict tracked tracks
            # ----------

            # Matching with Hungarian Algorithm
            dists = matching.iou_distance(track_pool_dict[cls_id], detections)
            # print(dists)

            dists = matching.fuse_score(dists, detections)

            matches, u_track, u_detection = matching.linear_assignment(dists,
                                                                       thresh=self.high_match_thresh)

            # --- process matched pairs between track pool and current frame detection
            for i_track, i_det in matches:
                track = track_pool_dict[cls_id][i_track]
                det = detections[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    retrieve_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with low score detection boxes'''
            # association the un-track to the low score detections
            if len(bboxes_2nd) > 0:
                '''Detections'''
                cls_detections_second = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                                         (tlbr, s) in zip(bboxes_2nd, scores_2nd)]
            else:
                cls_detections_second = []
            # print(cls_detections_second)

            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]

            dists = matching.iou_distance(r_tracked_tracks, cls_detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dists,
                                                                              thresh=self.low_match_thresh)  # thresh=0.5

            for i_track, i_det in matches:
                track = r_tracked_tracks[i_track]
                det = cls_detections_second[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    retrieve_tracks_dict[cls_id].append(track)

            # process unmatched tracks for two rounds
            for i_track in u_track:
                track = r_tracked_tracks[i_track]
                if not track.state == TrackState.Lost:
                    # mark unmatched track as lost track
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            # current frame's unmatched detection
            detections = [detections[i] for i in u_detection]

            # iou matching
            dists = matching.iou_distance(unconfirmed_tracks_dict[cls_id], detections)

            if not self.opt.mot20:
                dists = matching.fuse_score(dists, detections)

            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists,
                                                                             thresh=self.unconfirmed_match_thresh)  # 0.7

            for i_track, i_det in matches:
                track = unconfirmed_tracks_dict[cls_id][i_track]
                det = detections[i_det]
                track.update(det, self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_tracks_dict[cls_id][i_track])

            for i_track in u_unconfirmed:  # process unconfirmed tracks
                track = unconfirmed_tracks_dict[cls_id][i_track]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """Step 4: Init new tracks"""
            for i_new in u_detection:  # current frame's unmatched detection
                track = detections[i_new]
                if track.score < self.new_track_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                # if fr_id > 1, tracked but not activated
                track.activate(self.kalman_filter, self.frame_id)

                # activated_tracks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           retrieve_tracks_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

        ## ---------- Return final online targets of the frame
        return output_tracks_dict
        #################### MCMOT end

    def update_mcmot_emb(self, dets, feature_map, img_size, net_size):
        """
        :param dets:
        :param feature_map:
        :param img_size:
        :param net_size:
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        ## ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_id_dict(self.n_classes)
        ## -----

        ## ----- image width, height and net width, height
        img_h, img_w = img_size
        net_h, net_w = net_size
        feat_h, feat_w = feature_map.shape[2:]  # NCHW
        scale = min(net_h / float(img_h), net_w / float(img_w))

        ## ----- L2 normalize the feature map(feature map scale(1/4 of net input size))
        feature_map = F.normalize(feature_map, dim=1)

        ## ----- gpu ——> cpu
        with torch.no_grad():
            dets = dets.cpu().numpy()
            feature_map = feature_map.cpu().numpy()

        ## ----- Get dets dict and reid feature dict
        feat_dict = defaultdict(list)  # feature dict
        box_dict = defaultdict(list)  # dets dict
        score_dict = defaultdict(list)  # scores dict

        for det in dets:
            if det.size == 7:
                x1, y1, x2, y2, score1, score2, cls_id = det  # 7
                score = score1 * score2
            elif det.size == 6:
                x1, y1, x2, y2, score, cls_id = det  # 6

            ## ----- bbox scaling
            # x1 = x1 / float(net_w) * float(img_w)
            # x2 = x2 / float(net_w) * float(img_w)
            # y1 = y1 / float(net_h) * float(img_h)
            # y2 = y2 / float(net_h) * float(img_h)
            box = np.array([x1, y1, x2, y2])
            box /= scale

            ## ----- Fill the bbox dict
            box_dict[int(cls_id)].append(box)

            ## ----- Fill the score dict
            score_dict[int(cls_id)].append(score)

            # get center point
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            # map center point from net scale to feature map scale(1/4 of net input size)
            center_x = center_x / float(img_w) * float(feat_w)
            center_y = center_y / float(img_h) * float(feat_h)

            # rounding and converting to int64 for indexing
            center_x += 0.5
            center_y += 0.5
            center_x = int(center_x)
            center_y = int(center_y)

            # to avoid the object center out of reid feature map's range
            center_x = center_x if center_x >= 0 else 0
            center_x = center_x if center_x < feat_w else feat_w - 1
            center_y = center_y if center_y >= 0 else 0
            center_y = center_y if center_y < feat_h else feat_h - 1

            ## ----- Fill the feature dict
            id_feat_vect = feature_map[0, :, center_y, center_x]
            id_feat_vect = id_feat_vect.squeeze()
            feat_dict[int(cls_id)].append(id_feat_vect)  # put feat vect to dict(key: cls_id)

        ## ---------- Update tracking results of this frame
        # online_targets = self.update_with_emb(box_dict, score_dict, feat_dict)
        # online_targets = self.update_with_emb2(box_dict, score_dict, feat_dict)
        online_targets = self.update_with_fair_backend(box_dict, score_dict, feat_dict)
        ## ----------

        ## return the frame's tracking results
        return online_targets

    def update_with_fair_backend(self, box_dict, score_dict, feat_dict):
        """
        """
        ## ----- update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrackFeat.init_id_dict(self.n_classes)
        # -----

        # ----- The current frame tracking states recording
        unconfirmed_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        refind_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        #################### Even: Start MCMOT
        ## ---------- Process each object class
        for cls_id in range(self.n_classes):
            cls_dets = box_dict[cls_id]
            cls_dets = np.array(cls_dets)

            cls_scores = score_dict[cls_id]
            cls_scores = np.array(cls_scores)

            cls_feats = feat_dict[cls_id]  # n_objs × 128
            cls_feats = np.array(cls_feats)

            if len(cls_dets) > 0:
                '''Detections, tlbr: top left bottom right'''
                cls_detections = [
                    MCTrackFeat(MCTrackFeat.tlbr_to_tlwh(tlbr), score, feat, cls_id)
                    for (tlbr, score, feat) in zip(cls_dets, cls_scores, cls_feats)
                ]  # convert detection of current frame to track format
            else:
                cls_detections = []

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            '''Step 2: First association, with embedding'''
            ## ----- build current frame's track pool by joining tracked_tracks and lost tracks
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KF
            # for track in track_pool:

            ## ----- kalman prediction for track_pool
            MCTrackFeat.multi_predict(track_pool_dict[cls_id])  # predict all track-lets
            # MCTrackFeat.multi_predict(tracked_tracks_dict[cls_id])   # predict only activated track-lets

            dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, track_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7

            # --- process matched pairs between track pool and current frame detection
            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            '''Step 3: Second association, with IOU'''
            # match between track pool and unmatched detection in current frame
            cls_detections = [cls_detections[i] for i in u_detection]
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]

            dists = matching.iou_distance(r_tracked_tracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5

            ## ----- process matched tracks
            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detections[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            ## ----- mark the track lost if two matching rounds failed
            for i in u_track:
                track = r_tracked_tracks[i]
                if not track.state == TrackState.Lost:
                    track.mark_lost()  # mark unmatched track as lost track
                    lost_tracks_dict[cls_id].append(track)

            '''The 3rd matching(The final matching round):
             Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detections = [cls_detections[i] for i in u_detection]  # current frame's unmatched detection

            ## ----- compute iou matching cost
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7

            ## ----- process the matched
            for i_tracked, i_det in matches:
                unconfirmed_det = cls_detections[i_det]
                unconfirmed_track = unconfirmed_dict[cls_id][i_tracked]

                unconfirmed_track.update(cls_detections[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_track)

            ## ----- process the frame's [un-matched tracks]
            for i in u_unconfirmed:
                track = unconfirmed_dict[cls_id][i]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            ## ----- process the frame's [un-matched detections]
            for i in u_detection:
                track = cls_detections[i]
                if track.score < self.new_track_thresh:
                    continue

                # initial activation: tracked state
                track.activate(self.kalman_filter, self.frame_id)

                # activated_tracks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state for lost tracks: 
            remove some lost tracks that lost more than max_time(30 frames by default)
            """
            for lost_track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - lost_track.end_frame > self.max_time_lost:
                    lost_track.mark_removed()
                    removed_tracks_dict[cls_id].append(lost_track)
            # print('Remained match {} s'.format(t4-t3))

            """Final: Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])  # add activated track
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refind_tracks_dict[cls_id])  # add refined track
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])  # update lost tracks
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

        return output_tracks_dict

    def update_with_emb2(self, box_dict, score_dict, feat_dict):
        """
        :param box_dict:
        :param score_dict:
        :param feat_dict:
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_id_dict(self.n_classes)
        # -----

        # ----- The current frame tracking states recording
        unconfirmed_tracks_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        refind_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        #################### Even: Start MCMOT
        ## ---------- Process each object class
        for cls_id in range(self.n_classes):
            ## ----- class boxes
            cls_boxes = box_dict[cls_id]
            cls_boxes = np.array(cls_boxes)

            ## ----- class scores
            cls_scores = score_dict[cls_id]
            cls_scores = np.array(cls_scores)

            ## ----- class feature vectors
            cls_feats = feat_dict[cls_id]  # n_objs × 128
            cls_feats = np.array(cls_feats)

            cls_remain_1st = cls_scores > self.opt.track_thresh
            cls_inds_low = cls_scores > 0.1
            cls_inds_high = cls_scores < self.opt.track_thresh

            ## ---------- class second indices
            cls_inds_2nd = np.logical_and(cls_inds_low, cls_inds_high)

            ## ----- boxes
            cls_dets_boxes_1st = cls_boxes[cls_remain_1st]
            cls_dets_boxes_2nd = cls_boxes[cls_inds_2nd]

            ## ----- scores
            cls_scores_1st = cls_scores[cls_remain_1st]
            cls_scores_2nd = cls_scores[cls_inds_2nd]

            ## ----- features
            cls_feat_1st = cls_feats[cls_remain_1st]
            cls_feat_2nd = cls_feats[cls_inds_2nd]
            ## ----------

            if len(cls_dets_boxes_1st) > 0:
                '''Detections'''
                cls_dets_1st = [MCTrackEmb(MCTrackEmb.tlbr_to_tlwh(tlbr), s, feat, cls_id) for
                                (tlbr, s, feat) in zip(cls_dets_boxes_1st, cls_scores_1st, cls_feat_1st)]
            else:
                cls_dets_1st = []

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_tracks_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            '''Step 2: First association, with high score detection boxes'''
            ## ----- build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            '''---------- Predict the current location with KF
            Whether are lost tracks better with KF or not?
            '''
            MCTrackEmb.multi_predict(track_pool_dict[cls_id])  # predict all tracks in the track pool
            # MCTrackEmb.multi_predict(tracked_tracks_dict[cls_id])  # predict only tracks(not lost)

            # ---------- Matching with Hungarian Algorithm
            # ----- IOU matching
            dists_iou = matching.iou_distance(track_pool_dict[cls_id], cls_dets_1st)
            # print(dists_iou.shape)

            if not self.opt.mot20:
                if dists_iou.shape[0] > 0:
                    dists_iou = matching.fuse_score(dists_iou, cls_dets_1st)

            matches, u_track_1st, u_det_1st = matching.linear_assignment(dists_iou, thresh=self.opt.match_thresh)

            # --- process matched pairs between track pool and current frame detection
            for i_track, i_det in matches:
                track = track_pool_dict[cls_id][i_track]
                det = cls_dets_1st[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            '''Step 3: Second association, with low score detection boxes'''
            # association the un-track to the low score detections
            if len(cls_dets_boxes_2nd) > 0:
                '''Detections'''
                cls_dets_2nd = [MCTrackEmb(MCTrackEmb.tlbr_to_tlwh(tlbr), s, feat, cls_id) for
                                (tlbr, s, feat) in zip(cls_dets_boxes_2nd, cls_scores_2nd, cls_feat_2nd)]
            else:
                cls_dets_2nd = []

            ## The tracks that are not matched in the 1st round matching
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track_1st if track_pool_dict[cls_id][i].state == TrackState.Tracked]

            ## ----- IOU matching
            dists_iou = matching.iou_distance(r_tracked_tracks, cls_dets_2nd)
            matches, u_track_2nd, u_det_2nd = matching.linear_assignment(dists_iou, thresh=0.5)  # thresh=0.5

            for i_track, i_det in matches:
                track = r_tracked_tracks[i_track]
                det = cls_dets_2nd[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            # ## ------ @even: Match the unmatched dets and the lost tracks
            # # cls_dets_1st = [cls_dets_1st[i] for i in u_det_1st]
            # cls_dets_2nd = [cls_dets_2nd[i] for i in u_det_2nd]
            # # cls_dets_remain = cls_dets_1st + cls_dets_2nd
            # cls_lost_tracks = self.lost_tracks_dict[cls_id]
            # dists_emb = matching.embedding_distance(cls_lost_tracks, cls_dets_2nd)
            #
            # matches, u_track, u_det = matching.linear_assignment(dists_emb, thresh=0.9)
            #
            # for i_track, i_det in matches:
            #     track = cls_lost_tracks[i_track]
            #     det = cls_dets_2nd[i_det]
            #
            #     if track.state == TrackState.Tracked:
            #         track.update(det, self.frame_id)
            #         activated_tracks_dict[cls_id].append(track)
            #     else:
            #         track.re_activate(det, self.frame_id, new_id=False)
            #         refind_tracks_dict[cls_id].append(track)
            # ## ------

            ## ----- process unmatched tracks for 2 rounds: mark as lost
            for i in u_track_2nd:
                track = r_tracked_tracks[i]

                # mark unmatched track as lost track
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            ## ----- current frame's unmatched detection
            cls_dets_1st = [cls_dets_1st[i] for i in u_det_1st]
            cls_dets_2nd = [cls_dets_2nd[i] for i in u_det_2nd]
            cls_dets_unmatched = cls_dets_1st + cls_dets_2nd

            ## ----- IOU matching
            dists_iou = matching.iou_distance(unconfirmed_tracks_dict[cls_id], cls_dets_unmatched)

            if not self.opt.mot20:
                dists_iou = matching.fuse_score(dists_iou, cls_dets_unmatched)

            matches, u_unconfirmed, u_det_unconfirmed = matching.linear_assignment(dists_iou, thresh=0.7)  # 0.7

            for i_track, i_det in matches:
                unconfirmed_tracks_dict[cls_id][i_track].update(cls_dets_unmatched[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_tracks_dict[cls_id][i_track])

            for i in u_unconfirmed:
                track = unconfirmed_tracks_dict[cls_id][i]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """Step 4: Init new tracks"""
            for i_new in u_det_unconfirmed:  # current frame's unmatched detection
                track = cls_dets_unmatched[i_new]

                if track.score < self.new_track_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                track.activate(self.kalman_filter, self.frame_id)  # if fr_id > 1, tracked but not activated

                # activated_tarcks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id], refind_tracks_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

        ## ---------- Return final online targets of the frame
        return output_tracks_dict
        #################### MCMOT end

    def update_with_emb(self, boxes_dict, scores_dict, feats_dict):
        """
        :param boxes_dict:
        :param scores_dict:
        :param feats_dict:
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_id_dict(self.n_classes)
        # -----

        # ----- The current frame tracking states recording
        unconfirmed_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        refind_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        #################### Even: Start MCMOT
        ## ---------- Process each object class
        for cls_id in range(self.n_classes):
            ## ----- class boxes
            cls_boxes = boxes_dict[cls_id]
            cls_boxes = np.array(cls_boxes)

            ## ----- Scaling the boxes to image size
            # cls_boxes /= scale

            ## ----- class scores
            cls_scores = scores_dict[cls_id]
            cls_scores = np.array(cls_scores)

            ## ----- class feature vectors
            cls_feats = feats_dict[cls_id]  # n_objs × 128
            cls_feats = np.array(cls_feats)

            cls_remain_1st = cls_scores > self.opt.track_thresh
            cls_inds_low = cls_scores > 0.1
            cls_inds_high = cls_scores < self.opt.track_thresh

            ## ---------- class second indices
            cls_inds_2nd = np.logical_and(cls_inds_low, cls_inds_high)

            ## ----- boxes
            cls_dets_boxes_1st = cls_boxes[cls_remain_1st]
            cls_dets_boxes_2nd = cls_boxes[cls_inds_2nd]

            ## ----- scores
            cls_scores_1st = cls_scores[cls_remain_1st]
            cls_scores_2nd = cls_scores[cls_inds_2nd]

            ## ----- features
            cls_feat_1st = cls_feats[cls_remain_1st]
            cls_feat_2nd = cls_feats[cls_inds_2nd]
            ## ----------

            if len(cls_dets_boxes_1st) > 0:
                '''Detections'''
                cls_dets_1st = [MCTrackEmb(MCTrackEmb.tlbr_to_tlwh(tlbr), s, feat, cls_id) for
                                (tlbr, s, feat) in zip(cls_dets_boxes_1st, cls_scores_1st, cls_feat_1st)]
            else:
                cls_dets_1st = []

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            '''Step 2: First association, with high score detection boxes'''
            ## ----- build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            '''---------- Predict the current location with KF
            Whether are lost tracks better with KF or not?
            '''
            # MCTrackEmb.multi_predict(track_pool_dict[cls_id])    # predict all tracks in the track pool
            MCTrackEmb.multi_predict(tracked_tracks_dict[cls_id])  # predict only tracks(not lost)

            # ---------- Matching with Hungarian Algorithm
            # ----- IOU matching
            dists_iou = matching.iou_distance(track_pool_dict[cls_id], cls_dets_1st)
            # print(dists_iou.shape)

            # ----- Embedding matching
            dists_emb = matching.embedding_distance(track_pool_dict[cls_id], cls_dets_1st)

            if not self.opt.mot20:
                if dists_iou.shape[0] > 0:
                    dists_iou = matching.fuse_score(dists_iou, cls_dets_1st)

            # dists = matching.weight_sum_costs(dists_iou, dists_emb, alpha=0.9)
            dists = matching.fuse_costs(dists_iou, dists_emb)

            matches, u_track_1st, u_det_1st = matching.linear_assignment(dists, thresh=self.opt.match_thresh)
            # matches, u_track_1st, u_det_1st = matching.linear_assignment(dists_iou, thresh=self.args.match_thresh)

            # --- process matched pairs between track pool and current frame detection
            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_dets_1st[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            '''Step 3: Second association, with low score detection boxes'''
            # association the un-track to the low score detections
            if len(cls_dets_boxes_2nd) > 0:
                '''Detections'''
                cls_dets_2nd = [MCTrackEmb(MCTrackEmb.tlbr_to_tlwh(tlbr), s, feat, cls_id) for
                                (tlbr, s, feat) in zip(cls_dets_boxes_2nd, cls_scores_2nd, cls_feat_2nd)]
            else:
                cls_dets_2nd = []

            ## The tracks that are not matched in the 1st round matching
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track_1st if track_pool_dict[cls_id][i].state == TrackState.Tracked]

            ## ----- IOU matching
            dists_iou = matching.iou_distance(r_tracked_tracks, cls_dets_2nd)

            ## ----- embedding matching
            dists_emb = matching.embedding_distance(r_tracked_tracks, cls_dets_2nd)

            # dists = matching.weight_sum_costs(dists_iou, dists_emb, alpha=0.9)
            dists = matching.fuse_costs(dists_iou, dists_emb)

            matches, u_track_2nd, u_det_2nd = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5

            # matches, u_track_2nd, u_det_2nd = matching.linear_assignment(dists_iou, thresh=0.7)  # thresh=0.5

            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_dets_2nd[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            ## ----- process unmatched tracks for 2 rounds
            for i in u_track_2nd:
                track = r_tracked_tracks[i]

                # mark unmatched track as lost track
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            # current frame's unmatched detection
            cls_dets_1st = [cls_dets_1st[i] for i in u_det_1st]
            cls_dets_2nd = [cls_dets_2nd[i] for i in u_det_2nd]
            cls_dets_remain = cls_dets_1st + cls_dets_2nd

            ## -----IOU matching
            # dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_dets_1st)
            dists_iou = matching.iou_distance(unconfirmed_dict[cls_id], cls_dets_remain)

            ## ----- Embedding matching
            dists_emb = matching.embedding_distance(unconfirmed_dict[cls_id], cls_dets_remain)

            if not self.opt.mot20:
                # dists = matching.fuse_score(dists, cls_dets_1st)
                dists_iou = matching.fuse_score(dists_iou, cls_dets_remain)

            # dists = matching.weight_sum_costs(dists_iou, dists_emb, alpha=0.9)
            dists = matching.fuse_costs(dists_iou, dists_emb)

            matches, u_unconfirmed, u_det_unconfirmed = matching.linear_assignment(dists, thresh=0.7)  # 0.7

            for i_tracked, i_det in matches:
                # unconfirmed_dict[cls_id][i_tracked].update(cls_dets_1st[i_det], self.frame_id)
                unconfirmed_dict[cls_id][i_tracked].update(cls_dets_remain[i_det], self.frame_id)

                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])

            for i in u_unconfirmed:
                track = unconfirmed_dict[cls_id][i]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """Step 4: Init new tracks"""
            for i_new in u_det_unconfirmed:  # current frame's unmatched detection
                # track = cls_dets_1st[i_new]
                track = cls_dets_remain[i_new]

                if track.score < self.new_track_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                track.activate(self.kalman_filter, self.frame_id)  # if fr_id > 1, tracked but not activated

                # activated_tracks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id], refind_tracks_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

        ## ---------- Return final online targets of the frame
        return output_tracks_dict
        #################### MCMOT end

    def update(self, output_results, img_info, img_size):
        """
        :param output_results:
        :param img_info: img_height, img_width
        :param img_size: net_height, net_width
        :return:
        """
        self.frame_id += 1

        activated_tarcks = []
        refind_tracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        elif output_results.shape[1] == 7:  # x1, y1, x2, y2, score1, score2, cls
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            classes = output_results[:, -1]  # class ids

        # image width and image height
        img_h, img_w = img_info[0], img_info[1]

        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.opt.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.opt.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [Track(Track.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[Track]
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = join_tracks(tracked_stracks, self.lost_tracks)

        # Predict the current location with KF
        Track.multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections)
        if not self.opt.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.match_thresh)

        for i_tracked, i_det in matches:
            track = strack_pool[i_tracked]
            det = detections[i_det]
            if track.state == TrackState.Tracked:
                track.update(detections[i_det], self.frame_id)
                activated_tarcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [Track(Track.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_tracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_tracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for i_tracked, i_det in matches:
            track = r_tracked_tracks[i_tracked]
            det = detections_second[i_det]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tarcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)

        for it in u_track:
            track = r_tracked_tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        if not self.opt.mot20:
            dists = matching.fuse_score(dists, detections)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for i_tracked, i_det in matches:
            unconfirmed[i_tracked].update(detections[i_det], self.frame_id)
            activated_tarcks.append(unconfirmed[i_tracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_tarcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = join_tracks(self.tracked_tracks, activated_tarcks)
        self.tracked_tracks = join_tracks(self.tracked_tracks, refind_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_stracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(self.tracked_tracks, self.lost_tracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


def join_tracks(tlista, tlistb):
    """
    :param tlista:
    :param tlistb:
    :return:
    """
    exists = {}
    res = []

    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)

    return res


def sub_tracks(t_list_a, t_list_b):
    """
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    tracks = {}
    for t in t_list_a:
        tracks[t.track_id] = t
    for t in t_list_b:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    """
    :param tracks_a:
    :param tracks_b:
    :return:
    """
    pdist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()

    for p, q in zip(*pairs):
        timep = tracks_a[p].frame_id - tracks_a[p].start_frame
        timeq = tracks_b[q].frame_id - tracks_b[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(tracks_a) if not i in dupa]
    resb = [t for i, t in enumerate(tracks_b) if not i in dupb]

    return resa, resb
