# encoding=utf-8

from collections import OrderedDict, defaultdict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# Create a multi-object class BaseTrack class
class MCBaseTrack(object):
    """
    Multi-class Base track
    """
    _id_dict = defaultdict(int)  # the MCBaseTrack class owns this dict

    track_id = 0
    is_activated = False
    state = TrackState.New

    # history = OrderedDict()
    history = []

    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    # time_since_last_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """
        :return:
        """
        return self.frame_id

    # @even: reset track id
    @staticmethod
    def init_id_dict(n_classes):
        """
        Initiate _count for all object classes
        :param n_classes:
        """
        for cls_id in range(n_classes):
            MCBaseTrack._id_dict[cls_id] = 0

    @staticmethod
    def next_id(cls_id):
        """
        :param cls_id:
        :return:
        """
        MCBaseTrack._id_dict[cls_id] += 1
        return MCBaseTrack._id_dict[cls_id]

    @staticmethod
    def reset_track_id(cls_id):
        """
        :param cls_id:
        :return:
        """
        MCBaseTrack._id_dict[cls_id] = 0

    def activate(self, *args):
        """
        :param args:
        :return:
        """
        raise NotImplementedError

    def predict(self):
        """
        :return:
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def mark_lost(self):
        """
        :return:
        """
        self.state = TrackState.Lost

    def mark_removed(self):
        """
        :return:
        """
        self.state = TrackState.Removed


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_last_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        """
        :return:
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """
        :param args:
        :return:
        """
        raise NotImplementedError

    def predict(self):
        """
        :return:
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def mark_lost(self):
        """
        :return:
        """
        self.state = TrackState.Lost

    def mark_removed(self):
        """
        :return:
        """
        self.state = TrackState.Removed
