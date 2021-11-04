# encoding=utf-8

from collections import OrderedDict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


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
    time_since_update = 0

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
