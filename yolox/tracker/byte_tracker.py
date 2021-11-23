# encoding=utf-8

from collections import defaultdict

import numpy as np

from yolox.tracker import matching
from .basetrack import BaseTrack, MCBaseTrack, TrackState
from .kalman_filter import KalmanFilter


# Multi-class Track class
class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls_id):
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
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def reset_track_id(self):
        """
        :return:
        """
        self.reset_track_count(self.cls_id)

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
        Start a new tracklet
        :param kalman_filter:
        :param frame_id:
        :return:
        """
        self.kalman_filter = kalman_filter

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

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

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
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
        返回一个对象的 string 格式。
        :return:
        """
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class STrack(BaseTrack):
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

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

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
        :type new_track: STrack
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


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        """
        :param args:
        :param frame_rate:
        """
        self.frame_id = 0
        self.args = args
        print("args:\n", self.args)

        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # Get number of tracking object classes
        self.num_classes = args.n_classes

        # Define track lists for single object class
        self.tracked_tracks = []  # type: list[STrack]
        self.lost_tracks = []  # type: list[STrack]
        self.removed_tracks = []  # type: list[STrack]

        # Define tracks dict for multi-class objects
        self.tracked_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])
        self.lost_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])
        self.removed_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])

    def update_mcmot(self, output_results, img_size, net_size):
        """
        :param output_results:
        :param img_size: img_height, img_width
        :param net_size: net_height, net_width
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_count(self.num_classes)
        # -----

        ## ----- Get image height and image width
        img_h, img_w = img_size[0], img_size[1]

        # ----- Get net height and net width
        net_h, net_w = net_size[0], net_size[1]

        # record tracking states of the current frame
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        ## ----- Get the scale
        scale = min(float(net_h) / float(img_h), float(net_w) / float(img_w))

        #################### Even: MCMOT
        ## ---------- Get dets dict
        output_results = output_results.cpu().numpy()
        boxxes_dict = defaultdict(list)
        scores_dict = defaultdict(list)

        for det in output_results:
            if det.size == 7:
                x1, y1, x2, y2, score1, score2, cls_id = det  # 7
                score = score1 * score2
            elif det.size == 6:
                x1, y1, x2, y2, score, cls_id = det  # 6

            box = np.array([x1, y1, x2, y2])
            boxxes_dict[int(cls_id)].append(box)
            scores_dict[int(cls_id)].append(score)

        ## ---------- Process each object class
        for cls_id in range(self.num_classes):
            cls_boxes = boxxes_dict[cls_id]
            cls_boxes = np.array(cls_boxes)
            cls_boxes /= scale

            cls_scores = scores_dict[cls_id]
            cls_scores = np.array(cls_scores)

            cls_remain_inds = cls_scores > self.args.track_thresh
            cls_inds_low = cls_scores > 0.1
            cls_inds_high = cls_scores < self.args.track_thresh

            ## class second indices
            cls_inds_second = np.logical_and(cls_inds_low, cls_inds_high)

            cls_dets_boxes = cls_boxes[cls_remain_inds]
            cls_dets_boxes_second = cls_boxes[cls_inds_second]

            cls_scores_keep = cls_scores[cls_remain_inds]
            cls_scores_second = cls_scores[cls_inds_second]

            if len(cls_dets_boxes) > 0:
                '''Detections'''
                cls_detections = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                                  (tlbr, s) in zip(cls_dets_boxes, cls_scores_keep)]
            else:
                cls_detections = []
            # print(cls_detections)

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with high score detection boxes'''
            ## ----- build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KF
            MCTrack.multi_predict(track_pool_dict[cls_id])

            # Matching with Hungarian Algorithm
            dists = matching.iou_distance(track_pool_dict[cls_id], cls_detections)
            # print(dists)

            if not self.args.mot20:
                dists = matching.fuse_score(dists, cls_detections)

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

            # --- process matched pairs between track pool and current frame detection
            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with low score detection boxes'''
            # association the un-track to the low score detections
            if len(cls_dets_boxes_second) > 0:
                '''Detections'''
                cls_detections_second = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                                         (tlbr, s) in zip(cls_dets_boxes_second, cls_scores_second)]
            else:
                cls_detections_second = []
            # print(cls_detections_second)

            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_tracks, cls_detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detections_second[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            # process unmatched tracks for two rounds
            for it in u_track:
                track = r_tracked_tracks[it]
                if not track.state == TrackState.Lost:
                    # mark unmatched track as lost track
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            # current frame's unmatched detection
            cls_detections = [cls_detections[i] for i in u_detection]

            # iou matching
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)

            if not self.args.mot20:
                dists = matching.fuse_score(dists, cls_detections)

            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            for i_new in u_detection:  # current frame's unmatched detection
                track = cls_detections[i_new]
                if track.score < self.det_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                track.activate(self.kalman_filter, self.frame_id)

                # activated_tarcks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])  # add activated track
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])  # add refined track
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])  # update lost tracks
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]
            return output_tracks_dict

        #################### MCMOT end

        # ## ----- Get all infos of results
        # if output_results.shape[1] == 5:
        #     scores = output_results[:, 4]
        #     bboxes = output_results[:, :4]
        # elif output_results.shape[1] == 6:  # x1, y1, x2, y2, score cls
        #     output_results = output_results.cpu().numpy()
        #     scores = output_results[:, 4]
        #     bboxes = output_results[:, :4]  # x1y1x2y2
        #     # classes = output_results[:, -1]  # class ids
        # elif output_results.shape[1] == 7:  # x1, y1, x2, y2, score1, score2, cls
        #     output_results = output_results.cpu().numpy()
        #     scores = output_results[:, 4] * output_results[:, 5]
        #     bboxes = output_results[:, :4]  # x1y1x2y2
        #     # classes = output_results[:, -1]  # class ids
        #
        # bboxes /= scale
        #
        # remain_inds = scores > self.args.track_thresh
        # inds_low = scores > 0.1
        # inds_high = scores < self.args.track_thresh
        #
        # ## second indices
        # inds_second = np.logical_and(inds_low, inds_high)
        #
        # dets_boxes = bboxes[remain_inds]
        # dets_boxes_second = bboxes[inds_second]
        #
        # scores_keep = scores[remain_inds]
        # scores_second = scores[inds_second]
        #
        # ## ---------- Updating tracks
        # activated_tarcks = []
        # refind_tracks = []
        # lost_stracks = []
        # removed_stracks = []
        #
        # if len(dets_boxes) > 0:
        #     '''Detections'''
        #     detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
        #                   (tlbr, s) in zip(dets_boxes, scores_keep)]
        # else:
        #     detections = []
        #
        # ''' Add newly detected tracklets to tracked_stracks'''
        # unconfirmed = []
        # tracked_tracks = []  # type: list[STrack]
        # for track in self.tracked_tracks:
        #     if not track.is_activated:
        #         unconfirmed.append(track)
        #     else:
        #         tracked_tracks.append(track)
        #
        # ''' Step 2: First association, with high score detection boxes'''
        # track_pool = join_tracks(tracked_tracks, self.lost_tracks)
        #
        # # Predict the current location with KF
        # STrack.multi_predict(track_pool)
        #
        # # Matching with Hungarian Algorithm
        # dists = matching.iou_distance(track_pool, detections)
        #
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        #
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # for i_tracked, i_det in matches:
        #     track = track_pool[i_tracked]
        #     det = detections[i_det]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_tarcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_tracks.append(track)
        #
        # ''' Step 3: Second association, with low score detection boxes'''
        # # association the untrack to the low score detections
        # if len(dets_boxes_second) > 0:
        #     '''Detections'''
        #     detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
        #                          (tlbr, s) in zip(dets_boxes_second, scores_second)]
        # else:
        #     detections_second = []
        #
        # r_tracked_tracks = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.Tracked]
        # dists = matching.iou_distance(r_tracked_tracks, detections_second)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # for i_tracked, i_det in matches:
        #     track = r_tracked_tracks[i_tracked]
        #     det = detections_second[i_det]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_tarcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_tracks.append(track)
        #
        # # process unmatched tracks for two rounds
        # for it in u_track:
        #     track = r_tracked_tracks[it]
        #     if not track.state == TrackState.Lost:
        #         track.mark_lost()
        #         lost_stracks.append(track)
        #
        # '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # detections = [detections[i] for i in u_detection]
        # dists = matching.iou_distance(unconfirmed, detections)
        #
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        #
        # matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        #
        # for i_tracked, i_det in matches:
        #     unconfirmed[i_tracked].update(detections[i_det], self.frame_id)
        #     activated_tarcks.append(unconfirmed[i_tracked])
        # for it in u_unconfirmed:
        #     track = unconfirmed[it]
        #     track.mark_removed()
        #     removed_stracks.append(track)
        #
        # """ Step 4: Init new tracks"""
        # for i_new in u_detection:
        #     track = detections[i_new]
        #
        #     if track.score < self.det_thresh:
        #         continue
        #
        #     # tracked but not activated
        #     track.activate(self.kalman_filter, self.frame_id)
        #
        #     # activated_tarcks_dict may contain track with 'is_activated' False
        #     activated_tarcks.append(track)
        #
        # """ Step 5: Update state"""
        # for track in self.lost_tracks:
        #     if self.frame_id - track.end_frame > self.max_time_lost:
        #         track.mark_removed()
        #         removed_stracks.append(track)
        #
        # # print('Ramained match {} s'.format(t4-t3))
        #
        # self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        # self.tracked_tracks = join_tracks(self.tracked_tracks, activated_tarcks)
        # self.tracked_tracks = join_tracks(self.tracked_tracks, refind_tracks)
        # self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        # self.lost_tracks.extend(lost_stracks)
        # self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_stracks)
        # self.removed_stracks.extend(removed_stracks)
        # self.tracked_tracks, self.lost_tracks = remove_duplicate_stracks(self.tracked_tracks, self.lost_tracks)
        #
        # # get scores of lost tracks
        # output_stracks = [track for track in self.tracked_tracks if track.is_activated]
        #
        # return output_stracks

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

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = join_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

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
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
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

        if not self.args.mot20:
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
            if track.score < self.det_thresh:
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
    stracks = {}
    for t in t_list_a:
        stracks[t.track_id] = t
    for t in t_list_b:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


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
