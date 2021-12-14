# encoding=utf-8

import os

import cv2
import numpy as np


def cmp2VideosDir(src_dir1, src_dir2, dst_dir, ext=".mp4"):
    """
    :param src_dir1:
    :param src_dir2:
    :param dst_dir:
    :param ext:
    :return:
    """
    if not os.path.isdir(src_dir1):
        print("[Err]: invalid src dir 1!")
        return
    if not os.path.isdir(src_dir2):
        print("[Err]: invalid src dir 2!")
        return

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print("{:s} made.".format(dst_dir))

    video_paths1 = [src_dir1 + "/" + x for x in os.listdir(src_dir1) if x.endswith(ext)]
    video_paths2 = [src_dir2 + "/" + x for x in os.listdir(src_dir2) if x.endswith(ext)]
    video_paths1.sort()
    video_paths2.sort()

    assert len(video_paths1) == len(video_paths2)

    for vid_path1, vid_path2 in zip(video_paths1, video_paths2):
        if not (os.path.isfile(vid_path1) and os.path.isfile(vid_path2)):
            print("[Err]: Video path wrong!")
            continue

        vid_name1 = os.path.split(vid_path1)[-1]
        vid_name2 = os.path.split(vid_path2)[-1]

        assert vid_name1 == vid_name2

        ## ---------- Processing
        ## ----- 读取视频
        cap1 = cv2.VideoCapture(vid_path1)
        cap2 = cv2.VideoCapture(vid_path2)

        # 获取视频所有帧数
        FRAME_NUM1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        FRAME_NUM2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        assert FRAME_NUM1 == FRAME_NUM2
        print('Total {:d} frames'.format(FRAME_NUM1))

        if FRAME_NUM1 == 0:
            break

        for i in range(0, FRAME_NUM1):
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()

            if not (success1 and success2):  # 判断当前帧是否存在
                print("[Warning]: read frame-pair failed @frame{:d}!".format(i))
                break

            assert frame1.shape == frame2.shape

            ## ----- 设置输出帧
            H, W, C = frame1.shape
            if W >= H:
                res = np.zeros((H * 2, W, 3), dtype=np.uint8)
                res[:H, :, :] = frame1
                res[H:2 * H, :, :] = frame2
            else:
                res = np.zeros((H, W * 2, 3), dtype=np.uint8)
                res[:, :W, :] = frame1
                res[:, W:2 * W, :] = frame2

            ## ----- 输出到tmp目录
            res_sv_path = dst_dir + "/{:04d}.jpg".format(i)
            cv2.imwrite(res_sv_path, res)
            print("{:s} saved.".format(res_sv_path))

        ## ---------- 输出视频结果
        vid_sv_path = dst_dir + "/" + vid_name1[:-len(ext)] + "cmp" + ext
        cmd_str = 'ffmpeg -f image2 -r 6 -i {:s}/%04d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(dst_dir, vid_sv_path)
        print(cmd_str)
        os.system(cmd_str)

        cmd_str = "rm -rf {:s}/*.jpg".format(dst_dir)
        print(cmd_str)
        os.system(cmd_str)


def cmp2VideosExt(src_root, dst_root,
                  ext=".mp4", flag1="old", flag2="new"):
    """
    :param src_root:
    :param dst_root:
    :param ext:
    :param flag1:
    :param flag2:
    :return:
    """
    if not os.path.isdir(src_root):
        print("[Err]: invalid src root!")
        return

    parent_dir = os.path.abspath(os.path.join(src_root, ".."))
    tmp_dir = parent_dir + "/tmp"
    tmp_dir = os.path.abspath(tmp_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    if dst_root is None:
        # os.makedirs(dst_root)
        # print("{:s} made.".format(dst_root))
        dst_root = src_root

    videos1 = [src_root + "/" + x for x in os.listdir(src_root) if x.endswith(ext) and flag1 in x]
    videos2 = [src_root + "/" + x for x in os.listdir(src_root) if x.endswith(ext) and flag2 in x]

    # assert len(videos1) == len(videos2)

    videos1.sort()
    videos2.sort()

    for vid1_path in videos1:
        vid2_path = vid1_path.replace(flag1, flag2)
        if not (os.path.isfile(vid1_path) and os.path.isfile(vid2_path)):
            print("[Warning]: invalid file path.")
            continue

        cmd_str = "rm -rf {:s}/*.jpg".format(tmp_dir)
        print(cmd_str)
        os.system(cmd_str)

        vid1_name = os.path.split(vid1_path)[-1]
        vid_name = vid1_name.replace(flag1, "")

        ## ----- 读取视频
        cap1 = cv2.VideoCapture(vid1_path)
        cap2 = cv2.VideoCapture(vid2_path)

        # 获取视频所有帧数
        FRAME_NUM1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        FRAME_NUM2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        assert FRAME_NUM1 == FRAME_NUM2
        print('Total {:d} frames'.format(FRAME_NUM1))

        if FRAME_NUM1 == 0:
            break

        for i in range(0, FRAME_NUM1):
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()

            if not (success1 and success2):  # 判断当前帧是否存在
                print("[Warning]: read frame-pair failed @frame{:d}!".format(i))
                break

            assert frame1.shape == frame2.shape

            ## ----- 设置输出帧
            H, W, C = frame1.shape
            if W >= H:
                res = np.zeros((H * 2, W, 3), dtype=np.uint8)
                res[:H, :, :] = frame1
                res[H:2 * H, :, :] = frame2
            else:
                res = np.zeros((H, W * 2, 3), dtype=np.uint8)
                res[:, :W, :] = frame1
                res[:, W:2 * W, :] = frame2

            ## ----- 输出到tmp目录
            res_sv_path = tmp_dir + "/{:04d}.jpg".format(i)
            cv2.imwrite(res_sv_path, res)
            print("{:s} saved.".format(res_sv_path))

        ## ---------- 输出视频结果
        vid_sv_path = dst_root + "/" + vid_name[:-len(ext)] + "cmp" + ext
        cmd_str = 'ffmpeg -f image2 -r 6 -i {:s}/%04d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(tmp_dir, vid_sv_path)
        print(cmd_str)
        os.system(cmd_str)


def cmp2VideosForOutput(src_dir, dst_dir, ID0=0, ID1=2, ext=".mp4"):
    """
    :param src_dir:
    :param dst_dir:
    :param ID0:
    :param ID1:
    :param ext:
    :return:
    """
    if not os.path.isdir(src_dir):
        print("[Err]: invalid src dir.")
        return

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print("{:s} made.".format(dst_dir))

    parent_dir = os.path.abspath(os.path.join(src_dir, ".."))
    tmp_dir = parent_dir + "/tmp"
    tmp_dir = os.path.abspath(tmp_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    sub_dirs = [src_dir + "/" + x for x in os.listdir(src_dir)
                if os.path.isdir((src_dir + "/" + x))]
    for sub_dir_path in sub_dirs:
        video_path_pairs = [sub_dir_path + "/" + x for x in os.listdir(sub_dir_path)
                            if x.endswith(ext)]
        if len(video_path_pairs) < 2:
            print("[Err]: invalid video path pair!")
            continue

        ## ----- clearing
        cmd_str = "rm -rf {:s}/*.jpg".format(tmp_dir)
        print(cmd_str)
        os.system(cmd_str)

        video_path_pairs.sort()
        if ID0 >= len(video_path_pairs) - 1 or ID1 > len(video_path_pairs) - 1:
            continue
        vid1_path = video_path_pairs[ID0]
        vid2_path = video_path_pairs[ID1]

        vid_name = os.path.split(sub_dir_path)[-1]

        ## ----- 读取视频
        cap1 = cv2.VideoCapture(vid1_path)
        cap2 = cv2.VideoCapture(vid2_path)

        # 获取视频所有帧数
        FRAME_NUM1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        FRAME_NUM2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        assert FRAME_NUM1 == FRAME_NUM2
        print('Total {:d} frames'.format(FRAME_NUM1))

        if FRAME_NUM1 == 0:
            break

        for i in range(0, FRAME_NUM1):
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()

            if not (success1 and success2):  # 判断当前帧是否存在
                print("[Warning]: read frame-pair failed @frame{:d}!".format(i))
                break

            assert frame1.shape == frame2.shape

            ## ----- 设置输出帧
            H, W, C = frame1.shape
            if W >= H:
                res = np.zeros((H * 2, W, 3), dtype=np.uint8)
                res[:H, :, :] = frame1
                res[H:2 * H, :, :] = frame2
            else:
                res = np.zeros((H, W * 2, 3), dtype=np.uint8)
                res[:, :W, :] = frame1
                res[:, W:2 * W, :] = frame2

            ## ----- 输出到tmp目录
            res_sv_path = tmp_dir + "/{:04d}.jpg".format(i)
            cv2.imwrite(res_sv_path, res)
            print("{:s} saved.".format(res_sv_path))

        ## ---------- 输出视频结果
        vid_sv_path = dst_dir + "/" + vid_name + "_cmp" + ext
        cmd_str = 'ffmpeg -f image2 -r 6 -i {:s}/%04d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(tmp_dir, vid_sv_path)
        print(cmd_str)
        os.system(cmd_str)


if __name__ == "__main__":
    # cmp2VideosDir(src_dir1="/mnt/diskc/even/ByteTrack/YOLOX_outputs/output/vx",
    #               src_dir2="/mnt/diskc/even/ByteTrack/YOLOX_outputs/output/v4",
    #               dst_dir="/mnt/diskc/even/ByteTrack/output",
    #               ext=".mp4")

    cmp2VideosForOutput(src_dir="/mnt/diskc/even/ByteTrack/YOLOX_outputs/yolox_tiny_track_c5/track_vis",
                        dst_dir="/mnt/diskc/even/ByteTrack/YOLOX_outputs/",
                        ext=".mp4")
