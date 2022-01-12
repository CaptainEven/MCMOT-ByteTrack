python3 ./tools/train_det.py --ckpt ./YOLOX_outputs/yolox_tiny_det_c5_darknet/latest_ckpt.pth.tar \
                             --exp_file ./exps/example/mot/yolox_tiny_track_c5_darknet.py \
                             --cfg ./cfg/yolox_darknet_tiny.cfg \
                             --train_root /mnt/diskb/even/dataset/MCMOT \
                             --val_root /mnt/diskb/even/dataset/MCMOT_TEST \
                             --debug 0 \
                             --devices 3
