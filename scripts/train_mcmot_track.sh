python3 ./tools/train_track.py --ckpt ./pretrained/v5.45.weights \
                               --cutoff 44 \
                               --exp_file ./exps/example/mot/yolox_tiny_track_c5_darknet.py \
                               --cfg ./cfg/yolox_darknet_tiny.cfg \
                               --train_root /mnt/diskb/even/dataset/MCMOT \
                               --val_root /mnt/diskb/even/dataset/MCMOT_TEST \
                               --batch-size 16 \
                               --debug 0 \
                               --devices 0


##  ../YOLOX_outputs/yolox_tiny_track_c5_darknet/latest_ckpt.pth.tar
## ./pretrained/v5.45.weights
