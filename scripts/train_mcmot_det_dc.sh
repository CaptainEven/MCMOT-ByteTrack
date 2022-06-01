python3 ./tools/train_det.py --ckpt ./YOLOX_outputs/yolox_tiny_det_c5_dark/latest_ckpt.pth.tar \
                             --exp_file ./exps/example/mot/yolox_tiny_det_c5_dark.py \
                             --cfg ./cfg/yolox_darknet_tiny_bb46.cfg \
                             --batch-size 32 \
                             --cutoff 44 \
                             --debug 0 \
                             --devices 3,4,6,7


## /mnt/diskb/even/ByteTrack/YOLOX_outputs/yolox_tiny_det_c5_dark/latest_ckpt.pth.tar
## ./pretrained/v5.46.weights
## --devices 3,4,6,7