python3 ./tools/train_det.py --ckpt ./pretrained/v5.46.weights \
                             --exp_file ./exps/example/mot/yolox_det_c5_dark_ssl.py \
                             --cfg ./cfg/yolox_darknet_tiny_bb46.cfg \
                             --batch-size 10 \
                             --cutoff 44 \
                             --debug 0 \
                             --n_workers 2 \
                             --n_devices 2 \
                             --devices 5,6,7


## /mnt/diskb/even/ByteTrack/YOLOX_outputs/yolox_tiny_det_c5_dark/latest_ckpt.pth.tar
## ./YOLOX_outputs/yolox_det_c5_dark_ssl/ssl_ckpt.pth.tar
## ./pretrained/v5.46.weights
## --devices 3,4,6,7