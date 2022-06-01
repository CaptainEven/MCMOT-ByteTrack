python3 ./tools/train_det.py --ckpt ./pretrained/v5.46.weights \
                             --exp_file ./exps/example/mot/yolox_tiny_det_c5_dark.py \
                             --cfg ./cfg/yolox_darknet_tiny_bb46.cfg \
                             --batch-size 8 \
                             --cutoff 44 \
                             --debug 0 \
                             --devices 7