# MCMOT-ByteTrack: One-shot multi-class multi-object tracking </br>
单阶段实时多类别多目标跟踪
</br>
This is an extention work of ByteTrack, which extends the one-class multi-object tracking to multi-class multi-object tracking
</br>
You can refer to origin fork [ByteTrack](https://github.com/ifzhang/ByteTrack)
and the original fork of [OC_SORT](https://github.com/noahcao/OC_SORT)
## Tracking demo of C5(car, bicycle, person, cyclist, tricycle)
![image](https://github.com/CaptainEven/MCMOT-ByteTrack/blob/master/test_13.gif)

## update news! 2022/05/18 
Add OC_SORT as tracker's backend.
```
    parser.add_argument("--tracker",
                        type=str,
                        default="byte",
                        help="byte | oc")
```

## update news! 2021/12/01 TensorRT deployment updated! (Compile, release, debug with Visual Studio On Windows)
[TensorRT Deployment](https://github.com/CaptainEven/ByteTrack-MCMOT-TensorRT)

## How to Run the demo
Run the demo_mcmot.py python3 script for demo testing.

## Weights link
[checkpoint](https://pan.baidu.com/s/1PJc09vWK6UJEXp80y27b5g?pwd=ckpt)
### Weights extract code
ckpt

## Test video link
[Test Video for demo](https://pan.baidu.com/s/1RhT7UVtYK_3qiCg36GTb8Q?pwd=test)
### video extract code
test

## FairMOT's implemention of MCMOT: based on CenterNet
[MCMOT](https://github.com/CaptainEven/MCMOT)
