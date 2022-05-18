# MCMOT-ByteTrack: One-shot multi-class multi-object tracking </br>
单阶段实时多类别多目标跟踪
</br>
This is an extention work of ByteTrack, which extends the one-class multi-object tracking to multi-class multi-object tracking
</br>
You can refer to origin fork [ByteTrack](https://github.com/ifzhang/ByteTrack)
## Tracking demo of C5(car, bicycle, person, cyclist, tricycle)
![image](https://github.com/CaptainEven/MCMOT-ByteTrack/blob/master/test_13.gif)

## update! 2022/0518 
Add OC_SORT as tracker's backend.
```
    parser.add_argument("--tracker",
                        type=str,
                        default="byte",
                        help="byte | oc")
```

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

## TensorRT deployment updated! (Compile, release, debug with Visual Studio On Windows)
[TensorRT Deployment](https://github.com/CaptainEven/ByteTrack-MCMOT-TensorRT)
