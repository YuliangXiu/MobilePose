# MobilePose

MobilePose is a **Tiny** PyTorch implementation of single person 2D pose estimation framework. The aim is to provide the interface of the training/inference/evaluation, and the dataloader with various data augmentation options. And final trained model can satisfy basic requirements(speed+size+accuracy) for mobile device.

Some codes for mobilenetV2 and display are brought from [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2) and [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation). Thanks to the original authors.

## Functionality

1. **Tiny** trained model (Resnet18[43MB], MobilenetV2[8.9MB])
2. **Fast** inference speed (GPU>100FPS, CPU~30FPS)
3. **Accurate** keypoint estimation (75-85mAP(0.5IoU), 33-40mAP(0.5~0.95IoU))

## Requirements:

- Python 2.7.13
- Pytorch 0.2.0\_3 
- imgaug 0.2.5

## Todo List:

- [x] multi-thread dataloader
- [x] training and inference
- [x] performance evaluation
- [x] multi-scale training
- [x] support resnet18/mobilenetV2
- [x] data augmentation(rotate/shift/flip/multi-scale/noise)
- [x] Macbook camera realtime display script

## Usage

1. Training:
```shell
export CUDA_VISIBLE_DEVICES=0; python training.py --model=mobilenet/resnet --gpu=0 --retrain=True
```
2. Evaluation
```shell
ln -s cocoapi/PythonAPI/pycocotools
export CUDA_VISIBLE_DEVICES=0; python eval.py --model=mobilenet/resnet
```
4. Realtime visualization:
```shell
python run_webcam.py --model=mobilenet/resnet
```

## Contributors

MobilePose is developed and maintained by [Yuliang Xiu](http://xiuyuliang.cn/about/), [Zexin Chen](https://github.com/ZexinChen) and [Yinghong Fang](https://github.com/Fangyh09).

## License

MobilePose is freely available for free non-commercial use. For commercial queries, please contact [Cewu Lu](http://www.mvig.org/) or [SightPlus Co. Ltd](https://www.sightp.com/).

