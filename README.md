# DeepPose-pytorch
Reproduce Google's [DeepPose](https://arxiv.org/pdf/1312.4659.pdf) with Resnet18 by Pytorch

## Functionality

1. **Tiny** trained model (Resnet18[43MB], MobilenetV2[8.9MB])
2. **Fast** inference speed (>30FPS)
3. **Accurate** keypoint estimation (30~40mAP)

## Requirements:
- Python 3.6.2
- Pytorch 0.2.0\_3 

## Todo List:
- [x] accuracy(mAP) evaluation script
- [x] multi-scales training
- [x] data augmentation(rotate/shift/flip/multi-scale)
- [x] support LSP dataset
- [x] adding weighted loss(coco keypoints weight) 
- [x] support Macbook camera realtime skeleton display demo

## Usage

1. Training:
```shell
export CUDA_VISIBLE_DEVICES=0; python training.py --model=mobilenet/resnet --gpu=0
```
2. Evaluation
```shell
python eval.py --model=mobilenet/resnet
```
4. Realtime visualization:
```shell
python run_webcam.py --model=mobilenet/resnet
```

