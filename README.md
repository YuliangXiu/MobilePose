# MobilePose

MobilePose is a **Tiny** PyTorch implementation of single person 2D pose estimation framework. The aim is to provide the interface of the training/inference/evaluation, and the dataloader with various data augmentation options. And final trained model can satisfy basic requirements(speed+size+accuracy) for mobile device.

Some codes for networks and display are brought from:
1. [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)
2. [Vanilla FCN, U-Net, SegNet, PSPNet, GCN, DUC](https://github.com/zijundeng/pytorch-semantic-segmentation)
3. [Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch)
4. [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) 
5. [DSNTNN](https://github.com/anibali/dsntnn)

## Requirements

- Python 3.6
- PyTorch 0.4 

## Evaluation Results

|Model|Parmas(M)|Flops(G)|mAP(0.5IoU)|mAR(0.5IoU)|Google Drive|
|---|---|---|---|---|---|
|UNet+DSNTNN|29.60|9.37|||[119M]()|
|ResNet18+DUC+DSNTNN|12.26|1.64|||[50M]()|
|ResNet18(stride=2)+DSNTNN|10.66|32.66|0.901|0.947|[43M](https://drive.google.com/open?id=1MoAQoQyThrluGrRv6ZaKlrM14yvkCWrt)|
|MobileNetV2+DUC+DSNTNN|3.91|0.49|0.807|0.899|[16M](https://drive.google.com/open?id=1Meyz8Jg2aRe8ijeBAY1uCRpV9l5OJoXl)|
|ShuffleNetV2+DUC+DSNTNN|2.92|0.31|0.637|0.796|[12M](https://drive.google.com/open?id=1pKChewpUFA0CINdLUnV9sUxkscTF5Q_0)|

## Features

- [x] multi-thread dataloader with augmentations (dataloader.py)
- [x] training and inference (training.py)
- [x] performance evaluation (eval.py)
- [x] multiple models support (network.py)
- [x] ipython notebook visualization (demo.ipynb)
- [ ] Macbook camera realtime display script (run_webcam.py)

## Usage

1. Training:
```shell
python training.py --model=mobilenet2 --gpu=0 --inputsize=224 --lr 1e-3 --batchsize=128 --t7=./models/shufflenetv2_224_sgb_best.t7
```
2. Evaluation
```shell
ln -s cocoapi/PythonAPI/pycocotools
cd cocoapi/PythonAPI && make

python eval.py --t7=./models/resnet18_224_sgd_best.t7 --model=resnet18 --gpu=0
```

## Contributors

MobilePose is developed and maintained by [Yuliang Xiu](http://xiuyuliang.cn/about/), [Zexin Chen](https://github.com/ZexinChen) and [Yinghong Fang](https://github.com/Fangyh09).

## License

MobilePose is freely available for free non-commercial use. For commercial queries, please contact [Cewu Lu](http://www.mvig.org/).