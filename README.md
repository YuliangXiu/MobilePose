# DeepPose-pytorch
Reproduce Google's [DeepPose](https://arxiv.org/pdf/1312.4659.pdf) with Resnet18 by Pytorch

Note: This is a pretty coarse implementation of original paper, so it's far from Google's results.

## Requirements:
- Python 3.6.2
- Pytorch 0.2.0\_3 

## Todo List:
- [x] accuracy(mAP) evaluation script
- [ ] multi-scales training and testing
- [ ] data augmentation(rotate/shift/flip)
- [ ] support LSP COCO AIC dataset
- [ ] Teacher-Student Learning(For 4 times compression)

