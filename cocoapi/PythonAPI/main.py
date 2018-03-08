from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import json
import os

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)

PATH_PREFIX = "./txts/scale2"

annFile = os.path.join(PATH_PREFIX, "result-gt-scale2-100-json.txt")
cocoGt=COCO(annFile)

resFile = os.path.join(PATH_PREFIX, "result-pred-scale2-100-json.txt")

cocoDt=cocoGt.loadRes(resFile)
imgIds=sorted(cocoGt.getImgIds())

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
