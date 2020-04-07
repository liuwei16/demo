import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import pylab
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='data/coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
print(annFile)
# coco=COCO(annFile)
root = os.path.join(dataDir,dataType)
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = dset.CocoCaptions(root = root, annFile = annFile, transform=transform_train)
trainset = dset.CocoDetection(root = root, annFile = annFile, transform=transform_train)
print('样本数量: ', len(trainset))
img, target = trainset[3] # 加载第四个样本
print("数据类型：", type(img))
print("标注信息：",target[0])



# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
# print(len(cats))
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# print("类别编号：", catIds)
# imgIds = coco.getImgIds(catIds=catIds);
# print(type(imgIds),len(imgIds))
# print("图像编号：", imgIds)
# imgIds = coco.getImgIds(imgIds = [549220, 324158, 279278])
# print(imgIds)
# img = coco.loadImgs(imgIds[0])[0]

# path = os.path.join(dataDir,dataType,img['file_name'])
# I = io.imread(path)

# annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
# coco_kps=COCO(annFile)
# annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
# coco_caps=COCO(annFile)

# plt.imshow(I)
# plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)
# annIds = coco_caps.getAnnIds(imgIds=img['id']);
# anns = coco_caps.loadAnns(annIds)
# coco_caps.showAnns(anns)
# plt.savefig(img['file_name'])


