# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-11-15 18:06:20

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy
import os
import sys
from python_pfm import *


# output_dir=r'/home/lidong/Documents/datasets/kitti/'
# left_dir=r'/home/dataset/KITTI2015/data/training/image_2/'
# right_dir=r'/home/dataset/KITTI2015/data/training/image_3/'
# disparity_dir=r'/home/dataset/KITTI2015/data/training/disp_occ_0/'
# left_image=os.listdir(left_dir)
# left_image.sort()
# right_image=os.listdir(right_dir)
# right_image.sort()
# disparity_image=os.listdir(disparity_dir)
# disparity_image.sort()
# length=len(right_image)
# a=0
# #eval=np.random.randint(size=40,low=0,high=200)
# for f in range(length):
#     l_image=np.array(cv2.imread(os.path.join(left_dir,left_image[f])))[:,:,::-1]
#     r_image=np.array(cv2.imread(os.path.join(right_dir,right_image[f])))[:,:,::-1]
#     disparity=np.array(cv2.imread(os.path.join(disparity_dir,disparity_image[f])))[...,0]
#     print(os.path.join(left_dir,left_image[f]),os.path.join(right_dir,right_image[f]),os.path.join(disparity_dir,disparity_image[f]))
#     a=a+np.sum(np.where(disparity>128,1,0))

#     print(np.max(l_image),np.max(disparity),np.sum(np.where(disparity>120,1,0)))
#     #break
#     # if f in eval:
#     #     data=np.concatenate([l_image,
#     #                     r_image,
#     #                     np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1])
#     #                     ],
#     #                     axis=2)
#     #     np.save(os.path.join(output_dir,r'eval',str(f)+'.npy'),data)
#     #     print(os.path.join(output_dir,r'eval',str(f)+'.npy'))
#     # else:
#     #     data=np.concatenate([l_image,
#     #                     r_image,
#     #                     np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1])
#     #                     ],
#     #                     axis=2)
#     #     np.save(os.path.join(output_dir,r'train',str(f)+'.npy'),data)
#     #     print(os.path.join(output_dir,r'train',str(f)+'.npy'))
# print(a)
# a=450
output_dir=r'/home/lidong/Documents/datasets/kitti/'
left_dir=r'/home/dataset/KITTI2015/data/testing/image_2/'
right_dir=r'/home/dataset/KITTI2015/data/testing/image_3/'
disparity_dir=r'/home/dataset/KITTI2015/data/training/disp_occ_0/'
left_image=os.listdir(left_dir)
left_image.sort()
right_image=os.listdir(right_dir)
right_image.sort()
disparity_image=os.listdir(disparity_dir)
disparity_image.sort()
length=len(right_image)

for f in range(length):
    l_image=np.array(cv2.imread(os.path.join(left_dir,left_image[f])))[:,:,::-1]
    r_image=np.array(cv2.imread(os.path.join(right_dir,right_image[f])))[:,:,::-1]
    disparity=np.array(cv2.imread(os.path.join(right_dir,right_image[f])))[...,0]
    print(os.path.join(left_dir,left_image[f]),os.path.join(right_dir,right_image[f]),os.path.join(disparity_dir,disparity_image[f]))
    print(np.max(l_image),np.max(disparity))
    name=left_image[f].split('.')[0]
    print(name)
    data=np.concatenate([l_image,
                    r_image,
                    np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1])
                    ],
                    axis=2)
    np.save(os.path.join(output_dir,r'test',str(name)+'.npy'),data)
    print(os.path.join(output_dir,r'test',str(name)+'.npy'))
