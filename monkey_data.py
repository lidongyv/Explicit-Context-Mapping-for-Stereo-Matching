# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-29 20:17:52

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from multiprocessing import Process,Lock
from multiprocessing import Pool
import cv2
import numpy
import os
import sys
from python_pfm import *

thread_num=10
def pre_processing(start,end):
    output_dir=r'/home/lidong/Documents/datasets/monkey/'
    train=np.load(os.path.join(output_dir,'train.npy'))
    left=train[0]
    right=train[1]
    length=len(left)
    for f in range(start,end):
        start_time=time.time()
        #print(left[f][0],right[f][0],left[f][1])
        l_image=np.array(cv2.imread(left[f][0]))[:,:,::-1]
        r_image=np.array(cv2.imread(right[f][0]))[:,:,::-1]
        disparity=np.array(readPFM(left[f][1])[0])

        data=np.concatenate([l_image,
                        r_image,
                        np.reshape(disparity,[disparity.shape[0],disparity.shape[1],1])
                        ],
                        axis=2)
        np.save(os.path.join(output_dir,r'train',str(f)+'.npy'),data)
        print(f,start,end,time.time()-start_time)
        print(os.path.join(output_dir,r'train',str(f)+'.npy'))



if __name__=='__main__':
    process = []
    output_dir=r'/home/lidong/Documents/datasets/monkey/'
    train=np.load(os.path.join(output_dir,'train.npy'))
    left=train[0]
    right=train[1]
    length=len(left)
    start=[]
    end=[]
    p = Pool(thread_num)
    for z in range(thread_num):
        start.append(int(np.floor(z*length/thread_num)))
        end.append(int(np.ceil((z+1)*length/thread_num)))
    for z in range(thread_num):
        p.apply_async(pre_processing, args=(start[z],end[z]))

    p.close()
    p.join()
    #pre_processing(0,4248)
    print('end')