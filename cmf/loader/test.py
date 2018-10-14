# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-06 21:02:16
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-26 20:12:01
import numpy as np
import os
left_dir=r'/home/lidong/Documents/datasets/Driving/train_data_clean_pass/left/'
files=os.listdir(left_dir)
files.sort()
# max=0
# for i in range(len(files)):
#     P=np.load(os.path.join(left_dir,files[i]))[...,8]
#     print(i,np.max(P),max)
#     if np.max(P)>=max:
#         max=np.max(P)
#         print(max)
#max region: 142
P3=np.load(os.path.join(left_dir,files[0]))[8]
depth=np.load(os.path.join(left_dir,files[0]))[6]
for i in range(int(np.max(P3))):
    region=np.where(P3==i,depth,0)
    print(np.max(region))
    region=np.where(P3==i,depth,np.max(region))
    print(np.min(region))