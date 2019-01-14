# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-06-20 14:37:27
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-29 21:07:06

import numpy as np
import os
import time
import sys
from python_pfm import *

file=[]
for i in os.listdir('/home/lidong/Documents/datasets/flying3d/train/'):
    file.append(os.path.join('/home/lidong/Documents/datasets/flying3d/train/',i))
print(len(file),file[-1])
for i in os.listdir('/home/lidong/Documents/datasets/driving/train/'):
    file.append(os.path.join('/home/lidong/Documents/datasets/driving/train/',i))
print(len(file),file[-1])
for i in os.listdir('/home/lidong/Documents/datasets/monkey/train/'):
    file.append(os.path.join('/home/lidong/Documents/datasets/monkey/train/',i))
np.save('/home/lidong/Documents/datasets/flying3d/all.npy',file)