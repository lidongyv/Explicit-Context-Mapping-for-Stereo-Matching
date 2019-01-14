#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:52:01 2018

@author: lidong
"""
import argparse
import os
import sys
import numpy as np
import time

#/home/dataset2/sceneflow/monkey_clean_pass/a_rain_of_stones_x2/left
count=0
output_dir=r'/home/lidong/Documents/datasets/monkey/'
image_dir=r'/home/dataset2/sceneflow/monkey_clean_pass/'
disparity_dir=r'/home/dataset2/sceneflow/monkey_disparity/'
semantic_dir=r'/home/dataset2/sceneflow/monkey_semantic/'
left=[]
right=[]
check=[]
stage1=image_dir
stage1_files=os.listdir(stage1)
stage1_files.sort()
for i in range(len(stage1_files)):
    stage2=os.path.join(stage1,stage1_files[i])
    #/home/dataset2/sceneflow/monkey_clean_pass/a_rain_of_stones_x2/
    stage2_files=os.listdir(stage2)
    stage2_files.sort()
    left_stage=os.path.join(stage2,stage2_files[0])
    right_stage=os.path.join(stage2,stage2_files[1])
    image_files=os.listdir(left_stage)
    image_files.sort()
    pfm_files=os.listdir(os.path.join(disparity_dir,stage1_files[i],stage2_files[0]))
    pfm_files.sort()
    for n in range(len(image_files)):
        check=os.path.join(stage1_files[i],stage2_files[0],image_files[n])
        pfm=os.path.join(stage1_files[i],stage2_files[0],pfm_files[n])
        left.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])
        check=os.path.join(stage1_files[i],stage2_files[1],image_files[n])
        pfm=os.path.join(stage1_files[i],stage2_files[1],pfm_files[n])
        right.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])

np.save(os.path.join(output_dir,'train.npy'),[left,right])
