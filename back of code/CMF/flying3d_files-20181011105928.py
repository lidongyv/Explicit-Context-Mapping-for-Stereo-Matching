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
#remove=np.loadtxt(r'/home/dataset2/sceneflow/flying3d_clean_pass/all_unused_files.txt')
remove=[]
with open(r'/home/dataset2/sceneflow/flying3d_clean_pass/all_unused_files.txt','r') as f:
    data=f.readlines()
    for line in data:
        line=line.split()[0]
        remove.append(line)
        #print(line)
print(len(remove))

count=0
output_dir=r'/home/lidong/Documents/datasets/flying3d/train/'
image_dir=r'/home/dataset2/sceneflow/flying3d_clean_pass/'
disparity_dir=r'/home/dataset2/sceneflow/flying3d_dispairty/'
#semantic_dir=r'/home/dataset2/sceneflow/flying3d_semantic/'
task1=r'TRAIN'
task2=r'TEST'
left=[]
right=[]
check=[]
stage1=os.path.join(image_dir,task1)
stage1_files=os.listdir(stage1)
stage1_files.sort()
for i in range(len(stage1_files)):
    stage2=os.path.join(stage1,stage1_files[i])
    stage2_files=os.listdir(stage2)
    stage2_files.sort()
    for j in range(len(stage2_files)):
        stage3=os.path.join(stage2,stage2_files[j])
        stage3_files=os.listdir(stage3)
        stage3_files.sort()
        left_stage=os.path.join(stage3,stage3_files[0])
        right_stage=os.path.join(stage3,stage3_files[1])
        image_files=os.listdir(left_stage)
        image_files.sort()
        pfm_files=os.listdir(os.path.join(disparity_dir,task1,stage1_files[i],stage2_files[j],stage3_files[0]))
        pfm_files.sort()
        for m in range(len(image_files)):
            check=os.path.join(task1,stage1_files[i],stage2_files[j],stage3_files[0],image_files[m])
            pfm=os.path.join(task1,stage1_files[i],stage2_files[j],stage3_files[0],pfm_files[m])
            if check not in remove:
                left.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])
            else:
                count+=1
            check=os.path.join(task1,stage1_files[i],stage2_files[j],stage3_files[1],image_files[m])
            pfm=os.path.join(task1,stage1_files[i],stage2_files[j],stage3_files[1],pfm_files[m])
            if check not in remove:
                right.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])
            else:
                count+=1
np.save(os.path.join(output_dir,'train.npy'),[left,right])
print(count)
count=0
left=[]
right=[]
stage1=os.path.join(image_dir,task2)
stage1_files=os.listdir(stage1)
stage1_files.sort()
for i in range(len(stage1_files)):
    stage2=os.path.join(stage1,stage1_files[i])
    stage2_files=os.listdir(stage2)
    stage2_files.sort()
    for j in range(len(stage2_files)):
        stage3=os.path.join(stage2,stage2_files[j])
        stage3_files=os.listdir(stage3)
        stage3_files.sort()
        left_stage=os.path.join(stage3,stage3_files[0])
        right_stage=os.path.join(stage3,stage3_files[1])
        image_files=os.listdir(left_stage)
        image_files.sort()
        pfm_files=os.listdir(os.path.join(disparity_dir,task2,stage1_files[i],stage2_files[j],stage3_files[0]))
        pfm_files.sort()
        for m in range(len(image_files)):
            check=os.path.join(task2,stage1_files[i],stage2_files[j],stage3_files[0],image_files[m])
            pfm=os.path.join(task1,stage1_files[i],stage2_files[j],stage3_files[0],pfm_files[m])
            if check not in remove:
                left.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])
            else:
                count+=1
            check=os.path.join(task2,stage1_files[i],stage2_files[j],stage3_files[1],image_files[m])
            pfm=os.path.join(task1,stage1_files[i],stage2_files[j],stage3_files[1],pfm_files[m])
            if check not in remove:
                right.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])
            else:
                count+=1
np.save(os.path.join(output_dir,'test.npy'),[left,right])
print(count)