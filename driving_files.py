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

#/home/dataset2/sceneflow/driving_clean_pass/15mm_focallength/scene_backwards/fast/left
count=0
output_dir=r'/home/lidong/Documents/datasets/driving/'
image_dir=r'/home/dataset2/sceneflow/driving_clean_pass/'
disparity_dir=r'/home/dataset2/sceneflow/driving_dispairty/'
semantic_dir=r'/home/dataset2/sceneflow/driving_semantic/'
left=[]
right=[]
check=[]
stage1=image_dir
stage1_files=os.listdir(stage1)
stage1_files.sort()
for i in range(len(stage1_files)):
    stage2=os.path.join(stage1,stage1_files[i])
    #/home/dataset2/sceneflow/driving_clean_pass/15mm_focallength/
    stage2_files=os.listdir(stage2)
    stage2_files.sort()

    for j in range(len(stage2_files)):
        stage3=os.path.join(stage2,stage2_files[j])
        #/home/dataset2/sceneflow/driving_clean_pass/15mm_focallength/scene_backwards
        stage3_files=os.listdir(stage3)
        stage3_files.sort()
        for m in range(len(stage3_files)):
            stage4=os.path.join(stage3,stage3_files[m])
            #/home/dataset2/sceneflow/driving_clean_pass/15mm_focallength/scene_backwards/fast/
            stage4_files=os.listdir(stage4)
            stage4_files.sort()

            left_stage=os.path.join(stage4,stage4_files[0])
            right_stage=os.path.join(stage4,stage4_files[1])
            image_files=os.listdir(left_stage)
            image_files.sort()
            pfm_files=os.listdir(os.path.join(disparity_dir,stage1_files[i],stage2_files[j],stage3_files[m],stage4_files[0]))
            pfm_files.sort()
            for n in range(len(image_files)):
                check=os.path.join(stage1_files[i],stage2_files[j],stage3_files[m],stage4_files[0],image_files[m])
                pfm=os.path.join(stage1_files[i],stage2_files[j],stage3_files[m],stage4_files[0],pfm_files[m])
                left.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])
                check=os.path.join(stage1_files[i],stage2_files[j],stage3_files[m],stage4_files[1],image_files[m])
                pfm=os.path.join(stage1_files[i],stage2_files[j],stage3_files[m],stage4_files[1],pfm_files[m])
                right.append([os.path.join(image_dir,check),os.path.join(disparity_dir,pfm),os.path.join(semantic_dir,pfm)])

np.save(os.path.join(output_dir,'train.npy'),[left,right])
