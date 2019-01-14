# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 23:06:40
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-18 22:18:15


import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
from rsden.utils import recursive_glob
import torchvision.transforms as transforms

class NYU(data.Dataset):


    def __init__(self, root, split="train", is_transform=True, img_size=(480,640),task='depth'):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.num=0
        self.is_transform = is_transform
        self.n_classes = 9  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (480, 640)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.path=os.path.join(self.root,self.split)
        self.files=os.listdir(self.path)
        self.files.sort(key=lambda x:int(x[:-4]))
        if len(self.files)<1:
            raise Exception("No files for %s found in %s" % (split, self.path))

        print("Found %d in %s images" % (len(self.files), self.path))
        self.task=task
        if task=='depth':
            self.d=3
            self.r=5
        else:
            self.d=5
            self.r=7
        if task=='all':
            self.d=3
            self.r=7 
        if task=='visualize':
            self.d=3
            self.r=5      
    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        data=np.load(os.path.join(self.path,self.files[index]))
        if self.task=='visualize':
            data=data[0,:,:,:]
        print(data.shape)
        img = data[:,:,0:3]
        #dis=readPFM(disparity_path)
        #dis=np.array(dis[0], dtype=np.uint8)

        depth = data[:,:,self.d]
        segments = data[:,:,self.r]
        segments=np.reshape(segments,[1,segments.shape[0],segments.shape[1]])
        if self.task=='visualize':
            rgb=img
            img, depth,segments = self.transform(img, depth,segments)
            return img, depth,segments,data

        if self.is_transform:
            img, depth,segments = self.transform(img, depth,segments)

        return img, depth,segments

    def transform(self, img, depth,segments):
        """transform

        :param img:
        :param depth:
        """
        img = img[:,:,:]
        #print(img.shape)
        img = img.astype(np.float64)
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        #img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()
        segments=torch.from_numpy(segments).float()
        #img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #img = img.transpose(1,2,0)
        totensor=transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        img=totensor(img)
        img=normalize(img)
        #depth=depth[0,:,:]
        #depth = depth.astype(float)/32
        #depth = np.round(depth)
        #depth = m.imresize(depth, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        #depth = depth.astype(int)
        #depth=np.reshape(depth,[1,depth.shape[0],depth.shape[1]])
        #classes = np.unique(depth)
        #print(classes)
        #depth = depth.transpose(2,0,1)
        #if not np.all(classes == np.unique(depth)):
        #    print("WARN: resizing segmentss yielded fewer classes")

        #if not np.all(classes < self.n_classes):
        #    raise ValueError("Segmentation map contained invalid class values")



        return img, depth,segments
