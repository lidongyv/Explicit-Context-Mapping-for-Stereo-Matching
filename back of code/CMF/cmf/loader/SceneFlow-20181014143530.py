# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-22 19:35:42

import os
import torch
import numpy as np
from torch.utils import data
from pssm.utils import recursive_glob
import torchvision.transforms as transforms

class SceneFlow(data.Dataset):


    def __init__(self, root, split="train", is_transform=True, img_size=(540,960)):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.is_transform = is_transform
        self.n_classes = 9  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (540, 960)
        self.stats={'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        self.left_files = {}
        self.datapath=root
        if split=='train':
            self.left_files=os.listdir(os.path.join(self.datapath,'left_re'))
            self.match_files=os.listdir(os.path.join(self.datapath,'match_re'))
            self.left_files.sort()
        else:
            self.left_files=os.listdir(os.path.join(self.datapath,'test','left'))
            self.left_files.sort()            
        self.task='generation'
        self.split=split
        if len(self.left_files)<1:
            raise Exception("No files for ld=[%s] found in %s" % (split, self.ld))

        print("Found %d in %s data" % (len(self.left_files), self.datapath))

    def __len__(self):
        """__len__"""
        return len(self.left_files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        #index=58
        if self.split=='test':
            data=np.load(os.path.join(self.datapath,self.split,'left',self.left_files[index]))
            print(os.path.join(self.datapath,self.split,'left',self.left_files[index]))
            data=data[:540,:960,:]
            left=data[...,0:3]/255
            #print(data.shape)
            right=data[...,3:6]/255
            disparity=data[...,6]
            P=data[...,7:]
            pre_match=np.load(os.path.join(self.datapath,self.split,'match',self.left_files[index]))
        else:
            data=np.load(os.path.join(self.datapath,'left_re',self.left_files[index]))
            print(os.path.join(self.datapath,'left_re',self.left_files[index]))
            data=data[:540,:960,:]
            left=data[...,0:3]/255
            #print(data.shape)
            right=data[...,3:6]/255
            disparity=data[...,6]
            P=data[...,7:]
            pre_match=np.load(os.path.join(self.datapath,'match_re',self.left_files[index]))
        #matching=np.load(os.path.join(self.datapath,'matching',self.left_files[index]))
        # aggregation=np.load(os.path.join(self.datapath,'aggregation',self.left_files[index]))
        #print('load')
        if self.is_transform:
            left, right,disparity,P,pre_match,pre2 = self.transform(left, right,disparity,P,pre_match)
        if self.task=='generation':
            #print('value')
            return left, right,disparity,P,pre_match,pre2
        else:
            return left, right,disparity,P,pre_match,pre2
    def transform(self, left, right,disparity,P,pre):
        """transform
        """
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.stats),
        ])
        # left=transforms.ToTensor()(left)
        # left=transforms.Normalize(**self.stats)(left)
     
        left=trans(left).float()
        right=trans(right).float()

        disparity=torch.from_numpy(disparity).float()
        P=torch.from_numpy(P).float()
        #print(torch.max(disparity).item())

        pre1=torch.from_numpy(pre[1,0]).float()

        #max 142
        pre2=torch.from_numpy(pre[0,0][0]).float()
        #print(pre2.shape)
        #pre2=torch.cat([pre2,torch.zeros([pre2.shape[0],100-pre2.shape[1],pre2.shape[2],pre2.shape[3],pre2.shape[4],pre2.shape[5],pre2.shape[6]])])

        #print(pre1.shape)
        return left,right,disparity,P,pre2,pre1
