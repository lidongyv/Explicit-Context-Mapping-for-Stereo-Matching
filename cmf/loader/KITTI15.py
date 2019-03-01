# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-19 13:33:07
# @Last Modified by:   yulidong
# @Last Modified time: 2019-02-26 15:07:40

import os
import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torch.nn.functional as F
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.Tensor([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        # print(rgb.view(3, 1, 1).expand_as(img))
        # exit()
        return img.add(rgb.view(3, 1, 1).expand_as(img))

class KITTI15(data.Dataset):


    def __init__(self, root, split="train", is_transform=True, img_size=(540,960)):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (540, 960)
        self.stats={'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        self.pca = Lighting()
        self.files = {}
        self.datapath=root
        self.files=os.listdir(os.path.join(self.datapath,split))
        self.files.sort()          
        self.split=split
        if len(self.files)<1:
            raise Exception("No files for ld=[%s] found in %s" % (split, self.ld))
        self.length=self.__len__()
        print("Found %d in %s data" % (len(self.files), self.datapath))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        #index=58
        #print(os.path.join(self.datapath,'train_all',self.files[index]))
        data=np.load(os.path.join(self.datapath,self.split,self.files[index]))
        #print(os.path.join(self.datapath,self.split,self.files[index]))
        if self.split=='train' or self.split=='train_all':
            position=np.nonzero(data[...,6])
            hmin=np.min(position[0])
            hmax=np.max(position[0])
            wmin=np.min(position[1])
            wmax=np.max(position[1])
            if hmax-hmin<=256:
                hmin=hmax-256
            if wmax-wmin<=512:
                wmax=wmin+512
            th, tw = 256, 512
            x1 = random.randint(hmin, hmax - th)
            y1 = random.randint(wmin, wmax - tw)
            data=data[x1:x1+th,y1:y1+tw,:]
        else:
            h,w = data.shape[0],data.shape[1]
            th, tw = 384, 1248
            x1 = 0
            y1 = 0
            padding_h=data[:(th-h),:,:]
            padding_h[:,:,6]=0
            data=np.concatenate([padding_h,data],0)
            padding_w=data[:,:(tw-w),:]
            padding_w[:,:,6]=0
            data=np.concatenate([padding_w,data],1)
            #data[:(th-h),:(tw-w),6]=0
        #data=data[:540,:960,:]
        left=data[...,0:3]/255
        #
        image2=data[...,0:3]
        image2=transforms.ToTensor()(image2)
        #print(torch.max(image),torch.min(image))
        right=data[...,3:6]/255
        disparity=data[...,6]
        # print(np.sum(np.where(disparity[:540,...]==0,np.ones(1),np.zeros(1))))
        # print(np.sum(np.where(disparity[:540,...]<=1,np.ones(1),np.zeros(1))))
        # print(np.sum(np.where(disparity<=2,np.ones(1),np.zeros(1))))
        # print(np.sum(np.where(disparity<=3,np.ones(1),np.zeros(1))))
        # print(disparity.shape)
        if self.is_transform:
            left, right,disparity,image = self.transform(left, right,disparity)
        if self.split=='test':
            return left, right,disparity,image,self.files[index].split('.')[0],h,w

        #print(torch.max(left),torch.min(left))
        return left, right,disparity,image2
    def transform(self, left, right,disparity):
        """transform
        """
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])
        if self.split=='eval' or self.split=='test':
            image=left*255+0
            left=trans(left).float()
            right=trans(right).float()
            disparity=torch.from_numpy(disparity).float()
            #image=left+0
        else:

            disparity=torch.from_numpy(disparity).float()
            topil=transforms.ToPILImage()
            totensor=transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            left=totensor(left)
            right=totensor(right)
            one=torch.ones(1).float()
            zero=torch.zeros(1).float()
            #sigma=random.uniform(0, 0.04)
            # brightness=random.uniform(0, 0.4)
            # contrast=random.uniform(0, 0.4)
            # saturation=random.uniform(0, 0.4)
            # hue=random.uniform(0, 0.2)
            
            #variance=color(left)-left
            left=topil(left)
            right=topil(right)
            color=transforms.ColorJitter(0.4,0.4,0.4,0)
            left=color(left)
            right=color(right)

            # gamma=random.uniform(0.8, 1.2)
            # left=tf.adjust_gamma(left,gamma)
            # right=tf.adjust_gamma(right,gamma)
            left=totensor(left)
            right=totensor(right)
            left=self.pca(left)
            right=self.pca(right)
            # r=random.uniform(0.8, 1.2)
            # g=random.uniform(0.8, 1.2)
            # b=random.uniform(0.8, 1.2)
            # left[:,:,0]*=r
            # left[:,:,1]*=g
            # left[:,:,2]*=b
            # right[:,:,0]*=r
            # right[:,:,1]*=g
            # right[:,:,2]*=b
            # gaussian=torch.zeros_like(left).normal_()*sigma
            # left=left+gaussian
            left=left.clamp(min=0,max=1)
            # right=right+gaussian
            right=right.clamp(min=0,max=1)
            image=left+0
            left=normalize(left)
            right=normalize(right)

        return left,right,disparity,image
