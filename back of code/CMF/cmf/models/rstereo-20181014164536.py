# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-14 16:44:55
# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-20 18:01:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-16 22:16:14
import time
import torch
import numpy as np
import torch.nn as nn
import math
from math import ceil
from torch.autograd import Variable
from torch.nn.functional import cosine_similarity as cosine_s
from pssm import caffe_pb2
from pssm.models.utils import *
rsn_specs = {
    'scene': 
    {
         'n_classes': 9,
         'input_size': (540, 960),
         'block_config': [3, 4, 23, 3],
    },

}

group_dim=32
pramid_dim=8
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    if stride==1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    if stride==2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=2, bias=False) 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(group_dim,planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(group_dim,planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # print(residual.shape)
            # print(out.shape)
        out += residual
        out = self.relu(out)

        return out
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)

        self.branch1 = nn.Sequential(nn.AvgPool2d((54, 96), stride=(54,96)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(pramid_dim,8),
                                     nn.LeakyReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((27, 48), stride=(27,48)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(pramid_dim,8),
                                     nn.LeakyReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((36, 64), stride=(36,64)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(pramid_dim,8),
                                     nn.LeakyReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((18, 32), stride=(18,32)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(pramid_dim,8),
                                     nn.LeakyReLU(inplace=True))
        self.branch5 = nn.Sequential(nn.AvgPool2d((9, 16), stride=(9,16)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(pramid_dim,8),
                                     nn.LeakyReLU(inplace=True))
        self.branch6 = nn.Sequential(nn.AvgPool2d((3, 8), stride=(3,8)),
                                     nn.Conv2d(32, 8, 1, 1, 0, 1),
                                     nn.GroupNorm(pramid_dim,8),
                                     nn.LeakyReLU(inplace=True))


        self.lastconv = nn.Sequential(nn.Conv2d(80, 64, 3, 1, 1, 1),
                                      nn.GroupNorm(group_dim,64),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(64, 32, 3, 1, 1, 1),
                                      )
        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # output = self.conv1(x)
        # output = self.gn1(output)
        # output = self.relu1(output)
        # output = self.conv2(output)
        # output = self.gn2(output)
        # output = self.relu2(output)
        # output = self.conv3(output)
        # output = self.gn3(output)
        # output = self.relu3(output)
        output_skip = self.layer1(x)
        # output_skip=x

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch5 = self.branch5(output_skip)
        output_branch5 = F.interpolate(output_branch5, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_branch6 = self.branch6(output_skip)
        output_branch6 = F.interpolate(output_branch6, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=True)

        output_feature = torch.cat((output_skip, output_branch6, output_branch5, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        #print(output_feature.shape)
        return output_feature

class feature_extraction2(nn.Module):
    def __init__(self):
        super(feature_extraction2, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False,dilation=1)
        self.gn1 = nn.GroupNorm(group_dim,32)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False,dilation=1)
        self.gn2 = nn.GroupNorm(group_dim,32)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=6,
                               bias=False,dilation=2)
        self.gn3 = nn.GroupNorm(group_dim,32)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 32, 1, 1,1,1)
        self.lastconv = nn.Conv2d(32, 32, 3, 1, 1, 1)

        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.gn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.gn2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.gn3(output)
        output = self.relu3(output)
        #print(output.shape)
        output = self.layer1(output)
        output=self.lastconv(output)

        return output

class similarity_measure1(nn.Module):
    def __init__(self):
        super(similarity_measure1, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        # self.conv4 = nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)        
        # self.lastconv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)
        self.s1=nn.Parameter(torch.ones(1)).float()
        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        # output=self.conv4(output)
        # output=self.lastconv(output)
        output=output*self.s1
        return output
class similarity_measure2(nn.Module):
    def __init__(self):
        super(similarity_measure2, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        # self.conv4 = nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)        
        # self.lastconv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)
        self.s2=nn.Parameter(torch.ones(1)).float()

        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        # output=self.conv4(output)
        # output=self.lastconv(output)
        output=self.s2*output
        return output

class similarity_measure3(nn.Module):
    def __init__(self):
        super(similarity_measure3, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(69, 32, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        # self.relu5 = nn.ReLU(inplace=True)


        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        output = self.conv5(output)
        output = torch.abs(output)
        # output = self.relu5(output)
        # print(output.shape)
        # print(torch.mean(output).item(),torch.max(output).item(),torch.min(output).item())

        output = output/torch.max(output)
        # output = output-torch.min(output)
        # output = 1-output
        output = torch.exp(-output)
        #print(torch.mean(output).item(),torch.max(output).item(),torch.min(output).item())
        return output
class rstereo(nn.Module):


    def __init__(self, 
                 n_classes=9, 
                 block_config=[3, 4, 6, 3], 
                 input_size= (480, 640), 
                 version='scene'):

        super(rstereo, self).__init__()
        self.feature_extraction=feature_extraction().cuda(0)
        self.feature_extraction2=feature_extraction2().cuda(0)
        self.softmax= nn.Softmax(dim=-1)
        self.similarity1=similarity_measure1().cuda(1)
        self.similarity2=similarity_measure2().cuda(1)
        self.similarity3=similarity_measure3().cuda(1)

    def ss_argmin(self,x,index):
        one=torch.ones(1)
        zero=torch.zeros(1)
        #print(x.data.cpu())
        # exit()
        x=self.softmax(-x)
        #print(x)
        disparity= torch.sum(x*index.unsqueeze(0),dim=-1)
        return disparity     
    def cluster_vector(self,feature,x,y):
        one=torch.ones(1).cuda(1)
        zero=torch.zeros(1).cuda(1)
        cluster_feature=feature[...,x,y]
        mean=torch.sum(cluster_feature,dim=-1)/x.shape[0]
        mean=mean.view(cluster_feature.shape[0],cluster_feature.shape[1],1)
        #print(mean.shape)
        weights=torch.norm(cluster_feature-mean,dim=1)
        weights=torch.exp(-weights)
        return weights
    def forward(self, l,r,P,pre,pre2):
        #self.P=P[1,0]
        #0 l to r,1 min,2 max
        #[l_box,r_box,match],[min_d,max_d]
        #start_time=time.time()
        self.pre=pre.cuda(1)
        P1=P[...,0].cuda(1).squeeze()
        P2=P[...,1].cuda(1).squeeze()
        P3=P[...,2].cuda(1).squeeze()
        P4=P[...,3].cuda(1).squeeze()
        #feature extraction
        #P2=P2-P1
        #print(torch.min(P3),torch.max(P3))
        l_sf=self.feature_extraction2(l)
        l_lf=self.feature_extraction(l_sf)

        r_sf=self.feature_extraction2(r)
        r_lf=self.feature_extraction(r_sf)

        disparity=torch.zeros([540,960]).cuda(0)
        one=torch.ones(1).cuda(1)
        zero=torch.zeros(1).cuda(1)



        print(count/960/540,torch.sum(torch.where(disparity>0,one,zero)).item()/960/540)
        #time.sleep(1000)
        #exit()
        #print(torch.max(disparity).item(),torch.min(disparity).item())
        return disparity


