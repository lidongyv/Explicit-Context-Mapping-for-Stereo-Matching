# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2019-03-01 14:12:35
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
from cmf import caffe_pb2
from cmf.models.utils import *
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
group_norm_group_num = 32


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False), nn.GroupNorm(group_norm_group_num, out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride,
            bias=False), nn.GroupNorm(group_norm_group_num, out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation),
            nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(
                left, 3,
                Variable(torch.LongTensor(
                    [i for i in range(shift, width)])).cuda()),
            (shift, 0, 0, 0))
        shifted_right = F.pad(
            torch.index_select(
                right, 3,
                Variable(torch.LongTensor(
                    [i for i in range(width - shift)])).cuda()),
            (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(
            batch, filters * 2, 1, height, width)
        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = Variable(
            torch.Tensor(
                np.reshape(np.array(range(maxdisp)),
                           [1, maxdisp, 1, 1])).cuda(),
            requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 1, 1, 1),
            # nn.GroupNorm(group_dim, 32),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False))
        self.secondconv = nn.Sequential(
            nn.GroupNorm(group_dim, 32),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(
            convbn(320, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.GroupNorm(group_norm_group_num, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output_all = self.firstconv(x)
        output=self.secondconv(output_all)
        output_rt = self.layer1(output)
        output_raw = self.layer2(output_rt)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(
            output_branch1, (output_skip.size()[2], output_skip.size()[3]),
            mode='bilinear',
            align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(
            output_branch2, (output_skip.size()[2], output_skip.size()[3]),
            mode='bilinear',
            align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(
            output_branch3, (output_skip.size()[2], output_skip.size()[3]),
            mode='bilinear',
            align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(
            output_branch4, (output_skip.size()[2], output_skip.size()[3]),
            mode='bilinear',
            align_corners=False)

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3,
             output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature, output_rt,output_all



class hourglass(nn.Module):
    def __init__(self, inplanes):
        super().__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(
            inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(
            convbn_3d(
                inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(
                inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False), nn.GroupNorm(group_norm_group_num,
                                          inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=1,
                output_padding=(1,1,1),
                stride=2,
                bias=False), nn.GroupNorm(group_norm_group_num,
                                          inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(
                self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post
class similarity_measure1(nn.Module):
    def __init__(self):
        super(similarity_measure1, self).__init__()
        self.inplanes = 32
        self.conv0 = nn.Conv2d(66, 32, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu0 = nn.LeakyReLU(inplace=True)        
        self.conv1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)        
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        #self.relu3 = nn.Sigmoid()
        # self.conv4 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)
        # self.relu4 = nn.LeakyReLU(inplace=True)
        # self.conv5 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)
        # self.relu5 = nn.ReLU(inplace=True)
        #self.s1=nn.Parameter(torch.ones(1)).float()*0.5

        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):

        output = self.conv0(x)
        output = self.relu0(output)
        output = self.conv1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        #output = self.relu3(output)
        # output = self.conv4(output)
        # output = self.relu4(output)
        # output = self.conv5(output)
        # #output = torch.abs(output)
        # output = self.relu5(output)

        # print(output.shape)
        # print(torch.mean(output).item(),torch.max(output).item(),torch.min(output).item())

        # output = output/torch.max(output)
        # output = output-torch.min(output)
        # output = 1-output
        # output = torch.exp(-output)
        #print(torch.mean(output).item(),torch.max(output).item(),torch.min(output).item())
        return output
class similarity_measure2(nn.Module):
    def __init__(self):
        super(similarity_measure2, self).__init__()
        self.inplanes = 32
        self.conv0 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu0 = nn.LeakyReLU(inplace=True)        
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)        
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)        
        #self.s2=nn.Parameter(torch.ones(1)).float()*0.5

        for m in self.modules():
          if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
          elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):

        output = self.conv0(x)
        output = self.relu0(output)
        output = self.conv1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        return output


def matrix_generation():
    scale=4
    x=torch.arange(-scale//2,scale//2+1).float()
    x=torch.cat([x[:x.shape[0]//2],x[x.shape[0]//2+1:]]).unsqueeze(0)
    distance_matrix=x.expand(scale,scale).unsqueeze(0)

    distance_matrix=torch.cat([distance_matrix,distance_matrix.transpose(2,1)],0)
    distance_matrix=distance_matrix.unsqueeze(0)
    distance_matrix1=distance_matrix+0
    distance_matrix2=distance_matrix+0
    distance_matrix3=distance_matrix+0
    distance_matrix4=distance_matrix+0
    distance_matrix5=distance_matrix+0
    distance_matrix6=distance_matrix+0
    distance_matrix7=distance_matrix+0
    distance_matrix8=distance_matrix+0
    x=torch.arange(1,scale+1).float()
    x=x.expand(scale,scale).unsqueeze(0)
    #x=x.repeat(hr_feature.shape[0],hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float().cuda()
    distance_matrix1[:,0,:,:]=scale-x+1
    distance_matrix2[:,0,:,:]=x
    distance_matrix5[:,0,:,:]=distance_matrix2[:,0,:,:]
    distance_matrix6[:,0,:,:]=distance_matrix1[:,0,:,:]
    distance_matrix7[:,0,:,:]=distance_matrix2[:,0,:,:]
    distance_matrix8[:,0,:,:]=distance_matrix1[:,0,:,:]
    x=torch.arange(1,scale+1).float()
    x=x.expand(scale,scale).unsqueeze(0).transpose(2,1)

    distance_matrix3[:,1,:,:]=(scale-x+1)
    distance_matrix4[:,1,:,:]=x
    distance_matrix5[:,1,:,:]=distance_matrix3[:,1,:,:]
    distance_matrix6[:,1,:,:]=distance_matrix3[:,1,:,:]
    distance_matrix7[:,1,:,:]=distance_matrix4[:,1,:,:]
    distance_matrix8[:,1,:,:]=distance_matrix4[:,1,:,:]
    # print(distance_matrix3)
    
    return distance_matrix.cuda(),distance_matrix1.cuda(),distance_matrix2.cuda(),distance_matrix3.cuda(),distance_matrix4.cuda(), \
           distance_matrix5.cuda(),distance_matrix6.cuda(),distance_matrix7.cuda(),distance_matrix8.cuda()


class eight_related_context_mapping(nn.Module):
    def __init__(self):
        super(eight_related_context_mapping,self).__init__()
        self.similarity1=similarity_measure1()
        #need to remove
        #self.similarity2=similarity_measure2()
        # self.fuse=nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1),nn.LeakyReLU(inplace=True))
        #self.fuse.weight.data.fill_(1)
        self.sigmoid=nn.Sigmoid()
        self.distance_matrix,self.distance_matrix1,self.distance_matrix2,self.distance_matrix3,self.distance_matrix4, \
        self.distance_matrix5,self.distance_matrix6,self.distance_matrix7,self.distance_matrix8=matrix_generation()
    def forward(self, lr_feature, hr_feature,lr_feature_r, hr_feature_r):
        
        #self.fuse.weight.data=torch.abs(self.fuse.weight.data)
        with torch.no_grad():
            scale=hr_feature.shape[-1]//lr_feature.shape[-1]
            if scale%2!=0:
                exit()

            padding1=hr_feature[:,:1,:,:scale]*0-100
            padding2=hr_feature[:,:1,:scale,:]*0-100

            distance_matrix=self.distance_matrix.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix1=self.distance_matrix1.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix2=self.distance_matrix2.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix3=self.distance_matrix3.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix4=self.distance_matrix4.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix5=self.distance_matrix1.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix6=self.distance_matrix2.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix7=self.distance_matrix3.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
            distance_matrix8=self.distance_matrix4.repeat(hr_feature.shape[0],1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).float()
        #center
        #reference image
        lr_feature=lr_feature.unsqueeze(-1).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3],scale) \
                                     .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3]*scale) \
                                  .unsqueeze(-2).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],scale,lr_feature.shape[3]*scale) \
                                  .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2]*scale,lr_feature.shape[3]*scale)

        representation=torch.cat([lr_feature,hr_feature,distance_matrix],1)
        weight=self.similarity1(representation)

        #target image
        # lr_feature_r=lr_feature_r.unsqueeze(-1).expand(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2],lr_feature_r.shape[3],scale) \
        #                              .contiguous().view(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2],lr_feature_r.shape[3]*scale) \
        #                           .unsqueeze(-2).expand(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2],scale,lr_feature_r.shape[3]*scale) \
        #                           .contiguous().view(lr_feature_r.shape[0],lr_feature_r.shape[1],lr_feature_r.shape[2]*scale,lr_feature_r.shape[3]*scale)

        # representation_target=torch.cat([lr_feature_r,hr_feature_r,distance_matrix],1)
        # weight_target=self.similarity1(representation_target)

        #left
        #reference
        representation_l=torch.cat([lr_feature[:,:,:,:-scale],hr_feature[:,:,:,scale:],distance_matrix1[:,:,:,:-scale]],1)
        weight_l=self.similarity1(representation_l)
        weight_l=torch.cat([padding1,weight_l],-1)
        #target
        # representation_l_target=torch.cat([lr_feature_r[:,:,:,:-scale],hr_feature_r[:,:,:,scale:],distance_matrix2[:,:,:,:-scale]],1)
        # weight_l_target=self.similarity1(representation_l_target)
        # weight_l_target=torch.cat([padding1,weight_l_target],-1)
        #right
        #reference
        representation_r=torch.cat([lr_feature[:,:,:,scale:],hr_feature[:,:,:,:-scale],distance_matrix2[:,:,:,scale:]],1)
        weight_r=self.similarity1(representation_r)
        weight_r=torch.cat([weight_r,padding1],-1)

        #target image
        # representation_r_target=torch.cat([lr_feature_r[:,:,:,scale:],hr_feature_r[:,:,:,:-scale],distance_matrix1[:,:,:,scale:]],1)
        # weight_r_target=self.similarity1(representation_r_target)
        # weight_r_target=torch.cat([weight_r_target,padding1],-1)
        #top
        #reference
        representation_t=torch.cat([lr_feature[:,:,:-scale,:],hr_feature[:,:,scale:,:],distance_matrix3[:,:,:-scale,:]],1)
        weight_t=self.similarity1(representation_t)
        weight_t=torch.cat([padding2,weight_t],-2)
        #target
        # representation_t_target=torch.cat([lr_feature_r[:,:,:-scale,:],hr_feature_r[:,:,scale:,:],distance_matrix3[:,:,:-scale,:]],1)
        # weight_t_target=self.similarity1(representation_t_target)
        # weight_t_target=torch.cat([padding2,weight_t_target],-2)
        #bottom
        #reference
        representation_b=torch.cat([lr_feature[:,:,scale:,:],hr_feature[:,:,:-scale,:],distance_matrix4[:,:,scale:,:]],1)
        weight_b=self.similarity1(representation_b)
        weight_b=torch.cat([weight_b,padding2],-2)

        #left-top
        #reference
        representation_lt=torch.cat([lr_feature[:,:,:-scale,:-scale],hr_feature[:,:,scale:,scale:],distance_matrix5[:,:,:-scale,:-scale]],1)
        weight_lt=self.similarity1(representation_lt)
        weight_lt=torch.cat([padding2,torch.cat([padding1[...,scale:,:],weight_lt],-1)],-2)
        #target
        # representation_l_target=torch.cat([lr_feature_r[:,:,:,:-scale],hr_feature_r[:,:,:,scale:],distance_matrix2[:,:,:,:-scale]],1)
        # weight_l_target=self.similarity1(representation_l_target)
        # weight_l_target=torch.cat([padding1,weight_l_target],-1)
        #right-top
        #reference
        representation_rt=torch.cat([lr_feature[:,:,:-scale,scale:],hr_feature[:,:,scale:,:-scale],distance_matrix6[:,:,:-scale,scale:]],1)
        weight_rt=self.similarity1(representation_rt)
        weight_rt=torch.cat([padding2,torch.cat([weight_rt,padding1[...,scale:,:]],-1)],-2)

        #target image
        # representation_r_target=torch.cat([lr_feature_r[:,:,:,scale:],hr_feature_r[:,:,:,:-scale],distance_matrix1[:,:,:,scale:]],1)
        # weight_r_target=self.similarity1(representation_r_target)
        # weight_r_target=torch.cat([weight_r_target,padding1],-1)
        #left-bottom
        #reference
        representation_lb=torch.cat([lr_feature[:,:,scale:,:-scale],hr_feature[:,:,:-scale:,scale:],distance_matrix7[:,:,scale:,:-scale]],1)
        weight_lb=self.similarity1(representation_lb)
        weight_lb=torch.cat([torch.cat([padding1[...,scale:,:],weight_lb],-1),padding2],-2)
        #target
        # representation_t_target=torch.cat([lr_feature_r[:,:,:-scale,:],hr_feature_r[:,:,scale:,:],distance_matrix3[:,:,:-scale,:]],1)
        # weight_t_target=self.similarity1(representation_t_target)
        # weight_t_target=torch.cat([padding2,weight_t_target],-2)
        #right-bottom
        #reference
        representation_rb=torch.cat([lr_feature[:,:,scale:,scale:],hr_feature[:,:,:-scale,:-scale],distance_matrix8[:,:,scale:,scale:]],1)
        weight_rb=self.similarity1(representation_rb)
        weight_rb=torch.cat([torch.cat([weight_rb,padding1[...,:-scale,:]],-1),padding2],-2)


        weight_all=torch.cat([weight,weight_l,weight_r,weight_t,weight_b,weight_lt,weight_rt,weight_lb,weight_rb],dim=1)
        weight_norm=F.softmax(weight_all, dim=1)
        #weight_fuse=F.softmax(weight_norm*weight_all)
        #target
        # representation_b_target=torch.cat([lr_feature_r[:,:,scale:,:],hr_feature_r[:,:,:-scale,:],distance_matrix4[:,:,scale:,:]],1)
        # weight_b_target=self.similarity1(representation_b_target)
        # weight_b_target=torch.cat([weight_b_target,padding2],-2)

        # weight_all=torch.cat([weight,weight_r,weight_l,weight_t,weight_b],dim=1)
        # weight_norm=F.softmax(weight_all, dim=1)
        # weight_all_target=torch.cat([weight_target,weight_r_target,weight_l_target,weight_t_target,weight_b_target],dim=1)
        # weight_norm_target=F.softmax(weight_all_target, dim=1)

        # return weight*weight_norm[:,0:1,:,:],weight_target*weight_norm_target[:,0:1,:,:], \
        #         weight_r*weight_norm[:,1:2,:,:],weight_r_target*weight_norm_target[:,1:2,:,:], \
        #         weight_l*weight_norm[:,2:3,:,:],weight_l_target*weight_norm_target[:,2:3,:,:], \
        #         weight_t*weight_norm[:,3:4,:,:],weight_t_target*weight_norm_target[:,3:4,:,:], \
        #         weight_b*weight_norm[:,4:5,:,:],weight_b_target*weight_norm_target[:,4:5,:,:]
        # return  self.sigmoid(weight)*weight_norm[:,0:1,...], \
        #         self.sigmoid(weight_l)*weight_norm[:,1:2,...], \
        #         self.sigmoid(weight_r)*weight_norm[:,2:3,...], \
        #         self.sigmoid(weight_t)*weight_norm[:,3:4,...], \
        #         self.sigmoid(weight_b)*weight_norm[:,4:5,...],\
        #         self.sigmoid(weight_lt)*weight_norm[:,5:6,...], \
        #         self.sigmoid(weight_rt)*weight_norm[:,6:7,...], \
        #         self.sigmoid(weight_lb)*weight_norm[:,7:8,...], \
        #         self.sigmoid(weight_rb)*weight_norm[:,8:9,...]
        #print(torch.mean(torch.max(weight_norm,dim=1)[0]),torch.max(weight_all,dim=1)[0])
        #print(torch.mean(torch.topk(weight_all,3,dim=1)[0].float()),torch.mean(torch.topk(weight_all,3,dim=1)[1].float()))
        #print(torch.mean(torch.topk(weight_all,1,dim=1)[0].float()),torch.mean(torch.topk(weight_all,1,dim=1)[1].float()))
        if torch.mean(torch.topk(weight_all,1,dim=1)[0].float())<0:
            print(torch.mean(torch.topk(weight_all,3,dim=1)[0].float()),torch.mean(torch.topk(weight_all,3,dim=1)[1].float()))
            print(torch.mean(torch.topk(weight_all,1,dim=1)[0].float()),torch.mean(torch.topk(weight_all,1,dim=1)[1].float()))
        #print(torch.mean(torch.min(weight_norm,dim=1)[0]),torch.min(weight_all,dim=1)[0])
        return  weight_norm[:,0:1,...], \
                weight_norm[:,1:2,...], \
                weight_norm[:,2:3,...], \
                weight_norm[:,3:4,...], \
                weight_norm[:,4:5,...],\
                weight_norm[:,5:6,...], \
                weight_norm[:,6:7,...], \
                weight_norm[:,7:8,...], \
                weight_norm[:,8:9,...]
class cmfsm(nn.Module):


    def __init__(self, 
                maxdisp=192):

        super(cmfsm, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.mapping_matrix=eight_related_context_mapping()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        start=time.time()
        refimg_fea, half,all_feature= self.feature_extraction(left)
        targetimg_fea, _ ,all_feature_right= self.feature_extraction(right)
        scale=all_feature.shape[-1]//refimg_fea.shape[-1]
        #mapping,mapping_r,mapping_l,mapping_t,mapping_b=self.mapping_matrix(refimg_fea,all_feature)
        #target
        #[mapping,mapping_r,mapping_l,mapping_t,mapping_b],[mapping_target,mapping_target_r,mapping_target_l]=self.mapping_matrix(refimg_fea,all_feature,targetimg_fea,all_feature_right)
        #time=0.1s
        weight,weight_l,weight_r,weight_t,weight_b,weight_lt,weight_rt,weight_lb,weight_rb=self.mapping_matrix(refimg_fea,all_feature,targetimg_fea,all_feature_right)
        #mapping,mapping_target=self.mapping_matrix(refimg_fea,all_feature,targetimg_fea,all_feature_right)
        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0],
                              refimg_fea.size()[1] * 2, self.maxdisp // scale,
                              refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // scale):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :,
                                                                      i:]
                cost[:, refimg_fea.size()[1]:, i, :,
                     i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()
        
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        #cost2 = self.classif2(out2) + cost1
        #cost3 = self.classif3(out3) + cost2
        #torch.Size([1, 1, 256, 512])
        # weight_all=torch.cat([weight,weight_r,weight_l,weight_t,weight_b],dim=1)
        # weight_norm=F.softmax(weight_all, dim=1)

        # t=time.time()
        cost1 = torch.squeeze(cost1, 1)

        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparityregression(self.maxdisp//scale)(pred1)
        #torch.Size([1, 64, 128])

        pred1=scale*pred1.unsqueeze(-1).expand(pred1.shape[0],pred1.shape[1],pred1.shape[2],scale) \
                                     .contiguous().view(pred1.shape[0],pred1.shape[1],pred1.shape[2]*scale) \
                                  .unsqueeze(-2).expand(pred1.shape[0],pred1.shape[1],scale,pred1.shape[2]*scale) \
                                  .contiguous().view(pred1.shape[0],pred1.shape[1]*scale,pred1.shape[2]*scale)

        pred1_map=pred1*weight
        pred1_map[...,scale:]+=pred1[...,:-scale]*weight_l[...,scale:]
        pred1_map[...,:-scale]+=pred1[...,scale:]*weight_r[...,:-scale]
        pred1_map[...,scale:,:]+=pred1[...,:-scale,:]*weight_t[...,scale:,:]
        pred1_map[...,:-scale,:]+=pred1[...,scale:,:]*weight_b[...,:-scale,:]

        pred1_map[...,scale:,scale:]+=pred1[...,:-scale,:-scale]*weight_lt[...,scale:,scale:]
        pred1_map[...,scale:,:-scale]+=pred1[...,:-scale,scale:]*weight_rt[...,scale:,:-scale]
        pred1_map[...,:-scale,scale:]+=pred1[...,scale:,:-scale]*weight_lb[...,:-scale,scale:]
        pred1_map[...,:-scale,:-scale]+=pred1[...,scale:,scale:]*weight_rb[...,:-scale,:-scale]
        cost2 = self.classif2(out2)
        cost2 = torch.squeeze(cost2, 1)+cost1

        pred2 = F.softmax(cost2, dim=1)
        pred2 = disparityregression(self.maxdisp//scale)(pred2)

        pred2=scale*pred2.unsqueeze(-1).expand(pred2.shape[0],pred2.shape[1],pred2.shape[2],scale) \
                                     .contiguous().view(pred2.shape[0],pred2.shape[1],pred2.shape[2]*scale) \
                                  .unsqueeze(-2).expand(pred2.shape[0],pred2.shape[1],scale,pred2.shape[2]*scale) \
                                  .contiguous().view(pred2.shape[0],pred2.shape[1]*scale,pred2.shape[2]*scale)

        pred2_map=pred2*weight
        pred2_map[...,scale:]+=pred2[...,:-scale]*weight_l[...,scale:]
        pred2_map[...,:-scale]+=pred2[...,scale:]*weight_r[...,:-scale]
        pred2_map[...,scale:,:]+=pred2[...,:-scale,:]*weight_t[...,scale:,:]
        pred2_map[...,:-scale,:]+=pred2[...,scale:,:]*weight_b[...,:-scale,:]

        pred2_map[...,scale:,scale:]+=pred2[...,:-scale,:-scale]*weight_lt[...,scale:,scale:]
        pred2_map[...,scale:,:-scale]+=pred2[...,:-scale,scale:]*weight_rt[...,scale:,:-scale]
        pred2_map[...,:-scale,scale:]+=pred2[...,scale:,:-scale]*weight_lb[...,:-scale,scale:]
        pred2_map[...,:-scale,:-scale]+=pred2[...,scale:,scale:]*weight_rb[...,:-scale,:-scale]


        cost3 = self.classif3(out3)
        cost3 = torch.squeeze(cost3, 1)+cost2
        
        pred3 = F.softmax(cost3, dim=1)
        # print(torch.max(pred3,dim=1)[0])
        # print(torch.min(pred3,dim=1)[0])
        pred3 = disparityregression(self.maxdisp//scale)(pred3)

        pred3=scale*pred3.unsqueeze(-1).expand(pred3.shape[0],pred3.shape[1],pred3.shape[2],scale) \
                                     .contiguous().view(pred3.shape[0],pred3.shape[1],pred3.shape[2]*scale) \
                                  .unsqueeze(-2).expand(pred3.shape[0],pred3.shape[1],scale,pred3.shape[2]*scale) \
                                  .contiguous().view(pred3.shape[0],pred3.shape[1]*scale,pred3.shape[2]*scale)

        pred3_map=pred3*weight
        pred3_map[...,scale:]+=pred3[...,:-scale]*weight_l[...,scale:]
        pred3_map[...,:-scale]+=pred3[...,scale:]*weight_r[...,:-scale]
        pred3_map[...,scale:,:]+=pred3[...,:-scale,:]*weight_t[...,scale:,:]
        pred3_map[...,:-scale,:]+=pred3[...,scale:,:]*weight_b[...,:-scale,:]

        pred3_map[...,scale:,scale:]+=pred3[...,:-scale,:-scale]*weight_lt[...,scale:,scale:]
        pred3_map[...,scale:,:-scale]+=pred3[...,:-scale,scale:]*weight_rt[...,scale:,:-scale]
        pred3_map[...,:-scale,scale:]+=pred3[...,scale:,:-scale]*weight_lb[...,:-scale,scale:]
        pred3_map[...,:-scale,:-scale]+=pred3[...,scale:,scale:]*weight_rb[...,:-scale,:-scale]


        #pred3 = self.srr(pred3, left, refimg_fea, half)
        #print(time.time()-start)
        return pred1_map, pred2_map, pred3_map
        #return pred3



