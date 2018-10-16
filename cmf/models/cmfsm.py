# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-16 15:46:55
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
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True))
        self.secondconv = nn.Sequential(
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
        self.conv0 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu0 = nn.LeakyReLU(inplace=True)        
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)        
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        # self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
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
        output = self.relu3(output)
        output = self.conv4(output)
        # output = self.relu4(output)
        output = self.conv5(output)
        # output = torch.abs(output)
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

        # output = torch.abs(output)
        # output = self.relu5(output)
        # print(output.shape)
        # print(torch.mean(output).item(),torch.max(output).item(),torch.min(output).item())

        # output = output/torch.max(output)
        # output = output-torch.min(output)
        # output = 1-output
        # output = torch.exp(-output)
        #print(torch.mean(output).item(),torch.max(output).item(),torch.min(output).item())
        return output
class context_mapping(nn.Module):
    def __init__(self):
        super(context_mapping,self).__init__()
        self.similarity1=similarity_measure1()
        self.similarity2=similarity_measure2()
        #self.s1=nn.Parameter(torch.ones(1)).float()*0.5
        #self.s2=nn.Parameter(torch.ones(1)).float()*0.1
        self.fuse=nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        # self.s2=nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0,
        #                        bias=False,dilation=1)
        self.fuse.weight.data.fill_(1)
        # self.s2.weight.data.fill_(1)
        #print(self.s1.weight.data)
        #exit()
    def forward(self, lr_feature, hr_feature):
        # self.s1=self.s1.cuda()
        # self.s2=self.s2.cuda()
        self.fuse.weight.data=torch.abs(self.fuse.weight.data)
        scale=hr_feature.shape[-1]//lr_feature.shape[-1]
        if scale%2==0:
            x=torch.arange(-scale//2,scale//2+1).float()
            x=torch.cat([x[:x.shape[0]//2],x[x.shape[0]//2+1:]]).unsqueeze(0)
            distance_matrix=x.expand(scale,scale).unsqueeze(0)
            distance_matrix=torch.cat([distance_matrix,distance_matrix.transpose(2,1)],0)
            distance_matrix=torch.cat([distance_matrix,torch.sqrt(torch.pow(distance_matrix[0],2)+torch.pow(distance_matrix[1],2)).unsqueeze(0)],0)
        else:
            x=torch.arange(-scale//2,scale//2+1).unsqueeze(0)
            distance_matrix=x.expand(scale,scale).unsqueeze(0)
            distance_matrix=torch.cat([distance_matrix,distance_matrix.transpose(2,1)],0)

        #print(distance_matrix.shape)
        distance_matrix=distance_matrix.repeat(1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).unsqueeze(0)
        lr_feature=lr_feature.unsqueeze(-1).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3],scale) \
                                     .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3]*scale) \
                                  .unsqueeze(-2).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],scale,lr_feature.shape[3]*scale) \
                                  .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2]*scale,lr_feature.shape[3]*scale)
        #128
        representation=torch.cat([lr_feature,hr_feature,lr_feature*hr_feature,torch.pow(lr_feature-hr_feature,2)],1)
        #print(representation.shape)
        weights1=self.similarity1(representation)
        weights2=self.similarity2(distance_matrix.float().cuda())
        #self.s1=self.s1.view(1,1,1,1).repeat(weights1.shape[0],weights1.shape[1],weights1.shape[2],weights1.shape[3])
        # self.s2=self.s2.view(1,1,1,1).repeat(weights1.shape[0],weights1.shape[1],weights1.shape[2],weights1.shape[3])
        # mapping=(self.s1*weights1+self.s2*weights2)/(self.s1+self.s2)
        #print(self.s1[0,0,0,0].item(),self.s2[0,0,0,0].item())
        fuse=self.fuse(torch.ones(1,2,1,1).cuda())
        #s2=self.s2(torch.ones(1,1,1,1).cuda())
        print(self.fuse.weight.data.cpu().squeeze().numpy())
        #mapping=(self.s1(weights1)+self.s2(weights2))
        mapping=self.fuse(torch.cat([weights1,weights2],1))
        return mapping
class four_related_context_mapping(nn.Module):
    def __init__(self):
        super(four_related_context_mapping,self).__init__()
        self.similarity1=similarity_measure1()
        self.similarity2=similarity_measure2()
        self.fuse=nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0,
                               bias=False,dilation=1)
        self.fuse.weight.data.fill_(1)

    def forward(self, lr_feature, hr_feature):

        #self.fuse.weight.data=torch.abs(self.fuse.weight.data)
        scale=hr_feature.shape[-1]//lr_feature.shape[-1]
        if scale%2==0:
            x=torch.arange(-scale//2,scale//2+1).float()
            x=torch.cat([x[:x.shape[0]//2],x[x.shape[0]//2+1:]]).unsqueeze(0)
            distance_matrix=x.expand(scale,scale).unsqueeze(0)
            distance_matrix=torch.cat([distance_matrix,distance_matrix.transpose(2,1)],0)
            distance_matrix=torch.cat([distance_matrix,torch.sqrt(torch.pow(distance_matrix[0],2)+torch.pow(distance_matrix[1],2)).unsqueeze(0)],0)
        else:
            exit()
        #center
        distance_matrix=distance_matrix.repeat(1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).unsqueeze(0).float().cuda()
        lr_feature=lr_feature.unsqueeze(-1).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3],scale) \
                                     .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],lr_feature.shape[3]*scale) \
                                  .unsqueeze(-2).expand(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2],scale,lr_feature.shape[3]*scale) \
                                  .contiguous().view(lr_feature.shape[0],lr_feature.shape[1],lr_feature.shape[2]*scale,lr_feature.shape[3]*scale)
        #128
        representation=torch.cat([lr_feature,hr_feature,lr_feature*hr_feature,torch.pow(lr_feature-hr_feature,2)],1)
        weights1=self.similarity1(representation)
        weights2=self.similarity2(distance_matrix)
        mapping=self.fuse(torch.cat([weights1,weights2],1))
        #right
        x=torch.arange(1,scale+1).float()
        x=x.expand(scale,scale).unsqueeze(0)
        x=x.repeat(1,hr_feature.shape[-2]//scale,hr_feature.shape[-1]//scale).unsqueeze(0).float().cuda()
        distance_matrix1=distance_matrix
        distance_matrix1[:,0,:,:]=scale-x+1
        distance_matrix1[:,2,:,:]=torch.sqrt(torch.pow(distance_matrix1[:,0,:,:],2)+torch.pow(distance_matrix1[:,1,:,:],2)).unsqueeze(0)
        representation=torch.cat([lr_feature[:,:,:,scale:],hr_feature[:,:,:,:-scale],lr_feature[:,:,:,scale:]*hr_feature[:,:,:,:-scale], \
                       torch.pow(lr_feature[:,:,:,scale:]-hr_feature[:,:,:,:-scale],2)],1)
        weights1=self.similarity1(representation)
        weights2=self.similarity2(distance_matrix1[:,:,:,scale:])
        mapping_r=self.fuse(torch.cat([weights1,weights2],1))
        #left
        distance_matrix2=distance_matrix
        distance_matrix2[:,0,:,:]=x
        distance_matrix2[:,2,:,:]=torch.sqrt(torch.pow(distance_matrix2[:,0,:,:],2)+torch.pow(distance_matrix2[:,1,:,:],2)).unsqueeze(0)
        representation=torch.cat([lr_feature[:,:,:,:-scale],hr_feature[:,:,:,scale:],lr_feature[:,:,:,:-scale]*hr_feature[:,:,:,scale:], \
                       torch.pow(lr_feature[:,:,:,:-scale]-hr_feature[:,:,:,scale:],2)],1)
        weights1=self.similarity1(representation)
        weights2=self.similarity2(distance_matrix2[:,:,:,:-scale])
        mapping_l=self.fuse(torch.cat([weights1,weights2],1))
        #top
        distance_matrix3=distance_matrix
        distance_matrix3[:,1,:,:]=(scale-x+1).transpose(2,3)
        distance_matrix3[:,2,:,:]=torch.sqrt(torch.pow(distance_matrix3[:,0,:,:],2)+torch.pow(distance_matrix3[:,1,:,:],2)).unsqueeze(0)
        representation=torch.cat([lr_feature[:,:,:-scale,:],hr_feature[:,:,scale:,:],lr_feature[:,:,:-scale,:]*hr_feature[:,:,scale:,:], \
                       torch.pow(lr_feature[:,:,:-scale,:]-hr_feature[:,:,scale:,:],2)],1)
        weights1=self.similarity1(representation)
        weights2=self.similarity2(distance_matrix3[:,:,:-scale,:])
        mapping_t=self.fuse(torch.cat([weights1,weights2],1))
        #bottom
        distance_matrix4=distance_matrix
        distance_matrix4[:,1,:,:]=x.transpose(2,3)
        distance_matrix4[:,2,:,:]=torch.sqrt(torch.pow(distance_matrix4[:,0,:,:],2)+torch.pow(distance_matrix4[:,1,:,:],2)).unsqueeze(0)
        representation=torch.cat([lr_feature[:,:,scale:,:],hr_feature[:,:,:-scale,:],lr_feature[:,:,scale:,:]*hr_feature[:,:,:-scale,:], \
                       torch.pow(lr_feature[:,:,scale:,:]-hr_feature[:,:,:-scale,:],2)],1)
        weights1=self.similarity1(representation)
        weights2=self.similarity2(distance_matrix4[:,:,scale:,:])
        mapping_b=self.fuse(torch.cat([weights1,weights2],1))
      
        return mapping,mapping_r,mapping_l,mapping_t,mapping_b
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
        self.mapping_matrix=context_mapping()


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

        refimg_fea, half,all_feature= self.feature_extraction(left)
        targetimg_fea, _ ,_= self.feature_extraction(right)
        scale=all_feature.shape[-1]//refimg_fea.shape[-1]
        mapping=self.mapping_matrix(refimg_fea,all_feature)
        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0],
                              refimg_fea.size()[1] * 2, self.maxdisp // scale,
                              refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 4):
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
        #cost1 = self.classif1(out1)

        # cost1 = torch.squeeze(cost1, 1)
        # pred1 = F.softmax(cost1, dim=1)
        # pred1 = disparityregression(self.maxdisp // 4)(pred1)
        # #pred1 = self.srr(pred1, left, refimg_fea, half)
        # pred1 = self.srr(pred1, left, refimg_fea, half)


        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost3 = torch.squeeze(cost3, 1)
        cost3=cost3.unsqueeze(-1).expand(cost3.shape[0],cost3.shape[1],cost3.shape[2],cost3.shape[3],scale) \
                                     .contiguous().view(cost3.shape[0],cost3.shape[1],cost3.shape[2],cost3.shape[3]*scale) \
                                  .unsqueeze(-2).expand(cost3.shape[0],cost3.shape[1],cost3.shape[2],scale,cost3.shape[3]*scale) \
                                  .contiguous().view(cost3.shape[0],cost3.shape[1],cost3.shape[2]*scale,cost3.shape[3]*scale) \
                                  .unsqueeze(-3).expand(cost3.shape[0],cost3.shape[1],scale,cost3.shape[2]*scale,cost3.shape[3]*scale) \
                                  .contiguous().view(cost3.shape[0],cost3.shape[1]*scale,cost3.shape[2]*scale,cost3.shape[3]*scale)
        cost3=cost3*mapping
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)
        #pred3 = self.srr(pred3, left, refimg_fea, half)
        #pred3=
        return pred3



