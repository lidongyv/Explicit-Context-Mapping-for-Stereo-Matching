# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-17 10:44:43
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-22 18:21:06
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
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
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

        self.lastconv_16 = nn.Sequential(
            convbn(384, 128, 3, 1, 1, 1),
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
        #print(output_all.shape)
        output=self.secondconv(output_all)
        #print(output.shape)
        output_rt = self.layer1(output)
        output_raw = self.layer2(output_rt)
        output_raw = self.layer3(output_raw)
        output_skip = self.layer4(output_raw)
        #print(output_skip.shape)
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
        # print(output_branch4.shape, output_branch3.shape,
        #      output_branch2.shape, output_branch1.shape)
        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3,
             output_branch2, output_branch1), 1)
        output_feature = self.lastconv_16(output_feature)

        return output_feature, output_rt,output_all
class super_resolution_refinement(nn.Module):
    def __init__(self, dis_planes, twice_times):
        super().__init__()
        self.twice_times = twice_times
        self.conv1 = nn.Sequential(
            convbn(1, dis_planes * 2, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.deconv_module_list = nn.ModuleList()
        for _ in range(twice_times):
            deconv_i = nn.Sequential(
                nn.ConvTranspose2d(dis_planes * 3, dis_planes * 2, 3, 2, 1, 1),
                nn.GroupNorm(group_norm_group_num, dis_planes * 2),
                nn.ReLU(inplace=True))
            self.deconv_module_list.append(deconv_i)

        self.rgb_fea = nn.Sequential(
            convbn(3, dis_planes, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(dis_planes, dis_planes, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(dis_planes, dis_planes, 3, 1, 1, 1),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            convbn(dis_planes * 3, dis_planes * 3, 3, 1, 1, 1),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Conv2d(dis_planes * 3, 1, 3, 1, 1)
        self.crap=nn.ReLU(inplace=True)
    def forward(self, low_resolution_disparity, rgb, *rgb_zoom_feature):
        assert self.twice_times == len(rgb_zoom_feature)
        x = self.conv1(torch.unsqueeze(low_resolution_disparity, dim=1))
        for i, deconv_i in enumerate(self.deconv_module_list):
            x = deconv_i(torch.cat([x, rgb_zoom_feature[i]], dim=1))

        x = torch.cat([x, self.rgb_fea(rgb)], dim=1)
        x = self.conv_out(self.conv2(x))
        x=  self.crap(x)
        return x


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
                output_padding=1,
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


class bilinear_cmf_sub_16(nn.Module):


    def __init__(self, 
                maxdisp=192):

        super(bilinear_cmf_sub_16, self).__init__()
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

        #self.srr = super_resolution_refinement(32, 2)

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

        refimg_fea, half,all = self.feature_extraction(left)
        targetimg_fea, _,_ = self.feature_extraction(right)

        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0],
                              refimg_fea.size()[1] * 2, self.maxdisp // 16,
                              refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 16):
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
        #print(cost0.shape)
        out1, pre1, post1 = self.dres2(cost0, None, None)
        #print(out1.shape)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        #if self.training:
        cost1 = F.interpolate(
            cost1,
            [self.maxdisp, left.size()[2],
             left.size()[3]],
            mode='trilinear',align_corners=False)
        cost2 = F.interpolate(
            cost2,
            [self.maxdisp, left.size()[2],
             left.size()[3]],
            mode='trilinear',align_corners=False)
        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparityregression(self.maxdisp)(pred1)

        cost2 = torch.squeeze(cost2, 1)
        pred2 = F.softmax(cost2, dim=1)
        pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.interpolate(
            cost3, [self.maxdisp, left.size()[2],
                    left.size()[3]],
            mode='trilinear',align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)
        return pred1, pred2, pred3



