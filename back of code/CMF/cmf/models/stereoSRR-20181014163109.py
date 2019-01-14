# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import convbn_3d, feature_extraction, disparityregression
from .submodule import convbn

group_norm_group_num = 32


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

    def forward(self, low_resolution_disparity, rgb, *rgb_zoom_feature):
        assert self.twice_times == len(rgb_zoom_feature)
        x = self.conv1(torch.unsqueeze(low_resolution_disparity, dim=1))
        for i, deconv_i in enumerate(self.deconv_module_list):
            x = deconv_i(torch.cat([x, rgb_zoom_feature[i]], dim=1))

        x = torch.cat([x, self.rgb_fea(rgb)], dim=1)
        x = self.conv_out(self.conv2(x))
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


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
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

        self.srr = super_resolution_refinement(32, 2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Conv3d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * \
        #             m.kernel_size[2] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, left, right):

        refimg_fea, half = self.feature_extraction(left)
        targetimg_fea, _ = self.feature_extraction(right)

        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0],
                              refimg_fea.size()[1] * 2, self.maxdisp // 4,
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

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp // 4)(pred1)
            pred1 = self.srr(pred1, left, refimg_fea, half)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp // 4)(pred2)
            pred2 = self.srr(pred2, left, refimg_fea, half)

        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression(self.maxdisp // 4)(pred3)
        pred3 = self.srr(pred3, left, refimg_fea, half)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3
