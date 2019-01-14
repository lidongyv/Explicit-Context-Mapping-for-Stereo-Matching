# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 16:31:14
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-21 23:32:28

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def l1_r(input, target, weight=None, size_average=True):
    relation=[]
    loss=nn.MSELoss()

    for i in range(3):
        target=torch.reshape(target,(input[i].shape))
        #print(target.shape)
        t=loss(input[i],target)
        #print(t.item())
        relation.append(t)

    return relation
def l1_a(input, target, weight=None, size_average=True):
    relation=[]
    loss=nn.MSELoss()

    for i in range(4):
        target=torch.reshape(target,(input[i].shape))
        #print(target.shape)
        t=torch.sqrt(loss(input[i],target))
        #print(t.item())
        relation.append(t)

    return relation
def log_r(input, target, weight=None, size_average=True):
    relation=[]
    d=[]
    out=[]
    target=torch.reshape(target,(input[0].shape))
    target=torch.log(target+1e-6)
    loss=nn.MSELoss()
    for i in range(3):
        # pre=input[i]
        # num=torch.sum(torch.where(pre>0,torch.ones_like(pre),torch.zeros_like(pre)))/torch.sum(torch.ones_like(pre))
        # print(num)
        input[i]=torch.log(input[i]+1e-6)  
        relation.append(loss(input[i],target))
        #d.append(0.5*torch.pow(torch.sum(input[i]-target),2)/torch.pow(torch.sum(torch.ones_like(input[i])),2))
        #out.append(relation[i]-d[i])
    return relation     
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    #print(c,target.max().data.cpu().numpy())

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    #loss=log_p.sum()
    loss = F.nll_loss(log_p, target,
                      weight=weight, size_average=False)
    #print(loss, mask.data.sum())
    if size_average:
    #    print(mask.data.sum())
       loss /= mask.data.sum()
    #    loss=loss/(950*540)
    return loss
def l1(input, target,mask, weight=None, size_average=True):
    # P1=mask[...,0].cuda(0)
    # P2=mask[...,3].cuda(0)
    one,zero=torch.ones(1).cuda(0),torch.zeros(1).cuda(0)
    mask=torch.where(input>0,one,zero)
    #print(torch.sum(mask)/(input.shape[0]*input.shape[1]))
    mask=torch.reshape(mask,(input.shape))
    target=torch.reshape(target,(input.shape))
    loss=nn.L1Loss(reduction='none')
    #loss=nn.MSELoss(reduction='none')
    #print(torch.sqrt(loss(input,target)).shape)
    relation=torch.sum(mask*loss(input,target))/torch.sum(mask)
    #print(torch.max(torch.where(input>zero,input,zero)),torch.min(torch.where(P1>zero,input,192*one)))
    #relation=torch.sum(torch.sqrt(mask*loss(input,target)))/torch.sum(mask)
    return relation
def l2(input, target, weight=None, size_average=True):
    target=torch.reshape(target,(input.shape))
    #print(input.shape)
    #print(target.shape)
    # num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    # positive=num/torch.sum(torch.ones_like(input))
    #print(positive.item())
    loss=nn.MSELoss()
    relation=loss(input,target)
    #mean=torch.abs(torch.mean(input)-torch.mean(target))
    #print("pre_depth:%.4f,ground_depth:%.4f"%(torch.mean(input[1]).data.cpu().numpy().astype('float32'),torch.mean(target).data.cpu().numpy().astype('float32')))
    #output=relation+0.2*mean
    return relation
def log_loss(input, target, weight=None, size_average=True):
    # num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    # positive=num/torch.sum(torch.ones_like(input))
    # print(positive.item())
    target=torch.reshape(target,(input.shape))
    loss=nn.MSELoss() 
    input=torch.log(input+1e-12) 
    target=torch.log(target+1e-12) 
    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target) 
    d=0.5*torch.pow(torch.sum(input-target),2)/torch.pow(torch.sum(torch.ones_like(input)),2)
    #relation=relation-d 
    return relation

    # target=torch.reshape(target,(input.shape))
    # #loss=nn.MSELoss()
    # num=torch.sum(torch.where(input>0,torch.ones_like(input),torch.zeros_like(input)))
    # input=torch.log(torch.where(input>0,input,torch.ones_like(input)))
    # target=torch.log(torch.where(target>0,target,torch.ones_like(target)))
    # # #relation=torch.sqrt(loss(input,target))
    # relation=torch.sum(torch.pow(torch.where(input==0,input,input-target),2))/num
    # d=torch.pow(torch.sum(torch.where(input==0,input,input-target)),2)/torch.pow(num,2)*0.5
    # #positive=num/torch.sum(torch.ones_like(input))
    # #print(positive.item())
    # #-torch.sum(torch.where(input<0,input,torch.zeros_like(input)))/num
    # losses=relation+d
    # return losses

def log_l1(input, target, weight=None, size_average=True):
    l1loss=l1(input,target)
    logloss=log_loss(input,target)
    num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    positive=num/torch.sum(torch.ones_like(input))
    print(positive.item())
    loss=(1-positive)*logloss+positive*l1loss
    return loss
def l1_kitti(input, target, weight=None, size_average=True):
    zero=torch.zeros_like(input)
    target=torch.reshape(target,(input.shape))
    input=torch.where(target>0,input,zero)
    target=torch.where(target>0,target,zero)
    loss=nn.MSELoss(size_average=False) 
    relation=loss(input,target)/torch.sum(torch.where(target>0,torch.ones_like(input),zero))
    return relation
def log_kitti(input, target, weight=None, size_average=True):
    zero=torch.zeros_like(input)
    target=torch.reshape(target,(input.shape))
    loss=nn.MSELoss(size_average=False) 
    input=torch.where(target>0,torch.log(input),zero)
    target=torch.where(target>0,torch.log(target),zero)

    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target)/torch.sum(torch.where(target>0,torch.ones_like(input),zero))
    d=0.5*torch.pow(torch.sum(input-target),2)/torch.pow(torch.sum(torch.where(target>0,torch.ones_like(input),zero)),2)
 
    return relation-d 
# def region(input,target,instance):
#     loss=0
#     lf=nn.MSELoss(size_average=False,reduce=False)
#     target=torch.reshape(target,(input.shape))
#     instance=torch.reshape(instance,(input.shape))
#     zero=torch.zeros_like(input)
#     one=torch.ones_like(input)
#     dis=lf(input,target)
#     for i in range(0,int(torch.max(instance).item()+1)):
#         input_region=torch.where(instance==i,input,zero)
#         ground_region=torch.where(instance==i,target,zero)
#         m=torch.max(ground_region)
#         if m==0:
#             continue
#         num=torch.sum(torch.where(instance==i,one,zero))
#         loss+=lf(input_region,ground_region)/num
#         # average=torch.sum(input_region)/num
#         # input_region=input_region-average
#         # input_region=torch.pow(input_region,2)
#         # var=torch.sum(input_region)/num
#         # loss+=0.5*var
#     loss=loss/torch.max(instance)
#     return loss


def region(input,target,instance):
    loss=0
    lf=nn.MSELoss(size_average=False,reduce=False)
    target=torch.reshape(target,(input.shape))
    # input=torch.log(input+1e-12) 
    # target=torch.log(target+1e-12) 
    instance=torch.reshape(instance,(input.shape))
    zero=torch.zeros_like(input)
    one=torch.ones_like(input)
    dis=lf(input,target)
    for i in range(1,int(torch.max(instance).item()+1)):
        dis_region=torch.where(instance==i,dis,zero)
        num=torch.sum(torch.where(instance==i,one,zero))
        average=torch.sum(dis_region)/num
        loss=loss+average
        #dis_region=torch.where(instance==i,dis_region-average,zero)
        # var=0.1*torch.sqrt(torch.sum(torch.pow(dis_region,2))/num)/average
        # loss=loss+var
    loss=loss/(torch.max(instance))
    return loss

def region_log(input,target,instance):
    loss=0
    lf=nn.MSELoss(size_average=False,reduce=False)
    target=torch.reshape(target,(input.shape))
    input=torch.log(input+1e-6) 
    target=torch.log(target+1e-6) 
    instance=torch.reshape(instance,(input.shape))
    zero=torch.zeros_like(input)
    one=torch.ones_like(input)
    dis=lf(input,target)
    for i in range(1,int(torch.max(instance).item()+1)):
        dis_region=torch.where(instance==i,dis,zero)
        num=torch.sum(torch.where(instance==i,one,zero))
        average=torch.sum(dis_region)/num
        loss=loss+average
        # dis_region=torch.where(instance==i,dis_region-average,zero)
        # var=(torch.sum(torch.pow(dis_region,2))/num)/average
        # loss=loss+var
    loss=loss/(torch.max(instance))
    #print(torch.max(instance).item())
    return loss


def region_r(input,target,instance):
    loss=0
    relation=[]
    lf=nn.MSELoss(size_average=False,reduce=False)
    target=torch.reshape(target,(input[0].shape))
 
    target=torch.log(target+1e-6) 
    instance=torch.reshape(instance,(input[0].shape))
    zero=torch.zeros_like(input[0])
    one=torch.ones_like(input[0])
    for i in range(3):
        input[i]=torch.log(input[i]+1e-6)
        dis=lf(input[i],target)
        for i in range(1,int(torch.max(instance).item()+1)):
            dis_region=torch.where(instance==i,dis,zero)
            num=torch.sum(torch.where(instance==i,one,zero))
            average=torch.sum(dis_region)/num
            loss=loss+average
        #print(torch.max(instance).item())
        relation.append(loss/(torch.max(instance)))
        loss=0
    return relation