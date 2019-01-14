# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-11-12 23:00:30
import sys
import torch
import visdom
import argparse
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import torch.nn.functional as F
from cmf.models import get_model
from cmf.loader import get_loader, get_data_path
from cmf.loss import *
import os

def train(args):
    torch.backends.cudnn.benchmark=True
    # Setup Augmentations
    loss_rec=[0]
    best_error=2
    # Setup Dataloader
    data_path = get_data_path(args.dataset)
    data_loader = get_loader(args.dataset)
    v_loader = data_loader(data_path, is_transform=True,
                           split='test', img_size=(args.img_rows, args.img_cols))
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=2, shuffle=False)
    # Setup Model
    model = get_model(args.arch)

    model = torch.nn.DataParallel(
        model, device_ids=range(2))
    test_length=v_loader.__len__()/2
    #model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()


    saved_model_path=r'/home/lidong/Documents/CMF/all_data/test/'
    saved_model_dir=os.listdir(saved_model_path)
    saved_model_dir.sort()
    for s in range(len(saved_model_dir)):
        print("Loading model and optimizer from checkpoint '{}'".format(os.path.join(saved_model_path,saved_model_dir[s])))
        checkpoint = torch.load(os.path.join(saved_model_path,saved_model_dir[s]))
        model.load_state_dict(checkpoint['model_state'])
        print("Loaded checkpoint '{}' (epoch {})"
              .format(os.path.join(saved_model_path,saved_model_dir[s]), checkpoint['epoch']))
        epoch=checkpoint['epoch']
        error=0
        error_rec=[]
        error_rec_non=[]
        error_rec_true=[]
        #trained
        print('training!')
        model.eval()
        for i, (left, right,disparity,image) in enumerate(valloader):
            with torch.no_grad():
                start_time=time.time()
                left = left.cuda()
                right = right.cuda()
                disparity = disparity.cuda()[:,:540,:960]
                local=torch.arange(disparity.shape[-1]).repeat(disparity.shape[0],disparity.shape[1],1).view_as(disparity).float().cuda()
                mask_non = (disparity < 192) & (disparity >= 0) &((local-disparity)>=0)
                mask_true = (disparity < 192) & (disparity > 0)&((local-disparity)>=0)
                mask = (disparity < 192) & (disparity >= 0)
                mask.detach_()
                mask_non.detach_()
                mask_true.detach_()
                #print(P.shape)
                #print(left.shape)
                output1, output2, output3 = model(left,right)
                #output3 = model(left,right)
                #print(output3.shape)
                output1=output3
                output1 = torch.squeeze(output1, 1)[:,:540,:960]
                # print(output3.shape,disparity.shape)
                # exit()
                # print(torch.sum(torch.where(disparity==0,torch.ones(1).cuda(),torch.zeros(1).cuda())))
                # print(torch.sum(torch.where(disparity<=1,torch.ones(1).cuda(),torch.zeros(1).cuda())))
                # print(torch.sum(torch.where(disparity<=2,torch.ones(1).cuda(),torch.zeros(1).cuda())))
                # print(torch.sum(torch.where(disparity<=3,torch.ones(1).cuda(),torch.zeros(1).cuda())))
                # print(disparity.shape)
                #output3=torch.where(output3<1,torch.zeros(1).cuda(),output3)
                loss=torch.mean(torch.abs(output1[mask] - disparity[mask]))
                loss_non=torch.mean(torch.abs(output1[mask_non] - disparity[mask_non]))
                loss_true=torch.mean(torch.abs(output1[mask_true] - disparity[mask_true]))
                #loss = F.l1_loss(output3[mask], disparity[mask], reduction='elementwise_mean')
                error_rec.append(loss.item())
                error_rec_non.append(loss_non.item())
                error_rec_true.append(loss_true.item())
            print(time.time()-start_time)
            print("data [%d/%d/%d/%d] Loss: %.4f ,Loss_non: %.4f, Loss_true: %.4f" % (i, test_length,epoch, args.n_epoch,loss.item(),loss_non.item(),loss_true.item()))
            #break
        error=np.mean(error_rec)
        error_non=np.mean(error_rec_non)
        error_true=np.mean(error_rec_true)
        np.save('/home/lidong/Documents/CMF/all_data_test/4_sub/epoch:%d_error%.4f_non%.4f_true%.4f.npy'%(epoch-1,error,error_non,error_true),[error_rec,error_rec_non,error_rec_true])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='cmfsm',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='flying3d',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=540,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=960,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=4000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/CMF/1_cmf_flying3d_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/PSSM/rstereo_sceneflow_best_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
