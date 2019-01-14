# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-09-21 15:15:12
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

from pssm.models import get_model
from pssm.loader import get_loader, get_data_path
from pssm.metrics import runningScore
from pssm.loss import *
from pssm.augmentations import *
import os

def train(args):
    #torch.backends.cudnn.benchmark=True
    # Setup Augmentations
    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])
    loss_rec=[0]
    best_error=2
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True,
                           split='train', img_size=(args.img_rows, args.img_cols))
    v_loader = data_loader(data_path, is_transform=True,
                           split='test', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=0, shuffle=False)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=0)


    # Setup Model
    model = get_model(args.arch)
    # parameters=model.named_parameters()
    # for name,param in parameters:
    #     print(name)
    #     print(param.grad)
    # exit()

    # model = torch.nn.DataParallel(
    #     model, device_ids=range(torch.cuda.device_count()))
    #model = torch.nn.DataParallel(model, device_ids=[0])
    #model.cuda()

    # Check if model has custom optimizer / loss
    # modify to adam, modify the learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999),amsgrad=True)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.l_rate,momentum=0.90, weight_decay=5e-5)

    loss_fn = l1
    trained=0
    scale=100

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #model_dict=model.state_dict()  
            #opt=torch.load('/home/lidong/Documents/pssm/pssm/exp1/l2/sgd/log/83/rsnet_nyu_best_model.pkl')
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            #opt=None
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            trained=checkpoint['epoch']
    error=0

    #best_error=5
    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, trained+1):
    #for epoch in range(0, args.n_epoch):
        
        #trained
        print('training!')
        model.train()
        for i, (left, right,disparity,P,pre_match,pre2) in enumerate(valloader):
            with torch.no_grad():
                #print(left.shape)
                start_time=time.time()
                left = left.cuda(0)
                right = right.cuda(0)
                disparity = disparity.cuda(0)
                P = P.cuda(1)
                pre_match=pre_match.cuda(1)
                pre2=pre2.cuda(1)

                optimizer.zero_grad()
                #print(P.shape)
                outputs = model(left,right,P=P,pre=pre_match,pre2=pre2)

                #outputs=outputs
                loss = l1(input=outputs, target=disparity,mask=P)

                print(time.time()-start_time)
                torch.cuda.empty_cache()

            #     pre = outputs.data.cpu().numpy().astype('float32')
            #     pre = pre[0, :, :, :]
            #     #pre = np.argmax(pre, 0)
            #     pre = (np.reshape(pre, [480, 640]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))
            #     #pre = pre/np.max(pre)
            #     # print(type(pre[0,0]))
            #     vis.image(
            #         pre,
            #         opts=dict(title='predict!', caption='predict.'),
            #         win=pre_window,
            #     )
            #     ground=disparity.data.cpu().numpy().astype('float32')
            #     #print(ground.shape)
            #     ground = ground[0, :, :]
            #     ground = (np.reshape(ground, [480, 640]).astype('float32')-np.min(ground))/(np.max(ground)-np.min(ground))
            #     vis.image(
            #         ground,
            #         opts=dict(title='ground!', caption='ground.'),
            #         win=ground_window,
            #     )
            
            print("data [%d/812/%d/%d] Loss: %.4f" % (i, epoch, args.n_epoch,loss.item()))
            error+=loss.item()
    print("error is %.4f"%(error/812))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rstereo',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='sceneflow',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=480,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=640,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/PSSM/14_rstereo_sceneflow_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/PSSM/rstereo_sceneflow_best_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
