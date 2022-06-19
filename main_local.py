# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_local.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import DatasetSplit, LocalUpdate
from models.test import test_img_local
import os
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    # training
    loss_train = []
    acc_avg = 0

    lr = args.lr
    results = []

    criterion = nn.CrossEntropyLoss()
    indd = None
    accs_=[]
    start = time.time()
    for user in range(min(100,args.num_users)):
        model_save_path = os.path.join('local/model_user{}.pt'.format(user))#base_dir,
        net_best = None
        best_acc = None
        net_local = copy.deepcopy(net_glob)
        if 'sent140' in args.dataset:
            local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[user]], idxs=dict_users_train, indd=indd)
            for iter in range(args.epochs):
                w_local, loss, indd = local.train(net=net_local.to(args.device), lr=args.lr)
            
            acc_test, loss_test = test_img_local(net_local, dataset_test, args,indd=indd, idx=list(dataset_test.keys())[user], user_idx=user)
            accs_.append(acc_test)
            print('User {} acc: {}'.format(user, acc_test))
            print(sum(accs_)/len(accs_))
            continue
         
        if 'femnist' in args.dataset:
            dataset=dataset_train[list(dataset_train.keys())[user]]
            ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=args.dataset), batch_size=args.local_bs, shuffle=True)
        else:
            ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[user]), args.local_bs, shuffle=True)

        optimizer = torch.optim.SGD(net_local.parameters(), lr=lr, momentum=0.5)
        for iter in range(args.epochs):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                net_local.zero_grad()
                log_probs = net_local(images)
                labels = labels.long()
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test, loss_test = test_img_local(net_local, dataset_test, args,indd=indd, idx=list(dataset_test.keys())[user], user_idx=user)
        else:
            acc_test, loss_test = test_img_local(net_local, dataset_test, args, user_idx=user, idxs=dict_users_test[user])
        accs_.append(acc_test)
        print('Average accuracy: {}'.format(sum(accs_)/len(accs_)))
        print('User {}, Loss: {:.2f}, Accuracy: {:.2f}'.format(user, loss_test, acc_test))
        acc_avg += acc_test/args.num_users

        del net_local

    end = time.time()
    print(end - start)
    print(accs_)
    print('Average accuracy: {}'.format(acc_avg))
