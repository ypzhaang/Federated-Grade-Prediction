# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_mtl.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
import torch
from torch import nn

from utils.train_utils import get_model, get_data, read_data
from utils.options import args_parser
from models.Update import LocalUpdateMTL
from models.test import test_img_local_all

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

    if 'sent140' not in args.dataset:
        w_glob_keys = net_glob.weight_keys
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    else:
        net_keys = [*net_glob.state_dict().keys()]
        w_glob_keys = net_keys[1:]

    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))

    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            if key in w_glob_keys:
                w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    criterion = nn.CrossEntropyLoss()

    # training
    loss_train = []
    lr = args.lr
    accs10 = 0

    args.device = args.gpu
    m = max(int(args.frac * args.num_users), 1)
    i = torch.ones((m, 1))
    omega = I - 1 / m * i.mm(i.T)
    omega = omega ** 2
    omega = omega.to(args.device)

    W = [w_locals[0][key].flatten() for key in w_glob_keys]
    W = torch.cat(W)
    d = len(W)
    del W

    start = time.time()
    indd =None
    for iter in range(args.epochs + 1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        W = torch.zeros((d, m)).cuda()
        for idx, user in enumerate(idxs_users):
            W_local = [w_locals[user][key].flatten() for key in w_glob_keys]
            W_local = torch.cat(W_local)
            W[:, idx] = W_local

        for idx, user in enumerate(idxs_users):
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                local = LocalUpdateMTL(args=args,indd=indd, dataset=dataset_train[list(dataset_train.keys())[user]], idxs=dict_users_train)
            else:
                local = LocalUpdateMTL(args=args, dataset=dataset_train,indd=indd, idxs=dict_users_train[user])
           
            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            for k in w_locals[user].keys():
                w_local[k] = w_locals[user][k]
            net_local.load_state_dict(w_local)
            w_local, loss, indd = local.train(net=net_local.to(args.device), lr=lr,
                                        omega=omega, W_glob=W.clone(), idx=idx, w_glob_keys=w_glob_keys)
            for k in w_locals[user].keys():
                w_locals[user][k] = w_local[k]
            loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if 'sent140' in args.dataset:
            w_glob_new = copy.deepcopy(net_glob.state_dict())
            for user in range(args.num_users):       
                if user == 0:
                    for k in w_glob_keys:
                        w_glob_new[k] = w_locals[user][k]/args.num_users
                else:
                    for k in w_glob_keys:
                        w_glob_new[k] += w_locals[user][k]/args.num_users
            net_glob_new = copy.deepcopy(net_glob)    
            net_glob_new.load_state_dict(w_glob_new)
        else:
            net_glob_new = copy.deepcopy(net_glob)

        loss_max = max(loss_locals)
        user_max = idxs_users[np.argmax(np.asarray(loss_locals))]
        if iter % args.test_freq == 0 or iter >= args.epochs - 10:
            acc_test, loss_test = test_img_local_all(net_glob_new, args, dataset_test, dict_users_test,w_locals=w_locals,indd=indd,)
            
            if iter >= args.epochs - 10:
                accs10 += acc_pre/10
            if iter != args.epochs:
                print('Round {:3d}, Avg train loss {:.3f}, Test loss: {:.3f}, Test acc: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round, Avg train loss {:.3f}, Test loss: {:.3f}, Test acc: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))

    print('Average acc over final 10 rounds: {}'.format(accs10))
    end = time.time()
    print(end-start)
    print(accs)
    base_dir = './save/accs_mtl_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
