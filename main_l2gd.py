# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import pandas as pd
import numpy as np
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
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

    # generate list of local models for each user
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    net_local = copy.deepcopy(net_glob)

    loss_train = []
    test_freq = args.test_freq
    indd = None
    accs = []

    p = 0.9
    alpha = args.alpha_l2gd
    lr = alpha/(args.num_users*(1-p))
    lam = args.lambda_l2gd
    print("alpha, lambda: {}, {}".format(alpha, lam))
    accs10 = 0
    start = time.time()
    for iter in range(args.epochs+1):
        w_glob = {}
        loss_locals = []
        m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        
        u = np.random.uniform()
        # on the last iteration locally update all models
        if iter == args.epochs:
            u = 1
        # otherwise only update locally with probability 1-p which ensures fair computation budget in expectation with partial participation (note here is full participation)
        if u > p:
            
            for ind, idx in enumerate(idxs_users):
                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    local  = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])            

                net_local = copy.deepcopy(net_glob)
                w_local = net_local.state_dict()
                for k in w_locals[idx].keys():
                    w_local[k] = w_locals[idx][k]
                net_local.load_state_dict(w_local)
                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    w_local, loss, indd = local.train(net=net_local.to(args.device),ind=idx,mask=mask, idx=clients[idx], w_glob_keys=w_glob_keys, lr=lr)
                else:
                    w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx,mask=mask, w_glob_keys=w_glob_keys, lr=lr)
                loss_locals.append(copy.deepcopy(loss))
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_locals[idx][key] = w_local[key]
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

        else:
            for ind, idx in enumerate(idxs_users):
                if ind == 0:
                    w_glob = copy.deepcopy(w_locals[idx])
                    for k,key in enumerate(net_glob.state_dict().keys()):
                        w_glob[key] = w_glob[key]/args.num_users

                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_locals[idx][key]/args.num_users
            net_glob.load_state_dict(w_glob)
            # take step closer to average
            for ind, idx in enumerate(idxs_users):
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_locals[idx][key] = (1 - lr*lam)*w_locals[idx][key] + lr*lam*w_glob[key]
       
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                     w_locals=w_locals,fedavg=False,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
            accs.append(acc_test)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_ditto_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    base_dir = './save/accs_ditto_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
