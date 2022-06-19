# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate, LocalUpdateScaffold
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
    
    lr = args.lr
    indd = None

    # generate list of local models for each user
    c_list = []
    for user in range(args.num_users+1):
        ci = {}
        for k in net_glob.state_dict().keys():
            ci[k] = torch.zeros(net_glob.state_dict()[k].size()).to(args.device)
        c_list.append(copy.deepcopy(ci))

    accs = []
    accs10 = 0
    accs10_glob = 0
    start = time.time() 

    w_glob = net_glob.state_dict()
    for iter in range(args.epochs+1):
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
            w_locals = {}
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        delta_c = {}
        delta_y = {}
        for k in w_glob.keys():
            delta_c[k] = torch.zeros(w_glob[k].size()).to(args.device)
            delta_y[k] = torch.zeros(w_glob[k].size()).to(args.device)
        for ind,idx in enumerate(idxs_users):
            if iter != args.epochs and ('femnist' in args.dataset or 'sent140' in args.dataset):
                local = LocalUpdateScaffold(args=args,indd=indd, dataset=dataset_train[list(dataset_train.keys())[idx]], idxs=dict_users_train, optim= optim)
            elif iter != args.epochs:
                local = LocalUpdateScaffold(args=args, dataset=dataset_train, idxs=dict_users_train[idx], optim=optim)
            elif 'femnist' in args.dataset or 'sent140' in args.dataset:
                local = LocalUpdate(args=args,indd=indd, dataset=dataset_train[list(dataset_train.keys())[idx]], idxs=dict_users_train, optim= optim) 
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx], optim=optim)

            net_local = copy.deepcopy(net_glob)
            w_glob_here = copy.deepcopy(w_glob)

            if args.epochs != iter:
                w_local, loss, indd, count = local.train(net=net_local.to(args.device), idx=idx, lr=lr, c_list=c_list)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, lr=lr, last=True)

            loss_locals.append(copy.deepcopy(loss))
            if iter == args.epochs:
                w_locals[idx] = copy.deepcopy(w_local)
            
            if args.alg == 'scaf':
                ci_new = {}
                for jj,k in enumerate(w_glob.keys()):
                    ci_new[k] = c_list[idx][k] - c_list[-1][k] +torch.div( (w_glob_here[k] - w_local[k]), count*args.lr)
                    delta_y[k] = delta_y[k] +  w_local[k] - w_glob_here[k]
                    delta_c[k] = delta_c[k] + ci_new[k] - c_list[idx][k]
            else:
                ci_new = {}
                for jj,k in enumerate(w_glob.keys()):
                    ci_new[k] = c_list[idx][k] - torch.div(c_list[-1][k], count) +torch.div( (w_glob_here[k] - w_local[k]),0.1)
                    delta_y[k] = delta_y[k] +  w_glob_here[k] - w_local[k]

            c_list[idx] = copy.deepcopy(ci_new)

        # update global weights
        for k in w_glob.keys():
            if args.alg == 'scaf':
                w_glob[k] += torch.mul(delta_y[k], args.lr_g/(m))
                c_list[-1][k] += torch.div(delta_c[k], args.num_users)
            else:
                c_list[-1][k] = torch.div(delta_y[k], m*args.lr)
                w_glob[k] -= torch.mul(c_list[-1][k], args.lr_g*args.lr)

        # copy weight to net_glob 
        if iter != args.epochs:
            net_glob.load_state_dict(copy.deepcopy(w_glob))

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
            accs.append(acc_test)

            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round: Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

            # below prints the global accuracy of the single global model for the relevant algs
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None, indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
            if iter != args.epochs:
                print('Round {:3d}: Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round: Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)

