# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from matplotlib import rcParams
import torch
import matplotlib.pyplot as plt
from torch import nn
import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import itertools
import pandas as pd
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        # image, label = self.dataset[d]
        image = self.dataset[d][:-1]
        label = self.dataset[d][-1]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

# 生成混淆矩阵
def confusion_matrix(preds, labels, conf_matrix,flag=False):
    if(flag==1):
        a = len(preds)
        b = len(labels)
        A = preds[0]
        B = labels[0]
        for i in range(a):
            if i == a - 1:
                break
            else:
                A = torch.cat([A, preds[i + 1]], dim=0)
        for i in range(b):
            if i == b - 1:
                break
            else:
                B = torch.cat([B, labels[i + 1]], dim=0)

        for p, t in zip(A, B):
            conf_matrix[p, t] += 1
    return conf_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    config = {
        "font.family": 'serif',
        "font.size": 13,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('./save/confusion_matrix.png', transparent=True, dpi=800)
    # plt.show()

# num_li=0
def test_img_local(net_g, dataset, args,idx=None,indd=None, user_idx=-1, idxs=None,flag=False):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf=True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf=True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
    else:
        leaf=False
    
    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    label = []
    pre_label = []
    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            count += args.local_bs
            test_loss += loss.item()

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            # log_probs = net_g(data)
            log_probs, layer_hidden2, z_layer_hidden2, cat_layer, m = net_g(data)
            target = target.type(torch.LongTensor)
            label.append(target)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            pre_label.append(y_pred)
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss, pre_label, label

def test_img_local_all(net, num_users, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False,flag=False):
    tot = 0
    num_idxxs = num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    conf_matrix = torch.zeros(5, 5)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            a, b =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            a, b, pre_label, label = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx],flag=flag)
            tot += len(dict_users_test[idx])
            conf_matrix = confusion_matrix(pre_label, label, conf_matrix,flag=flag)
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = a*len(dict_users_test[idx])
            loss_test_local[idx] = b*len(dict_users_test[idx])
        del net_local

    if (flag == 1):
        attack_types = ['1', '2', '3', '4', '5']
        conf_matrix = conf_matrix.numpy()
        conf_matrix = conf_matrix.astype(np.int16)
        plot_confusion_matrix(conf_matrix, classes=attack_types, normalize=False)

    if return_all:
        return acc_test_local, loss_test_local
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot
