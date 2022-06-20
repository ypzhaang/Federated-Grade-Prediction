# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)


import copy
import itertools
import numpy as np
from PIL import Image
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st
from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all
import time
import random

if __name__ == '__main__':
    st.sidebar.title("页面设置")
    choice_1=st.sidebar.radio("选择页面",['主页','训练','可视化'])
    st.sidebar.title("训练参数设置")
    col5, col6 = st.sidebar.columns(2)
    with col5:
        num_users = st.number_input('客户端数量', min_value=5, max_value=200, step=5, format="%d")
        epochs = st.number_input('通信轮数', min_value=2, format="%d")
        local_ep = st.number_input('训练次数', min_value=10, format="%d")
    with col6:
        frac = st.slider('参与训练的比例', min_value=0.1, max_value=1.0, step=0.1)
        batch_size = st.slider('学习率', min_value=0.01, max_value=1.0, step=0.01)
        alg = st.selectbox('联邦学习算法', [("联邦成绩预测")])

    if(choice_1=='主页'):
        st.title("联邦学习可视化平台")
        st.subheader("题目：基于联邦表示学习的高校学生成绩预测研究")
        st.caption("设计者：李钰心")
        st.info('请在下方上传您的数据文件，在侧边栏中调整训练参数，点击“训练”按钮，将会立即开始训练')
        st.warning('以下是数据文件的样本格式，-1代表该学生未选择此门课程，样本数据中已清除敏感信息')
        df = pd.read_csv('data/样本数据.csv')
        st.dataframe(df)
        uploaded_files = st.file_uploader("请上传您的数据文件，支持csv、mat等二进制格式的数据文件", accept_multiple_files=True)
        # if uploaded_file is not None:
        for uploaded_file in uploaded_files:
            if len(uploaded_files) < num_users + 1:
                bytes_data = uploaded_file.read()
            else:
                st.error('您选择的文件数目与客户端设置数目不符，请重新上传文件！！！！！')
                st.stop()
                flag=1

    elif(choice_1=='训练'):
        st.title("开始联邦学习")
        st.subheader("模型参数如下所示：")

        # parse args
        args = args_parser()
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        args.local_ep = local_ep

        lens = np.ones(num_users)
        if 'cifar' in args.dataset or args.dataset == 'mnist':
            dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all = get_data(args)
            print('----rand_set_all--------')
            print(rand_set_all)
            for idx in dict_users_train.keys():
                np.random.shuffle(dict_users_train[idx])

        else:
            if 'femnist' in args.dataset:
                train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
                test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
            else:
                train_path = './data/' + args.dataset + '/train'
                test_path = './data/' + args.dataset + '/test'
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
        print(args.alg)

        # build model
        net_glob = get_model(args)
        st.code(net_glob, language='python')
        
        # 分为两部分，一部分客户端，一部分服务器
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("客户端训练情况")
        with col2:
            st.subheader("服务器训练情况")

        net_glob.train()
        if args.load_fed != 'n':
            fed_model_path = './save/' + args.load_fed + '.pt'
            net_glob.load_state_dict(torch.load(fed_model_path)) # 用于将预训练的参数权重加载到新的模型之中
        total_num_layers = len(net_glob.state_dict().keys())
        print(net_glob.state_dict().keys())
        net_keys = [*net_glob.state_dict().keys()]

        # specify the representation parameters (in w_glob_keys) and head parameters (all others)
        if args.alg == 'fedrep' or args.alg == 'fedper':
            if 'cifar' in  args.dataset:
                w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
            elif 'mnist' in args.dataset:
                w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
                w_opsubspace_keys = [net_glob.weight_keys[i] for i in [3, 4, 5]]  # [0,1,2]
            elif 'sent140' in args.dataset:
                w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
            else:
                w_glob_keys = net_keys[:-2]
        elif args.alg == 'lg':
            if 'cifar' in  args.dataset:
                w_glob_keys = [net_glob.weight_keys[i] for i in [1,2]]
            elif 'mnist' in args.dataset:
                w_glob_keys = [net_glob.weight_keys[i] for i in [2,3]]
            elif 'sent140' in args.dataset:
                w_glob_keys = [net_keys[i] for i in [0,6,7]]
            else:
                w_glob_keys = net_keys[total_num_layers - 2:]

        if args.alg == 'fedavg' or args.alg == 'prox':
            w_glob_keys = []
        if 'sent140' not in args.dataset:
            w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
            w_opsubspace_keys = list(itertools.chain.from_iterable(w_opsubspace_keys))  # 去除内嵌
    
        print(total_num_layers)
        print(w_glob_keys)
        print(w_opsubspace_keys)
        print(net_keys)
        if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
            num_param_glob = 0
            num_param_local = 0
            for key in net_glob.state_dict().keys():
                num_param_local += net_glob.state_dict()[key].numel()
                # print(num_param_local)
                if key in w_glob_keys:
                    num_param_glob += net_glob.state_dict()[key].numel()
            percentage_param = 100 * float(num_param_glob) / num_param_local
            print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
                num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
        print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

        # generate list of local models for each user
        net_local_list = []
        w_locals = {}
        for user in range(num_users):
            w_local_dict = {}
            for key in net_glob.state_dict().keys():
                w_local_dict[key] =net_glob.state_dict()[key]
            w_locals[user] = w_local_dict

        # training
        indd = None      # indices of embedding for sent140
        loss_train = []
        times = []
        accs10 = 0
        accs = []
        losses = []
        accs10_glob = 0
        start = time.time()
        for iter in range(epochs+1):
            w_glob = {}
            loss_locals = []
            m = max(int(frac * num_users), 1)
            if iter == epochs:
                m = num_users

            idxs_users = np.random.choice(range(num_users), m, replace=False)
            w_keys_epoch = w_glob_keys
            times_in = []
            total_len=0
            li_accuracy=[]
            print(idxs_users)
            for ind, idx in enumerate(idxs_users):
                start_in = time.time()
                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    if epochs == iter:
                        local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                    else:
                        local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
                else:
                    if epochs == iter:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                    else:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

                net_local = copy.deepcopy(net_glob)
                w_local = net_local.state_dict()
                if args.alg != 'fedavg' and args.alg != 'prox':
                    for k in w_locals[idx].keys():
                        if k not in w_glob_keys:
                            w_local[k] = w_locals[idx][k]
                net_local.load_state_dict(w_local)
                last = iter == epochs

                if 'femnist' in args.dataset or 'sent140' in args.dataset:
                    w_local, loss, indd, local_acc = local.train(net=net_local.to(args.device),ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, lr=args.lr,last=last)
                else: 
                    w_local, loss, indd, local_acc = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys,
                                                      w_opsubspace_keys=w_opsubspace_keys, lr=args.lr, last=last)
                loss_locals.append(copy.deepcopy(loss))
                li_accuracy.append(local_acc)

                total_len += lens[idx]
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                    for k,key in enumerate(net_glob.state_dict().keys()):
                        w_glob[key] = w_glob[key]*lens[idx]
                        w_locals[idx][key] = w_local[key]
                else:
                    for k,key in enumerate(net_glob.state_dict().keys()):
                        if key in w_glob_keys:
                            w_glob[key] += w_local[key]*lens[idx]
                        else:
                            w_glob[key] += w_local[key]*lens[idx]
                        w_locals[idx][key] = w_local[key]

                times_in.append( time.time() - start_in )
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            # get weighted average for global weights
            for k in net_glob.state_dict().keys():
                w_glob[k] = torch.div(w_glob[k], total_len)

            w_local = net_glob.state_dict()
            for k in w_glob.keys():
                w_local[k] = w_glob[k]
            if epochs != iter:
                net_glob.load_state_dict(w_glob)

            if iter % args.test_freq==args.test_freq-1 or iter>=epochs-10:
                if times == []:
                    times.append(max(times_in))
                else:
                    times.append(times[-1] + max(times_in))
                flag_2 = iter == epochs
                acc_test, loss_test = test_img_local_all(net_glob, num_users, args, dataset_test, dict_users_test,
                                                            w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False,flag=flag_2)
                placeholder = st.empty()
                with placeholder.container():
                    col3, col4 = st.columns(2)
                    with col3:
                        latest_iteration = st.empty()
                        bar = st.progress(0)

                        for i in range(m):
                            # Update the progress bar with each iteration.
                            latest_iteration.write(f'开始训练， 处理进度 {round(((i + 1) / m),1)*100} %')
                            bar.progress((i + 1) / m)
                            time.sleep(0.5)
                        '客户端训练已结束！客户端的训练精度如下：'
                        li_accuracy = np.array(li_accuracy).reshape(-1, (len(li_accuracy)))
                        li_accuracy = pd.DataFrame(li_accuracy).transpose()
                        st.write(li_accuracy)

                    with col4:
                        with st.spinner('正在上传服务器进行处理...'):
                            time.sleep(3)
                        st.success('服务器处理结束!')
                        st.write(f'交流轮数： {iter}，  预测精度：{round(acc_test, 3)}')

                accs.append(acc_test)
                losses.append(loss_test)

                # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
                if iter != epochs:
                    print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                            iter, loss_avg, loss_test, acc_test))
                else:
                    # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                    print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
                if iter >= epochs-10 and iter != epochs:
                    accs10 += acc_test/10



                # below prints the global accuracy of the single global model for the relevant algs
                if args.alg == 'fedavg' or args.alg == 'prox':
                    acc_test, loss_test, conf_matrix = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                            w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                    if iter != epochs:
                        print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            iter, loss_avg, loss_test, acc_test))
                    else:
                        print('Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
                if iter >= epochs-10 and iter != epochs:
                    accs10_glob += acc_test/10

            if iter % args.save_every==args.save_every-1:
                model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
                torch.save(net_glob.state_dict(), model_save_path)

        print('Average accuracy final 10 rounds: {}'.format(accs10))
        if args.alg == 'fedavg' or args.alg == 'prox':
            print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
        st.balloons()
        end = time.time()
        print(end-start)
        time_li = end-start
        print(times)
        st.success(f'全部训练已结束！本次训练总时长为 {round(time_li, 3)} s')
        base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(num_users) +'.csv'
        base_dir2 = './save/losses_' + args.alg + '_' +  args.dataset + str(num_users) +'.csv'
        accs = np.array(accs).reshape((len(accs), 1))
        losses = np.array(losses).reshape((len(losses), 1))
        accs = pd.DataFrame(accs,columns=['预测精度'])
        losses = pd.DataFrame(losses,columns=['损失函数'])
        accs.to_csv(base_dir, index=False)
        losses.to_csv(base_dir2, index=False)


    elif (choice_1 == '可视化'):
        st.title("可视化如下：")
        options = st.multiselect(
            '请选择您的可视化内容',
            ['预测精度', '损失函数', '混淆矩阵'])

        for i in range(len(options)):
            if (options[i] == '预测精度'):
                st.subheader("预测精度趋势图")
                accs = pd.read_csv('./save/accs_fedrep_mnist' + str(num_users) + '.csv')
                st.line_chart(accs)
            elif (options[i] == '损失函数'):
                st.subheader("函数损失趋势图")
                losses = pd.read_csv('./save/losses_fedrep_mnist' + str(num_users) + '.csv')
                st.line_chart(losses)
            elif (options[i] == '混淆矩阵'):
                st.subheader("混淆矩阵图")
                image = Image.open('./save/confusion_matrix.png')
                col7, col8 = st.columns([5, 1])
                col7.image(image, use_column_width=True)