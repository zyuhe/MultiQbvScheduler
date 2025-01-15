'''
-*- coding: utf-8 -*-
@Time    :   2025/1/15 17:24
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   qlearning.py
'''

import math
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx
import time
from typing import List

from fontTools.varLib.mutator import percents

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase
from common.funcs import *
from common.parser import check_and_draw_topology
from common.plot import draw_gantt_chart

class QLearning:
    def __init__(self, topology: TopologyBase, mstreams: List[MStream], gamma=0.3, alpha=0.3, epsilon=0.9, final_epsilon=0.05):
        self.mstreams = mstreams
        self.topology = topology
        self.topology_graph = check_and_draw_topology(topology)
        self.win_plus = 1000  # ns
        self.CityNum = 20  # stream number
        self.best_latency_history = []
        self.best_path = []
        # ql parameters
        self.gamma = gamma  # 折扣因子
        self.alpha = alpha  # 学习率
        self.epsilon = epsilon  # 初始探索率
        self.final_epsilon = final_epsilon  # 最终学习率
        # 记录训练得到的最优路线和最差路线
        self.good = {'path': [0], 'distance': 0, 'episode': 0}
        self.bad = {'path': [0], 'distance': 0, 'episode': 0}
        self.actions = np.arange(0, len(self.mstreams))  # 创建并初始化动作空间
        self.Qtable = np.zeros((len(self.mstreams), len(self.mstreams)))  # 创建并初始化q表

    # 选择动作-episilon-greedy方法
    def Choose_action(self, mstream_order, epsilon, qvalue):
        # 判断是否完成
        if len(mstream_order) == len(self.mstreams):  # 若mstream_order中包含了全部的mstream则结束本轮计算
            return -1
        # 获取上一个流量到其他流量的q值，以epsilon的概率随机选择下一条流量
        q = np.copy(qvalue[mstream_order[-1], :])
        if np.random.rand() > epsilon:
            q[mstream_order] = -np.inf  # Avoid already visited states
            action = np.argmax(q)
            # max_indices = np.where(q == np.max(q))[0]  # 找出 q 中的最大值的索引
            # next_mstream_id = np.random.choice(max_indices)  # 在所有最大值中随机选取一个
        else:
            action = np.random.choice([x for x in self.actions if x not in mstream_order])  # 在没走过的节点中选一个
        return action

    def update_stream_and_topology_winInfo(self):
        for mstream in self.mstreams:
            mstream.clean_winInfo()
        self.topology.clear_all_nodes_winInfo()

    def update_mstream_gcl(self, mstream):
        # 1. calc route
        best_paths = list()
        for dst_node_id in mstream.dst_node_ids:
            paths = list(
                networkx.all_simple_paths(self.topology_graph, source=mstream.src_node_id, target=dst_node_id))
            paths = sorted(paths, key=len)
            available_paths = paths.copy()
            for path in paths:
                for index in range(len(path) - 1):
                    if index != 0 and self.topology.get_node(path[index]).end_device == 1:
                        available_paths.remove(path)
                        break
                    if mstream.vlan_id not in self.topology.get_node(path[index]).get_port_by_neighbor_id(
                            path[index + 1]
                    ).allowed_vlans:
                        available_paths.remove(path)
                        break
            if len(available_paths) == 0:
                print(f"==>WARNING: no viable path from {mstream.src_node_id} to {dst_node_id}!")
                print("             Please check stream and topology settings.")
                # TODO: error re_choose route
            # TODO: route select algorithm
            best_path = available_paths[0]
            best_paths.append(best_path)
        mstream.seg_path_dict = compute_seg_path_dict(best_paths)

        # 2. update gcl and compute latency
        update_node_neigh_info(self.topology, mstream)
        ideal_add_latency = calc_ideal_add_latency(self.topology, mstream, self.win_plus)
        add_latency = update_node_win_info(self.topology, mstream, self.win_plus)  # update self.total_latency
        if add_latency < 0:
            print("error update qbv")
            return -1, -1
        return add_latency, ideal_add_latency

    # 和环境交互返回(s,a)下的reward和flag_done
    def Transform(self, path, action):
        mstream = self.mstreams[action]
        new_state = action
        add_latency, ideal_add_latency = self.update_mstream_gcl(mstream)
        if add_latency < 0:
            return -1, -1, -1, False
        # TODO：update reward
        reward = -1 * round(add_latency / ideal_add_latency, 2) * mstream.size
        # reward = -1000 * round(add_latency / ideal_add_latency, 2)
        # reward = -1000 * round(ideal_add_latency / add_latency, 2)
        if len(path) == len(self.mstreams):
            return new_state, add_latency, reward, True
        return new_state, add_latency, reward, False

    def greedy_policy(self, path, qvalue):
        if len(path) >= len(self.mstreams):
            return -1
        q = np.copy(qvalue[path[-1], :])
        q[path] = -np.inf  # Avoid already visited states
        action = np.argmax(q)
        # max_indices = np.where(q == np.max(q))[0]  # 找出 q 中的最大值的索引
        # action = np.random.choice(max_indices)  # 在所有最大值中随机选取一个
        return action

    def Plot_train_process(self, iter_nums, dists):
        # plt.ion()
        plt.figure(1)
        # plt.subplot(212)
        plt.title(f"qlearning node_num:{len(self.mstreams)}")
        plt.ylabel("distance")
        plt.xlabel("iter_order")
        plt.plot(iter_nums, dists, color='blue')

        try:
            os.mkdir(os.getcwd() + "/src/qlearning/" + "png")  # 创建指定名称的文件夹
            print("[Plot_train_process] png文件夹创建成功")
        except:
            pass
            # print("[Plot_train_process] png文件夹已存在")
        # timestr = time.strftime("%Y-%m-%d %H：%M：%S")
        # save_path = f"src/qlearning/png/process{len(self.mstreams)} {timestr}"
        # plt.savefig(save_path + '.png', format='png')
        plt.show()

    # 训练智能体 s a r s
    def Train_Qtable(self, iter_num=1000):
        # 训练参数
        gamma = self.gamma  # 折扣因子
        alpha = self.alpha  # 学习率
        epsilon = self.epsilon  # 初始探索率
        t1 = time.perf_counter()  # 用于进度条
        qvalue = self.Qtable.copy()
        plot_iter_nums = []  # 用于绘训练效果图，横坐标集合
        self.iter_num = iter_num
        # 大循环-走iter_num轮
        for iter in range(iter_num):
            mstream_order = []  # 重置路线记录
            state = random.randint(0, len(self.mstreams) - 1)  # 初始化状态，即选择第一条流
            add_latency, _ = self.update_mstream_gcl(self.mstreams[state])
            round_total_latency = add_latency  # 本轮距离累加统计
            mstream_order.append(state)
            flag_done = False  # 完成标志
            round_reward = 0
            # 小循环-走一轮
            while flag_done == False:  # 没完成
                action = self.Choose_action(mstream_order, epsilon, qvalue)
                if action == -1:
                    break
                state_next, add_latency, reward, flag_done = self.Transform(mstream_order, action)
                round_reward += reward
                round_total_latency += add_latency
                mstream_order.append(state_next)
                # 更新Qtable
                if flag_done:
                    q_target = reward
                    qvalue[state, action] = qvalue[state, action] + alpha * (q_target - qvalue[state, action])
                    break
                else:
                    action1 = self.greedy_policy(mstream_order, qvalue)
                    q_target = reward + gamma * qvalue[state_next, action1]
                qvalue[state, action] = qvalue[state, action] + alpha * (q_target - qvalue[state, action])
                state = state_next # 状态转移
            self.update_stream_and_topology_winInfo()
            # 衰减
            if epsilon > self.final_epsilon:
                epsilon *= 0.997
                # epsilon -=(self.epsilon-self.final_epsilon)/iter_num
            plot_iter_nums.append(iter + 1)
            self.best_latency_history.append(round_total_latency)
            # 记录最好成绩和最坏成绩
            if round_total_latency <= np.min(self.best_latency_history):
                self.Qtable = qvalue.copy()
                self.good['mstream_order'] = mstream_order.copy()
                self.good['total_latency'] = round_total_latency
                self.good['episode'] = iter + 1
            if round_total_latency >= np.max(self.best_latency_history):
                self.bad['mstream_order'] = mstream_order.copy()
                self.bad['total_latency'] = round_total_latency
                self.bad['episode'] = iter + 1
            # 训练进度条
            percent = (iter + 1) / iter_num
            bar = '*' * int(percent * 30) + '->'
            delta_t = time.perf_counter() - t1
            pre_total_t = (iter_num * delta_t) / (iter + 1)
            left_t = pre_total_t - delta_t
            print('\r{:6}/{:6}\t训练已完成了:{:5.2f}%[{:32}]已用时:{:5.2f}s,预计用时:{:.2f}s,预计剩余用时:{:.2f}s'
                  .format((iter + 1), iter_num, percent * 100, bar, delta_t,
                          pre_total_t, left_t), end='')
        # 打印训练结果
        print('\n', "qlearning_tsp result".center(40, '='))
        print('训练中的出现的最小时延：{},出现在第 {} 次训练中'.format(self.good['total_latency'], self.good['episode']))
        print("最短路线:", self.good['mstream_order'])
        print('训练中的出现的最大时延：{},出现在第 {} 次训练中'.format(self.bad['total_latency'], self.bad['episode']))
        print("最长路线:", self.bad['mstream_order'])
        # 画训练效果图
        self.Plot_train_process(plot_iter_nums, self.best_latency_history)

    # 将Q表存入本地
    def Write_Qtable(self):
        try:
            os.mkdir(os.getcwd() + "/src/qlearning/" + "data")  # 创建指定名称的文件夹
            print("[write_txt_Qtable] data文件夹创建成功")
        except:
            pass
            # print("[write_txt_Qtable] data文件夹已存在")
        filename = f"src/qlearning/data/Qtable{len(self.mstreams)}.txt"
        with open(filename, 'w') as f:
            f.write(f"{len(self.mstreams)}\n")
            # print(f"{self.tsp_map.node_num}")
            for i in range(len(self.mstreams)):
                for j in range(len(self.mstreams)):
                    f.write(f"{self.Qtable[i][j]}\t")
                    # print(f"{self.Qtable[i][j]}\t",end='')
                f.write("\n")
                # print()
        print(f"[write_txt_Qtable] 已写入{len(self.mstreams)}*{len(self.mstreams)}的Q表")
