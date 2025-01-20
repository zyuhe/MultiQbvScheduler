'''
-*- coding: utf-8 -*-
@Time    :   2025/1/19 14:42
@Auther  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   agent.py
'''

import os
import sys
import numpy as np
import random
from collections import namedtuple, deque
from typing import List
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import flatten
from torchsummary import summary

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase
from common.funcs import *
from common.parser import check_and_draw_topology
from common.plot import draw_gantt_chart

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 50
batch_size = 32
memory_size = 50000
learning_rate = 0.001
gamma = 0.65
input_channel = 2

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'long_term_reward', 'next_state'))

class BaseNet(nn.Module):
    def __init__(self, net_num_nodes, output_size):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(net_num_nodes * net_num_nodes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, long_term_reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, long_term_reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, long_term_reward, next_state = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        long_term_reward = torch.FloatTensor(long_term_reward)
        next_state = torch.FloatTensor(next_state)
        return state, action, reward, long_term_reward, next_state

    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, topology: TopologyBase, mstreams: List[MStream], gamma=0.3, alpha=0.3, epsilon=0.9, final_epsilon=0.05, buffer_size=10000, batch_size=32):
        self.mstreams = mstreams
        self.topology = topology
        self.topology_graph = check_and_draw_topology(topology)
        self.win_plus = 1000  # ns
        self.CityNum = 20  # stream number
        self.best_latency_history = []
        self.best_path = []
        # dqn parameters
        self.gamma = gamma  # 折扣因子
        self.alpha = alpha  # 学习率
        self.epsilon = epsilon  # 初始探索率
        self.final_epsilon = final_epsilon  # 最终学习率

        self.device = torch.device("mps" if torch.cuda.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size
        self.actions = np.arange(0, len(self.mstreams))  # 创建并初始化动作空间
        self.input_size = len(self.topology.nodes) * len(self.topology.nodes)  # 状态空间大小
        self.output_size = len(self.mstreams)  # 动作空间大小
        self.q_network = BaseNet(self.input_size, self.output_size).to(self.device)
        self.target_network = BaseNet(self.input_size, self.output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.replay_buffer = ReplayBuffer(buffer_size)
        # 记录训练得到的最优路线和最差路线
        self.good = {'path': [0], 'distance': 0, 'episode': 0}
        self.bad = {'path': [0], 'distance': 0, 'episode': 0}

    # TODO: wrong solution, BaseNet input and output?
    def Choose_action(self, mstream_order, state, epsilon):
        available_actions = [x for x in self.actions if x not in mstream_order]
        if np.random.rand() <= epsilon:
            # action = np.random.choice(available_actions)
            action = torch.tensor([[random.choice(available_actions)]], dtyp=torch.long, device=self.device)
        else:
            with torch.no_grad():
                action_values = self.q_network(state)
                for mstream_id in mstream_order:
                    action_values[0][mstream_id] = float('-inf')
                action = action_values.max(1)[1].view(1, 1)
        return int(action)

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
                    if index!= 0 and self.topology.get_node(path[index]).end_device == 1:
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

    def Transform(self, state, action, ok_num):
        mstream = self.mstreams[action]
        add_latency, ideal_add_latency = self.update_mstream_gcl(mstream)
        if add_latency <= 0:
            reward = -10000 # ???
        else:
            # TODO：update reward
            reward = -1 * round(add_latency / ideal_add_latency, 2) * mstream.size
        # compute new state
        next_state = state.clone()
        for node in self.topology.nodes:
            for nei_id in node.neighbor_node_ids:
                # update state[node.id][nei_id]
                port = node.get_port_by_neighbor_id(nei_id)
                next_state[node.id][nei_id] = port.remaining_resorce()
        done = True if ok_num == len(self.mstreams) - 1 else False
        return next_state, add_latency, reward, done

    def Train_Qtable(self, iter_num=1000, target_update=10):
        t1 = time.perf_counter()  # 用于进度条
        plot_iter_nums = []  # 用于绘训练效果图，横坐标集合
        self.iter_num = iter_num
        # 大循环-走iter_num轮
        for iter in range(iter_num):
            mstream_order = []
            round_total_latency = 0
            # 初始化狀態
            state = torch.full((len(self.topology.nodes), len(self.topology.nodes)), -1, device=self.device)
            for node in self.topology.nodes:
                for nei_id in node.neighbor_node_ids:
                    state[node.id][nei_id] = 1
            flag_done = False  # 完成标志
            round_reward = 0
            # 小循环-走一轮
            while flag_done == False:  # 没完成
                action = self.Choose_action(mstream_order, state, self.epsilon)
                next_state, add_latency, reward, flag_done = self.Transform(state, action, len(mstream_order))
                if add_latency == -1:
                    self.replay_buffer.push(state, action, reward, -np.inf, next_state)
                else:
                    round_reward += reward
                    round_total_latency += add_latency
                    if flag_done:
                        self.replay_buffer.push(state, action, reward, -round_total_latency / 1000, next_state)
                    else:
                        self.replay_buffer.push(state, action, reward, -np.inf, next_state)
                    mstream_order.append(action)
                    if len(self.replay_buffer) >= self.batch_size:
                        states, actions, rewards, long_term_reward, next_states = self.replay_buffer.sample(self.batch_size)
                        state_action_values = self.q_network(states).gather(1, actions)
                        with torch.no_grad():
                            next_state_values = self.target_network(next_states).max(1)[0]
                        expect_state_action_values = rewards + self.gamma * next_state_values
                        expect_state_action_values = expect_state_action_values.detach()
                        loss = F.mse_loss(state_action_values, expect_state_action_values.unsqueeze(1))
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                # 衰减
                if self.epsilon > self.final_epsilon:
                    self.epsilon *= 0.997
            # 更新目标网络
            if iter % target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            self.update_stream_and_topology_winInfo()
            plot_iter_nums.append(iter + 1)
            self.best_latency_history.append(round_total_latency)
            # 记录最好成绩和最坏成绩
            if round_total_latency <= np.min(self.best_latency_history):
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
        print('\n', "dqn_tsp result".center(40, '='))
        print('训练中出现的最小时延：{},出现在第 {} 次训练中'.format(self.good['total_latency'], self.good['episode']))
        print("最短路线:", self.good['mstream_order'])
        print('训练中出现的最大时延：{},出现在第 {} 次训练中'.format(self.bad['total_latency'], self.bad['episode']))
        print("最长路线:", self.bad['mstream_order'])


