'''
-*- coding: utf-8 -*-
@Time    :   2025/1/22 23:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   worker.py
'''

import torch
from typing import List
import numpy as np
import random
import networkx
from threading import Lock

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase
from common.funcs import compute_seg_path_dict, update_node_neigh_info, calc_ideal_add_latency, update_node_win_info
from common.parser import check_and_draw_topology
from src.a3c.net import ActorCriticNet

class Worker:
    def __init__(self, global_net, topology: TopologyBase, mstreams: List[MStream], optimizer, global_episode, device):
        self.mstreams = mstreams  # 多个流
        self.actions = np.arange(0, len(self.mstreams))  # 创建并初始化动作空间
        self.topology = topology  # 网络拓扑结构
        self.topology_graph = check_and_draw_topology(topology)
        self.win_plus = 1000
        self.global_net = global_net  # 全局网络
        self.epsilon = 0.9
        self.final_epsilon = 0.05
        self.local_net = ActorCriticNet(len(self.topology.nodes), len(self.mstreams))  # 每个工作线程一个本地网络
        self.local_net.load_state_dict(self.global_net.state_dict())  # 初始化为全局网络的权重
        # self.optimizer = optimizer    # 全局优化器
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=1e-4)  # 不共享优化器
        self.lock = Lock()  # 用于同步对优化器的访问
        self.global_episode = global_episode
        self.device = device

        self.best_latency_history = []
        # 记录训练得到的最优路线和最差路线
        self.good = {'mstream_order': [], 'total_latency': 0, 'episode': 0}
        self.bad = {'mstream_order': [], 'total_latency': 0, 'episode': 0}

    def choose_action(self, mstream_order, state, epsilon):
        # 使用 epsilon-greedy 策略
        with torch.no_grad():
            policy, _ = self.local_net(state)  # 获取当前状态下的每个动作的概率分布

            if random.random() <= epsilon:
                # 随机选择一个动作，确保它没有出现在已选择的 mstream_order 中
                available_actions = [x for x in self.actions if x not in mstream_order]
                action = random.choice(available_actions)  # 从可用的动作中选择
            else:
                # 选择最大概率的动作，但排除已经选择的 mstream
                for mstream_id in mstream_order:
                    policy[0][mstream_id] = float('-inf')  # 将已选的 mstream 对应的概率设为负无穷

                # 选择最大概率的动作
                action = torch.argmax(policy).item()

        return action

    def compute_loss(self, state, action, reward, next_state, done):
        policy, value = self.local_net(state)
        _, next_value = self.local_net(next_state)

        # 计算优势：Advantage = reward + gamma * next_value - value
        advantage = reward + 0.9 * next_value * (1 - done) - value
        value_loss = advantage.pow(2)  # 价值损失：目标值与当前值之间的差异

        # 策略损失：用log probability计算
        log_prob = torch.log(policy.squeeze(0)[action])
        policy_loss = -log_prob * advantage.detach()  # 策略梯度损失

        # 总损失：策略损失 + 价值损失
        loss = policy_loss + 0.5 * value_loss
        return loss

    def update_global(self, loss):
        with self.lock:
            # 反向传播更新全局网络
            self.optimizer.zero_grad()
            loss.backward()
            for local_param, global_param in zip(self.local_net.parameters(), self.global_net.parameters()):
                global_param.grad = local_param.grad
            self.optimizer.step()

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
        mstream = self.mstreams[int(action)]
        add_latency, ideal_add_latency = self.update_mstream_gcl(mstream)
        if add_latency <= 0:
            reward = -10000 # ???
        else:
            # TODO：update reward
            # reward = -1 * add_latency / ideal_add_latency * mstream.size
            reward = -2 * (add_latency - ideal_add_latency) * mstream.size
            reward = -1 * (add_latency - ideal_add_latency)
        reward = torch.tensor([reward], device=self.device)
        # compute new state
        next_state = state.clone().to(self.device)
        for node in self.topology.nodes:
            for nei_id in node.neighbor_node_ids:
                # update state[node.id][nei_id]
                port = node.get_port_by_neighbor_id(nei_id)
                next_state[0][node.id][nei_id] = port.remaining_resorce()
        done = True if ok_num == len(self.mstreams) - 1 else False
        return next_state, add_latency, reward, done

    def update_stream_and_topology_winInfo(self):
        for mstream in self.mstreams:
            mstream.clean_winInfo()
        self.topology.clear_all_nodes_winInfo()

    def train(self, iter_num=1000):
        for episode in range(iter_num):
            mstream_order = []
            add_latency_list = []
            round_total_latency = 0
            # 初始化狀態
            state = torch.full((1, len(self.topology.nodes), len(self.topology.nodes)), -1, device=self.device,
                               dtype=torch.float32)
            for node in self.topology.nodes:
                for nei_id in node.neighbor_node_ids:
                    state[0][node.id][nei_id] = 1
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(mstream_order, state, self.epsilon)
                next_state, add_latency, reward, done = self.Transform(state, action, len(mstream_order))
                if add_latency == -1:
                    # self.replay_buffer.push(state, action, reward, -np.inf, next_state)
                    pass
                else:
                    total_reward += reward
                    add_latency_list.append(add_latency)
                    round_total_latency += add_latency
                    if done:
                        pass
                        # long_term_reward = torch.tensor([-round_total_latency/1000], device=self.device)
                    else:
                        # long_term_reward = torch.tensor([-np.inf], device=self.device)
                        state = next_state
                    mstream_order.append(int(action))
                    loss = self.compute_loss(state, action, reward, next_state, done)
                    self.update_global(loss)
            # 衰减
            if self.epsilon > self.final_epsilon:
                self.epsilon *= 0.997
            # 每隔一定的 episode 更新全局网络
            if episode % 100 == 0:
                print(f"Episode {episode} total reward: {total_reward}")
            self.update_stream_and_topology_winInfo()
            self.best_latency_history.append(round_total_latency)
            # 记录最好成绩和最坏成绩
            if round_total_latency <= np.min(self.best_latency_history):
                self.good['mstream_order'] = mstream_order.copy()
                self.good['total_latency'] = round_total_latency
                self.good['all'] = add_latency_list.copy()
                self.good['episode'] = episode + 1
            if round_total_latency >= np.max(self.best_latency_history):
                self.bad['mstream_order'] = mstream_order.copy()
                self.bad['total_latency'] = round_total_latency
                self.bad['all'] = add_latency_list.copy()
                self.bad['episode'] = episode + 1
        # 打印训练结果
        print('\n', "result".center(40, '='))
        print('训练中出现的最小时延：{},出现在第 {} 次训练中'.format(self.good['total_latency'],
                                                                    self.good['episode']))
        print("最短路线:", self.good['mstream_order'], "all:", self.good['all'])
        print('训练中出现的最大时延：{},出现在第 {} 次训练中'.format(self.bad['total_latency'],
                                                                    self.bad['episode']))
        print("最长路线:", self.bad['mstream_order'], "all:", self.bad['all'])
        import matplotlib.pyplot as plt
        plt.plot(self.best_latency_history, color='green', linewidth=2)
        plt.show()


