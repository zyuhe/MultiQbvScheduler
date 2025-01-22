'''
-*- coding: utf-8 -*-
@Time    :   2025/1/22 23:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   net.py
'''

import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNet(nn.Module):
    def __init__(self, net_num_nodes, output_size):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(net_num_nodes * net_num_nodes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        # Actor: 策略网络
        self.policy_fc = nn.Linear(128, output_size)  # 输出动作的概率
        # Critic: 价值网络
        self.value_fc = nn.Linear(128, 1)  # 输出状态的价值

    def forward(self, x):
        x = x.view(-1, x.size(1) * x.size(2))  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 计算策略（概率分布）和价值（状态价值）
        policy = F.softmax(self.policy_fc(x), dim=-1)
        value = self.value_fc(x)
        return policy, value

