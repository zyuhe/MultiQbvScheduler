'''
-*- coding: utf-8 -*-
@Time    :   2025/1/22 23:24
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   a3c.py
'''

import threading
from typing import List
import torch
import torch.optim as optim
import copy

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase
from src.a3c.net import ActorCriticNet
from src.a3c.worker import Worker

class A3C:
    def __init__(self, topology: TopologyBase, mstreams: List[MStream]):
        self.topology = topology
        self.mstreams = mstreams
        self.global_net = ActorCriticNet(len(self.topology.nodes), len(self.mstreams))  # 全局网络
        self.optimizer = optim.Adam(self.global_net.parameters(), lr=0.001)
        self.global_episode = 0

        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device('cpu')

    def train(self, num_workers=1, iter_num=600):
        workers = []
        threads = []
        for _ in range(num_workers):
            # worker = Worker(self.global_net, self.topology, self.mstreams, self.optimizer,
            #                 self.global_episode, self.device)
            worker = Worker(self.global_net, copy.deepcopy(self.topology), copy.deepcopy(self.mstreams), self.optimizer,
                            self.global_episode, self.device)
            workers.append(worker)
            thread = threading.Thread(target=worker.train, args=(iter_num,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

