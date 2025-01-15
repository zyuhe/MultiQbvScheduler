'''
-*- coding: utf-8 -*-
@Time    :   2025/1/15 12:57
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   sa.py
'''

from math import exp
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx
import time
from typing import List

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase
from common.funcs import *
from common.parser import check_and_draw_topology
from common.plot import draw_gantt_chart

class SA(object):
    def __init__(self, topology: TopologyBase, mstreams: List[MStream]):
        self.topology = topology
        self.mstreams = mstreams
        self.topology_graph = check_and_draw_topology(topology)
        self.win_plus = 1000  # ns
        # sa parameters
        self.T_base = 50000
        self.T_end = 30
        self.anneal_rate = 0.95 #每次退火的比例 0.98
        self.generation = 100 # 每个温度的迭代次数

        self.total_latency = 0
        self.best_latency = np.inf
        self.best_latency_history = []
        self.best_path = []

    def init_stream_order(self):
        return random.sample([i for i in list(range(len(self.mstreams)))], len(self.mstreams))

    def generate_new_stream_order(self, stream_order_before):
        new_stream_order = []
        for i in range(len(stream_order_before)):
            new_stream_order.append(stream_order_before[i])
        cuta = random.randint(0, len(stream_order_before) - 1)
        cutb = random.randint(0, len(stream_order_before) - 1)
        new_stream_order[cuta], new_stream_order[cutb] = new_stream_order[cutb], new_stream_order[cuta]
        return new_stream_order

    def update_stream_and_topology_winInfo(self):
        for mstream in self.mstreams:
            mstream.clean_winInfo()
        self.topology.clear_all_nodes_winInfo()

    def calc_total_latency(self, mstream_order_list):
        total_latency = 0
        for mstream_id in mstream_order_list:
            mstream = self.mstreams[mstream_id]
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
            # result = self.update_node_win_info(mstream)
            self.total_latency = update_node_win_info(self.topology, mstream, self.win_plus, self.total_latency) # update self.total_latency
            if self.total_latency < 0:
                print("error update qbv")
                return -1
        total_latency = self.total_latency
        self.total_latency = 0
        self.update_stream_and_topology_winInfo()

        return total_latency

    def run(self):
        stream_order0 = self.init_stream_order()
        print("initial stream order,", stream_order0)
        T = self.T_base
        anneal_cnt = 0
        timeStart = time.time()
        while T > self.T_end:
            round_best_total_latency = np.inf
            old_total_latency = self.calc_total_latency(stream_order0)
            if old_total_latency < round_best_total_latency:
                round_best_total_latency = old_total_latency
            for i in range(self.generation):
                new_stream_order = self.generate_new_stream_order(stream_order0)
                new_total_latency = self.calc_total_latency(new_stream_order)
                # TODO： parameter to adjust
                df = (new_total_latency - old_total_latency) / 50
                if df >= 0:
                    rand = random.uniform(0, 1)
                    try:
                        exp_value = exp(df / T)
                    except Exception as e:
                        print(e)
                        break
                    if exp_value > 1e300:
                        break
                    if rand < 1 / (exp_value):
                        stream_order0 = new_stream_order
                        old_total_latency = new_total_latency
                        if new_total_latency < round_best_total_latency:
                            round_best_total_latency = new_total_latency
                else:
                    stream_order0 = new_stream_order
                    old_total_latency = new_total_latency
                    if new_total_latency < round_best_total_latency:
                        round_best_total_latency = new_total_latency
            T = T * self.anneal_rate
            anneal_cnt += 1
            self.best_latency_history.append(round_best_total_latency)
            if self.best_latency > round_best_total_latency:
                self.best_latency = round_best_total_latency
                self.best_path = stream_order0
            print(anneal_cnt, "annealing, T：", T, " best total latency：", round_best_total_latency)
        self.best_path = stream_order0
        timeEnd = time.time()
        print("algorithm use", timeEnd - timeStart, "seconds")




