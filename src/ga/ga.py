'''
-*- coding: utf-8 -*-
@Time    :   2025/1/14 17:58
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   ga.py
'''

import math
import random
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

class GA(object):
    def __init__(self, topology: TopologyBase, mstreams: List[MStream]):
        self.mstreams = mstreams
        self.topology = topology
        self.topology_graph = check_and_draw_topology(topology)
        self.win_plus = 1000 # ns
        self.CityNum = 20 # stream number
        # GA parameters
        self.generation = 100 # 迭代次数
        self.popsize = 100 # 种群大小
        self.tournament_size = 5 # 锦标赛小组大小
        self.pc = 0.95 # 交叉概率
        self.pm = 0.1 # 变异概率

        self.total_latency = 0
        self.best_latency_history = []
        self.best_path = []

    def update_node_neigh_info(self, mstream: MStream):
        # print("mstream.seg_path_dict", mstream.seg_path_dict)
        value_list = list(mstream.seg_path_dict.values())
        mstream.windowsInfo[value_list[0][0][0]] = [-1, -1]
        for v in value_list:
            for seg in v:
                for i in range(len(seg) - 1):
                    pre_node_id = seg[i]
                    pre_port = self.topology.get_node(pre_node_id).get_port_by_neighbor_id(seg[i + 1])
                    mstream.windowsInfo[seg[i + 1]] = [pre_node_id, pre_port.id]

    def update_node_win_info(self, mstream: MStream):
        # for segs from a node
        uni_id = -1
        delay_start = []
        delays = []
        seg_num = 0
        for v in mstream.seg_path_dict.values():
            seg_num += len(v)
        # print("mstream：", mstream.id, "has", seg_num, "segs")
        for v in mstream.seg_path_dict.values():
            # a seg of segs
            for seg in v:
                uni_id += 1
                # node with gcl except for last of this seg
                for i in range(len(seg) - 1):
                    last_hop = False
                    if i == len(seg) - 2:
                        tnode = self.topology.get_node(seg[i + 1])
                        if tnode.end_device == 1:
                            last_hop = True
                    node = self.topology.get_node(seg[i])
                    port = node.get_port_by_neighbor_id(seg[i+1])
                    port.expand_hyper_period(mstream.hyper_period)
                    # if port.hyper_period > mstream.hyper_period:
                    #     print(" == old stream hp：", mstream.hyper_period, "new hp：", port.hyper_period)
                    mstream.update_winInfo(port.hyper_period)
                    new_hyper = port.hyper_period
                    if mstream.hyper_period > new_hyper:
                        new_hyper = mstream.hyper_period
                    times = int(new_hyper / mstream.interval)
                    # print("mstream", mstream.id, "interval", mstream.interval, "ms hyper", mstream.hyper_period, "phyper", port.hyper_period, "times", times)
                    # print("node", seg[i], ", port", port.id, ", hyper_period", port.hyper_period, "ns, winInfo", port.windowsInfo)
                    # if mstream.windowsInfo[seg[i]][0] ！= -1: # source_device
                    # insert gate into port.windowsInfo
                    # find earliest ts, get win_len
                    win_len = math.ceil(mstream.size * 8 * 1000 / port.port_speed + self.win_plus)
                    # print("mstream.size", mstream.size, "bytes, port.speed", port.port_speed, "Mbps, window length", win_len, "us")
                    for ti in range(times):
                        # print("ti =", ti)
                        # get pre node's gate close time as temp ts open
                        tmp_ts_open = ti * mstream.interval
                        first_node = True
                        # print(mstream.windowsInfo, seg[i])
                        if mstream.windowsInfo[seg[i]][0] >= 0:
                            pre_node = self.topology.get_node(mstream.windowsInfo[seg[i]][0])
                            pre_port = pre_node.get_port(mstream.windowsInfo[seg[i]][1])
                            # print("get pre node", pre_node.id, "get pre port", pre_port.id)
                            first_node = False
                            get_pre_ts = False
                            for winInfo in pre_port.windowsInfo:
                                if pre_port.hyper_period >= port.hyper_period:
                                    if winInfo[pre_port.MSTREAM_ID] == mstream.id and ti * mstream.interval <= winInfo[pre_port.TS_OPEN] and \
                                            winInfo[pre_port.TS_OPEN] + winInfo[pre_port.WIN_LEN] < (ti+1) * mstream.interval:
                                        tmp_ts_open = round(winInfo[pre_port.TS_OPEN] + winInfo[pre_port.WIN_LEN], 1)
                                        get_pre_ts = True
                                        # print("1from pre node port get tmp ts open：", tmp_ts_open)
                                        break
                                else:
                                    if winInfo[pre_port.MSTREAM_ID] == mstream.id:
                                        tmp_pre_ts_open = winInfo[pre_port.TS_OPEN] + ti / (pre_port.hyper_period / mstream.interval) * pre_port.hyper_period
                                        if ti * mstream.interval <= tmp_pre_ts_open <= (ti+1) * mstream.interval - winInfo[pre_port.WIN_LEN]:
                                            tmp_ts_open = round(tmp_pre_ts_open + winInfo[pre_port.WIN_LEN], 1)
                                            get_pre_ts = True
                                            # print("2from pre node port get tmp ts open：", tmp_ts_open)
                                            break
                                #else:
                                #    print("error find mstream win info of previous port")
                                #    print(ti * mstream.interval, winInfo[pre_port.TS_OPEN], winInfo[pre_port.WIN_LEN], (ti+1) * mstream.interval)
                            if not get_pre_ts:
                                print("error get previous gcl info, ti", ti, "mstream interval", mstream.interval, "hyper", mstream.hyper_period)
                                print("pre port info", pre_port.windowsInfo, "hyper", pre_port.hyper_period)
                                print("now port hyper", port.hyper_period)
                                return False
                        right_limit = (ti+1) * mstream.interval
                        insert_ok = False
                        for win_index in range(len(port.windowsInfo)):
                            cmp_close = port.windowsInfo[win_index][port.TS_OPEN] + port.windowsInfo[win_index][port.WIN_LEN]
                            if right_limit >= port.windowsInfo[win_index][port.TS_OPEN] >= tmp_ts_open + win_len:
                                # insert and update win info of topology and mstream
                                if not first_node:
                                    port.windowsInfo.append([
                                        tmp_ts_open, win_len, mstream.id, mstream.pcp, pre_node.id, pre_port.id])
                                else:
                                    port.windowsInfo.append([
                                        tmp_ts_open, win_len, mstream.id, mstream.pcp, -1, -1])
                                port.update_windows_info()
                                mstream.windowsInfo[seg[i]].append([
                                    port.id, tmp_ts_open, win_len, uni_id + ti * seg_num])
                                # print("update port info1")
                                #print(port.windowsInfo)
                                #print(mstream.windowsInfo)
                                insert_ok = True
                                break
                            elif cmp_close >= right_limit:
                                pass
                                # print("！！！error find mstream win info of this port, cmp close：", cmp_close, "tmp ts open", tmp_ts_open, "right limit：", right_limit)
                            elif tmp_ts_open < cmp_close:
                                tmp_ts_open = port.windowsInfo[win_index][port.TS_OPEN] + port.windowsInfo[win_index][port.WIN_LEN]
                                # print("update tmp ts open：", tmp_ts_open, "from port wininfo", port.windowsInfo)
                        if not insert_ok and right_limit >= tmp_ts_open + win_len:
                            # insert and update win info of topology and mstream
                            if not first_node:
                                port.windowsInfo.append([
                                    tmp_ts_open, win_len, mstream.id, mstream.pcp, pre_node.id, pre_port.id])
                            else:
                                port.windowsInfo.append([
                                    tmp_ts_open, win_len, mstream.id, mstream.pcp, -1, -1])
                            port.update_windows_info()
                            mstream.windowsInfo[seg[i]].append([
                                port.id, tmp_ts_open, win_len, uni_id + ti * seg_num])
                            insert_ok = True
                            # print("update port info2")
                            #print(port.windowsInfo)
                            #print(mstream.windowsInfo)
                        if first_node:
                            # print("got first node：", node.id, "delay start：", tmp_ts_open)
                            delay_start.append(tmp_ts_open)
                        if not first_node and last_hop:
                            # print("got last node：", node.id, "end at：", tmp_ts_open + win_len)
                            if len(delay_start) > ti:
                                d = round(tmp_ts_open + win_len - delay_start[ti], 1)
                            else:
                                d = round(tmp_ts_open + win_len - delay_start[ti % len(delay_start)] - mstream.interval * (ti / len(delay_start)), 1)
                            delays.append(d)
                        if not insert_ok:
                            print("error insert_ok")
                            return False
                    # print("=============")
        # compute delay
        add_latency = sum(delays) / (len(delays) / len(mstream.dst_node_ids))
        self.total_latency += add_latency
        # print("mstream：", mstream.id, "src：", mstream.src_node_id, "dst：", mstream.multicast_id, "delay result：")
        # print("delays：", delays, "dst num", len(mstream.dst_node_ids), "hyper times", (len(delays) / len(mstream.dst_node_ids)), "now total latency：", self.total_latency)
        return True

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
            self.update_node_neigh_info(mstream)
            result = self.update_node_win_info(mstream) # update self.total_latency
            if not result:
                print("error update qbv")
                return -1
        total_latency = self.total_latency
        self.total_latency = 0

        self.update_stream_and_topology_winInfo()

        return total_latency

    def tournament_select(self, pops, popsize, fits, tournament_size):
        new_pops, new_fits = [], []
        while len(new_pops) < len(pops):
            tournament_list = random.sample(range(0, popsize), tournament_size)
            tournament_fit = [fits[i] for i in tournament_list]
            tournament_df = pd.DataFrame \
                ([tournament_list, tournament_fit]).transpose().sort_values(by=1).reset_index(drop=True)
            fit = tournament_df.iloc[0, 1]
            pop = pops[int(tournament_df.iloc[0, 0])]
            new_pops.append(pop)
            new_fits.append(fit)
        return new_pops, new_fits

    def crossover(self, popsize, parent1_pops, parent2_pops, pc):
        child_pops = []
        for i in range(popsize):
            child = [None] * len(parent1_pops[i])
            parent1 = parent1_pops[i]
            parent2 = parent2_pops[i]
            if random.random() >= pc:
                child = parent1.copy()
                # random.shuffle(child)
            else:
                # parent1
                start_pos = random.randint(0, len(parent1) - 1)
                end_pos = random.randint(0, len(parent1) - 1)
                if start_pos > end_pos:
                    start_pos, end_pos = end_pos, start_pos
                child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
                # parent2 -> child
                list1 = list(range(end_pos + 1, len(parent2)))
                list2 = list(range(0, start_pos))
                list_index = list1 + list2
                j = -1
                for i in list_index:
                    for j in range(j + 1, len(parent2)):
                        if parent2[j] not in child:
                            child[i] = parent2[j]
                            break
            child_pops.append(child)
        return child_pops

    def mutate(self, pops, pm):
        pops_mutate = []
        for i in range(len(pops)):
            pop = pops[i].copy()
            t = random.randint(1, 5)
            count = 0
            while count < t:
                if random.random() < pm:
                    mut_pos1 = random.randint(0, len(pop) - 1)
                    mut_pos2 = random.randint(0, len(pop) - 1)
                    if mut_pos1 != mut_pos2:
                        tem = pop[mut_pos1]
                        pop[mut_pos1] = pop[mut_pos2]
                        pop[mut_pos2] = tem
                pops_mutate.append(pop)
                count += 1
        return pops_mutate


    def run(self):
        iteration = 0
        timeStart = time.time()
        # 随机生成每个种群初始流顺序方案
        pops = \
            [random.sample([i for i in list(range(len(self.mstreams)))], len(self.mstreams)) for
             j in range(self.popsize)]
        fits = [None] * self.popsize
        for i in range(self.popsize):
            fits[i] = self.calc_total_latency(pops[i])

        best_fit = min(fits)
        best_pop = pops[fits.index(best_fit)]
        print('first best %.1f' % (best_fit))
        self.best_latency_history.append(best_fit)

        while iteration <= self.generation:
            pop1, fits1 = self.tournament_select(pops, self.popsize, fits, self.tournament_size)
            pop2, fits2 = self.tournament_select(pops, self.popsize, fits, self.tournament_size)
            # 交叉
            child_pops = self.crossover(self.popsize, pop1, pop2, self.pc)
            # 变异
            child_pops = self.mutate(child_pops, self.pm)
            # 计算子代适应度
            child_fits = [None] * self.popsize
            for i in range(self.popsize):
                child_fits[i] = self.calc_total_latency(child_pops[i])
            # 一对一生存者竞争
            for i in range(self.popsize):
                if fits[i] > child_fits[i]:
                    fits[i] = child_fits[i]
                    pops[i] = child_pops[i]
            if best_fit > min(fits):
                best_fit = min(fits)
                best_pop = pops[fits.index(best_fit)]
            self.best_latency_history.append(best_fit)
            self.best_path = best_pop
            print('No. %d version best %.1f' % (iteration, best_fit))
            iteration += 1

        timeEnd = time.time()
        print("compute ", timeEnd - timeStart, "seconds")

