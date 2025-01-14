'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:40
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   Ant.py
'''
import math
from logging import lastResort

import networkx
import numpy as np

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase
from common.funcs import *
from common.parser import check_and_draw_topology, mstream_parser
from common.plot import draw_gantt_chart
from src.aco.StreamGraph import StreamGraph


class Ant(object):
    def __init__(self, streamGraph: StreamGraph, topology: TopologyBase):
        self.streamGraph = streamGraph
        self.topology = topology
        self.current_stream_id = None
        self.path = []
        self.total_latency = 0
        self.unvisited_streams = set(range(streamGraph.num_streams))
        self.topology_graph = check_and_draw_topology(topology)

        self.win_plus = 1000 # ns

    def select_next_stream_id(self):
        probabilities = np.zeros(self.streamGraph.num_streams)
        if self.current_stream_id is not None:
            for stream in self.unvisited_streams:
                if self.streamGraph.distances[self.current_stream_id][stream] > 0:
                    probabilities[stream] = (self.streamGraph.pheromones[self.current_stream_id][stream] ** 2 /
                                            self.streamGraph.distances[self.current_stream_id][stream] )
            probabilities /= probabilities.sum()
            next_stream_id = np.random.choice(range(self.streamGraph.num_streams), p=probabilities)
        else:
            next_stream_id = np.random.randint(self.streamGraph.num_streams)
        return next_stream_id

    def update_node_neigh_info(self, mstream: MStream):
        # print("mstream.seg_path_dict", mstream.seg_path_dict)
        value_list = list(mstream.seg_path_dict.values())
        mstream.windowsInfo[value_list[0][0][0]] = [-1, -1]
        for v in value_list:
            for seg in v:
                for i in range(len(seg) - 1):
                    pre_node_id = seg[i]
                    pre_port = self.topology.get_node(pre_node_id).get_port_by_neighbor_id(seg[i+1])
                    mstream.windowsInfo[seg[i+1]] = [pre_node_id, pre_port.id]

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



    def update_qbv(self):
        next_stream_id = self.select_next_stream_id()
        next_stream = self.streamGraph.mstreams[next_stream_id]
        self.path.append(next_stream_id)
        # TODO: update qbv, total bandwidth, error return
        # get mstream, cal route, update windowInfo and hyper_period along the route
        # insert new gate/windows, update bandwidth
        # 1: calculate route
        best_paths = list()
        for dst_node_id in next_stream.dst_node_ids:
            paths = list(networkx.all_simple_paths(self.topology_graph, source=next_stream.src_node_id, target=dst_node_id))
            paths = sorted(paths, key=len)
            available_paths = paths.copy()
            for path in paths:
                for index in range(len(path) - 1):
                    if index != 0 and self.topology.get_node(path[index]).end_device == 1:
                        available_paths.remove(path)
                        break
                    if next_stream.vlan_id not in self.topology.get_node(path[index]).get_port_by_neighbor_id(
                        path[index + 1]
                    ).allowed_vlans:
                        available_paths.remove(path)
                        break
            if len(available_paths) == 0:
                print(f"==>WARNING: no viable path from {next_stream.src_node_id} to {dst_node_id}!")
                print("             Please check stream and topology settings.")
                # TODO: error re_choose route
            # TODO: route select algorithm
            best_path = available_paths[0]
            best_paths.append(best_path)
        next_stream.seg_path_dict = compute_seg_path_dict(best_paths)

        # 2.update windowInfo and TODO： bandwidth
        self.update_node_neigh_info(next_stream)
        # print("next stream", next_stream.id, "interval", next_stream.interval, "windowsInfo", next_stream.windowsInfo)
        # print("seg_paths", next_stream.seg_path_dict)
        result = self.update_node_win_info(next_stream)
        if not result:
            print("error update qbv")
            return False

        self.current_stream_id = next_stream_id
        self.unvisited_streams.remove(next_stream_id)
        return True

    def complete_solution(self):
        while self.unvisited_streams:
            result = self.update_qbv()
            if not result:
                print("error complete solution")
                return False
        # print("draw chart, total latency：", self.total_latency)
        # self.draw_chart()
        return True

    def draw_chart(self):
        # stream timeline
        for mstream in self.streamGraph.mstreams:
            name = f'mstream_{mstream.id}_src_{mstream.src_node_id}_dst_{mstream.multicast_id}_pcp_{mstream.pcp}'
            mstream_timeline = []
            for k, v in mstream.windowsInfo.items():
                for info in v[2:]:
                    mstream_timeline.append([info[1], round(info[1] + info[2], 1), k, 0, info[3]])
            draw_gantt_chart(name, mstream_timeline, mstream.hyper_period)

        # port timeline
        '''
        for node in self.topology.nodes:
            for port in node.ports:
                name = f'node_{node.id}_port_{port.id}'
                port_timeline = []
                for info in port.windowsInfo:
                    port_timeline.append([info[port.TS_OPEN], round(info[port.TS_OPEN] + info[port.WIN_LEN], 1), info[port.PCP], 0])
                draw_gantt_chart(name, port_timeline, port.hyper_period / 1000)
        '''
