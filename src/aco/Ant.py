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

    def update_qbv(self):
        next_stream_id = self.select_next_stream_id()
        next_stream = self.streamGraph.mstreams[next_stream_id]
        self.path.append(next_stream_id)
        # update qbv, total bandwidth, error return
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

        # 2.update windowInfo and latency
        update_node_neigh_info(self.topology, next_stream)
        # print("next stream", next_stream.id, "interval", next_stream.interval, "windowsInfo", next_stream.windowsInfo)
        # print("seg_paths", next_stream.seg_path_dict)
        add_latency = update_node_win_info(self.topology, next_stream, self.win_plus)
        if add_latency < 0:
            print("error update qbv")
            return False
        self.total_latency += add_latency
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
