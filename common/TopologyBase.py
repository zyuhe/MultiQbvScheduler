'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:19
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   TopologyBase.py
'''
import math

from z3 import *
from typing import List


class Port:
    '''
    port_speed : unit Mbps
    '''
    def __init__(self, port_id, node_id, allowed_vlans: List[int], port_speed=100):
        self.id = port_id
        self.node_id = node_id
        self.neighbor_node_id = -1
        self.neighbor_port_id = -1
        self.allowed_vlans = allowed_vlans
        self.port_speed = port_speed
        self.used_bandwidth = 0
        self.occupied_bandwidth = 0
        self.hyper_period = 1
        '''
        alpha_set omega_set unit: ns
        '''
        self.alpha_set = Array(f'node_{self.node_id}_port_{self.id}_alpha', IntSort(), IntSort())
        self.omega_set = Array(f'node_{self.node_id}_port_{self.id}_omega', IntSort(), IntSort())
        self.window2last_hop_constraint_info = []
        self.window2queue_set = []
        '''
        for aco
        [[ts_open, win_len, mstream_id, pcp, pre_node_id, pre_port_id],[],]
        '''
        self.TS_OPEN = 0
        self.WIN_LEN = 1
        self.MSTREAM_ID = 2
        self.PCP = 3
        self.PRE_NODE_ID = 4
        self.PRE_PORT_ID = 5
        self.windowsInfo = []

    def set_port_speed(self, port_speed):
        self.port_speed = port_speed

    def add_allowed_vlan(self, vlanId):
        self.allowed_vlans.append(vlanId)

    def set_neighbor_node_id(self, neighbor_node_id):
        self.neighbor_node_id = neighbor_node_id

    def set_neighbor_port_id(self, neighbor_port_id):
        self.neighbor_port_id = neighbor_port_id

    def add_used_bandwidth(self, bandwidth):
        self.used_bandwidth += bandwidth

    def add_occupied_bandwidth(self, bandwidth):
        self.occupied_bandwidth += bandwidth

    def expand_hyper_period(self, new_period):
        new_hyper_period = int(self.hyper_period) * int(new_period) / math.gcd(int(self.hyper_period), int(new_period))
        times = int(new_hyper_period / self.hyper_period)
        copy_windowsInfo = copy.deepcopy(self.windowsInfo)
        for win_info in copy_windowsInfo:
            for i in range(times-1):
                win_info[self.TS_OPEN] += self.hyper_period
                self.windowsInfo.append(win_info)
        self.hyper_period = int(new_hyper_period)
        self.update_windows_info()

    def update_windows_info(self):
        self.windowsInfo.sort(key=lambda x: x[self.TS_OPEN])

    def clear_windows_info(self):
        self.windowsInfo = []

    def remaining_resorce(self):
        pre_t = 0
        rr = 0
        min_slot = 800 / self.port_speed + 1
        for wf in self.windowsInfo:
            # 門控間隙能容納100bytes的數據包通過
            if wf[0] - pre_t > min_slot:
                rr += wf[0] - pre_t
            pre_t = wf[0] + wf[1]
        if self.hyper_period - pre_t > min_slot:
            rr += self.hyper_period - pre_t
        return round(rr / self.hyper_period, 3)

class Link:
    '''
    length : unit m
    link_speed : unit Mbps
    '''
    def __init__(self, src_node, src_port, dst_node, dst_port, length=1, link_speed=100):
        self.src_node = src_node
        self.src_port = src_port
        self.dst_node = dst_node
        self.dst_port = dst_port
        self.length = length
        self.link_speed = link_speed

    def get_propagation_interval(self):
        '''
        :return: unit: ns
        '''
        return self.length * 1000 / self.link_speed


class Node:
    '''
    end_device: 1 represents  end device, 0 not
    plane: 0 represents  plane A, 1 represents plane B
    '''
    def __init__(self, node_id, end_device, plane, proc_delay=10000):
        self.id = node_id
        self.neighbor_node_ids = list() #
        self.ports = list()
        self.proc_delay = proc_delay
        self.links = list()
        self.end_device = end_device
        self.plane = plane

    def add_port(self, port: Port):
        self.ports.append(port)

    def add_neighbor_node_id(self, neighbor_node_id):
        self.neighbor_node_ids.append(neighbor_node_id)

    def add_link(self, link: Link):
        self.links.append(link)

    def get_port(self, port_id):
        for port in self.ports:
            if port_id == port.id:
                return port

    def get_port_by_neighbor_id(self, neighbor_id):
        for port in self.ports:
            if port.neighbor_node_id == neighbor_id:
                return port

    def clear_all_ports_winInfo(self):
        for port in self.ports:
            port.clear_windows_info()


class TopologyBase:
    def __init__(self, topology_id):
        self.id = topology_id
        self.nodes = list()
        self.links = list()
        self.end_devices = list()

    def get_node(self, node_id):
        for node in self.nodes:
            if node_id == node.id:
                return node

    def get_link(self, link_id):
        return self.links.__getitem__(link_id)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_link(self, link: Link):
        self.links.append(link)

    def clear_all_nodes_winInfo(self):
        for node in self.nodes:
            node.clear_all_ports_winInfo()
