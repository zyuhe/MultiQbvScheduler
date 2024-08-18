'''
-*- coding: utf-8 -*-
@Time    :   2024/3/20 21:06
@Auther  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   Topology.py
'''
from z3 import *


class Port:
    '''
    port_speed : unit Mbps
    '''
    def __init__(self, port_id, node_id, allowed_vlans: list[int], port_speed=100):
        self.id = port_id
        self.node_id = node_id
        self.neighbor_node_id = -1
        self.neighbor_port_id = -1
        self.allowed_vlans = allowed_vlans
        self.port_speed = port_speed
        self.used_bandwidth = 0
        self.occupied_bandwidth = 0
        '''
        alpha_set omega_set unit: ns
        '''
        self.alpha_set = Array(f'node_{self.node_id}_port_{self.id}_alpha', IntSort(), IntSort())
        self.omega_set = Array(f'node_{self.node_id}_port_{self.id}_omega', IntSort(), IntSort())
        self.window2last_hop_constraint_info = []
        self.window2queue_set = []

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


class Topology:
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

