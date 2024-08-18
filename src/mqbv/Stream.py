'''
-*- coding: utf-8 -*-
@Time    :   2024/3/20 21:06
@Auther  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   Stream.py
'''
from z3 import *

class Route:
    def __init__(self, uni_stream_id, path: list[int]):
        self.uni_stream_id = uni_stream_id
        self.path = path

    def get_windows_len(self):
        return len(self.path) - 1


class StreamInstance:
    def __init__(self, stream_id, cycle_id):
        self.stream_id = stream_id
        self.cycle_id = cycle_id
        self.offset = Int(f'stream_{self.stream_id}_instance_{self.cycle_id}_offset')
        '''
        alpha_set omega_set unit: ns
        '''
        self.alpha_set = Array(f'stream_{self.stream_id}_instance_{self.cycle_id}_alpha', IntSort(), IntSort())
        self.omega_set = Array(f'stream_{self.stream_id}_instance_{self.cycle_id}_omega', IntSort(), IntSort())


class Stream:
    def __init__(self, stream_id, uni_stream_id, size, interval, vlan_id, pcp, src_node_id, multicast_id, dst_node_id, max_latency, max_jitter):
        self.uni_stream_id = uni_stream_id
        self.stream_id = stream_id
        self.size = size
        self.interval = interval
        self.vlan_id = vlan_id
        self.pcp = pcp
        self.src_node_id = src_node_id
        self.curr_src_port_id = -1
        self.multicast_id = multicast_id
        self.dst_node_id = dst_node_id
        self.curr_dst_port_id = -1
        self.max_latency = max_latency
        self.max_jitter = max_jitter
        self.current_streamInstance_id = 0
        self.streamInstances = list()
        self.current_route_id = 0
        self.routes = list()

    def get_bandwidth(self):
        return self.size * 8 / self.interval

    def get_current_route(self):
        if len(self.routes) != 0:
            return self.routes[self.current_route_id]
        else:
            return None

    def get_dst_count(self):
        return len(self.dst_node_id)

    def add_route(self, route: Route):
        self.routes.append(route)

    def add_stream_instance(self, stream_instance: StreamInstance):
        self.streamInstances.append(stream_instance)

    def set_current_stream_instance_id(self, curr_stream_instance_id):
        self.current_streamInstance_id = curr_stream_instance_id


class MStream:
    '''
    interval unit: ns
    max_latency unit: ns
    max_jitter unit: ns
    '''
    def __init__(self, stream_id, size, interval, vlan_id, pcp, src_node_id, dst_node_ids, multicast_id, max_latency, max_jitter):
        self.id = stream_id
        self.size = size
        self.interval = interval
        self.vlan_id = vlan_id
        self.pcp = pcp
        self.src_node_id = src_node_id
        self.dst_node_ids = dst_node_ids
        self.multicast_id = multicast_id
        self.max_latency = max_latency
        self.max_jitter = max_jitter
        self.shortest_routes = list()
        self.seg_streams = list() # Stream

