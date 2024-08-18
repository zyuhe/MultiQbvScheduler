'''
-*- coding: utf-8 -*-
@Time    :   2024/3/20 21:07
@Auther  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   parser.py
'''
import networkx
import yaml
import shutil
import matplotlib.pyplot as plt
from yaml import CLoader
from src.mqbv.Topology import *
from src.mqbv.Stream import *


def topology_parser(topology_file_path):
    topology = Topology(0)
    default_proc_delay = 10000
    with open(topology_file_path, "r", encoding="utf-8") as f:
        topology_info = yaml.load(f, Loader=CLoader)
        for node_info in topology_info:
            if node_info['end_device'] == 1:
                topology.end_devices.append(node_info['node_id'])
                default_proc_delay = 5000 # 5000
            node = Node(node_info['node_id'], node_info['end_device'], node_info['plane'], default_proc_delay)
            for port_info in node_info['ports']:
                port = Port(port_info['port_id'],
                            node_info['node_id'],
                            port_info['allowed_vlan'],
                            port_info['port_speed'])
                port.set_neighbor_node_id(port_info['neighbor_node'])
                port.set_neighbor_port_id(port_info['neighbor_port'])
                node.add_port(port)
                link = Link(node_info['node_id'],
                            port_info['port_id'],
                            port_info['neighbor_node'],
                            port_info['neighbor_port'])
                node.add_link(link)
                topology.add_link(link)
            topology.add_node(node)
    return topology


def stream_parser(stream_file_path):
    streams = list()
    with open(stream_file_path, "r", encoding="utf-8") as f:
        streams_info = yaml.load(f, Loader=CLoader)
        uni_stream_index = 0
        for stream_info in streams_info:
            for dst_id in stream_info['dst_node_ids']:
                stream = Stream(uni_stream_index,
                                stream_info['size'],
                                stream_info['interval'],
                                stream_info['vlan_id'],
                                stream_info['pcp'],
                                stream_info['src_node_id'],
                                dst_id,
                                stream_info['max_latency'],
                                stream_info['max_jitter'])
                uni_stream_index += 1
                streams.append(stream)
    return streams


def mstream_parser(stream_file_path):
    mstreams = list()
    with open(stream_file_path, "r", encoding="utf-8") as f:
        mstreams_info = yaml.load(f, Loader=CLoader)
        index = 0
        for mstream_info in mstreams_info:
            mstream = MStream(index,
                              mstream_info['size'],
                              mstream_info['interval'],
                              mstream_info['vlan_id'],
                              mstream_info['pcp'],
                              mstream_info['src_node_id'],
                              mstream_info['dst_node_ids'],
                              mstream_info['multicast_id'],
                              mstream_info['max_latency'],
                              mstream_info['max_jitter'])
            mstreams.append(mstream)
            index += 1
    return mstreams


def sanone_sw_converse_instruction(port_timelines, hyper_period):
    curr_node_id = -1
    if "output" in os.listdir("./"):
        shutil.rmtree("./output/")
    os.mkdir("./output/")
    for name in port_timelines:
        if int(name.split('_')[1]) != curr_node_id:
            curr_node_id = int(name.split('_')[1])
            print(f"Write GCL instruction of node_{curr_node_id}...")
        if port_timelines[name][0][3] == 1: # end_device
            with open(f"./output/ls1028_node_{curr_node_id}.txt", "a", encoding="utf-8") as f:
                # f.write("tc qdisc del dev eno0 root\n")
                _ls1028_converse_instruction(port_timelines[name], int(hyper_period), f)
            with open(f"./output/tc_taprio_node_{curr_node_id}.txt", "a", encoding="utf-8") as f:
                _tc_taprio_comverse_instruction(port_timelines[name], int(hyper_period), f)
        else:
            with open(f"./output/sanone_node_{curr_node_id}.txt", "a", encoding="utf-8") as f:
                f.write("conf t\n")
                _sanone_sw_converse_instruction(name, port_timelines[name], int(hyper_period), f)
                f.write("exit\n")
                f.write("write\n")


def _sanone_sw_converse_instruction(node_port_info, timeline: list[list[int]], hyper_period, f):
    # print(f"============== GCL instruction of {node_port_info} ==============")
    last_end_time = 0
    gcl_index = 0
    interval_len = 0
    gcl_set = list()
    queues_i = [i for i in range(8)]
    queues_to_remove = []
    f.write(f"interface ge {int(node_port_info.split('_')[3]) + 1}\n")
    # f.write("no tsn tas gate-enabled\n")
    # f.write(f"no tsn frame-preemption verify-disable\n")
    # f.write(f"no tsn frame-preemption ignore-lldp\n")
    # f.write(f"no tsn frame-preemption queue 1\n")
    # f.write(f"tsn frame-preemption\n")
    # f.write(f"tsn frame-preemption verify-disable\n")
    # f.write(f"tsn frame-preemption ignore-lldp\n")
    # f.write(f"tsn frame-preemption queue 1\n")
    f.write(f"tsn tas cycle-time us {int(hyper_period / 1000)}\n")
    f.write(f"tsn tas cycle-time-extension {hyper_period}\n")
    last_queue = -1
    for i in range(len(timeline)):
        _, _, queue, _ = timeline[i]
        queues_to_remove.append(queue)
    queues_to_remove = list(set(queues_to_remove))
    queues_i = [i for i in queues_i if i not in queues_to_remove]
    for i in range(len(timeline)):
        queues = queues_i.copy()
        start_time, end_time, queue, _ = timeline[i]
        if start_time > last_end_time:
            set_queues = ','.join([str(x) for x in queues_i])
            # guardband = 5000
            gcl_set.append(
                "tsn tas control-list index " + str(gcl_index) + f" gate-state queue {set_queues} open time-interval " +
                str(start_time - last_end_time) + " operation set\n")
            gcl_index += 1
            # gcl_set.append("tsn tas control-list index " + str(gcl_index) + f" gate-state queue 7 open time-interval " +
            #                str(guardband) + " operation set\n") # for guardband
            # gcl_index += 1 # for guardband
            interval_len += start_time - last_end_time

        gcl_set.append("tsn tas control-list index " + str(gcl_index) + f" gate-state queue {queue} open time-interval " +
              str(end_time - start_time) + " operation set\n")
        gcl_index += 1
        interval_len += end_time - start_time
        last_end_time = end_time
        last_queue = queue
    if last_end_time < hyper_period:
        set_queues = ','.join([str(x) for x in queues_i])
        gcl_set.append("tsn tas control-list index " + str(gcl_index) + f" gate-state queue {set_queues} open time-interval " +
              str(hyper_period - last_end_time) + " operation set\n")
        gcl_index += 1
        interval_len += hyper_period - last_end_time
    f.write(f"tsn tas control-list-length {gcl_index}\n")
    for gcl_instruction in gcl_set:
        f.write(gcl_instruction)
    f.write("tsn tas config-change\n")
    f.write("tsn tas gate-enabled\n")
    f.write("tsn tas config-change\n")

    f.write("exit\n")
    # print(f"total interval length is {interval_len}ns")


def _ls1028_converse_instruction(timeline: list[list[int]], hyper_period, f):
    '''
    t0 11111111b 10000
    t1 00000000b 99990000
    '''
    last_end_time = 0
    gcl_index = 0
    interval_len = 0
    gcl_set = list()
    for i in range(len(timeline)):
        start_time, end_time, queue, _ = timeline[i]
        if start_time > last_end_time:
            gcl_set.append(f"t{gcl_index} 10000111b {start_time - last_end_time}\n")
            gcl_index += 1
            interval_len += start_time - last_end_time
        if queue == 3:
            q_set = "10001111"
        elif queue == 4:
            q_set = "10010111"
        elif queue == 5:
            q_set = "10100111"
        elif queue == 6:
            q_set = "11000111"
        else:
            print("Waring! invalid queue!")
        # q_set = '0'*(7-queue)+'1'+'0'*(queue-1)+'1'
        gcl_set.append(f"t{gcl_index} {q_set}b {end_time - start_time}\n")
        gcl_index += 1
        interval_len += end_time - start_time
        last_end_time = end_time
    if last_end_time < hyper_period:
        gcl_set.append(f"t{gcl_index} 10000111b {hyper_period - last_end_time}\n")
        gcl_index += 1
        interval_len += hyper_period - last_end_time
    for gcl_instruction in gcl_set:
        f.write(gcl_instruction)


def _tc_taprio_comverse_instruction(timeline: list[list[int]], hyper_period, f):
    last_end_time = 0
    gcl_index = 0
    interval_len = 0
    gcl_set = list("tc qdisc del dev eno0 root\n"
                   "tc qdisc replace dev eno0 parent root handle 100 stab overhead 24 taprio num_tc 8 map 0 1 2 3 4 5 6 7 \\\n"
                   "queues 1@0 1@1 1@2 1@3 1@4 1@5 1@6 1@7 \\\n"
                   "max-sdu 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 1500 base-time 0 \\\n")
    for i in range(len(timeline)):
        start_time, end_time, queue, _ = timeline[i]
        if start_time > last_end_time:
            gcl_set.append(f"sched-entry S 87 {start_time - last_end_time} \\\n")
            gcl_index += 1
            interval_len += start_time - last_end_time
        if queue == 3:
            q_set = "8f"
        elif queue == 4:
            q_set = "97"
        elif queue == 5:
            q_set = "a7"
        elif queue == 6:
            q_set = "c7"
        else:
            print("Waring! invalid queue!")
        # q_set = '0'*(7-queue)+'1'+'0'*(queue-1)+'1'
        gcl_set.append(f"sched-entry S {q_set} {end_time - start_time} \\\n")
        gcl_index += 1
        interval_len += end_time - start_time
        last_end_time = end_time
    if last_end_time < hyper_period:
        gcl_set.append(f"sched-entry S 87 {hyper_period - last_end_time} \\\n")
        gcl_index += 1
        interval_len += hyper_period - last_end_time
    gcl_set.append("flags 0x2\n")
    # gcl_set.append("tc qdisc replace dev eno0 parent 100:4 etf clockid CLOCK_TAI delta 600000 offload skip_sock_check\n")
    # gcl_set.append("tc qdisc replace dev eno0 parent 100:5 etf clockid CLOCK_TAI delta 600000 offload skip_sock_check\n")
    # gcl_set.append("tc qdisc replace dev eno0 parent 100:6 etf clockid CLOCK_TAI delta 600000 offload skip_sock_check\n")
    # gcl_set.append("tc qdisc replace dev eno0 parent 100:7 etf clockid CLOCK_TAI delta 600000 offload skip_sock_check\n")
    for gcl_instruction in gcl_set:
        f.write(gcl_instruction)


def check_and_draw_topology(topology: Topology):
    G = networkx.DiGraph()
    for node in topology.nodes:
        G.add_node(node.id, node_id=node.id)
    for link in topology.links:
        G.add_edge(link.src_node, link.dst_node, port1=link.src_port, port2=link.dst_port)
    # pos = networkx.spring_layout(G, iterations=200)
    pos = networkx.circular_layout(G)
    networkx.draw(G, pos, with_labels=True)
    # for debug
    # plt.show()
    return G


def check_streams(streams: list[Stream]):
    for item in streams:
        print(item.uni_stream_id, item.src_node_id, item.dst_node_id, item.size, item.vlanid, item.interval)


