'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:22
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   conf_generator.py
'''

import time
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from common.parser import *


def generate_topology(size, yaml_file_path):
    begin_time = time.time_ns()
    topology_info = list()
    for i in range(size):
        topology_info.append({"node_id": i,
                              "ports": []})
    for node_id_i in range(0, size - 1):
        for node_id_j in range(node_id_i + 1, size):
            linked = random.randint(0, 100)
            curr_port_count_i = len(topology_info[node_id_i]["ports"])
            curr_port_count_j = len(topology_info[node_id_j]["ports"])
            if linked <= 20:
                topology_info[node_id_i]["ports"].append({
                    "port_id": curr_port_count_i,
                    "port_speed": 1000,
                    "allowed_vlan": [10, 20, 30],
                    "neighbor_node": node_id_j,
                    "neighbor_port": curr_port_count_j
                })
                topology_info[node_id_j]["ports"].append({
                    "port_id": curr_port_count_j,
                    "port_speed": 1000,
                    "allowed_vlan": [10, 20, 30],
                    "neighbor_node": node_id_i,
                    "neighbor_port": curr_port_count_i
                })
    # check end device
    for i in range(size):
        if len(topology_info[i]["ports"]) == 0 or len(topology_info[i]["ports"]) == 1:
            topology_info[i]["end_device"] = 1
        else:
            topology_info[i]["end_device"] = 0
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(topology_info, f)
    end_time = time.time_ns()
    print(f"generate topology yaml file takes {(end_time - begin_time) / 1000000000}s")
    return True


'''
store in yaml file
'''
def generate_streams(stream_count, topology: TopologyBase, yaml_file_path):
    begin_time = time.time_ns()
    streams_info = list()
    optional_size = [i*100 for i in range(1, 16)]
    optional_interval = [i*100000 for i in [2, 3, 4, 5, 10]]
    optional_vlan_id = [10, 20, 30]
    optional_pcp = [i for i in range(8)]
    count = 0
    if len(topology.end_devices) == 0:
        print("ERROR: no end device!")
    while count < stream_count:
        test_time = time.time_ns()
        if test_time - begin_time > 5000000000:
            print("ERROR: generate streams yaml file overtime!")
            return False
        src_node_id = random.choice(topology.end_devices)
        dst_node_ids = []
        topo_graph = check_and_draw_topology(topology)
        available = random.randint(0, 100)
        for node_id in topology.end_devices:
            if node_id != src_node_id and networkx.has_path(topo_graph, src_node_id, node_id) and available < 50 and count < stream_count:
                dst_node_ids.append(node_id)
                count += 1
            elif count == stream_count:
                break
        if len(dst_node_ids) != 0:
            streams_info.append({
                "src_node_id": src_node_id,
                "dst_node_ids": dst_node_ids,
                "size": random.choice(optional_size),
                "interval": random.choice(optional_interval),
                "vlan_id": random.choice(optional_vlan_id),
                "pcp": random.choice(optional_pcp),
                "max_latency": 1000000,
                "max_jitter": 1000000
            })
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(streams_info, f)
    end_time = time.time_ns()
    print(f"generate streams yaml file takes {(end_time - begin_time) / 1000000000}s")
    return True


'''
return a stream: Stream
'''
def gene_a_stream(topology: TopologyBase, uni_id):
    # decide sender and receiver
    if len(topology.end_devices) < 2:
        raise ValueError("No enough end devices to choice!")
    sender, receiver = random.sample(topology.end_devices, 2)
    topo_graph = check_and_draw_topology(topology)
    paths = networkx.all_simple_paths(topo_graph, source=sender, target=receiver)
    optional_vlan_id_all = []
    # only computed output interfaces
    for path in paths:
        curr_port = topology.get_node(path[0]).get_port_by_neighbor_id(path[1])
        optional_vlan_id = curr_port.allowed_vlans.copy()
        for index in range(1, len(path) - 1):
            curr_port = topology.get_node(path[index]).get_port_by_neighbor_id(path[index + 1])
            tmp_optional_vlan_id = optional_vlan_id.copy()
            for vlan_id in tmp_optional_vlan_id:
                if vlan_id not in curr_port.allowed_vlans:
                    optional_vlan_id.remove(vlan_id)
        optional_vlan_id_all.extend(optional_vlan_id)
    optional_vlan_id_all = list(set(optional_vlan_id_all))
    if len(optional_vlan_id_all) == 0:
        print(f"Error! no allowed vlan from {sender} to {receiver}")
    optional_size = [i * 100 for i in [1, 2, 5, 5, 8, 8, 8, 10, 10, 10, 10, 12, 12, 14]]
    optional_interval = [i * 100000 for i in [10, 10, 10, 10, 10, 20, 20, 20, 50, 50]]
    optional_pcp = [3, 4, 5, 6, 7]
    size = random.choice(optional_size)
    interval = random.choice(optional_interval)
    vlan = random.choice(optional_vlan_id_all)
    pcp = random.choice(optional_pcp)
    while 160 * size >= interval - 2000 or 800 * size >= interval:
        size = random.choice(optional_size)
        interval = random.choice(optional_interval)
    max_latency = size * 1600
    max_jitter = max_latency
    print(f"Generate stream {uni_id} from {sender} to {receiver}, allowed vlan: {optional_vlan_id_all}, "
          f"size: {size}, interval: {interval / 1000}us, pcp: {pcp}, vlan: {vlan}, "
          f"max_latency: {max_latency / 1000}us.")
    return Stream(uni_id, size, interval, vlan, pcp, sender, receiver, max_latency, max_jitter)


def parse_streams_to_yaml(streams: List[Stream], yaml_file_path):
    streams_info = list()
    for stream in streams:
        streams_info.append({
            "src_node_id": stream.src_node_id,
            "dst_node_ids": [stream.dst_node_id],
            "size": stream.size,
            "interval": stream.interval,
            "vlan_id": stream.vlanid,
            "pcp": stream.pcp,
            "max_latency": stream.max_latency,
            "max_jitter": stream.max_jitter
        })
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(streams_info, f)


def turn_stream_info_to_trdp_config_xml(streams: List[Stream], topology: TopologyBase, mapping_file_path1, mapping_file_path2, output_xml_path_pre):
    # Please input Ipv4 Address of node n:
    # Please input interface name of node n port n:
    # Please input Ipv4 Address of vlan interface ifname.vlan_id:
    # Or parse mapping file
    # TODO: Static Route Configuration
    with open(mapping_file_path2, "r", encoding="utf-8") as f:
        multicast_mapping_info = yaml.load(f, Loader=yaml.Loader)
    with open(mapping_file_path1, "r", encoding="utf-8") as f:
        mapping_info = yaml.load(f, Loader=yaml.Loader)
        # print(mapping_info)
        streams = [element for sublist in streams for element in sublist]
        sorted_streams_by_src = sorted(streams, key=lambda x: x.src_node_id)
        sorted_streams_by_src_copy = sorted_streams_by_src.copy()
        for s_stream in sorted_streams_by_src_copy:
            if not topology.get_node(s_stream.src_node_id).end_device:
                sorted_streams_by_src.remove(s_stream)
        sorted_streams_by_dst = sorted(streams, key=lambda x: x.dst_node_id)
        sorted_streams_by_dst_copy = sorted_streams_by_dst.copy()
        for s_stream in sorted_streams_by_dst_copy:
            if not topology.get_node(s_stream.dst_node_id).end_device:
                sorted_streams_by_dst.remove(s_stream)
        nodes_streams_by_src = dict()
        nodes_streams_by_dst = dict()
        flag_node_id = -1
        '''
        {src_id:[streams]}
        {dst_id:[streams]}
        '''
        for stream in sorted_streams_by_src:
            if stream.src_node_id != flag_node_id:
                nodes_streams_by_src[stream.src_node_id] = [stream]
            else:
                nodes_streams_by_src[stream.src_node_id].append(stream)
            flag_node_id = stream.src_node_id
        flag_node_id = -1
        for stream in sorted_streams_by_dst:
            if stream.dst_node_id != flag_node_id:
                nodes_streams_by_dst[stream.dst_node_id] = [stream]
            else:
                nodes_streams_by_dst[stream.dst_node_id].append(stream)
            flag_node_id = stream.dst_node_id
        # first add publish traffic
        for node_id in list(set(list(nodes_streams_by_src.keys()) + list(nodes_streams_by_dst.keys()))):
            if node_id in nodes_streams_by_src.keys():
                pub_streams = nodes_streams_by_src[node_id]
            else:
                pub_streams = []
            if node_id in nodes_streams_by_dst.keys():
                sub_streams = nodes_streams_by_dst[node_id]
            else:
                sub_streams = []
            if node_id not in mapping_info.keys():
                print(f"Warning! no viable node {node_id} info in mapping")
                continue
            node_info = mapping_info[node_id]
            root = ET.Element("device")
            if_list = ET.SubElement(root, "interface-list")
            # create interface by route (sort)
            sorted_node_streams_by_src = sorted(pub_streams, key=lambda x: x.routes[x.current_route_id].path[1])
            sorted_node_streams_by_dst = sorted(sub_streams, key=lambda x: x.routes[x.current_route_id].path[-2])
            ports_streams_pub = dict()
            ports_streams_sub = dict()
            this_node = topology.get_node(node_id)
            flag_neighbor_node_id = -1
            '''
            {port:[pub_streams]}
            {port:[sub_streams]}
            '''
            for stream in sorted_node_streams_by_src:
                neighbor_node_id = stream.routes[stream.current_route_id].path[1]
                tx_port = this_node.get_port_by_neighbor_id(neighbor_node_id)
                if neighbor_node_id != flag_neighbor_node_id:
                    ports_streams_pub[tx_port.id] = [stream]
                else:
                    ports_streams_pub[tx_port.id].append(stream)
                flag_neighbor_node_id = neighbor_node_id
            flag_neighbor_node_id = -1
            for stream in sorted_node_streams_by_dst:
                neighbor_node_id = stream.routes[stream.current_route_id].path[-2]
                rx_port = this_node.get_port_by_neighbor_id(neighbor_node_id)
                if neighbor_node_id != flag_neighbor_node_id:
                    ports_streams_sub[rx_port.id] = [stream]
                else:
                    ports_streams_sub[rx_port.id].append(stream)
                flag_neighbor_node_id = neighbor_node_id
            for port_id in list(set(list(ports_streams_pub.keys()) + list(ports_streams_sub.keys()))):
                if port_id not in node_info.keys():
                    print(f"Warning! no viable tx port {port_id} info in node {node_id} info")
                    continue
                port_info = node_info[port_id]
                if_config = ET.SubElement(if_list, "interface")
                if_config.set("host-ip", port_info["ipv4_addr"])
                if_config.set("name", port_info["if_name"])
                # add publish streams
                if port_id in ports_streams_pub.keys():
                    for stream in ports_streams_pub[port_id]:
                        stream_config = ET.SubElement(if_config, "traffic")
                        stream_config.set("com-id", str(stream.stream_id + 21020))
                        stream_config.set("type", "source")
                        stream_config.set("port", "17226")
                        com_parameter = ET.SubElement(stream_config, "com-parameter")
                        com_parameter.set("pcp", str(stream.pcp))
                        pd_parameter = ET.SubElement(stream_config, "pd-parameter")
                        # cycle of etf and non-etf is different
                        # non-etf:
                        pd_parameter.set("cycle", str(int(stream.interval / 1000)))
                        # etf:
                        # pd_parameter.set("cycle", str(int(hyper_period / 1000)))
                        pd_parameter.set("size", str(stream.size))
                        # get vlan and vlan ip
                        source_vlan_ip = port_info["vlans"][stream.vlan_id]
                        source_ip = ET.SubElement(stream_config, "source")
                        source_ip.set("ip", str(source_vlan_ip))
                        # destination_vlan_ip = mapping_info[stream.dst_node_id][stream.curr_dst_port_id]["vlans"][stream.vlanid]
                        destination_vlan_ip = multicast_mapping_info[stream.multicast_id]
                        destination_ip = ET.SubElement(stream_config, "destination")
                        destination_ip.set("ip", str(destination_vlan_ip))
                        pd_txtime_config = ET.SubElement(stream_config, "pd-txtime")
                        pd_txtime_config.set("starttime", str(0))
                        # txtime of etf and non-etf is different
                        # non-etf:
                        pd_txtime_config.set("txtime", str(0))
                        # etf:
                        # curr_cycle_id = -1
                        # txtime_list = []
                        # for stream_timeline_key in stream_timelines.keys():
                        #     if stream_timeline_key.split('_')[1] == str(stream.uni_stream_id):
                        #         for i in range(len(stream_timelines[stream_timeline_key])):
                        #             start_time, _, _, cycle_id = stream_timelines[stream_timeline_key][i]
                        #             if cycle_id != curr_cycle_id:
                        #                 txtime_list.append(str(int((start_time) / 1000)))
                        #                 curr_cycle_id = cycle_id
                        #         break
                        # pd_txtime_config.set("txtime", ' '.join(txtime_list))
                # add subscribe streams
                if port_id in ports_streams_sub.keys():
                    for stream in ports_streams_sub[port_id]:
                        stream_config = ET.SubElement(if_config, "traffic")
                        stream_config.set("com-id", str(stream.stream_id + 21020))
                        stream_config.set("type", "sink")
                        stream_config.set("port", "17226")
                        com_parameter = ET.SubElement(stream_config, "com-parameter")
                        com_parameter.set("pcp", str(stream.pcp))
                        pd_parameter = ET.SubElement(stream_config, "pd-parameter")
                        pd_parameter.set("cycle", str(int(stream.interval / 1000)))
                        pd_parameter.set("size", str(stream.size))
                        # get vlan and vlan ip
                        source_vlan_ip = port_info["vlans"][stream.vlan_id]
                        source_ip = ET.SubElement(stream_config, "source")
                        source_ip.set("ip", str(source_vlan_ip))
                        # destination_vlan_ip = mapping_info[stream.dst_node_id][stream.curr_dst_port_id]["vlans"][stream.vlanid]
                        destination_vlan_ip = multicast_mapping_info[stream.multicast_id]
                        destination_ip = ET.SubElement(stream_config, "destination")
                        destination_ip.set("ip", str(destination_vlan_ip))
            tree = ET.ElementTree(root)
            output_file = output_xml_path_pre + str(node_id) + ".xml"
            dom = minidom.parseString(ET.tostring(root, encoding='utf-8'))
            pretty_xml = dom.toprettyxml(indent='  ')
            with open(output_file, "w") as fo:
                fo.write(pretty_xml)
            # tree.write(output_file, encoding="utf-8", xml_declaration=True)

