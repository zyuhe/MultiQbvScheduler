'''
-*- coding: utf-8 -*-
@Time    :   2024/3/20 21:10
@Auther  :   zyh
@Email   :
@Project :   MultiQbvScheduler
@File    :   solver.py
'''
import time
from src.mqbv.parser import *

def compute_stream_omega(qbv_solver, topology: Topology, streams: list[Stream]):
    print("============== compute mapping relationship ==============")
    node_port_constraint_counter = {}
    for streams_in_m in streams:
        path_behind_mid_node = dict()
        seg_paths = list()
        for stream in streams_in_m:
            seg_paths.append(stream.routes[0].path)
        for path_f in seg_paths:
            for path_l in seg_paths:
                if path_f[len(path_f) - 1] == path_l[0]:
                    if path_f[len(path_f) - 1] not in path_behind_mid_node.keys():
                        path_behind_mid_node[path_f[len(path_f) - 1]] = list()
                    path_behind_mid_node[path_f[len(path_f) - 1]].append(path_l)
        # print(path_behind_mid_node)
        for stream in streams_in_m:
            if stream.get_current_route() is None:
                continue
            instance_window_count = stream.get_current_route().get_windows_len()
            path_end_node = stream.routes[0].path[instance_window_count]
            end_node_is_end_device = topology.get_node(path_end_node).end_device
            for stream_instance in stream.streamInstances:
                for win_index in range(instance_window_count):
                    stream_omega = stream_instance.omega_set[win_index]
                    stream_alpha = stream_instance.alpha_set[win_index]
                    curr_node_id = stream.get_current_route().path[win_index]
                    curr_node = topology.get_node(curr_node_id)
                    next_node_id = stream.get_current_route().path[win_index+1]
                    curr_port = curr_node.get_port_by_neighbor_id(next_node_id)
                    curr_port.window2queue_set.append(stream.pcp)
                    char_node_port = f"{curr_node_id}_{curr_port.id}"
                    if char_node_port in node_port_constraint_counter.keys():
                        node_port_constraint_counter[char_node_port] += 1
                    else:
                        node_port_constraint_counter[char_node_port] = 0
                    # store
                    if win_index != instance_window_count - 1:
                        next_node = topology.get_node(next_node_id)
                        next_next_node_id = stream.get_current_route().path[win_index + 2]
                        next_port = next_node.get_port_by_neighbor_id(next_next_node_id)
                        next_port.window2last_hop_constraint_info.append({
                            "node_id": curr_node_id,
                            "port_id": curr_port.id,
                            "constraint_id": node_port_constraint_counter[char_node_port]
                        })
                    if win_index == instance_window_count - 1 and not end_node_is_end_device:
                        next_node = topology.get_node(next_node_id)
                        if next_node_id in path_behind_mid_node.keys():
                            for forked_path in path_behind_mid_node[next_node_id]:
                                next_next_node_id = forked_path[1]
                                next_port = next_node.get_port_by_neighbor_id(next_next_node_id)
                                next_port.window2last_hop_constraint_info.append({
                                    "node_id": curr_node_id,
                                    "port_id": curr_port.id,
                                    "constraint_id": node_port_constraint_counter[char_node_port]
                                })
                    # print(node_port_constraint_counter)
                    # print(char_node_port, curr_port.window2queue_set)
                    # print(char_node_port, curr_port.window2last_hop_constraint_info)
                    win_length = stream.size * 8 * 1000 / curr_port.port_speed + curr_node.proc_delay # unit: ns

                    # print(stream_omega == stream_alpha + win_length)
                    '''
                    compute window omega
                    omega[i] = alpha[i] + window_len
                    '''
                    # print(win_index)
                    # print(stream_omega == stream_alpha + win_length)
                    qbv_solver.add(stream_omega == stream_alpha + win_length)
                    # map stream constraints to port constraints
                    '''
                    compute mapping relationship from stream to ports
                    '''
                    curr_constraint_index = node_port_constraint_counter[char_node_port]
                    # print(win_index)
                    # print(stream_alpha == curr_port.alpha_set[curr_constraint_index])
                    # print(stream_omega == curr_port.omega_set[curr_constraint_index])
                    qbv_solver.add(stream_alpha == curr_port.alpha_set[curr_constraint_index])
                    qbv_solver.add(stream_omega == curr_port.omega_set[curr_constraint_index])
    print(f"mapping {len(qbv_solver.assertions())} constrains.")


def constraints_constructor(qbv_solver, topology, streams):
    print("============== create constraints of solver ==============")
    begin_time = time.time_ns()
    sanone_constraint = 20000 # 20000
    for streams_in_m in streams:
        flag_m_first = True
        node_omega_dict = dict()
        first_seg_instance2first_alpha = dict()
        for stream in streams_in_m:
            instance_window_count = stream.get_current_route().get_windows_len()
            flag_jitter = 0
            curr_first_alpha = 0

            for stream_instance in stream.streamInstances:
                # constraint 0:
                # multicast fisrt node: offset = cycleid * interval
                # not multicast first nodes: offset > last_omega_by_node
                if flag_m_first is True:
                    constr = stream_instance.offset == stream_instance.cycle_id * stream.interval
                else:
                    '''
                    sanone constraint
                    '''
                    constr = stream_instance.offset >= node_omega_dict[stream.src_node_id][stream_instance.cycle_id] + sanone_constraint # + 20000
                qbv_solver.add(constr)
                # constraint 1:
                # offset(period*interval) <= alpha[0] && omega[win_count - 1] <= offset(period*interval) + interval
                first_alpha = stream_instance.alpha_set[0]
                if flag_m_first is True:
                    first_seg_instance2first_alpha[stream_instance.cycle_id] = first_alpha

                last_omega = stream_instance.omega_set[instance_window_count - 1]
                if stream.dst_node_id not in node_omega_dict.keys():
                    node_omega_dict[stream.dst_node_id] = dict()
                node_omega_dict[stream.dst_node_id][stream_instance.cycle_id] = last_omega #
                constr = And(first_alpha >= stream_instance.offset,
                             last_omega <= stream_instance.cycle_id * stream.interval + stream.interval)
                qbv_solver.add(constr)
                # constraint 4: jitter
                # interval + max_jitter >= alphai - alphai-1 >= interval - max_jitter
                if flag_jitter == 0:
                    flag_jitter = 1
                else:
                    qbv_solver.add(And(first_alpha - curr_first_alpha >= stream.interval - stream.max_jitter,
                                       first_alpha - curr_first_alpha <= stream.interval + stream.max_jitter))
                curr_first_alpha = first_alpha

                # constraint 2:
                # alpha[i+1] - alpha[i] >= window_len
                for win_index in range(instance_window_count - 1):
                    curr_alpha = stream_instance.alpha_set[win_index]
                    next_alpha = stream_instance.alpha_set[win_index+1]
                    curr_node_id = stream.get_current_route().path[win_index]
                    curr_node = topology.get_node(curr_node_id)
                    next_node_id = stream.get_current_route().path[win_index + 1]
                    curr_port = curr_node.get_port_by_neighbor_id(next_node_id)
                    win_length = stream.size * 8 * 1000 / curr_port.port_speed + curr_node.proc_delay  # unit: ns

                    qbv_solver.add(next_alpha - curr_alpha >= win_length + sanone_constraint) # + 20000

                # constraint 3:
                # omega[len-1] - alpha[0] <= max_latency
                if topology.get_node(stream.routes[0].path[-1]).end_device:
                    # qbv_solver.add(And(last_omega - first_seg_instance2first_alpha[stream_instance.cycle_id] <= stream.max_latency,
                    #                    last_omega <= stream_instance.cycle_id * stream.interval + stream.interval))
                    qbv_solver.add(last_omega - first_seg_instance2first_alpha[stream_instance.cycle_id] <= stream.max_latency)
            flag_m_first = False

    for node in topology.nodes:
        for port in node.ports:
            for c_i in range(0, len(port.window2queue_set) - 1):
                for c_j in range(c_i + 1, len(port.window2queue_set)):
                    # constraint 4:
                    # alpha[i] >= omega[j] or alpha[j] >= omega[i]
                    # print(c_i, c_j, Or(
                    #     port.alpha_set[c_i] >= port.omega_set[c_j],
                    #     port.alpha_set[c_j] >= port.omega_set[c_i]
                    # ))
                    '''
                    sanone constraint
                    '''
                    qbv_solver.add(Or(
                        port.alpha_set[c_i] >= port.omega_set[c_j] + sanone_constraint, # add something, 20000
                        port.alpha_set[c_j] >= port.omega_set[c_i] + sanone_constraint  # + 20000
                    ))
                    # constraint 5:
                    # if alpha[i] < alpha[j]
                    #     last_omega[i] < last_alpha[j]
                    # condition: same queue, not end device, last_hop not same
                    if port.window2queue_set[c_i] == port.window2queue_set[c_j] and \
                       len(port.window2last_hop_constraint_info) != 0 and \
                       port.window2last_hop_constraint_info[c_i]['node_id'] != port.window2last_hop_constraint_info[c_j]['node_id']:
                        # get last hops
                        last_node_id_i = port.window2last_hop_constraint_info[c_i]['node_id']
                        last_node_id_j = port.window2last_hop_constraint_info[c_j]['node_id']
                        last_port_id_i = port.window2last_hop_constraint_info[c_i]['port_id']
                        last_port_id_j = port.window2last_hop_constraint_info[c_j]['port_id']
                        last_constr_id_i = port.window2last_hop_constraint_info[c_i]['constraint_id']
                        last_constr_id_j = port.window2last_hop_constraint_info[c_j]['constraint_id']
                        last_alpha_i = topology.get_node(last_node_id_i).get_port(last_port_id_i).alpha_set[
                            last_constr_id_i]
                        last_omega_i = topology.get_node(last_node_id_i).get_port(last_port_id_i).omega_set[
                            last_constr_id_i]
                        last_alpha_j = topology.get_node(last_node_id_j).get_port(last_port_id_j).alpha_set[
                            last_constr_id_j]
                        last_omega_j = topology.get_node(last_node_id_j).get_port(last_port_id_j).alpha_set[
                            last_constr_id_j]
                        condition = port.alpha_set[c_i] < port.alpha_set[c_j]
                        then_value = last_omega_i < last_alpha_j
                        else_value = last_omega_j < last_alpha_i

                        qbv_solver.add(If(condition, then_value, else_value))
    end_time = time.time_ns()
    print(f"creating constraints takes {(end_time - begin_time) / 1000000000}s")


def constrains_solver(qbv_solver):
    print(f"============== solve {len(qbv_solver.assertions())} constraints ==============")
    begin_time = time.time_ns()
    res = qbv_solver.check()
    end_time = time.time_ns()
    print(f"checking takes {(end_time - begin_time) / 1000000000}s")
    if res == sat:
        m = qbv_solver.model()
        solution = {}
        for declare in m.decls():
            name = declare.name()
            value = m[declare]
            solution[name] = value
        return solution
    elif res == unsat:
        print("no solution!!!")
        return None
    elif res == unknown:
        print("solve failed!!!")
        None


def parse_solution_topo(solution: dict, topology: Topology):
    print("============== parse ports timeline ==============")
    port_res = dict()
    for node in topology.nodes:
        for port in node.ports:
            window_count = len(port.window2queue_set)
            name_alpha = str(port.alpha_set.decl())
            name_omega = str(port.omega_set.decl())
            if name_alpha in solution.keys():
                res_alpha = solution[name_alpha]
                res_omega = solution[name_omega]
                alpha_list = list(simplify(res_alpha[i]).as_long() for i in range(window_count))
                omega_list = list(simplify(res_omega[i]).as_long() for i in range(window_count))
                new_list = [[alpha_list[i], omega_list[i], port.window2queue_set[i], node.end_device] for i in range(window_count)]
                sorted_array = sorted(new_list, key=lambda x: x[0])
                port_res[f'node_{node.id}_port_{port.id}_{port.used_bandwidth:.0f}-{port.occupied_bandwidth:.2f}Mbps'] = sorted_array
    return port_res


def parse_solution_stream(solution: dict, streams: list[Stream]):
    print("============== parse stream timeline ==============")
    stream_res = dict()
    for streams_in_m in streams:
        m_uni_id = 0
        instance_res = list()
        for stream in streams_in_m:
            if stream.get_current_route() is None:
                continue
            instance_window_count = stream.get_current_route().get_windows_len()
            for stream_instance in stream.streamInstances:
                name_alpha = str(stream_instance.alpha_set.decl())
                name_omega = str(stream_instance.omega_set.decl())
                if name_alpha in solution.keys():
                    res_alpha = solution[name_alpha]
                    res_omega = solution[name_omega]
                    alpha_list = list(simplify(res_alpha[i]).as_long() for i in range(instance_window_count))
                    omega_list = list(simplify(res_omega[i]).as_long() for i in range(instance_window_count))
                    new_list = [[alpha_list[i], omega_list[i], stream.get_current_route().path[i], stream_instance.cycle_id, m_uni_id] for i in range(instance_window_count)]
                    instance_res.extend(new_list)
            m_uni_id += 1
        stream_res[f'stream_{stream.stream_id}_src_{stream.src_node_id}_dst_{stream.multicast_id}_pcp_{stream.pcp}'] = instance_res
    return stream_res


def seg_mstreams(topology: Topology, mstreams: list[MStream]):
    print("============== segmenting multicast traffic ==============")
    hyper_period = compute_hyper_period(mstreams)
    topology_graph = check_and_draw_topology(topology)
    streams = list()
    uni_stream_id = 0
    for mstream in mstreams:
        streams_in_m = list()
        best_paths = list() # to be segmented
        vlan_id = mstream.vlan_id
        for dst_node_id in mstream.dst_node_ids:
            paths = list(networkx.all_simple_paths(topology_graph, source=mstream.src_node_id, target=dst_node_id))
            paths = sorted(paths, key=len)
            available_paths = paths.copy()
            for path in paths:
                for index in range(len(path) - 1):
                    if index != 0 and topology.get_node(path[index]).end_device == 1:
                        available_paths.remove(path)
                        break
                    if vlan_id not in topology.get_node(path[index]).get_port_by_neighbor_id(
                            path[index + 1]).allowed_vlans:
                        available_paths.remove(path)
                        break
            if len(available_paths) == 0:
                print(f"==>WARNING: no viable path from {mstream.src_node_id} to {dst_node_id}!")
                print("             Please check stream and topology settings.")
            # calculate average bandwidth and select path
            # TODO: select vlan by path
            best_path = select_best_path(topology, available_paths, mstream, hyper_period)
            best_paths.append(best_path)
        seg_paths = seg_by_paths(best_paths)
        # create Stream for segmented stream
        if not isinstance(seg_paths[0], list):
            seg_paths = [seg_paths]
        for path in seg_paths:
            stream = Stream(stream_id=mstream.id,
                            uni_stream_id=uni_stream_id,
                            size=mstream.size,
                            interval=mstream.interval,
                            vlan_id=mstream.vlan_id,
                            pcp=mstream.pcp,
                            src_node_id=path[0],
                            multicast_id=mstream.multicast_id,
                            dst_node_id=path[len(path)-1],
                            max_latency=mstream.max_latency,
                            max_jitter=mstream.max_jitter)
            stream.add_route(Route(stream.uni_stream_id, path))
            stream.curr_src_port_id = topology.get_node(path[0]).get_port_by_neighbor_id(path[1]).id
            stream.curr_dst_port_id = topology.get_node(path[len(path) - 1]).get_port_by_neighbor_id(path[len(path) - 2]).id
            for cycle_id in range(int(hyper_period / stream.interval)):
                stream.add_stream_instance(StreamInstance(stream.uni_stream_id, cycle_id))
            instance_window_count = stream.get_current_route().get_windows_len()
            for win_index in range(instance_window_count):
                curr_node_id = stream.get_current_route().path[win_index]
                curr_node = topology.get_node(curr_node_id)
                next_node_id = stream.get_current_route().path[win_index + 1]
                curr_port = curr_node.get_port_by_neighbor_id(next_node_id)
                curr_port.add_used_bandwidth(stream.size * 8 * 1000 / stream.interval)
                occupied_window_len_in_hyper = (stream.size * 8 * 1000 / curr_port.port_speed + curr_node.proc_delay) * \
                                               int(hyper_period / stream.interval)
                curr_port.add_occupied_bandwidth(occupied_window_len_in_hyper * 100 / hyper_period) # 100?
            print("==>Uni stream id:", stream.uni_stream_id, "  Vlan:", stream.vlan_id,
                  "\n   Current Paths:", stream.routes[0].path)
            streams_in_m.append(stream)
            uni_stream_id += 1
        streams.append(streams_in_m)
    # print bandwidth occupancy rate
    max_occupied_bandwidth = 0
    bw_list = list()
    for node in topology.nodes:
        print(f"bandwidth usage of node {node.id}")
        for port in node.ports:
            max_occupied_bandwidth = max(max_occupied_bandwidth, port.occupied_bandwidth)
            if port.occupied_bandwidth > 0:
                bw_list.append(port.occupied_bandwidth)
                print(f"    port {port.id}: stream bandwidth: {port.used_bandwidth:.2f}M, occupied bandwidth: {port.occupied_bandwidth:.2f}M")
            if port.occupied_bandwidth >= port.port_speed:
                print("WARNING! Required bandwidth exceeds")
                return False
    if max_occupied_bandwidth >= 100:
        print(f"ERROR! occupied bandwidth:{max_occupied_bandwidth:.2f}M out of port capacity")
        return [], False
    print(f"maximum occupied bandwidth is {max_occupied_bandwidth:.2f}M, and average bandwidth usage is {sum(bw_list) / len(bw_list):.2f}M")
    return streams, True


# Greedy Algorithm
def select_best_path(topology, available_paths, new_stream, hyper_period):
    # cmp order: max_bandwidth > average bandwidth * path length ?
    cmp_dict = dict()
    for available_path in available_paths:
        new_max_bandwidth = 0
        new_bandwidth_list = []
        for curr_node_id_index in range(len(available_path) - 1):
            curr_node = topology.get_node(available_path[curr_node_id_index])
            curr_port = curr_node.get_port_by_neighbor_id(available_path[curr_node_id_index + 1])
            occupied_window_len_in_hyper = (new_stream.size * 8 * 1000 / curr_port.port_speed + curr_node.proc_delay) * \
                                           int(hyper_period / new_stream.interval)
            new_port_occupied_bandwidth = curr_port.occupied_bandwidth + occupied_window_len_in_hyper * 100 / hyper_period
            new_bandwidth_list.append(new_port_occupied_bandwidth)
            new_max_bandwidth = max(new_max_bandwidth, new_port_occupied_bandwidth)
        aver_bandwidth_cmp = sum(new_bandwidth_list) / len(new_bandwidth_list)
        cmp_dict[available_paths.index(available_path)] = [new_max_bandwidth, aver_bandwidth_cmp, len(available_path)]
        print(available_path, new_max_bandwidth, aver_bandwidth_cmp, len(available_path))
    sorted_cmp_dict = sorted(cmp_dict.items(), key=lambda x: (x[1][0], x[1][2] * x[1][1]))
    return available_paths[sorted_cmp_dict[0][0]]



def seg_by_paths(paths):
    if len(paths) == 1:
        return paths[0]
    res = list()
    paths_dict = dict()
    f = 1
    while f < len(paths[0]):
        cmp = paths[0][f]
        flag = True
        for path in paths:
            if path[f] != cmp:
                flag = False
                break
        if flag is True:
            f += 1
        else:
            break
    res.append(paths[0][:f])
    for path in paths:
        if path[f] in paths_dict.keys():
            paths_dict[path[f]].append(path[f - 1:])
        else:
            paths_dict[path[f]] = [path[f - 1:]]
    for s_paths in paths_dict.values():
        res_path = seg_by_paths(s_paths)
        res.append(res_path)
    return res


def compute_hyper_period(streams: list):
    interval_list = list()
    for stream in streams:
        interval_list.append(stream.interval)
    return _compute_hyper_period(interval_list)


def _compute_hyper_period(interval_list: list[int]):
    hyper_period = 1
    for interval in interval_list:
        hyper_period = int(interval) * int(hyper_period) / math.gcd(int(interval), int(hyper_period))
    return hyper_period
