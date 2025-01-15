'''
-*- coding: utf-8 -*-
@Time    :   2024/12/23 12:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   funcs.py
'''

import math
import numpy as np

from common.StreamBase import MStream
from common.TopologyBase import TopologyBase

'''
input:[list(path1), list(path2),]
'''
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

def all_1d_arrays(lists):
    for element in lists:
        if not (isinstance(element, list) and all(not isinstance(item, list) for item in element)):
            return False
    return True

def split_to_1d_arrays(nested_list):
    try:
        result = []
        for element in nested_list:
            if isinstance(element, list) and all(not isinstance(item, list) for item in element):
                result.append(element)
            else:
                result.extend(element)

        return result
    except Exception as e:
        print("error", nested_list)

def compute_seg_path_dict(paths):
    res = seg_by_paths(paths)
    if not isinstance(res[0], list):
        res = [res]
    seg_dict = dict()
    while not all_1d_arrays(res):
        res = split_to_1d_arrays(res)
    for seg in res:
        if seg[0] not in seg_dict.keys():
            seg_dict[seg[0]] = [seg]
        else:
            seg_dict[seg[0]].append(seg)
    return seg_dict

def update_node_neigh_info(topology: TopologyBase, mstream: MStream):
    # print("mstream.seg_path_dict", mstream.seg_path_dict)
    value_list = list(mstream.seg_path_dict.values())
    mstream.windowsInfo[value_list[0][0][0]] = [-1, -1]
    for v in value_list:
        for seg in v:
            for i in range(len(seg) - 1):
                pre_node_id = seg[i]
                pre_port = topology.get_node(pre_node_id).get_port_by_neighbor_id(seg[i + 1])
                mstream.windowsInfo[seg[i + 1]] = [pre_node_id, pre_port.id]

def update_node_win_info(topology: TopologyBase, mstream: MStream, win_plus, total_latency):
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
                    tnode = topology.get_node(seg[i + 1])
                    if tnode.end_device == 1:
                        last_hop = True
                node = topology.get_node(seg[i])
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
                win_len = math.ceil(mstream.size * 8 * 1000 / port.port_speed + win_plus)
                # print("mstream.size", mstream.size, "bytes, port.speed", port.port_speed, "Mbps, window length", win_len, "us")
                for ti in range(times):
                    # print("ti =", ti)
                    # get pre node's gate close time as temp ts open
                    tmp_ts_open = ti * mstream.interval
                    first_node = True
                    # print(mstream.windowsInfo, seg[i])
                    if mstream.windowsInfo[seg[i]][0] >= 0:
                        pre_node = topology.get_node(mstream.windowsInfo[seg[i]][0])
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
                            return -1
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
                        return -1
                # print("=============")
    # compute delay
    add_latency = sum(delays) / (len(delays) / len(mstream.dst_node_ids))
    total_latency += add_latency
    # print("mstream：", mstream.id, "src：", mstream.src_node_id, "dst：", mstream.multicast_id, "delay result：")
    # print("delays：", delays, "dst num", len(mstream.dst_node_ids), "hyper times", (len(delays) / len(mstream.dst_node_ids)), "now total latency：", total_latency)
    return total_latency

