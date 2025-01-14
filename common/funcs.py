'''
-*- coding: utf-8 -*-
@Time    :   2024/12/23 12:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   funcs.py
'''

import numpy as np

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
