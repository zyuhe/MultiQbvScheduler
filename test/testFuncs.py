'''
-*- coding: utf-8 -*-
@Time    :   2024/12/23 12:25
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   testFuncs.py
'''
from common.funcs import seg_by_paths, catch_segs


def test_seg_by_paths():
    paths = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6], [0, 1, 7, 8, 9]]
    # paths_dic = [[0,1,2,3]]
    res = seg_by_paths(paths)
    print(res)
    # [[0, 1], [[1, 2, 3, 4], [4, 5], [4, 6]], [1, 7, 8, 9]]

def test_compute_seg_path_dict():
    paths = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6], [0, 1, 7, 8, 9]]
    res = catch_segs(paths)
    print(res)
    # {0: [[0, 1]], 1: [[1, 2, 3, 4], [1, 7, 8, 9]], 4: [[4, 5], [4, 6]]}

if __name__ == '__main__':
    # test_seg_by_paths()
    test_compute_seg_path_dict()
