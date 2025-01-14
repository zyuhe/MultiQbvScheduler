'''
-*- coding: utf-8 -*-
@Time    :   2024/12/21 13:35
@Author  :   zyh
@Email   :   
@Project :   MultiQbvScheduler
@File    :   StreamGraph.py
'''
import numpy as np
from typing import List
from common.StreamBase import MStream


class StreamGraph:
    def __init__(self, mstreams: List[MStream], distances):
        self.mstreams = mstreams
        self.distances = distances
        self.num_streams = len(distances)
        self.pheromones = np.ones_like(distances, dtype=float)

    def clear_mstreams_winInfo(self):
        for mstream in self.mstreams:
            mstream.clean_winInfo()
